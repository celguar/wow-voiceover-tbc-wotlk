# import tts.vvembed.modular.modeling_vibevoice_inference
import glob
import hashlib
import json
import logging
import re
import shutil
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import random

import gradio
import numpy as np
import pandas as pd
import torch
import torchaudio
# from pydub import AudioSegment
from scipy.io import wavfile
# from scipy import signal
from slpp import slpp as lua
from tqdm import tqdm

from difflib import SequenceMatcher

import subprocess
from multiprocessing import Pool, cpu_count

import re, os
from num2words import num2words

from pydub import AudioSegment, silence
from pydub.effects import normalize

from scipy import signal
import soundfile as sf
import concurrent.futures

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name

_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)

from data_prep.env_vars import MODELS_DIR, ADDON_PATH, VOICE_SAMPLE_PATH, VOICE_PROMPTS_PATH, MODULE_NAMES, ADDON_PATH_EXPORT, ADDON_PATH_WORKS
from omnivoice import OmniVoice
import wowvo_client_omnivoice.audio_effects as ae
from wowvo_client_omnivoice.consts import RACE_DICT, GENDER_DICT, VOICE_MODEL_MAP, NPC_EFFECTS
from wowvo_client_omnivoice.length_table import write_sound_length_table_lua
from wowvo_client_omnivoice.utils import get_first_n_words, get_last_n_words, replace_dollar_bs_with_space
from data_prep.sql_queries import make_connection, make_connection_turtle, make_connection_vmangos, get_vanilla_quest_text

INTERRUPTION_SUPPORT = False

if torch.cuda.is_available():
    print(f"Using CUDA: YES")
else:
    print(f"Using CUDA: NO")

# from transformers import AutoFeatureExtractor, VibeVoiceAcousticTokenizerModel
# from transformers import VibeVoiceAcousticTokenizerModel, VibeVoiceAcousticTokenizerConfig

# from tts.vvembed.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference

# current_dir = os.path.dirname(os.path.abspath(__file__))
# vvembed_path = os.path.join(os.path.dirname(current_dir), 'tts', 'vvembed')
# if vvembed_path not in sys.path:
#     sys.path.insert(0, vvembed_path)
#
# from transformers import AutoConfig
#
# try:
#     from tts.vvembed.modular.configuration_vibevoice import VibeVoiceAcousticTokenizerConfig
#
#     # Пытаемся зарегистрировать его вручную
#     try:
#         AutoConfig.register("vibevoice_acoustic_tokenizer", VibeVoiceAcousticTokenizerConfig)
#         print("✅ Registered VibeVoice config manually")
#     except ValueError:
#         # Если уже зарегистрировано — это то, что нам нужно!
#         print("ℹ️ VibeVoice config already in registry, skipping...")
#
# except ImportError as e:
#     print(f"⚠️ Could not import VibeVoiceAcousticTokenizerConfig for registration: {e}")

# Setup logging
logging.basicConfig(
    level=logging.INFO
)
logger = logging.getLogger("OmniVoice")
logger.setLevel(logging.INFO)
logging.getLogger("faster_whisper").setLevel(logging.WARN)

models_dir = MODELS_DIR

def apply_timm_compatibility_patches():
    """Apply compatibility patches for timm package conflicts"""
    try:
        import timm.data
        
        # Patch missing functions that cause import errors
        patches = {
            'ImageNetInfo': lambda: type('ImageNetInfo', (), {'__init__': lambda self: None})(),
            'infer_imagenet_subset': lambda class_to_idx: 'imagenet',
            'get_imagenet_subset_labels': lambda *args, **kwargs: [],
            'get_imagenet_subset_info': lambda *args, **kwargs: {},
            'resolve_data_config': lambda *args, **kwargs: {}
        }
        
        for attr_name, patch_func in patches.items():
            if not hasattr(timm.data, attr_name):
                if attr_name == 'ImageNetInfo':
                    setattr(timm.data, attr_name, type('ImageNetInfo', (), {'__init__': lambda self: None}))
                else:
                    setattr(timm.data, attr_name, patch_func)
        
        return True
    except Exception as e:
        return False

def check_embedded_vibevoice():
    """Check if embedded VibeVoice is available"""
    vvembed_path = 'tts/vvembed'
    if not os.path.exists(vvembed_path):
        logger.error(f"Embedded VibeVoice not found at {vvembed_path}")
        return False
    
    # Add vvembed to path if not already there
    if vvembed_path not in sys.path:
        sys.path.insert(0, vvembed_path)
    
    logger.warning("Using embedded VibeVoice (MIT licensed)")
    return True

def ensure_dependencies():
    """Ensure required dependencies are installed"""
    try:
        import transformers
        from packaging import version
        if version.parse(transformers.__version__) < version.parse("4.44.0"):
            logger.warning("Transformers version < 4.44.0, some features may not work correctly")
    except ImportError:
        logger.warning("Transformers not installed. Please install: pip install transformers>=4.44.0")
        return False
    
    # Apply timm patches if needed
    apply_timm_compatibility_patches()
    
    return True


# if check_embedded_vibevoice() and ensure_dependencies():
#     logger.warning("VibeVoice Loaded")
# else:
#     logger.error("Failed to load VibeVoice or dependencies!")

# Import for interruption support
# try:
#     import execution
#     INTERRUPTION_SUPPORT = True
# except ImportError:
#     INTERRUPTION_SUPPORT = False
#     logger.warning("Interruption support not available")
#
# # Check for SageAttention availability
# try:
#     from sageattention import sageattn
#     SAGE_AVAILABLE = True
#     logger.info("SageAttention available for acceleration")
# except ImportError:
#     SAGE_AVAILABLE = False
#     logger.debug("SageAttention not available - install with: pip install sageattention")

def get_optimal_device():
    """Get the best available device (cuda, mps, or cpu)"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def get_device_map():
    """Get device map for model loading"""
    device = get_optimal_device()
    # Note: device_map "auto" might work better for MPS in some cases
    return device if device != "mps" else "mps"

# Cache for model scanning to avoid repeated scans
_model_cache = {
    "models": None,
    "last_scan_time": 0,
    "cache_duration": 5,  # Cache for 5 seconds
    "first_load_logged": False  # Track if we've logged the initial scan
}

def get_available_models() -> List[Tuple[str, str]]:
    """Scan models/vibevoice/ directory and return available models

    Returns:
        List of tuples (display_name, folder_path)
    """
    import time

    # Check if we have a valid cache
    current_time = time.time()
    if (_model_cache["models"] is not None and
        current_time - _model_cache["last_scan_time"] < _model_cache["cache_duration"]):
        # Return cached results
        return _model_cache["models"]

    try:
        vibevoice_dir = models_dir

        if not os.path.exists(vibevoice_dir):
            os.makedirs(vibevoice_dir, exist_ok=True)
            logger.info(f"Created vibevoice models directory: {vibevoice_dir}")
            _model_cache["models"] = []
            _model_cache["last_scan_time"] = current_time
            return ["OmniVoice"]

        # First, collect all valid model folders
        valid_folders = []
        logger.debug(f"Scanning vibevoice directory: {vibevoice_dir}")
        for folder in os.listdir(vibevoice_dir):
            folder_path = os.path.join(vibevoice_dir, folder)

            # Skip hidden folders, loras, and non-directories
            if folder.startswith(".") or folder == "loras" or not os.path.isdir(folder_path):
                logger.debug(f"Skipping: {folder}")
                continue

            logger.debug(f"Checking folder: {folder}")
            # Check if it's a valid model folder
            if is_valid_model_folder(folder_path):
                valid_folders.append(folder)
            else:
                logger.debug(f"Folder {folder} is not a valid model folder")

        # Now transform folder names with duplicate detection
        models = []
        models.append(("OmniVoice", "OmniVoice"))
        for folder in valid_folders:
            display_name = transform_folder_name(folder, valid_folders)
            models.append((folder, display_name))
            logger.info(f"Found model: {display_name} in folder: {folder}")

        # Sort by display name for consistent ordering
        models.sort(key=lambda x: x[1])

        # Only log on first scan to avoid spam
        if not _model_cache["first_load_logged"]:
            if not models:
                logger.warning("No valid models found in vibevoice directory")
                logger.info(f"Please download models to: {vibevoice_dir}")
            else:
                # Single summary message instead of individual logs
                logger.info(f"Found {len(models)} VibeVoice model(s) available")
            _model_cache["first_load_logged"] = True

        # Cache the results
        _model_cache["models"] = models
        _model_cache["last_scan_time"] = current_time

        return models

    except Exception as e:
        logger.error(f"Error scanning models directory: {e}")
        # Cache empty result on error to avoid repeated failures
        _model_cache["models"] = []
        _model_cache["last_scan_time"] = current_time
        return []

def extract_model_info(folder: str) -> Tuple[str, Optional[str]]:
    """Extract model name and author from folder name

    Args:
        folder: Folder name

    Returns:
        Tuple of (model_name, author_name)

    Examples:
        models--microsoft--VibeVoice-Large -> ('VibeVoice-Large', 'microsoft')
        models--aoi-ot--VibeVoice-Large -> ('VibeVoice-Large', 'aoi-ot')
        VibeVoice-1.5B -> ('VibeVoice-1.5B', None)
    """
    if "--" in folder:
        # HuggingFace cache format: models--author--model
        parts = folder.split("--")
        if len(parts) >= 3:
            author = parts[1]
            model = parts[-1]
            return model, author
        elif len(parts) == 2:
            return parts[-1], None
    return folder, None

def transform_folder_name(folder: str, all_folders: List[str]) -> str:
    """Transform folder name for display, adding author if there are duplicates

    Args:
        folder: Current folder name
        all_folders: List of all folder names to check for duplicates

    Returns:
        Display name with author in parentheses if needed
    """
    model_name, author = extract_model_info(folder)

    # Check if there are other folders with the same model name
    has_duplicate = False
    for other_folder in all_folders:
        if other_folder != folder:
            other_model_name, _ = extract_model_info(other_folder)
            if other_model_name == model_name:
                has_duplicate = True
                break

    # Add author in parentheses if there are duplicates and author is known
    if has_duplicate and author:
        return f"{model_name} ({author})"

    return model_name

def check_folder_has_model_files(folder_path: str) -> bool:
    """Check if a folder directly contains model files (not recursively)

    Args:
        folder_path: Path to check

    Returns:
        True if folder contains config.json and model files
    """
    if not os.path.isdir(folder_path):
        return False

    has_config = os.path.exists(os.path.join(folder_path, "config.json"))
    if not has_config:
        return False

    # Check for various model file formats
    files = os.listdir(folder_path)
    has_model = (
        "pytorch_model.bin" in files or
        "model.safetensors" in files or
        "pytorch_model.bin.index.json" in files or
        "model.safetensors.index.json" in files or
        any(f.startswith("pytorch_model-") and f.endswith(".bin") for f in files) or
        any(f.startswith("model-") and f.endswith(".safetensors") for f in files)
    )

    return has_model

def is_valid_model_folder(folder_path: str, max_depth: int = 4, current_depth: int = 0) -> bool:
    """Recursively check if a folder contains a valid VibeVoice model

    Args:
        folder_path: Path to the folder to check
        max_depth: Maximum recursion depth (default 3)
        current_depth: Current recursion depth

    Returns:
        True if folder or any subfolder contains valid model files
    """
    if current_depth >= max_depth:
        return False

    # Check if current folder has model files
    if check_folder_has_model_files(folder_path):
        return True

    # Recursively check subfolders
    try:
        for item in os.listdir(folder_path):
            # Skip hidden folders and specific folders we want to ignore
            if item.startswith(".") or item in ["loras", "__pycache__"]:
                continue

            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                # Recursively check subfolder
                if is_valid_model_folder(item_path, max_depth, current_depth + 1):
                    return True
    except (PermissionError, OSError):
        pass

    return False

def find_model_files_path_recursive(folder_path: str, max_depth: int = 4, current_depth: int = 0) -> Optional[str]:
    """Recursively find the path containing model files

    Args:
        folder_path: Path to search from
        max_depth: Maximum recursion depth
        current_depth: Current recursion depth

    Returns:
        Path to the directory containing model files, or None
    """
    if current_depth >= max_depth:
        return None

    # Check if current folder has model files
    if check_folder_has_model_files(folder_path):
        return folder_path

    # Recursively check subfolders
    try:
        for item in os.listdir(folder_path):
            # Skip hidden folders and specific folders we want to ignore
            if item.startswith(".") or item in ["loras", "__pycache__"]:
                continue

            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                # Recursively check subfolder
                result = find_model_files_path_recursive(item_path, max_depth, current_depth + 1)
                if result:
                    return result
    except (PermissionError, OSError):
        pass

    return None

def find_model_files_path(model_folder: str) -> Optional[str]:
    """Find the actual path containing model files

    Args:
        model_folder: Name of the folder in vibevoice directory

    Returns:
        Path to the directory containing model files, or None
    """
    try:
        # import folder_paths
        # models_dir = folder_paths.get_folder_paths("checkpoints")[0]
        # vibevoice_dir = os.path.join(os.path.dirname(models_dir), "vibevoice")
        base_path = model_folder #os.path.join(vibevoice_dir, model_folder)

        # Use recursive search to find model files
        result = find_model_files_path_recursive(base_path)

        if result:
            logger.info(f"Found model files at: {result}")
        else:
            logger.warning(f"No valid model files found for: {model_folder}")

        return result

    except Exception as e:
        logger.error(f"Error finding model files: {e}")
        return None

def find_qwen_tokenizer_path(MODELS_DIR: str) -> Optional[str]:
    """Find Qwen tokenizer using priority system

    Priority:
    1. ComfyUI/models/vibevoice/tokenizer/
    2. ComfyUI/models/vibevoice/models--Qwen--Qwen2.5-1.5B/
    3. HuggingFace cache (if exists)

    Returns:
        Path to tokenizer directory or None
    """
    # Priority 1: Check tokenizer folder
    tokenizer_dir = os.path.join(MODELS_DIR, "tokenizer")
    if os.path.exists(tokenizer_dir):
        # Check if it contains tokenizer files
        required_files = ["tokenizer_config.json", "vocab.json", "merges.txt"]
        if all(os.path.exists(os.path.join(tokenizer_dir, f)) for f in required_files):
            logger.info(f"Found Qwen tokenizer in: {tokenizer_dir}")
            return tokenizer_dir

    # Priority 2: Check models--Qwen--Qwen2.5-1.5B folder
    qwen_model_dir = os.path.join(MODELS_DIR, "models--Qwen--Qwen2.5-1.5B")
    if os.path.exists(qwen_model_dir):
        # Check snapshots folder
        snapshots_dir = os.path.join(qwen_model_dir, "snapshots")
        if os.path.exists(snapshots_dir):
            for snapshot in os.listdir(snapshots_dir):
                snapshot_path = os.path.join(snapshots_dir, snapshot)
                if os.path.isdir(snapshot_path):
                    # Check if it contains tokenizer files
                    if os.path.exists(os.path.join(snapshot_path, "tokenizer_config.json")):
                        logger.info(f"Found Qwen tokenizer in model cache: {snapshot_path}")
                        return snapshot_path

    # Priority 3: Check HuggingFace cache
    hf_cache_paths = [
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.path.join(os.environ.get("HF_HOME", ""), "hub") if os.environ.get("HF_HOME") else None,
    ]

    for cache_path in hf_cache_paths:
        if cache_path and os.path.exists(cache_path):
            qwen_cache = os.path.join(cache_path, "models--Qwen--Qwen2.5-1.5B")
            if os.path.exists(qwen_cache):
                snapshots_dir = os.path.join(qwen_cache, "snapshots")
                if os.path.exists(snapshots_dir):
                    for snapshot in os.listdir(snapshots_dir):
                        snapshot_path = os.path.join(snapshots_dir, snapshot)
                        if os.path.isdir(snapshot_path):
                            if os.path.exists(os.path.join(snapshot_path, "tokenizer_config.json")):
                                logger.info(f"Found Qwen tokenizer in HF cache: {snapshot_path}")
                                return snapshot_path

    return None

def detect_model_quantization(model_path: str) -> Optional[str]:
    """Detect if model is quantized from config files

    Args:
        model_path: Path to the model directory

    Returns:
        '4bit', '8bit', or None
    """
    try:
        # Check for quantization_config.json first
        quant_config_path = os.path.join(model_path, "quantization_config.json")
        if os.path.exists(quant_config_path):
            with open(quant_config_path, 'r') as f:
                quant_config = json.load(f)
                if quant_config.get("load_in_4bit"):
                    return "4bit"
                if quant_config.get("load_in_8bit"):
                    return "8bit"

        # Check main config.json
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                if "quantization_config" in config:
                    if config["quantization_config"].get("load_in_4bit"):
                        return "4bit"
                    if config["quantization_config"].get("load_in_8bit"):
                        return "8bit"
                    if config["quantization_config"].get("bits") == 4:
                        return "4bit"

    except Exception as e:
        logger.info(f"Could not detect quantization: {e}")

    return None

def find_voice_sample(voice_name, voice_type='standard'):
    samples = {}
    for module in MODULE_NAMES:
        sample_path = Path("voices") / module / f"{voice_name}.mp3"
        if sample_path.exists():
            samples[module] = str(sample_path)

    sample_path = Path("voices") / "GameVoice" / f"{voice_name}_{voice_type}.mp3"
    if voice_type != 'standard':
        if sample_path.exists():
            samples["FromGame"] = str(sample_path)
    else:
        sample_path = Path("voices") / "GameVoice" / f"{voice_name}.mp3"
        if sample_path.exists():
            samples["FromGame"] = str(sample_path)

    return samples

def get_default_sample(voice_name, voice_type='standard'):
    """
    Get default sample path for a given voice name
    
    Args:
        audio_name: Name of the voice (e.g., 'human_male', 'orc_female')
    
    Returns:
        Path to the default sample file if exists, None otherwise
    """
    game_samples = find_voice_sample(voice_name, voice_type)
    if game_samples.get('FromGame'):
        return str(game_samples.get('FromGame'))
    
    voice_sample_default_path = os.path.join(VOICE_SAMPLE_PATH, "default")
    default_file = Path(voice_sample_default_path) / f"{voice_name}.mp3"
    
    if default_file.exists():
        return str(default_file)
    
    return None

def extract_voice_type(voice_type):
    if not voice_type:
        return 'standard'
    
    voice_types = ['standard', 'official', 'noble', 'military', 'nerdy', 'happy', 'young', 'grim', 'maternal', 'guard', 'sentinel', 'priestess', 'shady', 'warrior', 'shaman', 'elder', 'magic', 'dark', 'old', 'zany', 'gruff', 'highelf']
    pattern = re.compile(r'(' + '|'.join(voice_types) + r')', re.IGNORECASE)
    match = pattern.search(str(voice_type))
    return match.group(1).lower() if match else 'standard'

# def merge_short_fragments(chunks, min_length=20):
#     """
#     Merge very short chunks (e.g. 'To.', 'All.') with the previous one
#     if they're too short to stand alone.
#     """
#     merged = []
#     buffer = ""
#
#     for chunk in chunks:
#         stripped = chunk.strip()
#         # If the current chunk is very short, merge it into buffer
#         if len(stripped) < min_length and merged:
#             merged[-1] = merged[-1].rstrip() + " " + stripped  # attach to previous
#         else:
#             merged.append(stripped)
#     return merged
#
#
# def split_into_sentences(text: str, lang: str = "en") -> List[str]:
#     # If text is short enough, skip splitting entirely
#     if len(text) <= 150:
#         return [text]
#
#     # Otherwise, use pysbd for sentence segmentation
#     segmenter = pysbd.Segmenter(language=lang, clean=True)
#     sentences = segmenter.segment(text)
#
#     # Post-process each sentence: remove commas if <= 60 chars
#     sentences = [
#         s.replace(",", "") if len(s) <= 60 else s
#         for s in sentences
#     ]
#     return sentences
#
#
# def get_or_create_latents(model, speaker_key, speaker_wav):
#     if speaker_key not in latents_cache:
#         gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(speaker_wav)
#         latents_cache[speaker_key] = (gpt_cond_latent, speaker_embedding)
#     return latents_cache[speaker_key]

# def load_default():
#     global default_model, vc
#     print("Loading default XTTS model from fine_tuned/base")

#     base_dir = "fine_tuned/base"
#     config_path = os.path.join(base_dir, "config.json")
#     vocab_path = os.path.join(base_dir, "vocab.json")
#     speakers_path = os.path.join(base_dir, "speakers_xtts.pth")

#     config = XttsConfig()
#     config.load_json(config_path)

#     model = Xtts.init_from_config(config)
#     model.load_checkpoint(
#         config,
#         checkpoint_dir=base_dir,
#         speaker_file_path=speakers_path,
#         vocab_path=vocab_path,
#         eval=True,
#     )
#     model.to(DEVICE)

#     default_model = model
#     print(" Default XTTS model loaded from local base folder.")
# load_default()

def get_game_sample_for_npc_id(npc_id):
    filename = f"npc_{npc_id}.mp3"
    full_path = Path(VOICE_SAMPLE_PATH) / "GameVoice" / filename
    if full_path.exists():
        return full_path
    
    return None

def generate_game_sample_for_npc_id(npc_id, module_name):
    
    if module_name == 'AI_VoiceOverData_Turtle':
        db_con = make_connection_turtle()
    else:
        db_con = make_connection()

    def get_row_as_dict(cursor):
        columns = [desc[0] for desc in cursor.description]
        row = cursor.fetchone()
        if row:
            return dict(zip(columns, row))
        return None
    
    if module_name == 'AI_VoiceOverData_Turtle':
        display_column = 'display_id1'
        creature_table = 'creature_template'
        creatureid_column = 'entry'
        npcsoundid_column = 'NPCSoundID'
        creature_di_table = 'creaturedisplayinfo'
    else:
        display_column = 'CreatureDisplayID as display_id1'
        creature_table = 'creature_template_model'
        npcsoundid_column = 'NPCSoundID'
        creatureid_column = 'CreatureID'
        creature_di_table = 'creaturedisplayinfo2'

    def get_npc_sound_files(npc_id: int, db_connection) -> List[Dict[str, str]]:
        result_files = []
        result_files_pissed = []
        
        try:
            with db_connection.cursor() as cursor:
                # 1. display_id1
                cursor.execute(f"""
                    SELECT {display_column} FROM {creature_table} WHERE {creatureid_column} = {npc_id}
                """)
                creature = get_row_as_dict(cursor)
                if not creature:
                    return []
                display_id = creature['display_id1']
                
                # 2. NPCSoundID
                cursor.execute(f"""
                    SELECT NPCSoundID FROM {creature_di_table} WHERE ID = {display_id}
                """)
                display_info = get_row_as_dict(cursor)
                if not display_info:
                    return []
                npc_sound_id = display_info['NPCSoundID']
                
                # 3. SoundID_1-4
                cursor.execute("""
                    SELECT SoundID_1, SoundID_2, SoundID_3
                    FROM dbc_npcsounds WHERE ID = %s
                """, (npc_sound_id,))
                npc_sounds = get_row_as_dict(cursor)
            
                if not npc_sounds:
                    print(f"❌ NPC {npc_id}: dbc_npcsounds entry not found (id={npc_sound_id})")
                    return [], []
            
                # 4. Обрабатываем каждый SoundID
                for i in range(1, 4):
                    sound_id = npc_sounds[f'SoundID_{i}']
                    if not sound_id or sound_id == 0:
                        continue
                    
                    # 5. Ищем в dbc_soundentries (SoundType = 17 - приветствие/прощание)
                    cursor.execute("""
                        SELECT DirectoryBase, File_1, File_2, File_3, File_4
                        FROM dbc_soundentries
                        WHERE ID = %s AND SoundType = 17
                    """, (sound_id,))
                    sound_entry = get_row_as_dict(cursor)
                    
                    if not sound_entry:
                        print(f"⚠️ NPC {npc_id}: sound_id={sound_id} not found or not SoundType=17")
                        continue
                    
                    # 6. Собираем файлы
                    directory = sound_entry['DirectoryBase']
                    for j in range(1, 5):
                        file_name = sound_entry[f'File_{j}']
                        if file_name and file_name.strip():
                            full_path = Path(directory) / file_name
                            if i == 3:
                                result_files_pissed.append({
                                    "full_path": str(full_path),
                                    "sound_id": sound_id,
                                    "file_index": j,
                                    "file_name": file_name,
                                    "directory": directory
                                })
                            else:
                                result_files.append({
                                    "full_path": str(full_path),
                                    "sound_id": sound_id,
                                    "file_index": j,
                                    "file_name": file_name,
                                    "directory": directory
                                })
        
        except Exception as e:
            print(f"❌ Error generating sample for NPC {npc_id}: {e}")
        
        finally:
            db_con.close()
        
        return result_files, result_files_pissed
    
    sound_files, sound_files_pissed  = get_npc_sound_files(npc_id, db_con)

    if not sound_files:
        print(f"❌ NPC {npc_id} sound files not found")
        return None
    
    # Объединяем все найденные звуки в один сэмпл
    combined = AudioSegment.empty()
    combined_pissed = AudioSegment.empty()
    
    base_path = os.path.join(VOICE_SAMPLE_PATH, "FromGame")
    if module_name == 'AI_VoiceOverData_Turtle':
        base_path = os.path.join(base_path, "SoundTurtle")
    else:
        base_path = os.path.join(base_path, "Sound")

    for sf in sound_files:
        relative_path = sf['full_path'].replace("Sound\\", "", 1)
        file_path = os.path.join(base_path, relative_path)
        if Path(file_path).exists():
            print(f"  + Adding: {sf['file_name']}")
            audio = AudioSegment.from_file(file_path)
            combined += audio
            combined += AudioSegment.silent(duration=500)  # Пауза 0.5 сек между звуками
        else:
            print(f"  ⚠️ File not found: {file_path}")
    
    if len(combined) == 0:
        print(f"❌ Failed to load any voice samples for NPC {npc_id}")
        return None
    
    for sf in sound_files_pissed:
        relative_path = sf['full_path'].replace("Sound\\", "", 1)
        file_path = os.path.join(base_path, relative_path)
        if Path(file_path).exists():
            print(f"  + Adding: {sf['file_name']}")
            audio = AudioSegment.from_file(file_path)
            combined_pissed += audio
            combined_pissed += AudioSegment.silent(duration=500)  # Пауза 0.5 сек между звуками
        else:
            print(f"  ⚠️ File not found: {file_path}")
    
    # Сохраняем
    output_subdir = os.path.join(VOICE_SAMPLE_PATH, "GameVoice")
    os.makedirs(output_subdir, exist_ok=True)
    output_path = os.path.join(output_subdir, f"npc_{npc_id}.mp3")
    
    combined.export(output_path, format="mp3", bitrate="64k")
    print(f"✅ Sample created: {output_path} (duration: {len(combined)/1000:.1f} sec)")

    if len(combined_pissed) > 0:
        output_path_pissed = os.path.join(output_subdir, f"npc_{npc_id}_pissed.mp3")
        combined_pissed.export(output_path_pissed, format="mp3", bitrate="64k")
        print(f"✅ Sample created (Pissed): {output_path_pissed} (duration: {len(combined_pissed)/1000:.1f} sec)")
    
    return output_path

from typing import List, Dict, Tuple

def get_quests_by_npc_model_id_lk(npc_id: int, module_name: str) -> Tuple[List[int], List[int]]:
    """
    1. Находит ModelID персонажа в базе текущего модуля (Turtle или vMangos).
    2. По этому ModelID ищет ВСЕ связанные квесты в базе Лича (make_connection).
    """
    # Определяем параметры для поиска ModelID в зависимости от исходного модуля
    if module_name == 'AI_VoiceOverData_Turtle':
        db_con_module = make_connection_turtle()
        creature_table = 'creature_template'
        creatureid_column = 'entry'
        display_column = 'display_id1'
        creature_di_table = 'creaturedisplayinfo'
    else:
        db_con_module = make_connection() # Если это и так лич/бк
        creature_table = 'creature_template_model'
        creatureid_column = 'CreatureID'
        display_column = 'CreatureDisplayID'
        creature_di_table = 'creaturedisplayinfo2'

    model_id = None

    # === ШАГ 1: Достаем ModelID из базы исходного модуля ===
    try:
        with db_con_module.cursor() as cursor:
            query_model = f"""
                SELECT cdi.ModelID 
                FROM {creature_table} ct
                JOIN {creature_di_table} cdi ON ct.{display_column} = cdi.ID
                WHERE ct.{creatureid_column} = %s
                LIMIT 1
            """
            cursor.execute(query_model, (npc_id,))
            row = cursor.fetchone()
            if row:
                model_id = row[0]
    except Exception as e:
        print(f"❌ Ошибка при поиске ModelID в модуле {module_name} для NPC {npc_id}: {e}")
    finally:
        # Если это было отдельное подключение (например, к Черепахе), закрываем его
        if module_name == 'AI_VoiceOverData_Turtle':
            db_con_module.close()

    # Если модельку не нашли, дальше искать в Личе нет смысла
    if model_id is None:
        print(f"⚠️ ModelID для NPC {npc_id} не найден в базе {module_name}.")
        return [], []

    # === ШАГ 2: Ищем квесты по этому ModelID в базе Лича ===
    starters = []
    enders = []
    
    # Всегда подключаемся к основной базе Лича
    db_con_lk = make_connection() 

    try:
        with db_con_lk.cursor() as cursor:
            # Ищем стартеры в Личе по creature_template_model и creaturedisplayinfo2
            query_starters = """
                SELECT DISTINCT qs.quest 
                FROM creature_queststarter qs
                JOIN creature_template_model ct ON qs.id = ct.CreatureID
                JOIN creaturedisplayinfo2 cdi ON ct.CreatureDisplayID = cdi.ID
                WHERE cdi.ModelID = %s
            """
            cursor.execute(query_starters, (model_id,))
            starters = [row[0] for row in cursor.fetchall()]

            # Ищем эндеры в Личе
            query_enders = """
                SELECT DISTINCT qe.quest 
                FROM creature_questender qe
                JOIN creature_template_model ct ON qe.id = ct.CreatureID
                JOIN creaturedisplayinfo2 cdi ON ct.CreatureDisplayID = cdi.ID
                WHERE cdi.ModelID = %s
            """
            cursor.execute(query_enders, (model_id,))
            enders = [row[0] for row in cursor.fetchall()]

    except Exception as e:
        print(f"❌ Ошибка при поиске квестов в базе Лича для ModelID {model_id}: {e}")
    # Будь аккуратен: если твой make_connection() — это глобальный синглтон, 
    # то закрывать его тут руками не нужно.

    print(f" Found {len(starters)}/{len(enders)} possible quest IDs for ModelID {model_id}!")
    return starters, enders

def compare_quest_texts(db_text, python_text, threshold=0.75):
    """Compare quest texts intelligently"""
    if not db_text or not python_text:
        return False
    
    # Normalize both texts
    def normalize(t):
        t = str(t)
        t = t.lower()
        t = t.replace('…', '...')
        t = re.sub(r'\s+', ' ', t)  # Normalize spaces
        t = re.sub(r'\.{2,}', '...', t)  # Normalize ellipsis
        t = re.sub(r'[^\w\s\.\?\!]', '', t)  # Remove special chars (optional)
        return t.strip()
    
    t1 = normalize(db_text)
    t2 = normalize(python_text)
    
    # Quick exact match first (faster)
    if t1 == t2:
        return True
    
    # Calculate similarity
    similarity = SequenceMatcher(None, t1, t2).ratio()
    
    # # Debug (optional)
    # if similarity < threshold and similarity > threshold - 0.15:
    #     print(f"⚠️ Low similarity: {similarity:.2f}")
    #     print(f"   DB: {t1[:100]}")
    #     print(f"   PY: {t2[:100]}")
    
    return similarity >= threshold

class SeedManager:
    def __init__(self, json_path="seeds_data.json"):
        self.json_path = Path(json_path)
        self.data = self._load_data()
    
    def _load_data(self):
        if self.json_path.exists():
            with open(self.json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_data(self):
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)
    
    def _get_key(self, voice_name, voice_type):
        """Генерирует ключ для словаря"""
        return f"{voice_name}_{voice_type}"
    
    def _ensure_voice_entry(self, voice_name, voice_type):
        """Убеждается, что запись для комбинации существует"""
        key = self._get_key(voice_name, voice_type)
        if key not in self.data:
            self.data[key] = {
                "voice_name": voice_name,
                "voice_type": voice_type,
                "default_seed": None,
                "seeds": {}
            }
        return key
    
    def add_or_update_seed(self, voice_name, voice_type, file_name, seed):
        """
        Добавляет или обновляет запись о сиде
        """
        key = self._ensure_voice_entry(voice_name, voice_type)
        
        # Проверяем, не является ли этот seed дефолтным
        is_default = (self.data[key]["default_seed"] == seed)
        
        self.data[key]["seeds"][file_name] = {
            "file_name": file_name,
            "seed": seed,
            "is_default": is_default
        }
        self._save_data()
        print(f"✅ Сохранено: {voice_name}/{voice_type}/{file_name} -> seed={seed}")
    
    def mark_as_default(self, voice_name, voice_type, seed_or_file_name):
        """
        Помечает seed как дефолтный для комбинации голоса и типа
        """
        key = self._ensure_voice_entry(voice_name, voice_type)
        
        # Сброс дефолта
        if seed_or_file_name is None:
            old_default = self.data[key]["default_seed"]
            self.data[key]["default_seed"] = None
            
            # Обновляем is_default для всех файлов
            for file_name, seed_data in self.data[key]["seeds"].items():
                seed_data["is_default"] = False
            
            self._save_data()
            print(f"🗑️ {voice_name}/{voice_type}: дефолтный seed сброшен (был {old_default})")
            return True
        
        target_seed = None
        
        # Если передан seed (число)
        if isinstance(seed_or_file_name, int):
            target_seed = seed_or_file_name
            found = False
            for file_name, seed_data in self.data[key]["seeds"].items():
                if seed_data["seed"] == target_seed:
                    print(f"🔍 Найден файл {file_name} с seed={target_seed}")
                    found = True
                    break
            if not found:
                print(f"❌ Не найден файл с seed={target_seed} для {voice_name}/{voice_type}")
                return False
        
        # Если передано имя файла
        elif isinstance(seed_or_file_name, str):
            file_name = seed_or_file_name
            if file_name in self.data[key]["seeds"]:
                target_seed = self.data[key]["seeds"][file_name]["seed"]
                print(f"🔍 Найден seed={target_seed} для файла {file_name}")
            else:
                print(f"❌ Не найден файл {file_name} для {voice_name}/{voice_type}")
                return False
        
        else:
            print(f"❌ Неверный тип аргумента: {type(seed_or_file_name)}")
            return False
        
        # Обновляем default_seed
        old_default = self.data[key]["default_seed"]
        self.data[key]["default_seed"] = target_seed
        
        # Обновляем is_default для всех файлов
        for file_name, seed_data in self.data[key]["seeds"].items():
            seed_data["is_default"] = (seed_data["seed"] == target_seed)
        
        self._save_data()
        print(f"⭐ {voice_name}/{voice_type}: дефолтный seed изменён с {old_default} на {target_seed}")
        return True
    
    def get_default_seed(self, voice_name, voice_type):
        """Возвращает дефолтный seed для комбинации"""
        key = self._get_key(voice_name, voice_type)
        if key not in self.data:
            return None
        return self.data[key]["default_seed"]
    
    def get_seed(self, voice_name, voice_type, file_name):
        """Получает seed для конкретного файла"""
        key = self._get_key(voice_name, voice_type)
        if key not in self.data:
            return None
        entry = self.data[key]["seeds"].get(file_name)
        return entry["seed"] if entry else None
    
    def get_all_for_voice(self, voice_name, voice_type):
        """Получает все записи для комбинации"""
        key = self._get_key(voice_name, voice_type)
        if key not in self.data:
            return {}
        return self.data[key]["seeds"]
    
    def get_all_voice_keys(self):
        """Возвращает все доступные комбинации"""
        return list(self.data.keys())
    
    def print_all(self):
        for key, value in self.data.items():
            print(f"\n🎤 {key}:")
            print(f"   Voice: {value['voice_name']}, Type: {value['voice_type']}")
            print(f"   Дефолтный seed: {value['default_seed']}")
            for file_name, seed_data in value["seeds"].items():
                default_mark = " ⭐" if seed_data["is_default"] else ""
                print(f"     {file_name}: seed={seed_data['seed']}{default_mark}")

class TTSEngine:
    def __init__(self):
        self.OUTPUT_DIR = "output"
        self.model = None
        self.model_whisper = None
        self._base_model_whisper = None
        self.processor = None
        self.current_model_folder = None
        self.current_attention_type = None
        self.current_quantize_llm = "full precision"
        self.current_lora_path = None
        # LoRA component flags (overridable by node instances)
        self.use_llm_lora = True
        self.use_diffusion_head_lora = True
        self.use_acoustic_connector_lora = True
        self.use_semantic_connector_lora = True
        self.seed_manager = SeedManager("generated_seeds.json")

    def make_output_dir(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

    def make_output_path(self, audio: str):
        self.output_path = os.path.join(self.OUTPUT_DIR, audio)
        return self.output_path

    # def synthesize(
    #     self,
    #     text: str = None,
    #     speaker_wav: str = None,
    #     language: str = "en",
    #     model_dir: str = None,
    #     speaker_id: str = None,
    #     voice_name: str = None,
    #
    #     # XTTS params
    #     temperature: float = 0.75,
    #     length_penalty: float = 1.2,
    #     repetition_penalty: float = 10.0,
    #     top_k: int = 1,
    #     top_p: float = 1.0,
    #     speed: float = 1.05,
    #
    #     # RVC params
    #     f0_up_key: int = 0,
    #     f0_method: str = "rmvpe",
    #     index_rate: float = 0.70,
    #     filter_radius: int = 3,
    #     resample_sr: int = 0,
    #     rms_mix_rate: float = 1,
    #     protect: float = 0.25
    # ):
    #
    #         self.make_output_dir()
    #
    #         global loaded_models, default_model
    #         # Save speaker reference audio
    #         ref_path = os.path.join(f"{speaker_wav}")
    #         audio = load_audio(ref_path, 22050)
    #
    #         # Choose model
    #         if model_dir:
    #             model_key = os.path.abspath(model_dir)
    #             with gpu_semaphore:
    #                 if model_key not in loaded_models:
    #
    #                     print(f"Loading fine-tuned model from {model_dir}")
    #                     config_path = os.path.join(model_dir, "config.json")
    #                     vocab_path = os.path.join(model_dir, "vocab.json")   # fine-tuned model
    #
    #                     speakers_path = os.path.join(model_dir, "speakers_xtts.pth")
    #
    #                     config = XttsConfig()
    #                     config.load_json(config_path)
    #
    #                     model = Xtts.init_from_config(config)
    #                     model.load_checkpoint(
    #                         config,
    #                         checkpoint_dir=model_dir,
    #                         speaker_file_path=speakers_path,
    #                         vocab_path=vocab_path,
    #                         eval=True
    #                     )
    #                     model.to(DEVICE)
    #                     loaded_models[model_key] = model
    #                 tts = loaded_models.get(model_key, default_model)
    #
    #         else:
    #             tts = default_model
    #         # Prune cache: keep only default and the current model_key
    #         keys_to_keep = set()
    #         if model_dir:
    #             keys_to_keep.add(model_key)
    #         # default_model is stored separately, so we don't remove it
    #         keys_to_remove = []
    #         with loaded_models_lock:
    #             for key in list(loaded_models.keys()):
    #                 if key not in keys_to_keep:
    #                     print(f"Unloading model from cache: {key}")
    #                     # remove mapping from dict now (so other threads won't try to use it)
    #                     model_obj = loaded_models.pop(key, None)
    #                     if model_obj is not None:
    #                         keys_to_remove.append((key, model_obj))
    #
    #         for key, model_obj in keys_to_remove:
    #             try:
    #                 print(f"Unloading model from cache: {key}")
    #                 # move off GPU
    #                 model_obj.to("cpu")
    #             except Exception as e:
    #                 print(f" Error moving {key} to CPU: {e}")
    #             finally:
    #                 # free cuda cache after model moved
    #                 try:
    #                     torch.cuda.empty_cache()
    #                 except Exception:
    #                     pass
    #
    #
    #         # Generate audio
    #         #This is why we need an inherited function, this needs to be passed back to TTSProcessor to export the file with its name to its final folder
    #         output_path = self.make_output_path(f"{uuid.uuid4()}.wav")
    #
    #         if speaker_id and speaker_id in tts.speaker_manager.speakers:
    #             # Inbuild / fine-tuned speaker from speakers_xtts.pth
    #             speaker_id = ref_path
    #             gpt_cond_latent, speaker_embedding = tts.speaker_manager.speakers[speaker_id].values()
    #         else:
    #             # On-the-fly from reference wav, cached
    #             gpt_cond_latent, speaker_embedding = get_or_create_latents(tts, ref_path, ref_path)
    #
    #         chunks = split_into_sentences(text)
    #         chunks = merge_short_fragments(chunks)
    #         print(chunks)
    #
    #         #default value from slider in webui (10) interpreted as int so we force as float
    #         repetition_penalty = float(repetition_penalty)
    #
    #         all_audio = []
    #         for chunk in chunks:
    #             with gpu_semaphore:
    #                 wav = tts.inference(
    #                     text=chunk,
    #                     language=language,
    #                     gpt_cond_latent = gpt_cond_latent,
    #                     speaker_embedding = speaker_embedding,
    #                     enable_text_splitting = False,
    #                     temperature = temperature,
    #                     length_penalty = length_penalty,
    #                     repetition_penalty = repetition_penalty,
    #                     top_k = top_k,
    #                     top_p = top_p,
    #                     speed = speed
    #                 )["wav"]
    #                 all_audio.append(wav)
    #             final_audio = np.concatenate(all_audio)
    #             sf.write(output_path, final_audio, 24000)  # XTTS uses 24kHz
    #
    #         # Base path where RVC models are stored
    #         rvc_base = "fine_tuned/_rvc"
    #         rvc_model_path = os.path.join(rvc_base,"weights", f"{voice_name}.pth")
    #         rvc_index_path = os.path.join(rvc_base,"indices", f"{voice_name}.index")
    #
    #         # Set environment variable for RVC
    #         os.environ.setdefault("rmvpe_root", "rvc/assets/rmvpe")
    #
    #         if os.path.isfile(rvc_model_path):
    #             if index_rate == 0:
    #                 print(f"Skipping RVC post-processing for {voice_name}")
    #                 sf.write(output_path, final_audio, 24000),
    #                 return output_path
    #             else:
    #                 print(f"Running RVC post-processing for {voice_name}")
    #
    #                 # Load the RVC model if needed
    #                 if not hasattr(vc, "model") or vc.model_path != rvc_model_path:
    #                     vc.get_vc(rvc_model_path)
    #                 file_index = rvc_index_path if os.path.isfile(rvc_index_path) else None
    #                 # Apply conversion
    #                 info, audio_data = vc.vc_single(
    #                     sid=0,
    #                     input_audio_path=output_path,
    #                     f0_up_key=f0_up_key,
    #                     f0_file=None,
    #                     f0_method=f0_method,  # or "crepe"
    #                     file_index=file_index,
    #                     file_index2="",
    #                     index_rate=index_rate,
    #                     filter_radius=3,
    #                     resample_sr=resample_sr,
    #                     rms_mix_rate=1,
    #                     protect=protect
    #                 )
    #
    #                 #audio = np.array(audio)
    #                 if audio_data is None:
    #                     print(f"Error in voice conversion: {info}")
    #                     return  # or handle error appropriately
    #                 sr, audio = audio_data
    #
    #                 # overwrite output_path with converted audio
    #                 wavfile.write(output_path, sr, audio)
    #                 return output_path
    #         else:
    #             print(f"⚠️ No RVC model found for {voice_name} at {rvc_model_path}")
    #             sf.write(output_path, final_audio, 24000),
    #             return output_path

    def free_memory(self):
        """Free model and processor from memory"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.processor is not None:
                del self.processor
                self.processor = None

            if self.model_whisper is not None:
                del self.model_whisper
                self.model_whisper = None
                
            if self._base_model_whisper is not None:
                del self._base_model_whisper
                self._base_model_whisper = None
            
            self.current_model_folder = None
            self.current_quantize_llm = "full precision"

            if self.emotion_classifier:
                # Close the pipeline
                if hasattr(self.emotion_classifier, 'model'):
                    del self.emotion_classifier.model
                if hasattr(self.emotion_classifier, 'tokenizer'):
                    del self.emotion_classifier.tokenizer
                self.emotion_classifier = None
            
            # Force garbage collection and clear CUDA cache if available
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info("Model and processor memory freed successfully")
            
        except Exception as e:
            logger.error(f"Error freeing memory: {e}")

    def free_memory_whisper(self):
        """Free model and processor from memory"""
        try:

            if self.model_whisper is not None:
                del self.model_whisper
                self.model_whisper = None
                
            if self._base_model_whisper is not None:
                del self._base_model_whisper
                self._base_model_whisper = None
            
            # Force garbage collection and clear CUDA cache if available
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
            logger.info("Whisper Model memory freed successfully")
            
        except Exception as e:
            logger.error(f"Error freeing memory: {e}")
    
    def _check_dependencies(self):
        """Check if VibeVoice is available and import it with fallback installation"""
        try:
            import sys
            import os
            
            # Add vvembed to path
            # current_dir = os.path.dirname(os.path.abspath(__file__))
            # parent_dir = os.path.dirname(current_dir)
            # vvembed_path = os.path.join(parent_dir, 'tts/vvembed')
            #
            # if vvembed_path not in sys.path:
            #     sys.path.insert(0, vvembed_path)
            
            # Import from embedded version
            # from modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
            
            # logger.info(f"Using embedded VibeVoice from {vvembed_path}")
            # return None, VibeVoiceForConditionalGenerationInference
            
        except ImportError as e:
            logger.error(f"Embedded VibeVoice import failed: {e}")
            
            # Try fallback to installed version if available
            try:
                import vibevoice
                from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
                logger.warning("Falling back to system-installed VibeVoice")
                return vibevoice, VibeVoiceForConditionalGenerationInference
            except ImportError:
                pass
            
            raise Exception(
                "VibeVoice embedded module import failed. Please ensure the vvembed folder exists "
                "and transformers>=4.51.3 is installed."
            )

    def match_peak_volume(self, original_file, volume_match_sample, generated_file, output_file, target_dbfs_override=None):
        # Загружаем
        target_dbfs = -16.0

        if target_dbfs_override:
            target_dbfs = target_dbfs_override

        if volume_match_sample:
            og_file = AudioSegment.from_mp3(original_file)
            target_dbfs = og_file.dBFS
            print(f"target_dbfs: {target_dbfs} (Sample)")
        else:
            print(f"target dbfs: {target_dbfs} (Default)")

        gen = AudioSegment.from_mp3(generated_file)
        gen_dbfs = gen.dBFS
        print(f"gen_dbfs: {gen_dbfs}")

        # Выравниваем
        adjusted = gen.apply_gain(target_dbfs - gen_dbfs)

        # Сохраняем
        adjusted.export(output_file, format="mp3", bitrate="64k")

        # Возвращаем коэффициент усиления
        gain_linear = 10 ** ((target_dbfs - gen_dbfs) / 20)
        print(f"Volume diff {gain_linear} normalized.")
        return gain_linear

    def get_emotion_preset(self, text):
        """
        Detect emotion from text and return appropriate TTS parameters
        """
        if not hasattr(self, 'emotion_classifier') or self.emotion_classifier is None:
            try:
                from transformers import pipeline
                self.emotion_classifier = pipeline(
                    "text-classification",
                    model='bhadresh-savani/distilbert-base-uncased-emotion',
                    device="cpu"
                )
                print("✅ Emotion classifier loaded")
            except Exception as e:
                print(f"⚠️ Failed to load emotion classifier: {e}")
                self.emotion_classifier = False  # Mark as failed
                return {'temperature': 5.0, 'class_temperature': 0.0, 'cfg_scale': 2.0, 'voice_speed_factor': 1.0, 'label': 'Default', 'score': 0}

         # Detect emotion
        try:
            result = self.emotion_classifier(text, top_k=1)[0]
            emotion = result['label']
            score = result['score']
        except Exception as e:
            print(f"⚠️ Emotion detection failed: {e}")
            return {'temperature': 5.0, 'class_temperature': 0.0, 'cfg_scale': 2.0, 'voice_speed_factor': 1.0, 'label': 'Default', 'score': 0}

        emotion_map = {
            # Гнев: рваный темп, хрипота + ускорение речи на 15%
            'anger':    {'position_temp': 6.2, 'class_temp': 0.18, 'cps_modifier': 1.15, 'label': 'Angry'},
            
            # Радость: живые интонации + легкое ускорение на 5%
            'joy':      {'position_temp': 5.8, 'class_temp': 0.05, 'cps_modifier': 1.05, 'label': 'Joyful'},
            
            # Удивление: резкие скачки тона + ускорение на 10%
            'surprise': {'position_temp': 6.5, 'class_temp': 0.10, 'cps_modifier': 1.10, 'label': 'Surprised'},
            
            # Страх: голос дрожит, срывается + жесткое ускорение на 20% (паника)
            'fear':     {'position_temp': 6.0, 'class_temp': 0.20, 'cps_modifier': 1.20, 'label': 'Frightened'},
            
            # Любовь/Доброта: спокойная, плавная речь + замедление на 10%
            'love':     {'position_temp': 4.5, 'class_temp': 0.00, 'cps_modifier': 0.90, 'label': 'Loving'},
            
            # Грусть: монотонный голос + сильное замедление на 15%
            'sadness':  {'position_temp': 4.2, 'class_temp': 0.08, 'cps_modifier': 0.85, 'label': 'Sad'},
            
            # Нейтральный: ничего не меняем (коэффициент 1.0)
            'neutral':  {'position_temp': 5.0, 'class_temp': 0.0, 'cps_modifier': 1.00, 'label': 'Neutral'}
        }

        BASE_CPS = 14

        if score < 0.58:
            current_emotion = 'neutral'
        else:
            current_emotion = emotion

        emotion_params = emotion_map.get(current_emotion, emotion_map['neutral']).copy()

        if score > 0.82 and current_emotion != 'neutral':
            if current_emotion in ['anger', 'fear', 'surprise']:
                emotion_params['position_temp'] = min(6.8, emotion_params['position_temp'] + 0.3)
                emotion_params['class_temp'] = min(0.25, emotion_params['class_temp'] + 0.05)
                emotion_params['cps_modifier'] += 0.05
            elif current_emotion in ['sadness', 'love']:
                emotion_params['position_temp'] = max(3.8, emotion_params['position_temp'] - 0.4)
                emotion_params['cps_modifier'] -= 0.05

        final_cps = BASE_CPS * emotion_params['cps_modifier']
        final_cps = round(final_cps, 1)

        params = {
            'temperature': max(3.5, min(7.0, emotion_params['position_temp'])),
            'class_temperature':    max(0.0, min(0.25, emotion_params['class_temp'])),
            'cps':           final_cps,
            'label':         emotion_params['label'],
            'score':         score
        }
        
        print(f"🎭 Emotion detected: {params['label']} (conf: {score:.2f})")
        print(f"   → P Temp: {params['temperature']}, C Temp : {params['class_temperature']}, CPS: {params['cps']}")
        
        return params
    
    def generate_speech(self, text: str = "", model: str = "VibeVoice-1.5B",
                       attention_type: str = "auto", quantize_llm: str = "full precision",
                       free_memory_after_generate: bool = True,
                       diffusion_steps: int = 32, seed: int = 42, cfg_scale: float = 2.0,
                       use_sampling: bool = True, voice_to_clone=None, voice_to_clone_2=None, lora=None,
                       temperature: float = 5.0, class_temperature: float = 0.3,
                       max_words_per_chunk: int = 75, voice_speed_factor: float = 1.0, cps = 15,
                       voice_key = None, questgiver_id = None, apply_effects: bool = True, apply_effects_override: str = None, detect_emotions: bool = False, volume_match: bool = True):
        """Generate speech from text using VibeVoice"""

        # Fix short texts cfg scale
        def fix_cfg_scale_by_word_count(text, cfg_scale):
            words = len(text.split())
            if words <= 3:
                print(f" {words} words in text : cfg_scale changed {cfg_scale} -> 1.9")
                return 1.9
            elif words <= 6:
                print(f" {words} words in text : cfg_scale changed {cfg_scale} -> 1.5")
                return 1.5
            elif words <= 15:
                print(f" {words} words in text : cfg_scale changed {cfg_scale} -> 1.4")
                return 1.4
            else:
                return cfg_scale
            
        def get_duration_with_emotion(text: str, cps: float = 15, min_chars: int = 30):
            char_count = len(text)    
            if char_count < min_chars:
                return None
            
            base_duration = len(text) / cps
            
            # Считаем эмоциональные маркеры
            excitement = text.count('!') + text.count('?')
            repeats = text.lower().count('! ') + text.lower().count('? ')
            
            # Коэффициент замедления (чем больше маркеров, тем медленнее)
            slow_factor = 1.0
            if excitement >= 2:
                slow_factor = 1.05   # +15% к длительности
            if repeats >= 2:
                slow_factor = 1.10   # +25% к длительности
            
            duration = base_duration * slow_factor
            return duration
        


        def get_duration(text: str, cps: float = 15, min_chars: int = 30) -> float:
            char_count = len(text)    
            if char_count < min_chars:
                return None
            return char_count / cps
            
        # use stricter cfg_scale when using custom voices
        # if voice_key == 'custom_custom':
        #     print(f"custom voice: cfg_scale changed {cfg_scale} -> 1.5")
        #     cfg_scale = 1.5

        print("\n" + "=" * 50)
        print("TTS GENERATION DEBUG".center(50))
        print("=" * 50)
        print(f" TEXT LENGTH: {len(text)} chars")
        # print(f" SEED       : {seed}")
        # print(f" SAMPLING   : {use_sampling}")
        print(f" POS TEMP       : {temperature}")
        print(f" CLS TEMP       : {class_temperature}")
        # print(f" TOP_P      : {top_p}")
        print(f" CFG_SCALE  : {cfg_scale}")
        print(f" STEPS      : {diffusion_steps}")
        print(f" Chars p/s  : {cps}")
        print(f" SPEED      : {voice_speed_factor}")
        # print(f" ATTENTION  : {attention_type}")
        print(f" EFFECTS    : {apply_effects}")
        print(f" FX OVERRIDE : {apply_effects_override}")
        print(f" DETECT EMOTIONS: {detect_emotions}")
        print("=" * 50 + "\n")
        
        try:
            # Check text input
            if not text or not text.strip():
                raise Exception(
                    "No text provided. Please enter text with speaker labels (e.g., '[1]: Hello' or '[2]: Hi')")


            # model = OmniVoice.from_pretrained(
            #     "k2-fsa/OmniVoice",
            #     device_map="cuda:0",
            #     dtype=torch.float16,
            #     load_asr=True,
            #     asr_model_name="openai/whisper-large-v3-turbo",
            # )

            bracket_pattern = r'\[(\d+)\]\s*:'
            speakers_numbers = sorted(list(set([int(m) for m in re.findall(bracket_pattern, text)])))

            if not speakers_numbers:
                num_speakers = 1  # Default to 1 if no speaker format found
            else:
                num_speakers = min(max(speakers_numbers), 2)  # Max speaker number, capped at 4
                if max(speakers_numbers) > 2:
                    print(f"[OmniVoice] Warning: Found {max(speakers_numbers)} speakers, limiting to 2")
            
            if not voice_to_clone_2:
                num_speakers = 1  # Default to 1 if no speaker format found

            if voice_to_clone is not None:
                mode = "clone"

            choices=['ghost', 'robot', 'undead', 'demon', 'giant', 'small', 'letter', 'comms', 'dk', 'ancient', 'ethereal', 'earthen', 'naaru', 'wretched']

            converted_text = text
            speakers_in_text = sorted(list(set([int(m) for m in re.findall(bracket_pattern, text)])))
            if voice_to_clone_2:
                # Convert [N]: directly to Speaker (N-1): and handle multi-line text
                # Split text to preserve speaker segments while cleaning up newlines within each segment
                segments = []

                # Find all speaker markers with their positions
                speaker_matches = list(
                    re.finditer(f'\\[({"|".join(map(str, speakers_in_text))})\\]\\s*:', converted_text))

                # Store speaker segments for pause processing
                speaker_segments_with_pauses = []

                for i, match in enumerate(speaker_matches):
                    speaker_num = int(match.group(1))
                    start = match.end()

                    # Find where this speaker's text ends (at next speaker or end of text)
                    if i + 1 < len(speaker_matches):
                        end = speaker_matches[i + 1].start()
                    else:
                        end = len(converted_text)

                    # Extract the speaker's text (keep pause keywords for now)
                    speaker_text = converted_text[start:end].strip()

                    # Parse pause keywords within this speaker's text
                    pause_segments = self._parse_pause_keywords(speaker_text)

                    # Process each segment (text or pause) for this speaker
                    for seg_type, seg_content in pause_segments:
                        if seg_type == 'pause':
                            # Add pause segment
                            speaker_segments_with_pauses.append(('pause', seg_content, None))
                        else:
                            # Clean up the text segment
                            text_clean = seg_content.replace('\n', ' ').replace('\r', ' ')
                            text_clean = ' '.join(text_clean.split())

                            if text_clean:  # Only add non-empty text
                                # Add text segment with speaker info
                                speaker_segments_with_pauses.append(('text', text_clean, speaker_num))
                                # Also build the traditional segments for fallback
                                segments.append(text_clean)

                # Join all segments with newlines (required for multi-speaker format) - for fallback
                converted_text = '\n'.join(segments) if segments else ""
            else:
                speakers_in_text = [1]

            speakers = [f"Speaker {i}" for i in range(len(speakers_in_text))]

            self.load_model()
            # _ = model.generate(text="Warmup", ref_audio=voice_to_clone, temperature=0.1)

            from omnivoice import VoiceClonePrompt

            voice_inputs = [voice_to_clone, voice_to_clone_2]
            voice_samples = []
            for i, speaker_num in enumerate(speakers_in_text):
                idx = speaker_num - 1  # Convert to 0-based for voice array

                # Try to use provided voice sample
                if idx < len(voice_inputs) and voice_inputs[idx] is not None:

                    # Save prompt to cache
                    voice_sample_key = voice_key
                    if idx == 1:
                        voice_sample_key = "narrator"

                    sample_path = self._get_prompt_path(str(voice_inputs[idx]))

                    if os.path.exists(sample_path):
                        voice_sample = VoiceClonePrompt.load(sample_path)
                        print(f"✅ Prompt loaded: {sample_path}")
                    else:
                        voice_sample = self.model.create_voice_clone_prompt(
                            ref_audio=str(voice_inputs[idx]),
                            ref_text=None,
                        )
                        os.makedirs(os.path.join(VOICE_SAMPLE_PATH, f"prompts"), exist_ok=True)
                        voice_sample.save(sample_path)
                        print(f"✅ Prompt saved: {sample_path}")

                    voice_samples.append(voice_sample)

            if len(voice_samples) != len(speakers_in_text):
                logger.error(f"Mismatch: {len(speakers_in_text)} speakers but {len(voice_samples)} voice samples!")
                raise Exception(
                    f"Voice sample count mismatch: expected {len(speakers_in_text)}, got {len(voice_samples)}")

            if voice_to_clone_2:
                if 'speaker_segments_with_pauses' in locals() and speaker_segments_with_pauses:
                    # Process segments with pauses
                    all_audio_segments = []
                    sample_rate = 24000  # VibeVoice uses 24kHz

                    # Group consecutive text segments from same speaker for efficiency
                    grouped_segments = []
                    current_group = []
                    current_speaker = None

                    for seg_type, seg_content, speaker_num in speaker_segments_with_pauses:
                        if seg_type == 'pause':
                            # Save current group if any
                            if current_group:
                                grouped_segments.append(('text_group', current_group, current_speaker))
                                current_group = []
                                current_speaker = None
                            # Add pause
                            grouped_segments.append(('pause', seg_content, None))
                        else:
                            # Text segment
                            if speaker_num == current_speaker:
                                # Same speaker, add to current group
                                current_group.append(seg_content)
                            else:
                                # Different speaker, save current group and start new one
                                if current_group:
                                    grouped_segments.append(('text_group', current_group, current_speaker))
                                current_group = [seg_content]
                                current_speaker = speaker_num

                    # Save last group if any
                    if current_group:
                        grouped_segments.append(('text_group', current_group, current_speaker))

                    # Process grouped segments
                    for seg_type, seg_content, speaker_num in grouped_segments:
                        if seg_type == 'pause':
                            # Generate silence
                            duration_ms = seg_content
                            logger.info(f"Adding {duration_ms}ms pause")
                            silence_audio = self._generate_silence(duration_ms, sample_rate)
                            all_audio_segments.append(silence_audio)
                        else:
                            # Process text group for a speaker
                            combined_text = ' '.join(seg_content)
                            formatted_text = f"{combined_text}"

                            temp_temp = temperature
                            temp_class_temp = class_temperature
                            temp_cps = cps

                            if detect_emotions:                                
                                # Detect emotion from text
                                emotion_params = self.get_emotion_preset(text)
                                
                                # Override temperature and top_p with detected values
                                temp_temp = emotion_params['temperature']
                                temp_class_temp = emotion_params['class_temperature']
                                temp_cps = emotion_params['cps']
                            
                            gen_config = OmniVoiceGenerationConfig(
                                num_step=int(diffusion_steps),
                                class_temperature=temp_class_temp,
                                position_temperature=temp_temp,
                                guidance_scale=float(cfg_scale) if cfg_scale is not None else 2.0,
                                denoise=True, #bool(denoise) if denoise is not None else True,
                                preprocess_prompt=bool(True),
                                postprocess_output=bool(True),
                            )

                            lang = "english" #language if (language and language != "Auto") else None

                            kw: Dict[str, Any] = dict(
                                language=lang, generation_config=gen_config
                            )

                            kw["text"] = formatted_text.strip()
                            if voice_speed_factor is not None and float(voice_speed_factor) != 1.0:
                                kw["speed"] = float(voice_speed_factor)
                            else:
                                duration = get_duration_with_emotion(kw['text'], temp_cps)
                                kw["duration"] = float(duration) if duration else None

                            # Get voice sample for this speaker
                            speaker_idx = speakers_in_text.index(speaker_num)
                            speaker_voice_samples = voice_samples[speaker_idx]

                            logger.info(
                                f"Generating audio for Speaker {speaker_num}: {len(combined_text.split())} words")

                            if mode == "clone":
                                if not speaker_voice_samples:
                                    return None, "Please upload a reference audio."
                                
                            kw["voice_clone_prompt"] = speaker_voice_samples

                            if apply_effects_override and apply_effects_override.strip() and not ae.is_sox_command(apply_effects_override) and apply_effects_override not in choices:
                                kw["instruct"] = apply_effects_override.strip()

                            try:
                                generated_audio = self.model.generate(**kw)
                                
                            except Exception as e:
                                return None, f"Error: {type(e).__name__}: {e}"
                            
                            if isinstance(generated_audio, list) and len(generated_audio) > 0:
                                if isinstance(generated_audio[0], dict):
                                    segment_audio = generated_audio[0]["waveform"]
                                else:
                                    segment_audio = generated_audio[0]

                            if speaker_idx != 1 and apply_effects and not apply_effects_override:
                                # after voice generation, apply sfx based on certain race-gender combos
                                if voice_key.endswith("_dk"):
                                    print(f"DK post-processing")
                                    segment_audio = ae.dk_effects_on_waveform(segment_audio)
                                elif voice_key in ("sylvanas", "forsaken_male", "forsaken_female"):
                                    print(f"Undead post-processing")
                                    segment_audio = ae.undead_effects_on_waveform(segment_audio)
                                elif voice_key in ("mechanical", "titan_male"):
                                    print(f"Robot post-processing")
                                    segment_audio = ae.robot_effects_on_waveform(segment_audio)
                                elif voice_key in ("demon_male", "keeper", "dragon_male", "dragon_female"):
                                    print(f"Demon post-processing")
                                    segment_audio = ae.demon_effects_on_waveform(segment_audio)
                                elif voice_key in ("giant_male", "ogre_male", "ogrila_ogre", "ancient"):
                                    print(f"Giant post-processing")
                                    segment_audio = ae.giant_effects_on_waveform(segment_audio, voice_key)
                                elif voice_key in ("wolvar_male", "gorloc_male"):
                                    print(f"Small creature post-processing")
                                    segment_audio = ae.small_effects_on_waveform(segment_audio)
                                elif voice_key in ("earthen"):
                                    print(f"Earthen creature post-processing")
                                    segment_audio = ae.earthen_effects_on_waveform(segment_audio)
                                elif voice_key in ("naaru"):
                                    print(f"Naaru creature post-processing")
                                    segment_audio = ae.naaru_effects_on_waveform(segment_audio)
                                elif voice_key in ("ethereal_male"):
                                    print(f"Ethereal creature post-processing")
                                    segment_audio = ae.ethereal_effects_on_waveform(segment_audio)
                                elif voice_key in ("wretched"):
                                    print(f"Ethereal wretched post-processing")
                                    segment_audio = ae.wretched_effect_on_waveform(segment_audio)
                                # Apply questgiver-specific effects that don't conform to race_gender categories
                                if questgiver_id and questgiver_id in NPC_EFFECTS:
                                    effect_type = NPC_EFFECTS[questgiver_id]
                                    if effect_type == "ghost":
                                        print("Doing ghost effects", flush=True)
                                        segment_audio = ae.ghost_effects_on_waveform(segment_audio)
                                    elif effect_type == "demon":
                                        print("Doing demon effects", flush=True)
                                        segment_audio = ae.demon_effects_on_waveform(segment_audio)
                                    elif effect_type == "giant":
                                        print("Doing giant effects", flush=True)
                                        segment_audio = ae.giant_effects_on_waveform(segment_audio, voice_key)
                                    elif effect_type == "ancient":
                                        print("Doing ancient effects", flush=True)
                                        segment_audio = ae.giant_effects_on_waveform(segment_audio, effect_type)
                                    elif effect_type == "undead":
                                        print("Doing undead effects", flush=True)
                                        segment_audio = ae.undead_effects_on_waveform(segment_audio)
                                    elif effect_type == "small":
                                        print(f"Small creature effects", flush=True)
                                        segment_audio = ae.small_effects_on_waveform(segment_audio)
                                    elif effect_type == "comms":
                                        print(f"Telephone effects ...", flush=True)
                                        segment_audio = ae.comms_effects_on_waveform(segment_audio)
                                    elif effect_type == "wretched":
                                        print(f"Wretched effects ...", flush=True)
                                        segment_audio = ae.wretched_effect_on_waveform(segment_audio)

                            if speaker_idx != 1 and apply_effects and apply_effects_override:
                                effect_type = apply_effects_override
                                if ae.is_sox_command(effect_type):
                                    print(f"Doing custom effects", flush=True)
                                    raw_stuff = str(effect_type).strip('"').strip()
                                    raw_stuff = raw_stuff.replace('[', '').replace(']', '')
                                    args = raw_stuff.replace(',', ' ').replace("'", "").replace('"', '').split()
                                    segment_audio = ae._apply_sox_effects_to_waveform(segment_audio, args)
                                else:
                                    if effect_type == "ghost":
                                        print("Doing ghost effects", flush=True)
                                        segment_audio = ae.ghost_effects_on_waveform(segment_audio)
                                    elif effect_type == "demon":
                                        print("Doing demon effects", flush=True)
                                        segment_audio = ae.demon_effects_on_waveform(segment_audio)
                                    elif effect_type == "robot":
                                        print("Doing robot effects", flush=True)
                                        segment_audio = ae.robot_effects_on_waveform(segment_audio)
                                    elif effect_type == "ethereal":
                                        print("Doing ethereal effects", flush=True)
                                        segment_audio = ae.ethereal_effects_on_waveform(segment_audio)
                                    elif effect_type == "giant":
                                        print("Doing giant effects", flush=True)
                                        segment_audio = ae.giant_effects_on_waveform(segment_audio, voice_key)
                                    elif effect_type == "ancient":
                                        print("Doing ancient effects", flush=True)
                                        segment_audio = ae.giant_effects_on_waveform(segment_audio, effect_type)
                                    elif effect_type == "undead":
                                        print("Doing undead effects", flush=True)
                                        segment_audio = ae.undead_effects_on_waveform(segment_audio)
                                    elif effect_type == "small":
                                        print(f"Small creature effects", flush=True)
                                        segment_audio = ae.small_effects_on_waveform(segment_audio)
                                    elif effect_type == "comms":
                                        print(f"Telephone effects ...", flush=True)
                                        segment_audio = ae.comms_effects_on_waveform(segment_audio)
                                    elif effect_type == "naaru":
                                        print(f"Doing naaru effects", flush=True)
                                        segment_audio = ae.naaru_effects_on_waveform(segment_audio)
                                    elif effect_type == "dk":
                                        print(f"Doing dk effects", flush=True)
                                        segment_audio = ae.dk_effects_on_waveform(segment_audio)
                                    elif effect_type == "earthen":
                                        print(f"Doing earthen effects", flush=True)
                                        segment_audio = ae.earthen_effects_on_waveform(segment_audio)
                                    elif effect_type == "wretched":
                                        print(f"Doing wretched effects", flush=True)
                                        segment_audio = ae.wretched_effect_on_waveform(segment_audio)
                                    elif effect_type == "letter":
                                        print(f"Doing letter effects", flush=True)
                                        segment_audio = ae.letter_effect_on_waveform(segment_audio)

                            if volume_match:
                                segment_audio = ae.match_peak_volume_on_waveform(segment_audio)

                            all_audio_segments.append(segment_audio)

                    # Concatenate all audio segments
                    if all_audio_segments:
                        normalized_segments = []
                        for seg in all_audio_segments:
                            # 1. Если это тензор PyTorch
                            if torch.is_tensor(seg):
                                seg = seg.detach().cpu().numpy()
                            
                            # 2. Если это список, превращаем в массив
                            if isinstance(seg, list):
                                seg = np.array(seg)
                                
                            # 3. Убираем лишние измерения (превращаем (1, 1, L) или (1, L) в просто (L,))
                            seg = np.squeeze(seg)
                            
                            # 4. Если после squeeze всё еще остались измерения (например, стерео), берем один канал
                            if seg.ndim > 1:
                                seg = seg[0]
                                
                            normalized_segments.append(seg)

                        # Теперь склейка пройдет идеально
                        full_waveform = np.concatenate(normalized_segments)

                        # full_waveform = np.concatenate(all_audio_segments)
                        del all_audio_segments
                        if full_waveform.size > 0:
                            logger.info(f"Successfully generated multi-speaker audio with pauses")
                        else:
                            raise Exception("Failed to extract waveforms from audio segments")
                    else:
                        raise Exception("No audio segments generated")
                else:
                    # Fallback to original method without pause support
                    logger.info("Processing without pause support (no pause keywords found)")
                    # fix cfg scale for short texts
                    # cfg_scale_temp = fix_cfg_scale_by_word_count(text, cfg_scale)
                    if mode == "clone":
                        if not voice_samples[0]:
                            return None, "Please upload a reference audio."
                                
                    temp_temp = temperature
                    temp_class_temp = class_temperature
                    temp_cps = cps

                    if detect_emotions:                                
                        # Detect emotion from text
                        emotion_params = self.get_emotion_preset(text)
                        
                        # Override temperature and top_p with detected values
                        temp_temp = emotion_params['temperature']
                        temp_class_temp = emotion_params['class_temperature']
                        temp_cps = emotion_params['cps']
                    
                    gen_config = OmniVoiceGenerationConfig(
                        num_step=int(diffusion_steps),
                        class_temperature=temp_class_temp,
                        position_temperature=temp_temp,
                        guidance_scale=float(cfg_scale) if cfg_scale is not None else 2.0,
                        denoise=True, #bool(denoise) if denoise is not None else True,
                        preprocess_prompt=bool(True),
                        postprocess_output=bool(True),
                    )

                    lang = "english" #language if (language and language != "Auto") else None

                    kw: Dict[str, Any] = dict(
                        language=lang, generation_config=gen_config
                    )

                    kw["text"] = text.strip()
                    if voice_speed_factor is not None and float(voice_speed_factor) != 1.0:
                        kw["speed"] = float(voice_speed_factor)
                    else:
                        duration = get_duration_with_emotion(kw['text'], temp_cps)
                        kw["duration"] = float(duration) if duration else None
                    
                    kw["voice_clone_prompt"] = voice_samples[0]

                    if apply_effects_override and apply_effects_override.strip() and not ae.is_sox_command(apply_effects_override) and apply_effects_override not in choices:
                        kw["instruct"] = apply_effects_override.strip()

                    try:
                        generated_audio = self.model.generate(**kw)
                        if isinstance(generated_audio, list) and len(generated_audio) > 0:
                            if isinstance(generated_audio[0], dict):
                                segment_audio = generated_audio[0]["waveform"]
                            else:
                                segment_audio = generated_audio[0]
                        
                        if volume_match:
                            segment_audio = ae.match_peak_volume_on_waveform(segment_audio)

                        full_waveform = segment_audio
                        
                    except Exception as e:
                        return None, f"Error: {type(e).__name__}: {e}"
            else:

                if mode == "clone":
                    if not voice_to_clone:
                        return None, "Please upload a reference audio."
                    
                temp_temp = temperature
                temp_class_temp = class_temperature
                temp_cps = cps

                if detect_emotions:                                
                    # Detect emotion from text
                    emotion_params = self.get_emotion_preset(text)
                    
                    # Override temperature and top_p with detected values
                    temp_temp = emotion_params['temperature']
                    temp_class_temp = emotion_params['class_temperature']
                    temp_cps = emotion_params['cps']
                
                gen_config = OmniVoiceGenerationConfig(
                    num_step=int(diffusion_steps),
                    class_temperature=temp_class_temp,
                    position_temperature=temp_temp,
                    guidance_scale=float(cfg_scale) if cfg_scale is not None else 2.0,
                    denoise=True, #bool(denoise) if denoise is not None else True,
                    preprocess_prompt=bool(True),
                    postprocess_output=bool(True),
                )

                lang = "english" #language if (language and language != "Auto") else None

                kw: Dict[str, Any] = dict(
                    language=lang, generation_config=gen_config
                )

                kw["text"] = text.strip()
                if voice_speed_factor is not None and float(voice_speed_factor) != 1.0:
                    kw["speed"] = float(voice_speed_factor)
                else:
                    duration = get_duration_with_emotion(kw['text'], temp_cps)
                    kw["duration"] = float(duration) if duration else None

                kw["voice_clone_prompt"] = voice_samples[0]

                if apply_effects_override and apply_effects_override.strip() and not ae.is_sox_command(apply_effects_override) and apply_effects_override not in choices:
                    kw["instruct"] = apply_effects_override.strip()

                try:
                    generated_audio = self.model.generate(**kw)
                    if isinstance(generated_audio, list) and len(generated_audio) > 0:
                        if isinstance(generated_audio[0], dict):
                            segment_audio = generated_audio[0]["waveform"]
                        else:
                            segment_audio = generated_audio[0]
                    
                except Exception as e:
                    return None, f"Error: {type(e).__name__}: {e}"

                if voice_key != 'narrator' and apply_effects and not apply_effects_override:
                    # after voice generation, apply sfx based on certain race-gender combos
                    if voice_key.endswith("_dk"):
                        print(f"DK post-processing")
                        segment_audio = ae.dk_effects_on_waveform(segment_audio)
                    elif voice_key in ("sylvanas", "forsaken_male", "forsaken_female"):
                        print(f"Undead post-processing")
                        segment_audio = ae.undead_effects_on_waveform(segment_audio)
                    elif voice_key in ("mechanical", "titan_male"):
                        print(f"Robot post-processing")
                        segment_audio = ae.robot_effects_on_waveform(segment_audio)
                    elif voice_key in ("demon_male", "keeper", "dragon_male", "dragon_female"):
                        print(f"Demon post-processing")
                        segment_audio = ae.demon_effects_on_waveform(segment_audio)
                    elif voice_key in ("giant_male", "ogre_male", "ogrila_ogre", "ancient"):
                        print(f"Giant post-processing")
                        segment_audio = ae.giant_effects_on_waveform(segment_audio, voice_key)
                    elif voice_key in ("wolvar_male", "gorloc_male"):
                        print(f"Small creature post-processing")
                        segment_audio = ae.small_effects_on_waveform(segment_audio)
                    elif voice_key in ("earthen"):
                        print(f"Earthen creature post-processing")
                        segment_audio = ae.earthen_effects_on_waveform(segment_audio)
                    elif voice_key in ("naaru"):
                        print(f"Naaru creature post-processing")
                        segment_audio = ae.naaru_effects_on_waveform(segment_audio)
                    elif voice_key in ("ethereal_male"):
                        print(f"Ethereal creature post-processing")
                        segment_audio = ae.ethereal_effects_on_waveform(segment_audio)
                    elif voice_key in ("wretched"):
                        print(f"Ethereal wretched post-processing")
                        segment_audio = ae.wretched_effect_on_waveform(segment_audio)
                    # Apply questgiver-specific effects that don't conform to race_gender categories
                    if questgiver_id and questgiver_id in NPC_EFFECTS:
                        effect_type = NPC_EFFECTS[questgiver_id]
                        if effect_type == "ghost":
                            print("Doing ghost effects", flush=True)
                            segment_audio = ae.ghost_effects_on_waveform(segment_audio)
                        elif effect_type == "demon":
                            print("Doing demon effects", flush=True)
                            segment_audio = ae.demon_effects_on_waveform(segment_audio)
                        elif effect_type == "giant":
                            print("Doing giant effects", flush=True)
                            segment_audio = ae.giant_effects_on_waveform(segment_audio, voice_key)
                        elif effect_type == "ancient":
                            print("Doing ancient effects", flush=True)
                            segment_audio = ae.giant_effects_on_waveform(segment_audio, effect_type)
                        elif effect_type == "undead":
                            print("Doing undead effects", flush=True)
                            segment_audio = ae.undead_effects_on_waveform(segment_audio)
                        elif effect_type == "small":
                            print(f"Small creature effects", flush=True)
                            segment_audio = ae.small_effects_on_waveform(segment_audio)
                        elif effect_type == "comms":
                            print(f"Telephone effects ...", flush=True)
                            segment_audio = ae.comms_effects_on_waveform(segment_audio)
                        elif effect_type == "wretched":
                            print(f"Wretched effects ...", flush=True)
                            segment_audio = ae.wretched_effect_on_waveform(segment_audio)

                if voice_key != 'narrator' and apply_effects and apply_effects_override:
                    effect_type = apply_effects_override
                    if ae.is_sox_command(effect_type):
                        print(f"Doing custom effects", flush=True)
                        raw_stuff = str(effect_type).strip('"').strip()
                        raw_stuff = raw_stuff.replace('[', '').replace(']', '')
                        args = raw_stuff.replace(',', ' ').replace("'", "").replace('"', '').split()
                        segment_audio = ae._apply_sox_effects_to_waveform(segment_audio, args)
                    else:
                        if effect_type == "ghost":
                            print("Doing ghost effects", flush=True)
                            segment_audio = ae.ghost_effects_on_waveform(segment_audio)
                        elif effect_type == "demon":
                            print("Doing demon effects", flush=True)
                            segment_audio = ae.demon_effects_on_waveform(segment_audio)
                        elif effect_type == "robot":
                            print("Doing robot effects", flush=True)
                            segment_audio = ae.robot_effects_on_waveform(segment_audio)
                        elif effect_type == "ethereal":
                            print("Doing ethereal effects", flush=True)
                            segment_audio = ae.ethereal_effects_on_waveform(segment_audio)
                        elif effect_type == "giant":
                            print("Doing giant effects", flush=True)
                            segment_audio = ae.giant_effects_on_waveform(segment_audio, voice_key)
                        elif effect_type == "ancient":
                            print("Doing ancient effects", flush=True)
                            segment_audio = ae.giant_effects_on_waveform(segment_audio, effect_type)
                        elif effect_type == "undead":
                            print("Doing undead effects", flush=True)
                            segment_audio = ae.undead_effects_on_waveform(segment_audio)
                        elif effect_type == "small":
                            print(f"Small creature effects", flush=True)
                            segment_audio = ae.small_effects_on_waveform(segment_audio)
                        elif effect_type == "comms":
                            print(f"Telephone effects ...", flush=True)
                            segment_audio = ae.comms_effects_on_waveform(segment_audio)
                        elif effect_type == "naaru":
                            print(f"Doing naaru effects", flush=True)
                            segment_audio = ae.naaru_effects_on_waveform(segment_audio)
                        elif effect_type == "dk":
                            print(f"Doing dk effects", flush=True)
                            segment_audio = ae.dk_effects_on_waveform(segment_audio)
                        elif effect_type == "earthen":
                            print(f"Doing earthen effects", flush=True)
                            segment_audio = ae.earthen_effects_on_waveform(segment_audio)
                        elif effect_type == "wretched":
                            print(f"Doing wretched effects", flush=True)
                            segment_audio = ae.wretched_effect_on_waveform(segment_audio)
                        elif effect_type == "letter":
                            print(f"Doing letter effects", flush=True)
                            segment_audio = ae.letter_effect_on_waveform(segment_audio)

                if volume_match:
                    segment_audio = ae.match_peak_volume_on_waveform(segment_audio)

                full_waveform = segment_audio

            # waveform = (segment_audio[0] * 32767).astype(np.int16)
            # audio = (self.model.sampling_rate, waveform)

            if isinstance(full_waveform, torch.Tensor):
                audio_np = full_waveform.cpu().float().numpy()
            else:
                audio_np = np.array(full_waveform)

            audio_np = np.squeeze(audio_np)
            audio_int16 = np.clip(audio_np, -1.0, 1.0)
            audio_int16 = (audio_int16 * 32767).astype(np.int16)

            output_path = self.make_output_path(f"{uuid.uuid4()}.wav")
            full_abs_path = os.path.abspath(output_path)
            os.makedirs(os.path.dirname(full_abs_path), exist_ok=True)
            print(f"--- DEBUG INFO ---")
            print(f"Current Working Directory: {os.getcwd()}")
            print(f"Output Path (raw): {output_path}")
            print(f"Full Absolute Path: {full_abs_path}")
            print(f"Directory exists: {os.path.exists(os.path.dirname(full_abs_path))}")
            print(f"------------------")
            # wavfile.write(output_path, 24000, audio_int16)
            sf.write(full_abs_path, audio_int16, 24000)
            
            del full_waveform
            # torch.cuda.empty_cache()
            # os.makedirs(os.path.dirname(full_abs_path), exist_ok=True)
            return output_path

                    
        except Exception as e:
            raise Exception(f"Error generating speech: {str(e)}")
    
    def load_model(self):
        if self.model is None:
            
            try:
                # Import time for timing
                import time
                start_time = time.time()
                
                # Get the actual model path using our discovery function
                model_full_path = r"I:\voiceover\model\Omnivoice"
                
                # Prepare attention implementation kwargs
                model_kwargs = {
                    "device_map": "cuda:0",
                    "dtype": torch.float16,
                    "load_asr": True,
                    "asr_model_name": r"I:\voiceover\model\Whisper"
                }

                try:
                    # Use the correct path (parent if 4bit subfolder is used)
                    logger.info(f"Loading model from: {model_full_path}")

                    self.model = OmniVoice.from_pretrained(
                        model_full_path,
                        **model_kwargs
                    )
                except Exception as e:
                    logger.error(f"Failed to load model from {model_full_path}: {e}")
                    raise Exception(
                        f"Failed to load model from {model_full_path}.\n"
                        f"Please ensure the model files are complete and properly downloaded.\n"
                        f"Required files: config.json, pytorch_model.bin or model safetensors\n"
                        f"Error: {str(e)}"
                    )

                elapsed = time.time() - start_time
                logger.info(f"Model loaded in {elapsed:.2f} seconds")

                # _ = self.model.generate(text="Warmup", temperature=0.1)

                # Verify model was loaded
                if self.model is None:
                    raise Exception("Model failed to load - model is None after loading")

                # self.current_model_folder = model_folder
                # self.current_attention_type = attention_type
                # self.current_quantize_llm = quantize_llm
                # self.current_lora_path = lora_path
                
            except Exception as e:
                logger.error(f"Failed to load VibeVoice model: {str(e)}")
                raise Exception(f"Model loading failed: {str(e)}")
            
    def load_model_whisper(self):
        if self.model_whisper is None:
            try:
                # Import time for timing
                import time
                start_time = time.time()

                from faster_whisper import WhisperModel, BatchedInferencePipeline
                model_name = r"I:\voiceover\model\Faster-Whisper"

                try:
                    # Use the correct path (parent if 4bit subfolder is used)
                    logger.info(f"Loading model from: {model_name}")

                    self._base_model_whisper = WhisperModel(model_name, device="cuda", compute_type="float16", local_files_only=True)
                    self.model_whisper = BatchedInferencePipeline(model=self._base_model_whisper)

                except Exception as e:
                    logger.error(f"Failed to load model from {model_name}: {e}")
                    raise Exception(
                        f"Failed to load model from {model_name}.\n"
                        f"Error: {str(e)}"
                    )

                elapsed = time.time() - start_time
                logger.info(f"Model loaded in {elapsed:.2f} seconds")

                # _ = self.model.generate(text="Warmup", temperature=0.1)

                # Verify model was loaded
                if self.model_whisper is None:
                    raise Exception("Model failed to load - model is None after loading")

                # self.current_model_folder = model_folder
                # self.current_attention_type = attention_type
                # self.current_quantize_llm = quantize_llm
                # self.current_lora_path = lora_path
                
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {str(e)}")
                raise Exception(f"Model loading failed: {str(e)}")
    
    def _create_synthetic_voice_sample(self, speaker_idx: int) -> np.ndarray:
        """Create synthetic voice sample for a specific speaker"""
        sample_rate = 24000
        duration = 1.0
        samples = int(sample_rate * duration)
        
        t = np.linspace(0, duration, samples, False)
        
        # Create realistic voice-like characteristics for each speaker
        # Use different base frequencies for different speaker types
        base_frequencies = [120, 180, 140, 200]  # Mix of male/female-like frequencies
        base_freq = base_frequencies[speaker_idx % len(base_frequencies)]
        
        # Create vowel-like formants (like "ah" sound) - unique per speaker
        formant1 = 800 + speaker_idx * 100  # First formant
        formant2 = 1200 + speaker_idx * 150  # Second formant
        
        # Generate more voice-like waveform
        voice_sample = (
            # Fundamental with harmonics (voice-like)
            0.6 * np.sin(2 * np.pi * base_freq * t) +
            0.25 * np.sin(2 * np.pi * base_freq * 2 * t) +
            0.15 * np.sin(2 * np.pi * base_freq * 3 * t) +
            
            # Formant resonances (vowel-like characteristics)
            0.1 * np.sin(2 * np.pi * formant1 * t) * np.exp(-t * 2) +
            0.05 * np.sin(2 * np.pi * formant2 * t) * np.exp(-t * 3) +
            
            # Natural breath noise (reduced)
            0.02 * np.random.normal(0, 1, len(t))
        )
        
        # Add natural envelope (like human speech pattern)
        # Quick attack, slower decay with slight vibrato (unique per speaker)
        vibrato_freq = 4 + speaker_idx * 0.3  # Slightly different vibrato per speaker
        envelope = (np.exp(-t * 0.3) * (1 + 0.1 * np.sin(2 * np.pi * vibrato_freq * t)))
        voice_sample *= envelope * 0.08  # Lower volume
        
        return voice_sample.astype(np.float32)

    def _adjust_voice_speed(self, audio_np: np.ndarray, speed_factor: float, sample_rate: int = 24000) -> np.ndarray:
        """Adjust voice speed using time-stretching without changing pitch significantly

        Args:
            audio_np: Input audio array
            speed_factor: Speed adjustment (0.75 = 25% slower, 1.25 = 25% faster)
            sample_rate: Sample rate of the audio

        Returns:
            Speed-adjusted audio array
        """
        if speed_factor == 1.0:
            return audio_np  # No change needed

        # Calculate new length
        original_length = len(audio_np)
        target_length = int(original_length / speed_factor)

        # Use linear interpolation for time-stretching
        # This is a simple approach that works reasonably well for small speed changes
        original_indices = np.arange(original_length)
        target_indices = np.linspace(0, original_length - 1, target_length)

        # Interpolate the audio to the new length
        adjusted_audio = np.interp(target_indices, original_indices, audio_np)

        logger.info(f"Adjusted voice speed by factor {speed_factor:.2f} ({original_length} -> {target_length} samples)")

        return adjusted_audio.astype(np.float32)

    def _prepare_audio_from_comfyui(self, voice_audio, target_sample_rate: int = 24000, speed_factor: float = 1.0) -> Optional[np.ndarray]:
        """Prepare audio from ComfyUI format to numpy array"""
        if voice_audio is None:
            return None
            
        # Extract waveform from ComfyUI audio format
        if isinstance(voice_audio, dict) and "waveform" in voice_audio:
            waveform = voice_audio["waveform"]
            input_sample_rate = voice_audio.get("sample_rate", target_sample_rate)
            
            # Convert to numpy (handling BFloat16 tensors)
            if isinstance(waveform, torch.Tensor):
                # Convert to float32 first as numpy doesn't support BFloat16
                audio_np = waveform.cpu().float().numpy()
            else:
                audio_np = np.array(waveform)
            
            # Handle different audio shapes
            if audio_np.ndim == 3:  # (batch, channels, samples)
                audio_np = audio_np[0, 0, :]  # Take first batch, first channel
            elif audio_np.ndim == 2:  # (channels, samples)
                audio_np = audio_np[0, :]  # Take first channel
            # If 1D, leave as is
            
            # Resample if needed
            if input_sample_rate != target_sample_rate:
                target_length = int(len(audio_np) * target_sample_rate / input_sample_rate)
                audio_np = np.interp(np.linspace(0, len(audio_np), target_length), 
                                   np.arange(len(audio_np)), audio_np)
            
            # Ensure audio is in correct range [-1, 1]
            audio_max = np.abs(audio_np).max()
            if audio_max > 0:
                audio_np = audio_np / max(audio_max, 1.0)  # Normalize

            # Apply speed adjustment if requested
            if speed_factor != 1.0:
                audio_np = self._adjust_voice_speed(audio_np, speed_factor, target_sample_rate)
                speed_percent = int((speed_factor - 1.0) * 100)
                if speed_percent > 0:
                    logger.info(f"Applied voice speed adjustment: +{speed_percent}% faster")
                else:
                    logger.info(f"Applied voice speed adjustment: {speed_percent}% slower")

            return audio_np.astype(np.float32)
        
        return None

    def prepare_audio_from_path_torch(self, audio_path: str, target_sample_rate: int = 24000):
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # В моно
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        # Ресемплинг
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
            waveform = resampler(waveform)
            
        return waveform.squeeze().numpy().astype(np.float32)

    def prepare_audio_from_path_librosa(self, audio_path: str, target_sample_rate: int = 24000,
                                        speed_factor: float = 1.0) -> np.ndarray:
        # import librosa
        # import numpy as np
        import os

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Аудиофайл не найден: {audio_path}")

        # 1. Загрузка и автоматический ресемплинг (High Quality)
        # Используем res_type='kaiser_fast' для баланса скорости и качества
        audio_np, _ = librosa.load(audio_path, sr=target_sample_rate, mono=True, res_type='kaiser_fast')

        # 2. Масштабирование амплитуды (Точно как в ComfyUI)
        # Это предотвращает клиппинг, но не усиливает тихий звук до максимума
        audio_max = np.abs(audio_np).max()
        if audio_max > 0:
            audio_np = audio_np / max(audio_max, 1.0)

        # 3. Регулировка скорости
        if speed_factor != 1.0:
            if hasattr(self, '_adjust_voice_speed'):
                audio_np = self._adjust_voice_speed(audio_np, speed_factor, target_sample_rate)
                # Логируем как в оригинале
                speed_percent = int((speed_factor - 1.0) * 100)
                sign = "+" if speed_percent > 0 else ""
                print(f"INFO: Applied voice speed adjustment: {sign}{speed_percent}%")
            else:
                print("WARNING: _adjust_voice_speed not found, skipping speed adjustment")

        return audio_np.astype(np.float32)

    def prepare_audio_from_path_sf(self, audio_path, target_sr=24000, speed_factor=1.0):
        # Читаем файл. soundfile сам найдет нужные DLL внутри своего пакета
        data, samplerate = sf.read(audio_path)

        # Конвертация в моно, если нужно
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # Если частота совпадает (а для образцов VibeVoice это обычно 24кГц),
        # то мы просто отдаем массив.
        if samplerate == target_sr:
            return data.astype(np.float32)

        # Если ресемплинг всё же нужен, используем простейший метод через torch
        import torch.nn.functional as F
        import torch

        input_tensor = torch.from_numpy(data).float().view(1, 1, -1)
        new_size = int((len(data) * target_sr / samplerate) / speed_factor)
        # Линейная интерполяция — быстро и достаточно для голоса
        resampled = F.interpolate(input_tensor,
                                  size=new_size,
                                  mode='linear',
                                  align_corners=False)
        return resampled.view(-1).numpy().astype(np.float32)

    def _split_text_into_chunks(self, text: str, max_words: int = 250) -> List[str]:
        """Split long text into manageable chunks at sentence boundaries
        
        Args:
            text: The text to split
            max_words: Maximum words per chunk (default 250 for safety)
        
        Returns:
            List of text chunks
        """
        import re
        
        # Split into sentences (handling common abbreviations)
        # This regex tries to split on sentence endings while avoiding common abbreviations
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        # If regex split didn't work well, fall back to simple split
        if len(sentences) == 1 and len(text.split()) > max_words:
            # Fall back to splitting on any period followed by space
            sentences = text.replace('. ', '.|').split('|')
            sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_words = sentence.split()
            sentence_word_count = len(sentence_words)
            
            # If single sentence is too long, split it further
            if sentence_word_count > max_words:
                # Split long sentence at commas or semicolons
                sub_parts = re.split(r'[,;]', sentence)
                for part in sub_parts:
                    part = part.strip()
                    if not part:
                        continue
                    part_words = part.split()
                    part_word_count = len(part_words)
                    
                    if current_word_count + part_word_count > max_words and current_chunk:
                        # Save current chunk
                        chunks.append(' '.join(current_chunk))
                        current_chunk = [part]
                        current_word_count = part_word_count
                    else:
                        current_chunk.append(part)
                        current_word_count += part_word_count
            else:
                # Check if adding this sentence would exceed the limit
                if current_word_count + sentence_word_count > max_words and current_chunk:
                    # Save current chunk and start new one
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_word_count = sentence_word_count
                else:
                    # Add sentence to current chunk
                    current_chunk.append(sentence)
                    current_word_count += sentence_word_count
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # If no chunks were created, return the original text
        if not chunks:
            chunks = [text]
        
        logger.info(f"Split text into {len(chunks)} chunks (max {max_words} words each)")
        for i, chunk in enumerate(chunks):
            word_count = len(chunk.split())
            logger.debug(f"Chunk {i+1}: {word_count} words")
        
        return chunks
    
    def _parse_pause_keywords(self, text: str) -> List[Tuple[str, Any]]:
        """Parse [pause] and [pause:ms] keywords from text
        
        Args:
            text: Text potentially containing pause keywords
            
        Returns:
            List of tuples: ('text', str) or ('pause', duration_ms)
        """
        segments = []
        # Pattern matches [pause] or [pause:1500] where 1500 is milliseconds
        pattern = r'\[pause(?::(\d+))?\]'
        
        last_end = 0
        for match in re.finditer(pattern, text):
            # Add text segment before pause (if any)
            if match.start() > last_end:
                text_segment = text[last_end:match.start()].strip()
                if text_segment:  # Only add non-empty text segments
                    segments.append(('text', text_segment))
            
            # Add pause segment with duration (default 1000ms = 1 second)
            duration_ms = int(match.group(1)) if match.group(1) else 1000
            segments.append(('pause', duration_ms))
            last_end = match.end()
        
        # Add remaining text after last pause (if any)
        if last_end < len(text):
            remaining_text = text[last_end:].strip()
            if remaining_text:
                segments.append(('text', remaining_text))
        
        # If no pauses found, return original text as single segment
        if not segments:
            segments.append(('text', text))
        
        logger.debug(f"Parsed text into {len(segments)} segments (including pauses)")
        return segments

    def _get_prompt_path(self, sample_path: str) -> str:
        import os
        
        # Берём только имя файла без расширения
        filename = os.path.basename(sample_path)
        name_no_ext = os.path.splitext(filename)[0]
        
        # Формируем путь к prompt'у
        prompt_name = f"{name_no_ext}.prompt"
        prompt_path = os.path.join(VOICE_PROMPTS_PATH, prompt_name)
        
        return prompt_path
    
    def _generate_silence(self, duration_ms: int, sample_rate: int = 24000) -> dict:
        """Generate silence audio tensor for specified duration
        
        Args:
            duration_ms: Duration of silence in milliseconds
            sample_rate: Sample rate (default 24000 Hz for VibeVoice)
            
        Returns:
            Audio dict with silence waveform
        """
        # Calculate number of samples for the duration
        num_samples = int(sample_rate * duration_ms / 1000.0)
        
        # Create silence tensor with shape (1, 1, num_samples) to match audio format
        silence_waveform = torch.zeros(1, 1, num_samples, dtype=torch.float32)
        
        logger.info(f"Generated {duration_ms}ms silence ({num_samples} samples)")
        
        return {
            "waveform": silence_waveform,
            "sample_rate": sample_rate
        }
    
    def _format_text_for_vibevoice(self, text: str, speakers: list) -> str:
        """Format text with speaker information for VibeVoice using correct format"""
        # Remove any newlines from the text to prevent parsing issues
        # The processor splits by newline and expects each line to have "Speaker N:" format
        text = text.replace('\n', ' ').replace('\r', ' ')
        # Clean up multiple spaces
        text = ' '.join(text.split())
        
        # VibeVoice expects format: "Speaker 1: text" not "Name: text"
        if len(speakers) == 1:
            return f"Speaker 1: {text}"
        else:
            # Check if text already has proper Speaker N: format
            if re.match(r'^\s*Speaker\s+\d+\s*:', text, re.IGNORECASE):
                return text
            # If text has name format, convert to Speaker N format
            elif any(f"{speaker}:" in text for speaker in speakers):
                formatted_text = text
                for i, speaker in enumerate(speakers):
                    formatted_text = formatted_text.replace(f"{speaker}:", f"Speaker {i+1}:")
                return formatted_text
            else:
                # Plain text, assign to first speaker
                return f"Speaker 1: {text}"
    
    def _generate_with_vibevoice(self, formatted_text: str, voice_samples: List[np.ndarray],
                                cfg_scale: float, seed: int, diffusion_steps: int, use_sampling: bool,
                                temperature: float = 0.95, top_p: float = 0.95, llm_lora_strength: float = 1.0) -> dict:
        """Generate audio using VibeVoice model"""
        try:
            # Ensure model and processor are loaded
            if self.model is None or self.processor is None:
                raise Exception("Model or processor not loaded")
            
            # Set seeds for reproducibility
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)  # For multi-GPU
            
            # Also set numpy seed for any numpy operations
            np.random.seed(seed)
            
            # Set diffusion steps
            self.model.set_ddpm_inference_steps(diffusion_steps)
            logger.info(f"Starting audio generation with {diffusion_steps} diffusion steps...")
            
            # Check for interruption before starting generation
            if INTERRUPTION_SUPPORT:
                try:
                    import comfy.model_management as mm
                    
                    # Check if we're being interrupted right now
                    # The interrupt flag is reset by ComfyUI before each node execution
                    # So we only check model_management's throw_exception_if_processing_interrupted
                    # which is the proper way to check for interruption
                    mm.throw_exception_if_processing_interrupted()
                    
                except ImportError:
                    # If comfy.model_management is not available, skip this check
                    pass
            
            # Prepare inputs using processor
            inputs = self.processor(
                [formatted_text],  # Wrap text in list
                voice_samples=[voice_samples], # Provide voice samples for reference
                return_tensors="pt",
                return_attention_mask=True
            )
            
            # Move to device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Estimate tokens for user information (not used as limit)
            text_length = len(formatted_text.split())
            estimated_tokens = int(text_length * 2.5)  # More accurate estimate for display
            
            # Log generation start with explanation
            logger.info(f"Generating audio with {diffusion_steps} diffusion steps...")
            logger.info(f"Note: Progress bar shows max possible tokens, not actual needed (~{estimated_tokens} estimated)")
            logger.info("The generation will stop automatically when audio is complete")
            
            # Create stop check function for interruption support
            stop_check_fn = None
            if INTERRUPTION_SUPPORT:
                def check_comfyui_interrupt():
                    """Check if ComfyUI has requested interruption"""
                    try:
                        if hasattr(execution, 'PromptExecutor') and hasattr(execution.PromptExecutor, 'interrupted'):
                            interrupted = execution.PromptExecutor.interrupted
                            if interrupted:
                                logger.info("Generation interrupted by user via stop_check_fn")
                            return interrupted
                    except:
                        pass
                    return False
                
                stop_check_fn = check_comfyui_interrupt
            
            # Generate with official parameters
            with torch.no_grad():
                if use_sampling:
                    # Use sampling mode (less stable but more varied)
                    output = self.model.generate(
                        **inputs,
                        tokenizer=self.processor.tokenizer,
                        cfg_scale=cfg_scale,
                        max_new_tokens=None,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        stop_check_fn=stop_check_fn,
                    )
                else:
                    # Use deterministic mode like official examples
                    output = self.model.generate(
                        **inputs,
                        tokenizer=self.processor.tokenizer,
                        cfg_scale=cfg_scale,
                        max_new_tokens=None,
                        do_sample=False,  # More deterministic generation
                        stop_check_fn=stop_check_fn,
                    )
                
                # Check if we got actual audio output
                if hasattr(output, 'speech_outputs') and output.speech_outputs:
                    speech_tensors = output.speech_outputs

                    if isinstance(speech_tensors, list) and len(speech_tensors) > 0:
                        audio_tensor = torch.cat(speech_tensors, dim=-1)
                    else:
                        audio_tensor = speech_tensors
                    
                    # Ensure proper format (1, 1, samples)
                    if audio_tensor.dim() == 1:
                        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
                    elif audio_tensor.dim() == 2:
                        audio_tensor = audio_tensor.unsqueeze(0)
                    
                    # Convert to float32 for compatibility with downstream nodes (Save Audio, etc.)
                    # Many audio processing nodes don't support BFloat16
                    return {
                        "waveform": audio_tensor.cpu().float(),
                        "sample_rate": 24000
                    }
                    
                elif hasattr(output, 'sequences'):
                    logger.error("VibeVoice returned only text tokens, no audio generated")
                    raise Exception("VibeVoice failed to generate audio - only text tokens returned")
                    
                else:
                    logger.error(f"Unexpected output format from VibeVoice: {type(output)}")
                    raise Exception(f"VibeVoice returned unexpected output format: {type(output)}")
                
        except Exception as e:
            # For real errors, log and re-raise with context
            logger.error(f"VibeVoice generation failed: {e}")
            raise Exception(f"VibeVoice generation failed: {str(e)}")
    
    def _generate_with_vibevoice_batch(self, formatted_texts: List[str], voice_samples: List[np.ndarray],
                                    cfg_scale: float, seed: int, diffusion_steps: int, use_sampling: bool,
                                    temperature: float = 0.95, top_p: float = 0.95, llm_lora_strength: float = 1.0) -> dict:
        """Generate audio using VibeVoice model"""
        try:
            # Ensure model and processor are loaded
            if self.model is None or self.processor is None:
                raise Exception("Model or processor not loaded")
            
            # Set seeds for reproducibility
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)  # For multi-GPU
            
            # Also set numpy seed for any numpy operations
            np.random.seed(seed)
            
            # Set diffusion steps
            self.model.set_ddpm_inference_steps(diffusion_steps)
            logger.info(f"Starting audio generation with {diffusion_steps} diffusion steps...")
            
            # Check for interruption before starting generation
            if INTERRUPTION_SUPPORT:
                try:
                    import comfy.model_management as mm
                    
                    # Check if we're being interrupted right now
                    # The interrupt flag is reset by ComfyUI before each node execution
                    # So we only check model_management's throw_exception_if_processing_interrupted
                    # which is the proper way to check for interruption
                    mm.throw_exception_if_processing_interrupted()
                    
                except ImportError:
                    # If comfy.model_management is not available, skip this check
                    pass
            
            # Prepare inputs using processor
            inputs = self.processor(
                formatted_texts,  # Wrap text in list
                voice_samples=voice_samples, # Provide voice samples for reference
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_attention_mask=True
            )
            
            # Move to device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            lengths = [len(t.split()) for t in formatted_texts]
            total_words = sum(lengths)
            estimated_tokens = int(total_words * 2.5)
            
            # Log generation start with explanation
            logger.info(f"Generating audio with {diffusion_steps} diffusion steps...")
            logger.info(f"Note: Progress bar shows max possible tokens, not actual needed (~{estimated_tokens} estimated)")
            print(f"Note: Progress bar shows max possible tokens, not actual needed (~{estimated_tokens} estimated)")
            logger.info("The generation will stop automatically when audio is complete")
            
            # Create stop check function for interruption support
            stop_check_fn = None
            if INTERRUPTION_SUPPORT:
                def check_comfyui_interrupt():
                    """Check if ComfyUI has requested interruption"""
                    try:
                        if hasattr(execution, 'PromptExecutor') and hasattr(execution.PromptExecutor, 'interrupted'):
                            interrupted = execution.PromptExecutor.interrupted
                            if interrupted:
                                logger.info("Generation interrupted by user via stop_check_fn")
                            return interrupted
                    except:
                        pass
                    return False
                
                stop_check_fn = check_comfyui_interrupt
            
            # Generate with official parameters
            with torch.no_grad():
                if use_sampling:
                    # Use sampling mode (less stable but more varied)
                    output = self.model.generate(
                        **inputs,
                        tokenizer=self.processor.tokenizer,
                        cfg_scale=cfg_scale,
                        max_new_tokens=None,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        stop_check_fn=stop_check_fn,
                    )
                else:
                    # Use deterministic mode like official examples
                    output = self.model.generate(
                        **inputs,
                        tokenizer=self.processor.tokenizer,
                        cfg_scale=cfg_scale,
                        max_new_tokens=None,
                        do_sample=False,  # More deterministic generation
                        stop_check_fn=stop_check_fn,
                    )
                
                # Check if we got actual audio output
                if hasattr(output, 'speech_outputs') and output.speech_outputs:
                    speech_tensors = output.speech_outputs
                    
                    # speech_tensors в батче обычно выглядит как список списков или тензор
                    # Нам нужно убедиться, что мы не склеиваем РАЗНЫЕ реплики в одну
                    
                    processed_waveforms = []
                    
                    # Если это батч, то speech_tensors будет иметь размерность [batch_size, sequence_length]
                    # или это будет список, где каждый элемент — аудио для одного текста из батча

                    # if isinstance(speech_tensors, list) and len(speech_tensors) > 0:
                    #     audio_tensor = torch.cat(speech_tensors, dim=-1)
                    # else:
                    #     audio_tensor = speech_tensors
                    
                    # # Ensure proper format (1, 1, samples)
                    # if audio_tensor.dim() == 1:
                    #     audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
                    # elif audio_tensor.dim() == 2:
                    #     audio_tensor = audio_tensor.unsqueeze(0)
                    
                    # # Convert to float32 for compatibility with downstream nodes (Save Audio, etc.)
                    # # Many audio processing nodes don't support BFloat16
                    # return {
                    #     "waveform": audio_tensor.cpu().float(),
                    #     "sample_rate": 24000
                    # }
                    
                    # Допустим, output.speech_outputs — это наш батч результатов
                    for t in output.speech_outputs:
                        t_clean = t.cpu().float()
                        # Приводим к нужному формату (1, samples)
                        if t_clean.dim() == 1:
                            t_clean = t_clean.unsqueeze(0)
                        processed_waveforms.append(t_clean)

                    full_waveform = torch.cat(processed_waveforms, dim=-1)
                    
                    return {
                        "waveform": full_waveform,
                        "sample_rate": 24000
                    }
                    
                elif hasattr(output, 'sequences'):
                    logger.error("VibeVoice returned only text tokens, no audio generated")
                    raise Exception("VibeVoice failed to generate audio - only text tokens returned")
                    
                else:
                    logger.error(f"Unexpected output format from VibeVoice: {type(output)}")
                    raise Exception(f"VibeVoice returned unexpected output format: {type(output)}")
                
        except Exception as e:
            # For real errors, log and re-raise with context
            logger.error(f"VibeVoice generation failed: {e}")
            raise Exception(f"VibeVoice generation failed: {str(e)}")


DATAMODULE_TABLE_GUARD_CLAUSE = 'if not VoiceOver or not VoiceOver.DataModules then return end'
REPLACE_DICT = {'$b': '\n', '$B': '\n', '$n': 'adventurer', '$N': 'Adventurer',
                '$C': 'Adventurer', '$c': 'adventurer', '$R': 'Traveler', '$r': 'traveler',
                '<name>': 'adventurer', '<Name>': 'Adventurer',
                '<race>': 'traveler', '<Race>': 'Traveler',
                '<class>': 'adventurer', '<Class>': 'Adventurer',
                '$t citizen : citizen': 'citizen',
                '$T Civvy : Civvy;': '',
                ' -- ': ', ', '—': ', ', '--': ', ', " - ": ", ",
                '*Hic*': 'Hick!',
                # Factions / Regions
                "Draenei": "Dray-nai",
                "Lordaeron": "Law-da-ron",
                "Quel'Thalas": "Kwel-tha-las",
                "Dalaran": "Dalah-ran",
                "Naxxramas": "Nax-ramas",
                "Scholomance": "Skolo-mance",
                "Stratholme": "Strath-holm",
                "Atal'ai":"Ata-lai",
                "Naaru":"Naroo",
                "Dragonflight": "Dragon-flight",
                "Necrolord":"necro-lord",
                "bloodmage":"blood-mage",
                "taunka'le":"taunka-lay",
                "Gnomeregan":"Gnome-ree-gahn",
                "wyrm":"werm",
                "Oneqwah":"Ohnay-kwah",
                "vrykul":"vrye-kool",
                "thor modan":"thormoe-dhaan",
                "ursoc":"ursok",
                "dun argol":"duunar-goll",
                "earthen":"ehrrthin",



                # Bosses / NPCs
                "Malygos": "Maali-goss",
                "Kel'Thuzad": "Kel-thu-zahd",
                "Anub'arak": "Anub-araak",
                "Kael'thas": "Kale-thoss",
                "Mok'Nathal":"Mockna-tholl",
                "orcish":"orkish",
                "Kil'jaeden": "Kil-jayden",
                "Archimonde": "Arki-mond",
                "C'Thun": "Kuh-thoon",
                "Yogg-Saron": "Yog-suh-ron",
                "Gjalerbron": "Yal-er-bron",
                "Heb'Drakkar": "Heb-drah-kar",
                "Rageclaw": "Rage-claw",
                "Ragemane": "Rage-mane",
                "Verna":"Vur-nah",
                "Pathaleon":"Pathalion",
                "Demetrian":"Deh-mee-tree-ahn",
                "Zul'Marosh": "Zool-marosh",
                "Medivh":"Medaeve",
                "Dar'Khan":"DarKahn",
                "Stormrage": "Storm-rayge",
                "Gul'dan":"Gool dan",
                "undead":"on-ded",
                "undeath":"on-deth",
                "Lok'tar ogar":"Loktaro garr",
                "Dughan": "Doo-ghan",
                "Taxxlo": "Taxlo",

                # Places
                "Icecrown": "Ice crown",
                "Dragonshrine": "Dragon-shrine",
                "Auchindoun": "Aw-kin-doon",
                "Hyjal": "High-jahl",
                "Mathystra": "Math-is-trah",
                "Ulduar": "Ool-dwar",
                "Utgarde": "Oot-guard",
                "Zul'Aman": "Zool-ahmaan",
                "Zul'Drak": "Zool-drak",
                "Ahn'kahet": "On-ka-het",
                "Gundrak": "Gun-drak",
                "Modan":"Moe-dahn",
                "Ahn'Qiraj":"On-kee-rahj",
                "Elwynn":"Elwin",
                "Arcatraz":"Arc-a-traz",
                "Stonetalon":"stone-talon",
                "Kalimdor":"Kalim-dor",

                # Titans / Lore
                "Tyr": "Teer",
                "Freya": "Fray-ah",
                "Thorim": "Thor-rim",
                "Hodir": "Ho-deer",
                #"Loken": "Low-ken",
                "Ymiron": "Yee-miron",
                "Elune":"Ehloon",

                # Misc
                "Felwood": "Fell-wood",
                "Ashenvale": "Ashen-veil",
                #"Grizzly Hills": "Grizz-lee Hills",
                "Sha'naar":"Shanar",
                "Sin'dorei":"Sindoh-rye",
                "Gorefiend":"Gorfeend",
                "Indu'le": "Indu-lay",
                "Venture Co.": "Venture Company"
                }

def get_hash(text):
    hash_object = hashlib.md5(text.encode())
    return hash_object.hexdigest()


def prune_quest_id_table(quest_id_table):
    def is_single_quest_id(nested_dict):
        if isinstance(nested_dict, dict):
            if len(nested_dict) == 1:
                return is_single_quest_id(next(iter(nested_dict.values())))
            else:
                return False
        else:
            return True

    def single_quest_id(nested_dict):
        if isinstance(nested_dict, dict):
            return single_quest_id(next(iter(nested_dict.values())))
        else:
            return nested_dict

    pruned_table = {}
    for source_key, source_value in quest_id_table.items():
        pruned_table[source_key] = {}
        for title_key, title_value in source_value.items():
            if is_single_quest_id(title_value):
                pruned_table[source_key][title_key] = single_quest_id(title_value)
            else:
                pruned_table[source_key][title_key] = {}
                for npc_key, npc_value in title_value.items():
                    if is_single_quest_id(npc_value):
                        pruned_table[source_key][title_key][npc_key] = single_quest_id(npc_value)
                    else:
                        pruned_table[source_key][title_key][npc_key] = npc_value

    return pruned_table


def clean_folder(df, expansions=[0, 1, 2], module_name="AI_VoiceOverData_TBC"):
    # Filter DF by expansion
    df = df[df['expansion'].isin(expansions)]

    # Collect all expected filenames
    expected_files = set()
    for _, row in df.iterrows():
        quest_id = str(row.get("quest", "")).strip()
        source = str(row.get("source", "")).strip().lower()

        if quest_id:  # quest dialogue
            base_name = f"{quest_id}-{source}.mp3"
            subfolder = "quests"
        else:  # gossip dialogue
            base_name = f"{row['templateText_race_gender_hash']}.mp3"
            subfolder = "gossip"

        # If gender is specified, add gendered versions
        if row.get("player_gender") in ["m", "f"]:
            filename = f"{row['player_gender']}-{base_name}"
            expected_files.add(os.path.join(subfolder, filename))
        else:
            # Add both gendered and non-gendered, just in case
            expected_files.add(os.path.join(subfolder, base_name))
            expected_files.add(os.path.join(subfolder, f"m-{base_name}"))
            expected_files.add(os.path.join(subfolder, f"f-{base_name}"))

    # Define base sound folder
    sound_folder = os.path.join(module_name, "generated", "sounds")

    # Walk through quests & gossip folders and clean up
    for subfolder in ["quests", "gossip"]:
        folder_path = os.path.join(sound_folder, subfolder)
        if not os.path.isdir(folder_path):
            continue  # skip missing subfolders

        for fname in os.listdir(folder_path):
            fpath = os.path.join(subfolder, fname)  # relative path
            full_path = os.path.join(folder_path, fname)

            if fpath not in expected_files:
                print(f"🗑️ Deleting extra file: {full_path}")
                os.remove(full_path)

    print("✅ Folder cleanup complete.")


def clean_brackets(s):
    if pd.isna(s):
        return s

    # If ENTIRE string is just <...>, keep inside
    m = re.fullmatch(r'<\s*(.*?)\s*>', s)
    if m:
        return m.group(1)  # return only inner text

    # Otherwise remove <...> completely
    return re.sub(r'<.*?>', '', s)

def convert_numbers_to_words(text):
    """Convert all numbers in text to their word representation"""

    def replace_number(match):
        num = match.group()
        words = num2words(num)
        # Remove all commas
        words = words.replace(',', '')
        return words

    # Find all numbers in text
    pattern = r'\b\d+\b'
    return re.sub(pattern, replace_number, text)

def split_by_speakers_quotes(text):
    """
    Process text with quotes.
    - "quotes" = NPC dialogue (speaker 1)
    - Everything else = narrator (speaker 2)
    Returns: (processed_text, has_narrator)
    """
    # Find all quoted text
    quoted_matches = list(re.finditer(r'"([^"]+)"', text))
    
    # Check if any quoted text is a proper sentence
    has_sentence = False
    for match in quoted_matches:
        quoted_text = match.group(1).strip()
        # Has multiple words and decent length
        if len(quoted_text) > 20 and len(quoted_text.split()) > 3:
            has_sentence = True
            break
    
    if not has_sentence:
        return text, False

    # # Check if there are any quotes
    # has_quotes = bool(re.search(r'"[^"]+"', text))
    
    # if not has_quotes:
    #     return text, False
    
    # Pattern to match quoted text
    pattern = r'"(?:[^"\\]|\\.)*"'
    
    parts = []
    last_end = 0
    processed = []
    
    for match in re.finditer(pattern, text):
        # Add text before quote (narrator)
        if match.start() > last_end:
            narrator_text = text[last_end:match.start()].strip()
            if narrator_text:
                sighs_or_laughes = 'sighs' in narrator_text or ' sigh' in narrator_text or 'laughs' in narrator_text or ' laugh' in narrator_text or '*gasp*' in narrator_text
                sound_word = '[sigh]' if ('sighs' in narrator_text or ' sigh' in narrator_text ) else '[laughter]' if ('laughs' in narrator_text or ' laugh' in narrator_text) else '[surprise-ah]' if '*gasp*' in narrator_text else ''
                sound_needed = sighs_or_laughes
                # replace with sighs or laughs if it's short
                if sighs_or_laughes and len(narrator_text.split()) < 5:
                    processed.append(f"[1]: {sound_word}")
                    sound_needed = False
                else:
                    processed.append(f"[2]: {narrator_text}")
                    # add after if was long text
                    if sighs_or_laughes and sound_needed:
                        processed.append(f"[1]: {sound_word}")
        
        # Add the quoted text (NPC)
        quoted = match.group()
        processed.append(f"[1]: {quoted[1:-1]}")
        
        last_end = match.end()
    
    # Add remaining text after last quote (narrator)
    if last_end < len(text):
        remaining = text[last_end:].strip()
        if remaining:
            processed.append(f"[2]: {remaining}")
    
    return ' '.join(processed), True

def split_by_speakers_brackets(text):
    """
    Convert text with <...> tags to speaker-separated format
    Example: "Hello <Something> bye" -> "[1]: Hello [2]: Something [1]: bye"
    Returns: (processed_text, has_narrator)
    """

    has_brackets = bool(re.search(r'<[^>]+>', text))
    # Check if there are any <...> tags
    if not has_brackets:
        return text, False  # Return original text without labels

    # Split by <...> tags, keeping the tags
    parts = re.split(r'(<[^>]+>)', text)

    result = []
    current_speaker = 1

    for part in parts:
        if not part:  # Skip empty parts
            continue

        if part.startswith('<') and part.endswith('>'):
            # This is speaker 2 (remove brackets)
            content = part[1:-1].strip()
            if content:  # Only add if not empty
                sighs_or_laughes = 'sighs' in content or ' sigh' in content or 'laughs' in content or ' laugh' in content or '*gasp*' in content
                sound_word = '[sigh]' if ('sighs' in content or ' sigh' in content ) else '[laughter]' if ('laughs' in content or ' laugh' in content) else '[surprise-ah]' if '*gasp*' in content else ''
                sound_needed = sighs_or_laughes
                # replace with sighs or laughs if it's short
                if sound_word and len(content.split()) < 5:
                    result.append(f"[1]: {sound_word}")
                    sound_needed = False
                else:
                    result.append(f"[2]: {content}")
                    # add after if was long text
                    if sighs_or_laughes and sound_needed:
                        result.append(f"[1]: {sound_word}")
        else:
            # This is speaker 1
            content = part.strip()
            if content:  # Only add if not empty
                result.append(f"[1]: {content}")

    return ' '.join(result), True

class TTSProcessor(TTSEngine):
    def __init__(self, module_name='AI_VoiceOverData_TBC', problematic_quest_path="problematic_quests.json", problematic_gossip_path="problematic_gossip.json"):
        self.output_path = None
        self.available_models = get_available_models()
        self.model_choices = [display_name for _, display_name in self.available_models]
        self.default_model = self.model_choices[0] if self.model_choices else "No models found"
        self.module_name = module_name
        self.output_folder = os.path.join(self.module_name, 'generated')
        self.sound_output_folder = os.path.join(self.output_folder, 'sounds')
        self.module_name_export = module_name
        self.output_folder_export = os.path.join(ADDON_PATH_EXPORT, self.module_name_export)
        self.output_folder_export = os.path.join(self.output_folder_export, 'generated')
        self.sound_output_folder_export = os.path.join(self.output_folder_export, 'sounds')
        self.create_output_dirs()
        self.setup_voice_presets()
        self.problematic_quests = {}  # {quest_id: {'reason': '...', 'failed': True}}
        self.problematic_gossip = {}
        self.manual_override_quests = set()  # {"quest_id_accept", "quest_id_complete"}
        self.manual_override_gossip = set()
        self.problematic_quest_path = problematic_quest_path
        self.problematic_gossip_path = problematic_gossip_path
        self.load_problematic_quests_cache(self.problematic_quest_path)
        self.load_problematic_gossip_cache(self.problematic_gossip_path)
        super().__init__()

    # def __del__(self):
        # self.save_problematic_quests_cache(self.problematic_quest_path)
        # self.save_problematic_gossip_cache(self.problematic_gossip_path)

    def set_module_name(self, m_name):
        self.module_name = m_name

    def get_module_name(self):
        return self.module_name

    def create_output_subdirs(self, subdir: str):
        output_subdir = os.path.join(self.sound_output_folder, subdir)
        os.makedirs(output_subdir, exist_ok=True)

    def create_export_subdirs(self, subdir: str):
        output_subdir = os.path.join(self.sound_output_folder_export, subdir)
        os.makedirs(output_subdir, exist_ok=True)

    def setup_voice_presets(self):
        """Setup voice presets by scanning the voices directory."""

        self.voice_presets = {}
        self.available_voices = {}

        # Check if voices directory exists
        if not os.path.exists(VOICE_SAMPLE_PATH):
            print(f"Warning: Voices directory not found at {VOICE_SAMPLE_PATH}")
            return

        for voice_path in MODULE_NAMES:

            if voice_path not in self.voice_presets:
                self.voice_presets[voice_path] = {}

            voices_dir = os.path.join(VOICE_SAMPLE_PATH, voice_path)
            # Get all .pt files in the voices directory
            pt_files = glob.glob(os.path.join(voices_dir, "**", "*.mp3"), recursive=True)

            # Create dictionary with filename (without extension) as key
            for pt_file in pt_files:
                # key: filename without extension
                name = os.path.splitext(os.path.basename(pt_file))[0].lower()
                full_path = os.path.abspath(pt_file)
                self.voice_presets[voice_path][name] = full_path

            # Sort the voice presets alphabetically by name for better UI
            self.voice_presets[voice_path] = dict(sorted(self.voice_presets[voice_path].items()))

            if voice_path not in self.available_voices:
                self.available_voices[voice_path] = {}

            # Filter out voices that don't exist (this is now redundant but kept for safety)
            self.available_voices[voice_path] = {
                name: path for name, path in self.voice_presets[voice_path].items()
                if os.path.exists(path)
            }

            print(f"Module: {voice_path}")
            print(f"Found {len(self.available_voices[voice_path])} voice files")
            print(f"Available voices: {', '.join(self.available_voices[voice_path].keys())}")

    # def _prepare_voice_samples(self, speakers: list, voice_to_clone, voice_speed_factor: float = 1.0) -> List[np.ndarray]:
    #     """Prepare voice samples from input audio or create synthetic ones"""

    #     if voice_to_clone is not None:
    #         # Use the base class method to prepare audio with speed adjustment
    #         audio_np = self._prepare_audio_from_comfyui(voice_to_clone, speed_factor=voice_speed_factor)
    #         if audio_np is not None:
    #             return [audio_np]

    #     # Create synthetic voice samples for speakers
    #     voice_samples = []
    #     for i, speaker in enumerate(speakers):
    #         voice_sample = self._create_synthetic_voice_sample(i)
    #         voice_samples.append(voice_sample)

    #     return voice_samples

    def is_already_correct_format(self, file_path):
        """Uses ffprobe to check if the file is already 64kbps and 44.1kHz."""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(file_path)
        ]
        
        try:
            # Run ffprobe and capture the JSON output
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            
            streams = data.get("streams", [])
            if not streams:
                return False # No streams found, force processing
                
            audio_stream = streams[0]
            
            # Safely extract sample rate and bit rate
            sample_rate_str = audio_stream.get("sample_rate", "0")
            
            # Bitrate can sometimes be stored in the 'format' block rather than the 'stream' block for MP3s
            bit_rate_str = audio_stream.get("bit_rate") or data.get("format", {}).get("bit_rate", "0")
            
            sample_rate = int(sample_rate_str) if sample_rate_str else 0
            bit_rate = int(bit_rate_str) if bit_rate_str else 0
            bit_rate_kbps = round(bit_rate / 1000)

            if sample_rate != 44100:
                logger.info(f"Sample rate mismatch: {sample_rate}Hz. Needs re-encode.")
                return False

            kbps = bit_rate // 1000
            if not (62 <= kbps <= 66):
                logger.info(f"Bitrate mismatch: {kbps}kbps. Needs re-encode.")
                return False

            channels = int(audio_stream.get("channels", 0))
            if channels == 2:
                logger.info(f"Stereo detected: {channels} channels. Needs re-encode.")
                return False
                
            return True

        except Exception:
            # If ffprobe fails for any reason (corrupt header, etc.), return False so FFmpeg attempts to fix/convert it.
            return False
        
    def correct_format(self, file_path):
        """Worker function to check and process a single file."""
        # 1. Skip files we already converted previously by looking at the filename
        if file_path.stem.endswith("_64k"):
            return None

        # 2. Check if the original file is already 64k/44.1kHz natively
        if self.is_already_correct_format(file_path):
            return f"Skipped (Already correct format): {file_path.name}"

        # Define output filename
        output_file = file_path.with_name(f"{file_path.stem}_64k.mp3")
        
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(file_path),
            "-c:a", "libmp3lame",
            "-b:a", "64k",
            "-ar", "44100",
            "-ac", "1",
            "-map_metadata", "-1",
            str(output_file)
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # =====================================================================
            # OPTIONAL: Uncomment the line below to OVERWRITE original files.
            # =====================================================================
            os.replace(output_file, file_path)
            
            return f"Converted: {file_path.name}"
        except subprocess.CalledProcessError:
            return f"ERROR: Failed to convert {file_path.name}."
        except FileNotFoundError:
            return "FATAL_ERROR"

    def trim_long_silence(self, audio_path, min_silence_len=2000, silence_thresh=-35):
        """
        Remove silences longer than min_silence_len completely
        """
        audio = AudioSegment.from_file(audio_path)
        
        # Detect long silences
        silences = silence.detect_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh
        )
        
        if not silences:
            return audio_path
        
        # Build audio without long silences
        result = AudioSegment.empty()
        last_end = 0
        
        for start, end in silences:
            # Add audio before silence
            result += audio[last_end:start]
            # Skip silence (don't add it)
            last_end = end
        
        # Add remaining audio
        result += audio[last_end:]
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            result.export(f.name, format="mp3", bitrate="64k")
            return f.name

    def remove_long_silence_ffmpeg(self, file_path, output_path, min_silence_len=500, silence_thresh=-35):       
        # FFmpeg command:
        # silenceremove filter:
        # stop_periods=-1 (process all silences)
        # stop_duration (how long silence must be to be cut)
        # stop_threshold (volume level to consider silence)
        # -y (overwrite existing)
        # -v error (only show errors)
        
        # Convert ms to seconds
        silence_duration = min_silence_len / 1000.0
        cmd = [
            'ffmpeg',
            '-y',
            '-i', str(file_path),
            '-af', f'silenceremove=stop_periods=-1:stop_duration={silence_duration}:stop_threshold={silence_thresh}dB',
            '-v', 'error',
            str(output_path)
        ]
        
        try:
            # We use subprocess.run for efficiency
            subprocess.run(cmd, check=True, capture_output=True)
            return f"✓ {output_path.name}"
        except subprocess.CalledProcessError as e:
            return f"✗ {output_path.name}: {e.stderr.decode()}"
        
    def check_for_long_silence_ffmpeg(self, file_path, min_silence_len=2000, silence_thresh=-35):
        """
        Check for silence using FFmpeg (faster than pydub)
        """
        # Convert ms to seconds
        silence_duration = min_silence_len / 1000.0
        
        cmd = [
            'ffmpeg',
            '-i', str(file_path),
            '-af', f'silencedetect=noise={silence_thresh}dB:d={silence_duration}',
            '-f', 'null',
            '-'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if 'silence_start' in result.stderr:
            # # Optional: parse and print details
            # for line in result.stderr.split('\n'):
            #     if 'silence_start' in line:
            #         print(f"  {line.strip()}")
            print(f"⚠️  Silence found in {file_path}")
            return True
        
        return False

    def check_for_long_silence(self, file_path, min_silence_len=2000, silence_thresh=-35):
        """
        Detect actual silence, not low energy pauses
        min_silence_len: 2000ms = 2 seconds of continuous silence to flag
        silence_thresh: -35 dBFS is good for voice (more sensitive than -40)
        """
        audio = AudioSegment.from_file(file_path)

        # Get audio's average loudness
        avg_dBFS = audio.dBFS

        # Adjust threshold relative to audio (more adaptive)
        # If audio is quiet overall, use a more sensitive threshold
        if avg_dBFS < -30:
            # Audio is quiet, use more sensitive detection
            threshold = silence_thresh - 5  # -40 dBFS
        else:
            threshold = silence_thresh  # -35 dBFS
        
        # Detect silence
        silences = silence.detect_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=threshold
        )
        
        # Also detect sections that are VERY quiet (almost silence)
        very_quiet = silence.detect_nonsilent(
            audio,
            min_silence_len=500,  # 0.5 seconds
            silence_thresh=threshold - 10  # Even more sensitive
        )
        
        # Log what was found
        if silences:
            for start, end in silences:
                duration = (end - start) / 1000
                print(f"⚠️  Silence from {start/1000}s to {end/1000}s ({duration}s)")
            return True
        
        # Check if there are LONG gaps between voice sections
        if very_quiet:
            last_end = 0
            for start, end in very_quiet:
                gap = start - last_end
                if gap > min_silence_len:
                    print(f"⚠️  {gap/1000}s gap detected between {last_end/1000}s and {start/1000}s")
                    return True
                last_end = end
        
        return False

        # # detect_silence returns a list of [start, end] in milliseconds
        # silences = silence.detect_silence(
        #     audio,
        #     min_silence_len=min_silence_len,  # 3000ms = 3 seconds
        #     silence_thresh=silence_thresh  # Adjust this: -40 is quite quiet
        # )

        # if silences:
        #     for start, end in silences:
        #         print(f"Silence found from {start / 1000}s to {end / 1000}s")
        #     return True
        # return False

    def check_for_long_silence_rms(self, file_path, min_silence_len=3000):
        """Check for sections that are too quiet to be voice"""
        audio = AudioSegment.from_file(file_path)
        
        # Convert to numpy for RMS calculation
        samples = np.array(audio.get_array_of_samples())
        
        # Calculate RMS in 1-second chunks
        chunk_size = audio.frame_rate  # 1 second worth of samples
        chunks = len(samples) // chunk_size
        
        for i in range(chunks):
            chunk = samples[i*chunk_size:(i+1)*chunk_size]
            rms = np.sqrt(np.mean(chunk**2))
            
            # If RMS is very low (quiet)
            if rms < 500:  # Adjust this threshold based on your audio
                # Check if consecutive chunks are quiet
                consecutive_quiet = 1
                for j in range(i+1, chunks):
                    next_chunk = samples[j*chunk_size:(j+1)*chunk_size]
                    next_rms = np.sqrt(np.mean(next_chunk**2))
                    if next_rms < 500:
                        consecutive_quiet += 1
                    else:
                        break
                
                if consecutive_quiet >= min_silence_len / 1000:
                    print(f"Low energy section detected for {consecutive_quiet} seconds")
                    return True
        
        return False

    def detect_repeating_pattern(self, audio_path, min_repeat_duration=1.0, similarity_threshold=0.95):
        """
        Detect if audio has repeating patterns (stuttering, looping)
        Returns: (has_repeats, repeat_info)
        """
        audio, sr = sf.read(audio_path)
        
        # Ensure mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Calculate autocorrelation
        autocorr = signal.correlate(audio, audio, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Take positive half
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Look for peaks that indicate repeating patterns
        # Peak at delay t means pattern repeats every t samples
        min_delay = int(min_repeat_duration * sr)  # minimum repeat duration in samples
        
        # Find peaks in autocorrelation
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(autocorr[min_delay:], height=similarity_threshold)
        
        if len(peaks) > 0:
            # Convert peak positions to seconds
            repeat_times = [(p + min_delay) / sr for p in peaks]
            print(f"Found repeating patterns at intervals: {repeat_times}")
            return True, repeat_times
    
        return False, []
    
    def detect_audio_stutter(self, audio_path, window_ms=500, similarity_threshold=0.95):
        """
        Detect if audio has stuttering (repeated similar segments)
        """
        audio, sr = sf.read(audio_path)
        
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        window_samples = int(window_ms * sr / 1000)
        step = window_samples // 2  # 50% overlap
        
        # Split into windows and compare consecutive windows
        stutter_segments = []
        
        for i in range(0, len(audio) - window_samples, step):
            window1 = audio[i:i+window_samples]
            window2 = audio[i+step:i+step+window_samples]
            
            # Skip if windows are too short
            if len(window1) < window_samples or len(window2) < window_samples:
                continue
            
            # Calculate similarity (cross-correlation)
            correlation = np.corrcoef(window1, window2)[0, 1]
            
            if correlation > similarity_threshold:
                start_time = i / sr
                end_time = (i + window_samples) / sr
                stutter_segments.append((start_time, end_time, correlation))
        
        # Merge consecutive stutters
        merged = []
        for seg in stutter_segments:
            if not merged or seg[0] > merged[-1][1] + 1.0:  # More than 1 second gap
                merged.append([seg[0], seg[1]])
            else:
                merged[-1][1] = max(merged[-1][1], seg[1])
        
        if merged:
            print(f"Found {len(merged)} stutter segments")
            for start, end in merged:
                print(f"  Stutter from {start:.2f}s to {end:.2f}s")
            return True, merged
        
        return False, []
    
    def detect_looped_audio(self, audio_path, min_loop_duration=0.5, max_loop_duration=3.0):
        """
        Detect if audio contains a looped pattern (repeating identical segment)
        """
        audio, sr = sf.read(audio_path)
        
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Try different loop lengths
        min_loop_samples = int(min_loop_duration * sr)
        max_loop_samples = int(max_loop_duration * sr)
        
        best_match = None
        best_score = 0
        
        for loop_len in range(min_loop_samples, max_loop_samples, sr // 20):  # 50ms steps
            if len(audio) < loop_len * 3:
                continue
            
            # Compare first segment with second segment
            segment1 = audio[:loop_len]
            segment2 = audio[loop_len:loop_len*2]
            
            # Trim to same length
            min_len = min(len(segment1), len(segment2))
            seg1 = segment1[:min_len]
            seg2 = segment2[:min_len]
            
            # Calculate similarity
            similarity = np.corrcoef(seg1, seg2)[0, 1]
            
            if similarity > best_score:
                best_score = similarity
                best_match = loop_len / sr
        
        if best_score > 0.9:
            print(f"Found possible loop of {best_match:.2f}s (similarity: {best_score:.3f})")
            return True, best_match
        
        return False, None
    
    def detect_drone_noise(self, audio_path, duration_threshold=5.0, variance_threshold=0.01):
        """
        Detect if audio has long sections of constant drone noise
        (like "hhhhhhh" continuous sound)
        """
        audio, sr = sf.read(audio_path)
        
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Split into 1-second chunks
        chunk_size = sr
        chunks = len(audio) // chunk_size
        
        drone_segments = []
        total_drone_duration = 0
        
        for i in range(chunks):
            chunk = audio[i*chunk_size:(i+1)*chunk_size]
            
            # Calculate variance (how much variation in amplitude)
            variance = np.var(chunk)
            
            # Also check if amplitude is stable (no large changes)
            amplitude_range = np.max(chunk) - np.min(chunk)
            
            # Drone detection: low variance AND small amplitude range
            if variance < variance_threshold and amplitude_range < 0.2:
                start_time = i
                end_time = i + 1
                drone_segments.append((start_time, end_time))
                total_drone_duration += 1
                print(f"  Drone detected at {start_time}s-{end_time}s (var={variance:.6f})")
        
        if total_drone_duration >= duration_threshold:
            print(f"⚠️  Found drone noise for {total_drone_duration} seconds")
            return True, drone_segments
        
        return False, []
    
    def detect_sustained_noise(self, audio_path, min_duration=5.0, freq_band=(100, 1000)):
        """
        Detect sustained noise in a specific frequency band
        """
        audio, sr = sf.read(audio_path)
        
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Split into 1-second chunks
        chunk_size = sr
        chunks = len(audio) // chunk_size
        
        noisy_chunks = []
        
        for i in range(chunks):
            chunk = audio[i*chunk_size:(i+1)*chunk_size]
            
            # Compute FFT
            fft = np.fft.rfft(chunk)
            freqs = np.fft.rfftfreq(len(chunk), 1/sr)
            
            # Find dominant frequency
            magnitude = np.abs(fft)
            dominant_idx = np.argmax(magnitude[1:]) + 1  # Skip DC
            dominant_freq = freqs[dominant_idx]
            
            # Check if dominant frequency is in the "noise" range
            # (constant humming is usually in 100-1000Hz range)
            if freq_band[0] <= dominant_freq <= freq_band[1]:
                # Also check how concentrated the energy is
                energy_ratio = magnitude[dominant_idx] / np.sum(magnitude[1:])
                
                if energy_ratio > 0.3:  # Single frequency dominates
                    noisy_chunks.append(i)
                    print(f"  Sustained frequency at {dominant_freq:.1f}Hz (energy: {energy_ratio:.2f})")
        
        if len(noisy_chunks) >= min_duration:
            print(f"⚠️  Found sustained noise for {len(noisy_chunks)} seconds")
            return True, noisy_chunks
        
        return False, []
    
    def detect_blank_noise(self, audio_path, min_duration=5.0):
        """
        Detect sections that have consistent amplitude but no voice-like modulation
        """
        audio, sr = sf.read(audio_path)
        
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Use voice activity detection
        from pydub import AudioSegment, silence
        import io
        
        # Convert to AudioSegment
        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format='wav')
        buffer.seek(0)
        audio_seg = AudioSegment.from_file(buffer, format='wav')
        
        # Detect voice sections
        voice_sections = silence.detect_nonsilent(
            audio_seg,
            min_silence_len=500,
            silence_thresh=-35
        )
        
        # Check for long gaps between voice sections
        if not voice_sections:
            # No voice at all
            if len(audio) / sr >= min_duration:
                print("⚠️  No voice detected at all!")
                return True, []
        
        # Check for long stretches without voice
        for i in range(len(voice_sections) - 1):
            gap = voice_sections[i+1][0] - voice_sections[i][1]
            if gap >= min_duration * 1000:  # Convert to ms
                print(f"⚠️  Long gap without voice: {gap/1000:.1f} seconds")
                return True, [(voice_sections[i][1], voice_sections[i+1][0])]
        
        return False, []
    
    def detect_breathing_noise(self, audio_path, min_duration=3.0):
        """
        Detect sustained breath/humming sounds like "hhhhhhh"
        """
        audio, sr = sf.read(audio_path)
        
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Split into 0.5-second chunks
        chunk_size = sr // 2
        chunks = len(audio) // chunk_size
        
        suspicious_chunks = []
        
        for i in range(chunks):
            chunk = audio[i*chunk_size:(i+1)*chunk_size]
            
            # Calculate features
            variance = np.var(chunk)
            max_amp = np.max(np.abs(chunk))
            
            # Breath/hum detection:
            # - Low variance (steady sound)
            # - Non-zero amplitude (not silence)
            # - Not too loud (not voice peaks)
            if 0.001 < variance < 0.01 and max_amp > 0.05:
                suspicious_chunks.append(i)
        
        # Check for consecutive suspicious chunks
        consecutive = 0
        max_consecutive = 0
        
        for i in range(len(suspicious_chunks)):
            if i > 0 and suspicious_chunks[i] == suspicious_chunks[i-1] + 1:
                consecutive += 1
            else:
                consecutive = 1
            max_consecutive = max(max_consecutive, consecutive)
        
        if max_consecutive >= min_duration * 2:  # 0.5s chunks
            print(f"⚠️  Found {max_consecutive * 0.5:.1f}s of breath/hum noise")
            return True
        
        return False
    
    def detect_hum_noise(self, audio_path, min_duration=3.0):
        """
        Detect continuous "hhhhhhh" humming/breathing noise
        """
        audio, sr = sf.read(audio_path)
        
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Split into 0.5-second chunks
        chunk_size = sr // 2
        chunks = len(audio) // chunk_size
        
        hum_segments = []
        
        for i in range(chunks):
            chunk = audio[i*chunk_size:(i+1)*chunk_size]
            
            # 1. Zero-crossing rate (low for hum, high for speech)
            zero_crossings = np.sum(np.abs(np.diff(np.sign(chunk)))) / (2 * len(chunk))
            
            # 2. Spectral centroid (where most energy is)
            fft = np.fft.rfft(chunk)
            freqs = np.fft.rfftfreq(len(chunk), 1/sr)
            magnitude = np.abs(fft)
            spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            
            # 3. Crest factor (peak-to-average ratio)
            rms = np.sqrt(np.mean(chunk**2))
            crest_factor = np.max(np.abs(chunk)) / (rms + 1e-10)
            
            # 4. Variance
            variance = np.var(chunk)
            
            # Hum detection criteria:
            # - Low zero-crossing rate (< 0.1 for hum)
            # - Spectral centroid in lower range (100-600Hz)
            # - Medium crest factor (not too spiky, not too flat)
            # - Moderate variance
            is_hum = (
                zero_crossings < 0.05 and
                100 < spectral_centroid < 600 and
                1.2 < crest_factor < 2.5 and
                0.001 < variance < 0.01
            )
            
            if is_hum:
                hum_segments.append(i)
                print(f"  Segment {i}: ZCR={zero_crossings:.3f}, Centroid={spectral_centroid:.1f}Hz, Crest={crest_factor:.2f}")
        
        # Find consecutive hum segments
        if hum_segments:
            consecutive = []
            current = [hum_segments[0]]
            
            for j in range(1, len(hum_segments)):
                if hum_segments[j] == hum_segments[j-1] + 1:
                    current.append(hum_segments[j])
                else:
                    if len(current) > 1:
                        consecutive.append(current)
                    current = [hum_segments[j]]
            
            if len(current) > 1:
                consecutive.append(current)
            
            # Check if any hum segment lasts long enough
            for seg in consecutive:
                duration = len(seg) * 0.5  # 0.5 seconds per chunk
                if duration >= min_duration:
                    start_time = seg[0] * 0.5
                    end_time = seg[-1] * 0.5 + 0.5
                    print(f"⚠️  Hum detected from {start_time:.1f}s to {end_time:.1f}s ({duration:.1f}s)")
                    return True, (start_time, end_time)
        
        return False, None
    
    def detect_steady_noise_spectrogram(self, audio_path, min_duration=3.0, freq_band=(100, 600)):
        """
        Use spectrogram to find long-lasting steady frequencies
        Only flag if the steady frequency lasts for >= min_duration seconds
        """
        audio, sr = sf.read(audio_path)
        
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Compute spectrogram
        f, t, Sxx = signal.spectrogram(audio, sr, nperseg=1024, noverlap=512)
        
        # Find dominant frequency at each time
        dominant_freqs = np.argmax(Sxx, axis=0)
        dominant_freq_values = f[dominant_freqs]
        
        # Look for stable frequencies over time
        stable_regions = []
        current_start = 0
        current_freq = dominant_freq_values[0]
        current_count = 1
        min_samples_for_duration = int(min_duration / (t[1] - t[0]))  # Convert seconds to samples
        
        for i in range(1, len(dominant_freq_values)):
            if abs(dominant_freq_values[i] - current_freq) < 50:  # Similar frequency
                current_count += 1
            else:
                if current_count >= min_samples_for_duration:  # Only if long enough
                    if freq_band[0] <= current_freq <= freq_band[1]:
                        stable_regions.append((current_start, i, current_freq))
                current_start = i
                current_freq = dominant_freq_values[i]
                current_count = 1
        
        # Check last region
        if current_count >= min_samples_for_duration:
            if freq_band[0] <= current_freq <= freq_band[1]:
                stable_regions.append((current_start, len(dominant_freq_values), current_freq))
        
        # Filter regions by duration
        time_per_sample = t[1] - t[0]
        long_regions = []
        
        for start, end, freq in stable_regions:
            duration = (end - start) * time_per_sample
            if duration >= min_duration:
                long_regions.append((start, end, freq, duration))
        
        if long_regions:
            return True, long_regions
        
        return False, []
    
    def check_for_audio_artifacts(self, audio_path):
        """
        Comprehensive audio artifact detection
        """
        issues = []
        
        # Check for long silence (your existing function)
        if self.check_for_long_silence(audio_path):
            issues.append("Long silence detected")
            return True, issues

        # skip the rest
        return False, []
        
        has_hum, _ = self.detect_steady_noise_spectrogram(audio_path)
        if has_hum:
            issues.append("Repeated noise detected")
            return True, issues
        
        # skip the rest
        return False, []

        # 2. Check for drone noise ("hhhhhh")
        has_drone, drone_segments = self.detect_drone_noise(audio_path, duration_threshold=3.0)
        if has_drone:
            issues.append(f"Drone noise for {len(drone_segments)} seconds")
            return True, issues
        
        # 3. Check for sustained frequency
        has_sustained, _ = self.detect_sustained_noise(audio_path, min_duration=3.0)
        if has_sustained:
            issues.append("Sustained frequency noise")
            return True, issues
        
        # 4. Check for no voice activity
        has_no_voice, _ = self.detect_blank_noise(audio_path)
        if has_no_voice:
            issues.append("No voice activity detected")
            return True, issues
        
        # Check for stuttering
        has_stutter, stutters = self.detect_audio_stutter(audio_path)
        if has_stutter:
            issues.append(f"Stuttering detected: {len(stutters)} segments")
            return True, issues
        
        # Check for repeating patterns
        has_repeats, repeats = self.detect_repeating_pattern(audio_path)
        if has_repeats:
            issues.append(f"Repeating patterns at {repeats}")
            return True, issues
        
        # Check for loops
        has_loop, loop_len = self.detect_looped_audio(audio_path)
        if has_loop:
            issues.append(f"Possible loop: {loop_len:.2f}s")
            return True, issues
        
        if issues:
            print(f"⚠️  Audio artifacts found in {Path(audio_path).name}:")
            for issue in issues:
                print(f"   - {issue}")
            return True, issues
        
        return False, []
    
    def normalize_text_before_check(self, text: str) -> str:
        """
        Normalizes text to minimize false positives caused by casing and punctuation.
        """
        if not text:
            return ""
        text = text.lower()
        # Remove punctuation except spaces
        text = re.sub(r'[^\w\s]', '', text)
        # Collapse multiple spaces into one
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def evaluate_quality_label(self, wer, cer, thresholds=None):
        if thresholds is None:
            thresholds = {
                "wer_excellent": 0.15,
                "wer_good": 0.25,
                "wer_bad": 0.35,
                "cer_excellent": 0.08,
                "cer_good": 0.12,
                "cer_bad": 0.18
            }
        
        if wer < thresholds["wer_excellent"] and cer < thresholds["cer_excellent"]:
            return "excellent", wer, cer
        elif wer < thresholds["wer_good"] and cer < thresholds["cer_good"]:
            return "good", wer, cer
        elif wer < thresholds["wer_bad"] or cer < thresholds["cer_bad"]:
            return "mediocre", wer, cer
        else:
            return "bad", wer, cer

    def evaluate_audio_quality(self, audio_path: str, reference_text: str, model_size: str = "large-v3-turbo") -> dict:
        from faster_whisper import WhisperModel, BatchedInferencePipeline
        from jiwer import wer, cer
        """
        Transcribes audio with Whisper and compares it against target text to catch artifacts.
        """
        self.load_model_whisper()
        # 2. Transcribe the OmniVoice generated speech
        segments, info = self.model_whisper.transcribe(audio_path, batch_size=16)
        transcription = " ".join([segment.text for segment in segments])
        
        # 3. Clean both texts
        clean_ref = self.normalize_text_before_check(reference_text)
        clean_trans = self.normalize_text_before_check(transcription)
        
        # 4. Handle empty edge case (Whisper hallucinated absolute silence or failed completely)
        if not clean_trans:
            return {
                "wer": 1.0, "cer": 1.0, "decision": "REGENERATE", 
                "reason": "Whisper produced an empty transcript (extreme artifact/mute)",
                "label": "bad"
            }
        
        # 5. Compute metrics
        error_rate_word = wer(clean_ref, clean_trans)
        error_rate_char = cer(clean_ref, clean_trans)

        label, _, _ = self.evaluate_quality_label(error_rate_word, error_rate_char)
        
        # 6. Apply Decision Boundaries
        # WER > 0.15 (15% missed words) or CER > 0.08 (8% missed characters)
        # If OmniVoice stutters, adds robotic clicks, or skips lines, these spike.
        if label == "bad" or label == "mediocre":
            decision = "REGENERATE"
            reason = "Acoustic glitch or text mismatch detected."
        else:
            decision = "KEEP"
            reason = "Passed alignment check."
            
        return {
            "reference": clean_ref,
            "transcription": clean_trans,
            "wer": round(error_rate_word, 4),
            "cer": round(error_rate_char, 4),
            "decision": decision,
            "reason": reason,
            "label": label
        }
    
    def load_problematic_quests_cache(self, path="problematic_quests.json"):
        """
        Загружает список проблемных квестов из JSON файла.
        """
        import json
        import os
        
        if not os.path.exists(path):
            print(f"ℹ️ Файл {path} не найден. Начинаем с пустого списка проблем.")
            self.problematic_quests = {}
            return
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.problematic_quests = json.load(f)
            print(f"✅ Loaded {len(self.problematic_quests)} quest quality status from {path}")
        except json.JSONDecodeError as e:
            print(f"❌ Ошибка чтения {path}: {e}. Начинаем с пустого списка.")
            self.problematic_quests = {}
        except Exception as e:
            print(f"❌ Неизвестная ошибка при загрузке {path}: {e}")
            self.problematic_quests = {}

    def load_problematic_gossip_cache(self, path="problematic_gossip.json"):
        """
        Загружает список проблемных квестов из JSON файла.
        """
        import json
        import os
        
        if not os.path.exists(path):
            print(f"ℹ️ Файл {path} не найден. Начинаем с пустого списка проблем.")
            self.problematic_gossip = {}
            return
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.problematic_gossip = json.load(f)
            print(f"✅ Loaded {len(self.problematic_gossip)} gossip quality status from {path}")
        except json.JSONDecodeError as e:
            print(f"❌ Ошибка чтения {path}: {e}. Начинаем с пустого списка.")
            self.problematic_gossip = {}
        except Exception as e:
            print(f"❌ Неизвестная ошибка при загрузке {path}: {e}")
            self.problematic_gossip = {}
    
    def save_problematic_quests_cache(self, path="problematic_quests.json"):
        with open(path, "w") as f:
            json.dump(self.problematic_quests, f, indent=2)
            # print(f"✅ Saved {len(self.problematic_quests)} quest quality status to {path}")
    
    def save_problematic_gossip_cache(self, path="problematic_gossip.json"):
        with open(path, "w") as f:
            json.dump(self.problematic_gossip, f, indent=2)
            # print(f"✅ Saved {len(self.problematic_gossip)} gossip quality status to {path}")
    
    def evaluate_and_register_quest(self, quest_id, text_type, generation_result, reference_text, remove_file: bool = False):
        # generation_result — это путь к файлу или словарь с waveform, который ты проверяешь
        # text_type: 'accept', 'complete', 'progress'
        key = f"{quest_id}_{text_type}"
        
        # Skip if was marked as good
        if key in self.problematic_quests and self.problematic_quests[key]["decision"] == "KEEP":
            return True, self.problematic_quests[key]
        
        analysis = self.evaluate_audio_quality(generation_result, reference_text)  # твоя функция
        if key not in self.problematic_quests:
            attempts = 1
        else:
            attempts = self.problematic_quests[key]["attempts"]
        # 2. Принимаем решение
        if analysis["decision"] == "REGENERATE":

            # 2a. Добавляем/обновляем запись в проблемных
            self.problematic_quests[key] = {
                "reason": analysis["reason"],
                "wer": analysis["wer"],
                "cer": analysis["cer"],
                "attempts": attempts,
                "decision": analysis["decision"],
                "label": analysis["label"]
            }
            # 2b. Удаляем файл
            if remove_file and os.path.exists(generation_result):
                os.remove(generation_result)
                print(f"🗑️ 👎 Bad file removed: {generation_result} ({analysis["label"]})")
            else:
                print(f"👎 Bad file found: {generation_result} ({analysis["label"]})")

            self.save_problematic_quests_cache(self.problematic_quest_path)
            return False, analysis
        else:
            # 3. Успешно! Если был в проблемных — удаляем оттуда
            self.problematic_quests[key] = {
                "reason": analysis["reason"],
                "wer": analysis["wer"],
                "cer": analysis["cer"],
                "attempts": attempts,
                "decision": "KEEP",
                "label": analysis["label"]
            }
            self.save_problematic_quests_cache(self.problematic_quest_path)
            return True, analysis
        
    def evaluate_and_register_gossip(self, npc_name, hash, generation_result, reference_text, remove_file: bool = False):
        # generation_result — это путь к файлу или словарь с waveform, который ты проверяешь
        # text_type: 'accept', 'complete', 'progress'
        key = f"{npc_name}_{hash}"
        
        # Skip if was marked as good
        if key in self.problematic_gossip and self.problematic_gossip[key]["decision"] == "KEEP":
            return True, self.problematic_gossip[key]
        
        analysis = self.evaluate_audio_quality(generation_result, reference_text)  # твоя функция
        if key not in self.problematic_gossip:
            attempts = 1
        else:
            attempts = self.problematic_gossip[key]["attempts"]
        # 2. Принимаем решение
        if analysis["decision"] == "REGENERATE":

            # 2a. Добавляем/обновляем запись в проблемных
            self.problematic_gossip[key] = {
                "reason": analysis["reason"],
                "wer": analysis["wer"],
                "cer": analysis["cer"],
                "attempts": attempts,
                "decision": analysis["decision"],
                "label": analysis["label"]
            }
            # 2b. Удаляем файл
            if remove_file and os.path.exists(generation_result):
                os.remove(generation_result)
                print(f"🗑️ 👎 Bad file removed: {generation_result} ({analysis["label"]})")
            else:
                print(f"👎 Bad file found: {generation_result} ({analysis["label"]})")

            self.save_problematic_gossip_cache(self.problematic_gossip_path)
            return False, analysis
        else:
            # 3. Успешно! Если был в проблемных — удаляем оттуда
            self.problematic_gossip[key] = {
                "reason": analysis["reason"],
                "wer": analysis["wer"],
                "cer": analysis["cer"],
                "attempts": attempts,
                "decision": "KEEP",
                "label": analysis["label"]
            }
            self.save_problematic_gossip_cache(self.problematic_gossip_path)
            return True, analysis
        
    def should_regenerate_quest(self, quest_id, text_type):
        """Возвращает True, если файл нужно перегенерировать"""
        key = f"{quest_id}_{text_type}"
        # Проблемный И не помечен ручным оверрайдом
        return key in self.problematic_quests and key not in self.manual_override_quests and self.problematic_quests[key]["decision"] == "REGENERATE"

    def mark_as_fixed_quest(self, quest_id, text_type):
        """Вручную помечаем квест как исправленный"""
        key = f"{quest_id}_{text_type}"
        self.problematic_quests[key]["decision"] = "KEEP"
        self.manual_override_quests.discard(key)

    def skip_for_now_quest(self, quest_id, text_type):
        """Временно игнорировать проблемный квест"""
        key = f"{quest_id}_{text_type}"
        self.manual_override_quests.add(key)

    def should_regenerate_gossip(self, npc_name, hash):
        """Возвращает True, если файл нужно перегенерировать"""
        key = f"{npc_name}_{hash}"
        # Проблемный И не помечен ручным оверрайдом
        return key in self.problematic_gossip and key not in self.manual_override_gossip and self.problematic_gossip[key]["decision"] == "REGENERATE"

    def mark_as_fixed_gossip(self, npc_name, hash):
        """Вручную помечаем квест как исправленный"""
        key = f"{npc_name}_{hash}"
        self.problematic_gossip[key]["decision"] = "KEEP"
        self.manual_override_gossip.discard(key)

    def skip_for_now_gossip(self, npc_name, hash):
        """Временно игнорировать проблемный квест"""
        key = f"{npc_name}_{hash}"
        self.manual_override_gossip.add(key)
    
    def check_audio_file(self, file_info):
        """Check a single audio file for artifacts"""
        size, path, src, who, module_name = file_info
        try:
            has_artifacts, issues = self.check_for_audio_artifacts(path)
            return (size, path, src, who, module_name, has_artifacts, issues)
        except Exception as e:
            print(f"Error checking {path}: {e}")
            return (size, path, src, who, module_name, True, [str(e)])
        
    def filter_good_samples_multithread(self, voice_samples_list, max_workers=8):
        """
        Check samples in parallel using multiple threads
        """
        good_samples = []
        bad_samples = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(self.check_audio_file, info): info for info in voice_samples_list}
            
            # Process with progress bar
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Checking audio"):
                info = futures[future]
                try:
                    size, path, src, who, module_name, has_artifacts, issues = future.result()
                    
                    if has_artifacts:
                        tqdm.write(f"❌ Skipping {path.name}: {issues}")
                        bad_samples.append((size, path, src, who, module_name))
                    else:
                        good_samples.append((size, path, src, who, module_name))
                except Exception as e:
                    tqdm.write(f"Error checking {info[1]}: {e}")
                    bad_samples.append(info)
        
        print(f"✅ Found {len(good_samples)} good samples, {len(bad_samples)} bad samples")
        return good_samples

    def get_game_voice_sample(self, voice_key):

        voice_types = ['standard', 'official', 'noble', 'military', 'nerdy', 'happy', 'young', 'grim', 'maternal', 'guard', 'sentinel', 'priestess', 'shady', 'warrior', 'shaman', 'elder', 'magic', 'dark', 'old', 'zany', 'gruff', 'highelf']
        voice_samples = []
        # 1. Clean the key: "human_male" -> ["human", "male"]

        for voice_type in voice_types:

            voice_type_copy = voice_type
            voice_key_copy = voice_key
            if voice_key == 'tauren_male':
                if voice_type == 'standard':
                    voice_type_copy = 'warrior'
            if voice_key == 'goblin_male':
                if voice_type == 'standard':
                    voice_type_copy = 'gruff'
            if voice_key == 'goblin_female':
                if voice_type == 'standard':
                    voice_type_copy = 'zany'
            if voice_key == 'tauren_female':
                if voice_type == 'grim':
                    voice_type_copy = 'grimtotem'
                    voice_key_copy = 'magatha'
            if voice_key == 'forsaken_male':
                voice_key_copy = 'undead_male'
            if voice_key == 'forsaken_female':
                voice_key_copy = 'undead_female'

            parts = voice_key_copy.lower().split('_')
            base_path = os.path.join(VOICE_SAMPLE_PATH, "FromGame/Sound/Creature")
            # 2. Find the right folder by scanning the directory
            target_folder = None
            if os.path.exists(base_path):
                for folder in os.listdir(base_path):
                    folder_lower = folder.lower()
                    # Check if all parts (e.g., 'human' AND 'male') are in the folder name
                    if all(p in folder_lower for p in parts) and ((len(parts) > 1 and ((parts[1] == 'female') or (
                            parts[1] == 'male' and folder_lower.find("female") == -1))) or (len(parts) == 1)) and (folder_lower.find(voice_type_copy) != -1 or voice_key.find('male') == -1):
                        target_folder = os.path.join(base_path, folder)
                        break

            if not target_folder:
                continue

            # 3. Grab all Greeting files (Windows ignores case, but glob helps)
            # We use a case-insensitive search pattern
            greeting_pattern = os.path.join(target_folder, "*[Gg][Rr][Ee][Ee][Tt][Ii][Nn][Gg]*.wav")
            # farewell_pattern = os.path.join(target_folder, "*[Ff][Aa][Rr][Ee][Ww][Ee][Ll][Ll]*.wav")

            greeting_files = glob.glob(greeting_pattern)
            # farewell_files = glob.glob(farewell_pattern)

            # Combine both lists
            all_files = greeting_files

            if not all_files:
                continue

            # Sort so 01 comes before 02
            all_files.sort()

            # 4. Merge
            combined = AudioSegment.empty()
            for file in all_files:
                audio = AudioSegment.from_wav(file)
                combined += audio # + AudioSegment.silent(duration=400)  # Gap for clarity

            # 5. Save to a Temp or Cache folder for Gradio
            out_path = f"temp_samples/{voice_key}_{voice_type}_game_ref.mp3"
            os.makedirs("temp_samples", exist_ok=True)
            combined.export(out_path, format="mp3")
            print(f"found {voice_key}_{voice_type}")
            voice_samples.append((out_path, voice_type))

            if voice_key.find('male') == -1:
                    break

        return voice_samples, "Success"

    def collect_best_references(self, df):
        # os.makedirs(output_path, exist_ok=True)

        def get_hash(text):
            return hashlib.md5(text.encode()).hexdigest()

        working_df = df.copy()
        unique_voices = working_df['voice_name'].unique()

        for voice in unique_voices:
            voice_data = working_df[working_df['voice_name'] == voice]

            for voice_path in MODULE_NAMES:

                # Skip game for now
                if voice_path == "FromGame":
                    continue

                copied_count = 0
                print(f"Searching: {voice}")

                found_file = None
                best_fallback = None

                output_subdir = os.path.join(VOICE_SAMPLE_PATH, voice_path)
                os.makedirs(output_subdir, exist_ok=True)

                # --- ШАГ 1: Ищем в квестах (по ID) ---
                quest_dir = Path(ADDON_PATH) / voice_path / "generated/sounds/quests"
                if quest_dir.exists():
                    # Пробуем найти любой accept или complete для этого NPC
                    for _, row in voice_data.iterrows():
                        quest_id = row['quest']
                        source_type = row['source']  # accept, complete и т.д.

                        if pd.isna(quest_id) or quest_id == "": continue

                        # Формируем имя: 12345-accept.wav
                        target_name = f"{int(quest_id)}-{source_type}.mp3"
                        full_path = quest_dir / target_name

                        if full_path.exists():
                            copied_count += 1
                            found_file = full_path
                            dest_file = Path(output_subdir) / f"{voice}.mp3"
                            if voice_path == "AI_VoiceOverData_Vanilla" or (50 * 1024 <= full_path.stat().st_size <= 150 * 1024):
                                if dest_file.exists():
                                    if found_file.stat().st_size > dest_file.stat().st_size:
                                        shutil.copy2(found_file, dest_file)
                                else:
                                    shutil.copy2(found_file, dest_file)
                            else:
                                if best_fallback and best_fallback.exists():
                                    if found_file.stat().st_size > best_fallback.stat().st_size:
                                        best_fallback = full_path
                                else:
                                    best_fallback = full_path

                print(f"quests found: {copied_count}")
                # --- ШАГ 2: Если в квестах не нашли, ищем в gossip (по хэшу) ---

                copied_count = 0
                gossip_dir = Path(ADDON_PATH) / voice_path / "generated/sounds/gossip"
                if gossip_dir.exists():
                    for _, row in voice_data.iterrows():
                        target_hash_name = f"{row['templateText_race_gender_hash']}.mp3"
                        full_path = gossip_dir / target_hash_name

                        # print(f"Searching: {full_path}")

                        if full_path.exists():
                            copied_count += 1
                            found_file = full_path
                            dest_file = Path(output_subdir) / f"{voice}.mp3"
                            if voice_path == "AI_VoiceOverData_Vanilla" or (50 * 1024 <= full_path.stat().st_size <= 150 * 1024):
                                if dest_file.exists():
                                    if found_file.stat().st_size > dest_file.stat().st_size:
                                        shutil.copy2(found_file, dest_file)
                                else:
                                    shutil.copy2(found_file, dest_file)
                            else:
                                if best_fallback and best_fallback.exists():
                                    if found_file.stat().st_size > best_fallback.stat().st_size:
                                        best_fallback = full_path
                                else:
                                    best_fallback = full_path

                print(f"gossip found: {copied_count}")

                if not found_file and best_fallback:
                    found_file = best_fallback
                    shutil.copy2(found_file, dest_file)
                    print(f"⚠️ Good voice for {voice} not found, using just biggest file")

            # From game sounds
            copied_count = 0
            game_voice_check, _ = self.get_game_voice_sample(voice)
            print(f"size of game voices {len(game_voice_check)}")
            if len(game_voice_check) > 0:
                for game_voice_path, voice_type in game_voice_check:
                    game_voice_path = Path(game_voice_path)
                    print(f"Processing: {game_voice_path.name} ({voice_type})")
                    if game_voice_path.exists():
                        output_subdir = os.path.join(VOICE_SAMPLE_PATH, "GameVoice")
                        os.makedirs(output_subdir, exist_ok=True)
                        if voice_type != 'standard':
                            suffix = f"_{voice_type}"
                        else:
                            suffix = ""
                        dest_file = Path(output_subdir) / f"{voice}{suffix}.mp3"
                        shutil.copy2(game_voice_path, dest_file)

        for voice_path in MODULE_NAMES:
            # Skip game for now
            if voice_path == "FromGame":
                continue

            output_subdir = os.path.join(VOICE_SAMPLE_PATH, voice_path)
            count = len(os.listdir(output_subdir))
            print(f"Voices from {voice_path} found: {count}")

        output_subdir = os.path.join(VOICE_SAMPLE_PATH, "GameVoice")
        count = len(os.listdir(output_subdir))
        print(f"Voices from Game found: {count}")

        self.setup_voice_presets()

    def get_voice_path(self, voice_name: str, module_name: str) -> str:
        """Get voice file path for a given speaker name"""
        # First try exact match
        voice_name = voice_name.lower()
        if voice_name in self.voice_presets[module_name]:
            return self.voice_presets[module_name][voice_name]

    def tts(self, text, voice_name, voice_type, voice_to_clone, voice_to_clone_2, output_name, output_subfolder, forceGen=False, volume_match=True, volume_match_sample=False, questgiver_id=None,
            model_name="VibeVoice-1.5B", seed: int = 42, temperature=5.0, class_temperature=0.3, speed=1.0, cps=15,
            max_words_per_chunk=75, attention_type="auto",
            quantize_llm="full precision", free_memory=True, steps: int = 32,
            cfg_scale: float = 2.0, use_sampling: bool = False, apply_effects: bool = True, apply_effects_override: str = None, detect_emotions: bool = False
            ):

        SOUND_OUTPUT_FOLDER = os.path.join(self.sound_output_folder, output_subfolder)
        VOICE_SAMPLE_FOLDER = "voices"

        outpath = os.path.join(SOUND_OUTPUT_FOLDER, output_name)
        # Keep original voice_name for effects further below
        voice_key = voice_name

        # remode dk suffix from dk so model search matches with the corresponding
        # race-gender model on file
        if voice_name.endswith("_dk"):
            voice_name = voice_name[:-3]

        mapped_voice = VOICE_MODEL_MAP.get(voice_name, voice_name)

        if os.path.isfile(outpath) and not forceGen:
            print("duplicate generation, skipping")
            return

        voice_found = False
        if os.path.exists(voice_to_clone):
            voice_found = True

        print("=" * 30)
        print(f"🔍 DEBUG AUDIO PATH:")
        print(f"  - VOICE_SAMPLE_FOLDER: {os.path.abspath(VOICE_SAMPLE_FOLDER)}")
        print(f"  - voice_name:          {voice_name}")
        print(f"  - FINAL FULL PATH:     {os.path.abspath(voice_to_clone)}")
        print(f"  - FILE EXISTS?         {'✅ YES' if os.path.exists(voice_to_clone) else '❌ NO'}")
        print("=" * 30)

        if not voice_found:
            print(f"Voice sample not found: {voice_name}")
            return None

        # Set model_dir based on mapped voice, or voice_name
        model_dir = f"fine_tuned/{mapped_voice}"

        if not os.path.isdir(model_dir):
            model_dir = None  # fallback in FastAPI will handle this

        files = {
            "speaker_wav": voice_to_clone
        }
        data = {
            "text": text,
            "language": "en",
            "model_dir": model_dir or "",  # If None, FastAPI treats it as base
            "voice_name": mapped_voice
        }

        os.makedirs(os.path.dirname(outpath), exist_ok=True)

        if len(data["text"]) == 0:
            print(f"Empty Text, skipping!")
            return None

        response = self.generate_speech(
            text=data["text"],
            model=model_name,
            attention_type=attention_type,
            quantize_llm=quantize_llm,
            free_memory_after_generate=free_memory,
            diffusion_steps=steps,
            seed=seed,
            cfg_scale=cfg_scale,
            use_sampling=use_sampling,
            voice_to_clone=voice_to_clone,
            voice_to_clone_2=voice_to_clone_2,
            temperature=temperature,
            class_temperature=class_temperature,
            max_words_per_chunk=max_words_per_chunk,
            voice_speed_factor=speed,
            cps=cps,
            voice_key=voice_key,
            questgiver_id=questgiver_id,
            apply_effects=apply_effects,
            apply_effects_override=apply_effects_override,
            detect_emotions=detect_emotions,
            volume_match=volume_match
        )

        # Convert response.content (which is WAV) into an MP3
        audio = AudioSegment.from_file(response, format="wav")
        audio = audio.set_frame_rate(44100).set_channels(1)
        audio.export(outpath, format="mp3", bitrate="64k")

        print(f"Audio saved and processed: {outpath}")

        self.seed_manager.add_or_update_seed(voice_name, voice_type, output_name, seed)

        self.output_path = outpath

        return outpath

    def handle_gender_options(self, text):
        # Pattern 1: Original with semicolon
        pattern1 = re.compile(r'\$[Gg]\s*([^:;]+?)\s*:\s*([^:;]+?)\s*;')
        
        # Pattern 2: Without semicolon (matches until comma or end)
        pattern2 = re.compile(r'\$[Gg]\s*([^:;\s]+)\s*:\s*([^,;\s]+)')
        
        # Try pattern1 first
        match1 = pattern1.search(text)
        if match1:
            male_text = pattern1.sub(r'\1', text)
            female_text = pattern1.sub(r'\2', text)
            return male_text, female_text
        
        # Fallback to pattern2
        match2 = pattern2.search(text)
        if match2:
            male_text = pattern2.sub(r'\1', text)
            female_text = pattern2.sub(r'\2', text)
            return male_text, female_text
        
        # No gender tag found
        return text, text

    def clean_text_preserve_gender(self, text):
        """
        Remove semicolons that aren't part of $g tags
        """
        if pd.isna(text) or not isinstance(text, str):
            return text

        # Protect gender tags by replacing their semicolons with a placeholder
        def protect_gender(match):
            return match.group(0).replace(';', '§')

        # Pattern for gender tags
        gender_pattern = re.compile(r'\$[Gg]\s*\w+:\s*\w+;')

        # Protect gender tag semicolons
        text = gender_pattern.sub(protect_gender, text)

        # Remove all remaining semicolons
        text = text.replace(';', '.')

        # Restore gender tag semicolons
        text = text.replace('§', ';')

        return text

    def process_text_column(self, df, column_name):
        df = df.copy()  # prevent mutation on original df for safety
        for k, v in REPLACE_DICT.items():
            df[column_name] = df[column_name].str.replace(k, v, regex=False, flags=re.IGNORECASE)

        # Remove qpercentage
        df[column_name] = df[column_name].str.replace(r'\$\d+w', '', regex=True)

        # VibeVoice doesn't seem to understand numbers
        df[column_name] = df[column_name].apply(convert_numbers_to_words)

        # Detect narrator text (brackets or quotes) if needed
        df['has_narrator'] = False
        speakers_check = df[column_name].apply(split_by_speakers_brackets)
        df[column_name] = speakers_check.apply(lambda x: x[0])
        df['has_narrator'] = df['has_narrator'] | speakers_check.apply(lambda x: x[1])
        speakers_check_quotes = df[column_name].apply(split_by_speakers_quotes)
        df[column_name] = speakers_check_quotes.apply(lambda x: x[0])
        df['has_narrator'] = df['has_narrator'] | speakers_check_quotes.apply(lambda x: x[1])

        # deal with text in brackets: if its all the text for that NPC, keep the text, otherwise remove it
        df[column_name] = df[column_name].apply(clean_brackets)

        # Remove quotes
        df[column_name] = df[column_name].str.replace(r'(?<!^)"(?!$)', '', regex=True)

        # Removes commas
        df[column_name] = df[column_name].str.replace(
            r',\s+(and|but|you|he|she|they|we|i)\b',
            r' \1',
            regex=True
        )

        # Remove appositive commas (optional)
        # df[column_name] = df[column_name].str.replace(
        #     r',\s*([^,]{1,60}?)\s*,',
        #     r' \1',
        #     regex=True
        # )

        # no commas for short sentences
        df[column_name] = df[column_name].apply(lambda x: x.replace(",", "") if len(x) <= 50 else x)

        # deal with ellipses
        # normalize spaced ellipses like ". . ." or ". ." into "..."
        # Replace unicode ellipsis with three dots
        df[column_name] = df[column_name].str.replace(r'…', '...', regex=True)  # ". . ." → "..."
        df[column_name] = df[column_name].str.replace(r"\.\s+\.\s+\.", "...", regex=True)  # ". . ." → "..."
        df[column_name] = df[column_name].str.replace(r"\.\s+\.", "...", regex=True)  # ". ."   → "..."

        # "..." inside a sentence → ","
        df[column_name] = df[column_name].str.replace(r"\.\.\.(\w)", r", \1", regex=True)
        # "..." at the end → "."
        df[column_name] = df[column_name].replace(r"\.\.\.$", ".", regex=True)

        # Remove newlines and normalize spaces
        df[column_name] = df[column_name].str.replace(r'\n+', '. ', regex=True)  # remove newlines
        df[column_name] = df[column_name].str.replace(r'\s+', ' ', regex=True)  # normalize multiple spaces
        df[column_name] = df[column_name].str.strip()  # remove leading/trailing spaces

        # ".." inside a sentence → ","
        df[column_name] = df[column_name].str.replace(r"\.\.(\w)", r", \1", regex=True)
        # ".." at the end → ","
        df[column_name] = df[column_name].replace(r"\.\.$", ".", regex=True)

        df[column_name] = df[column_name].replace(r"\.{2,}", ".", regex=True)

        df[column_name] = df[column_name].replace(r"\.\?", "?", regex=True)
        df[column_name] = df[column_name].replace(r"\.\!", "!", regex=True)

        df[column_name] = df[column_name].replace(r"\?\.", "?", regex=True)
        df[column_name] = df[column_name].replace(r"\!\.", "!", regex=True)

        # remove illegal characters
        df[column_name] = (
            df[column_name]
            .str.replace(r"<.*?>|\*\*.*?\*\*|\*.*?\*", "", regex=True)
            .str.replace(r"\s+", " ", regex=True)  # collapse leftover spaces
            .str.strip()
        )

        # Remove binary characters (for the punchographs in Gnomeregan)
        df[column_name] = df[column_name].str.replace(r'(?:\b[01]{8}\b\s*)+', '', regex=True)

        df[column_name] = df[column_name].str.replace(r"\.\s+\.\s+\.", "...", regex=True)  # ". . ." → "..."
        df[column_name] = df[column_name].str.replace(r"\.\s+\.", "...", regex=True)  # ". ."   → "..."
        df[column_name] = df[column_name].str.replace(r"\s\.\s", ". ", regex=True)  # ". ."   → "..."

        # fix space after g
        # df[column_name] = df[column_name].str.replace(r'\$[Gg]\s+', '$g', regex=True)
        # df['player_gender'] = None
        # rows = []
        # for _, row in df.iterrows():
        #     if re.search(r'\$[Gg]', row[column_name]):
        #         male_text, female_text = self.handle_gender_options(row[column_name])
        #
        #         row_male = row.copy()
        #         row_male[column_name] = male_text
        #         row_male['player_gender'] = 'm'
        #
        #         row_female = row.copy()
        #         row_female[column_name] = female_text
        #         row_female['player_gender'] = 'f'
        #
        #         rows.extend([row_male, row_female])
        #     else:
        #         rows.append(row)

        # move gender generation to tts_
        new_df = df.copy()
        new_df[column_name] = new_df[column_name].apply(self.clean_text_preserve_gender)
        return new_df
    
    def process_text_row(self, text: str) -> str:
        """
        Process a single text string through all cleaning steps.
        Returns cleaned text.
        """

        if text == '0':
            return text
        if text == 0:
            return text
        if not isinstance(text, str):
            return ""
        
        # Apply replacements
        for k, v in REPLACE_DICT.items():
            text = text.replace(k, v)
        
        # Remove qpercentage
        text = re.sub(r'\$\d+w', '', text)
        
        # Convert numbers to words
        text = convert_numbers_to_words(text)
        
        # Detect narrator text from brackets
        speakers_check = split_by_speakers_brackets(text)
        text = speakers_check[0]
        has_narrator = speakers_check[1]
        
        # Detect narrator text from quotes
        speakers_check_quotes = split_by_speakers_quotes(text)
        text = speakers_check_quotes[0]
        has_narrator = has_narrator or speakers_check_quotes[1]
        
        # Deal with text in brackets
        text = clean_brackets(text)
        
        # Remove quotes (but not whole string quotes)
        text = re.sub(r'(?<!^)"(?!$)', '', text)
        
        # Remove commas before conjunctions
        text = re.sub(r',\s+(and|but|you|he|she|they|we|i)\b', r' \1', text, flags=re.IGNORECASE)
        
        # No commas for short sentences
        if len(text) <= 50:
            text = text.replace(",", "")
        
        # Deal with ellipses
        text = re.sub(r"\.\s+\.\s+\.", "...", text)
        text = re.sub(r"\.\s+\.", "...", text)
        text = re.sub(r"\.\.\.(\w)", r", \1", text)
        text = re.sub(r"\.\.\.$", ".", text)
        
        # Remove newlines and normalize spaces
        text = re.sub(r'\n+', '. ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Deal with double dots
        text = re.sub(r"\.\.(\w)", r", \1", text)
        text = re.sub(r"\.\.$", ".", text)
        text = re.sub(r"\.{2,}", ".", text)
        
        # Fix punctuation combinations
        text = re.sub(r"\.\?", "?", text)
        text = re.sub(r"\.\!", "!", text)
        text = re.sub(r"\?\.", "?", text)
        text = re.sub(r"\!\.", "!", text)
        
        # Remove illegal characters and binary
        text = re.sub(r"<.*?>|\*\*.*?\*\*|\*.*?\*", "", text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove binary characters (punchographs in Gnomeregan)
        text = re.sub(r'(?:\b[01]{8}\b\s*)+', '', text)
        
        # Final ellipsis cleanup
        text = re.sub(r"\.\s+\.\s+\.", "...", text)
        text = re.sub(r"\.\s+\.", "...", text)
        text = re.sub(r"\s\.\s", ". ", text)
        
        # Preserve gender tags
        text = self.clean_text_preserve_gender(text)
        
        return text

    def preprocess_dataframe(self, df):
        df = df.copy()  # prevent mutation on original df for safety
        df['race'] = df['DisplayRaceID'].map(RACE_DICT)
        df['gender'] = df['DisplaySexID'].map(GENDER_DICT)
        df['voice_name'] = df['race'] + '_' + df['gender']

        # Multi-speaker: model-name-based speaker key for multi-speaker NPCs.
        # For Forsaken, keep 'scourge_male'/'scourge_female' — in-game model
        # is scourge_male.m2, not forsaken_male.m2.
        df['multi_name'] = df['voice_name'].copy()

        df['zone'] = df['zone'].fillna('Unknown')

        df['templateText_race_gender'] = df['original_text'] + df['race'] + df['gender']
        df['templateText_race_gender_hash'] = df['templateText_race_gender'].apply(get_hash)

        df['cleanedText'] = df['text'].copy()

        for k, v in REPLACE_DICT.items():
            df['cleanedText'] = df['cleanedText'].str.replace(k, v, regex=False, flags=re.IGNORECASE)

        # Remove qpercentage
        df['cleanedText'] = df['cleanedText'].str.replace(r'\$\d+w', '', regex=True)

        # VibeVoice doesn't seem to understand numbers
        df['cleanedText'] = df['cleanedText'].apply(convert_numbers_to_words)

        # convert laughter/sigh to omnivoice supported tags
        

        # Detect narrator text (brackets or quotes) if needed
        df['has_narrator'] = False
        speakers_check = df['cleanedText'].apply(split_by_speakers_brackets)
        df['cleanedText'] = speakers_check.apply(lambda x: x[0])
        df['has_narrator'] = df['has_narrator'] | speakers_check.apply(lambda x: x[1])
        speakers_check_quotes = df['cleanedText'].apply(split_by_speakers_quotes)
        df['cleanedText'] = speakers_check_quotes.apply(lambda x: x[0])
        df['has_narrator'] = df['has_narrator'] | speakers_check_quotes.apply(lambda x: x[1])

        # deal with text in brackets: if its all the text for that NPC, keep the text, otherwise remove it
        df['cleanedText'] = df['cleanedText'].apply(clean_brackets)

        # Remove quotes
        df['cleanedText'] = df['cleanedText'].str.replace(r'(?<!^)"(?!$)', '', regex=True)

        # Removes commas
        # df['cleanedText'] = df['cleanedText'].str.replace(
        #     r',\s+(and|but|you|he|she|they|we|i)\b',
        #     r' \1',
        #     regex=True
        # )

        # Remove appositive commas (optional)
        # df['cleanedText'] = df['cleanedText'].str.replace(
        #     r',\s*([^,]{1,60}?)\s*,',
        #     r' \1',
        #     regex=True
        # )

        # no commas for short sentences
        # df['cleanedText'] = df['cleanedText'].apply(lambda x: x.replace(",", "") if len(x) <= 50 else x)

        # deal with ellipses
        # normalize spaced ellipses like ". . ." or ". ." into "..."
        df['cleanedText'] = df['cleanedText'].str.replace(r"\.\s+\.\s+\.", "...", regex=True)  # ". . ." → "..."
        df['cleanedText'] = df['cleanedText'].str.replace(r"\.\s+\.", "...", regex=True)  # ". ."   → "..."

        # "..." inside a sentence → ","
        df['cleanedText'] = df['cleanedText'].str.replace(r"\.\.\.(\w)", r", \1", regex=True)
        # "..." at the end → "."
        df['cleanedText'] = df['cleanedText'].replace(r"\.\.\.$", ".", regex=True)

        # Remove newlines and normalize spaces
        df['cleanedText'] = df['cleanedText'].str.replace(r'\n+', '. ', regex=True)  # remove newlines
        df['cleanedText'] = df['cleanedText'].str.replace(r'\s+', ' ', regex=True)  # normalize multiple spaces
        df['cleanedText'] = df['cleanedText'].str.strip()  # remove leading/trailing spaces

        # ".." inside a sentence → ","
        df['cleanedText'] = df['cleanedText'].str.replace(r"\.\.(\w)", r", \1", regex=True)
        # ".." at the end → ","
        df['cleanedText'] = df['cleanedText'].replace(r"\.\.$", ".", regex=True)

        df['cleanedText'] = df['cleanedText'].replace(r"\.{2,}", ".", regex=True)

        df['cleanedText'] = df['cleanedText'].replace(r"\.\?", "?", regex=True)
        df['cleanedText'] = df['cleanedText'].replace(r"\.\!", "!", regex=True)

        df['cleanedText'] = df['cleanedText'].replace(r"\?\.", "?", regex=True)
        df['cleanedText'] = df['cleanedText'].replace(r"\!\.", "!", regex=True)
        df['cleanedText'] = df['cleanedText'].replace(r"\:\.", ":", regex=True)

        # remove illegal characters
        df["cleanedText"] = (
            df["cleanedText"]
            .str.replace(r"<.*?>|\*\*.*?\*\*|\*.*?\*", "", regex=True)
            .str.replace(r"\s+", " ", regex=True)  # collapse leftover spaces
            .str.strip()
        )

        # Remove binary characters (for the punchographs in Gnomeregan)
        df["cleanedText"] = df["cleanedText"].str.replace(r'(?:\b[01]{8}\b\s*)+', '', regex=True)

        df['cleanedText'] = df['cleanedText'].str.replace(r"\.\s+\.\s+\.", "...", regex=True)  # ". . ." → "..."
        df['cleanedText'] = df['cleanedText'].str.replace(r"\.\s+\.", "...", regex=True)  # ". ."   → "..."
        df['cleanedText'] = df['cleanedText'].str.replace(r"\s\.\s", ". ", regex=True)  # ". ."   → "..."

        # fix space after g
        # df['cleanedText'] = df['cleanedText'].str.replace(r'\$[Gg]\s+', '$g', regex=True)
        # df['player_gender'] = None
        # rows = []
        # for _, row in df.iterrows():
        #     if re.search(r'\$[Gg]', row['cleanedText']):
        #         male_text, female_text = self.handle_gender_options(row['cleanedText'])
        #
        #         row_male = row.copy()
        #         row_male['cleanedText'] = male_text
        #         row_male['player_gender'] = 'm'
        #
        #         row_female = row.copy()
        #         row_female['cleanedText'] = female_text
        #         row_female['player_gender'] = 'f'
        #
        #         rows.extend([row_male, row_female])
        #     else:
        #         rows.append(row)

        # move gender generation to tts_
        new_df = df.copy()
        new_df["cleanedText"] = new_df["cleanedText"].apply(self.clean_text_preserve_gender)

        def get_vanilla_quests_texts(df):
            df_copy = df.copy()
            db_con = make_connection_vmangos()

            print(f"Checking different vanilla quest texts...")
            
            def get_vanilla_text(row):
                if not row.get('source') in ('accept', 'complete', 'progress'):
                    return 0
                
                quest_id = row.get('quest', 0)
                if not quest_id:
                    return 0

                if row['expansion'] > 0:
                    return 0

                vanilla_text = get_vanilla_quest_text(db_con, quest_id, row['source'])
                if not vanilla_text or not vanilla_text[0]:
                    return 0

                if compare_quest_texts(vanilla_text[0], row['text']):
                    return 0
            
                vanilla_text = self.process_text_row(vanilla_text[0])
                return vanilla_text            
            
            df_copy['vanilla_text'] = df_copy.apply(get_vanilla_text, axis=1)


            count = (df_copy['vanilla_text'] != 0).sum()
            count_a = ((df_copy['vanilla_text'] != 0) & (df_copy['source'] == 'accept')).sum()
            count_c = ((df_copy['vanilla_text'] != 0) & (df_copy['source'] == 'complete')).sum()
            count_p = ((df_copy['vanilla_text'] != 0) & (df_copy['source'] == 'progress')).sum()
            print(f"Found {count} different text quests")
            print(f"Accept: {count_a}")
            print(f"Complete: {count_c}")
            print(f"Progress: {count_p}")
            db_con.close()
            return df_copy

        if self.get_module_name() != 'AI_VoiceOverData_Turtle':
            new_df = get_vanilla_quests_texts(new_df)

        # process vanilla text if any
        # new_df = self.process_text_column(new_df, 'vanilla_text')
        # if self.get_module_name() != 'AI_VoiceOverData_Turtle':
        #     new_df['vanilla_text'] = new_df['vanilla_text'].apply(self.process_text_row)
        
        new_df.reset_index(drop=True, inplace=True)
        return new_df

    def process_row(
            self,
            row_tuple,
            temperature=0.75,
            length_penalty=1.0,
            repetition_penalty=10.0,
            top_k=1,
            top_p=1.0,
            speed=1.0,
            cps=15,
            f0_up_key=0,
            f0_method="rmvpe",
            index_rate=0.70,
            filter_radius=3,
            resample_sr=0,
            rms_mix_rate=1,
            protect=0.25
    ):
        row = pd.Series(row_tuple[1:], index=row_tuple._fields[1:])
        # Strip '_dk' suffix for DK voice samples — they are stored under the base voice name
        if row['voice_name'].endswith("_dk"):
            voice_name = f'{row["voice_name"][:-3]}'
        else:
            voice_name = f'{row["voice_name"]}'

        custom_message = ""
        if "$" in row["cleanedText"] or "<" in row["cleanedText"] or ">" in row["cleanedText"]:
            custom_message = f'skipping due to invalid chars: {row["cleanedText"]}'
        elif voice_name not in self.selected_voice_names:
            custom_message = f'skipping due to voice being unselected or unavailable: {voice_name}'
        elif row[
            'source'] == "progress":  # skip progress text (progress text is usually better left unread since its always played before quest completion)
            custom_message = f'skipping progress text: {row["quest"]}-{row["source"]}'
        else:
            self.tts_row(row, voice_name, temperature=temperature, length_penalty=length_penalty,
                         repetition_penalty=repetition_penalty, top_k=top_k, top_p=top_p, speed=speed, cps=cps,
                         f0_up_key=f0_up_key, f0_method=f0_method, index_rate=index_rate, filter_radius=filter_radius,
                         resample_sr=resample_sr, rms_mix_rate=rms_mix_rate, protect=protect)
        return custom_message

    def tts_row(
            self, row, voice_name, **kwargs
    ):
        tts_text = row['cleanedText']
        file_name = f'{row["quest"]}-{row["source"]}' if row['quest'] else f'{row["templateText_race_gender_hash"]}'
        if row['player_gender'] is not None:
            file_name = row['player_gender'] + '-' + file_name
        # Multi-speaker: prepend model-name prefix for quest audio filenames
        if row.get('multi_name') is not None and row['multi_name'] != '':
            if row.get('source') != 'gossip':
                file_name = row['multi_name'] + '-' + file_name
        file_name = file_name + '.mp3'
        subfolder = 'quests' if row['quest'] else 'gossip'
        questgiver_id = row['id'] if row['id'] else None
        self.tts(tts_text, voice_name, file_name, subfolder, questgiver_id=questgiver_id,
                 **kwargs)

    def create_output_dirs(self):
        self.create_output_subdirs('')
        self.create_output_subdirs('quests')
        self.create_output_subdirs('gossip')

    def create_export_dirs(self):
        self.create_export_subdirs('')
        self.create_export_subdirs('quests')
        self.create_export_subdirs('gossip')

    def process_rows_in_parallel(self, df, row_proccesing_fn, selected_voice_names: list[str], max_workers=1, **kwargs):

        total_rows = len(df)
        bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
        self.selected_voice_names = set(selected_voice_names)

        with tqdm(total=total_rows, unit='rows', ncols=100, desc='Generating Audio', ascii=False, bar_format=bar_format,
                  dynamic_ncols=True) as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for row, custom_message in zip(df.iterrows(), executor.map(row_proccesing_fn, df.itertuples())):
                    pbar.set_postfix_str(custom_message)
                    pbar.update(1)

    # --- Multi-speaker table writers ---

    def write_multispeaker_quest_table(self, df, module_name, source_type, filename):
        """Write MultiSpeakerQuests table for a given quest source (accept/complete/progress)."""
        output_file = self.output_folder + f"/{filename}.lua"
        multispeaker_table = {}

        quest_df = df[(df['source'] == source_type) & (df['multi_name'].notna()) & (df['multi_name'] != '') & (df['quest'] != '')]

        for _, row in tqdm(quest_df.iterrows(), total=len(quest_df), desc=f"MultiSpeaker {source_type}"):
            quest_id = int(row['quest'])
            voicename = row['multi_name']

            # Strip '_dk' suffix — DK voice samples are stored under the base voice name
            if voicename.endswith('_dk'):
                voicename = voicename[:-3]

            if source_type not in multispeaker_table:
                multispeaker_table[source_type] = {}
            if quest_id not in multispeaker_table[source_type]:
                multispeaker_table[source_type][quest_id] = {}

            multispeaker_table[source_type][quest_id][voicename] = voicename

        with open(output_file, 'w', encoding='UTF-8') as f:
            f.write(DATAMODULE_TABLE_GUARD_CLAUSE + '\n')
            f.write(f'{module_name}.{filename} = ')
            f.write(lua.encode(multispeaker_table))
            f.write('\n')

        print(f'Finished writing {filename}.lua')

    def write_multispeaker_questlog_table(self, df, module_name, filename):
        """Write MultiSpeakerQuestLog table — default prefix per quest for questlog playback."""
        output_file = self.output_folder + f"/{filename}.lua"
        questlog_table = {}

        accept_df = df[(df['source'] == 'accept') & (df['multi_name'].notna()) & (df['multi_name'] != '') & (df['quest'] != '')]

        for _, row in tqdm(accept_df.iterrows(), total=len(accept_df), desc='MultiSpeaker QuestLog'):
            quest_id = int(row['quest'])
            voicename = row['multi_name']

            # Strip '_dk' suffix
            if voicename.endswith('_dk'):
                voicename = voicename[:-3]

            if quest_id not in questlog_table:
                questlog_table[quest_id] = voicename

        with open(output_file, 'w', encoding='UTF-8') as f:
            f.write(DATAMODULE_TABLE_GUARD_CLAUSE + '\n')
            f.write(f'{module_name}.{filename} = ')
            f.write(lua.encode(questlog_table))
            f.write('\n')

        print(f'Finished writing {filename}.lua')

    def write_multispeaker_gossip_table(self, df, module_name, filename):
        """Write MultiSpeakerGossip table — per-NPC, per-speaker gossip hash lookup."""
        output_file = self.output_folder + f"/{filename}.lua"
        multispeaker_table = {}

        gossip_df = df[(df['source'] == 'gossip') & (df['multi_name'].notna()) & (df['multi_name'] != '')]

        for _, row in tqdm(gossip_df.iterrows(), total=len(gossip_df), desc='MultiSpeaker Gossip'):
            npc_name = row['name']
            voicename = row['multi_name']
            gossip_hash = row['templateText_race_gender_hash']
            escaped_text = row['text'].replace('"', "'").replace('\r', ' ').replace('\n', ' ')

            # Strip '_dk' suffix
            if voicename.endswith('_dk'):
                voicename = voicename[:-3]

            escaped_npc_name = npc_name.replace('"', "'").replace('\r', ' ').replace('\n', ' ')

            if escaped_npc_name not in multispeaker_table:
                multispeaker_table[escaped_npc_name] = {}
            if voicename not in multispeaker_table[escaped_npc_name]:
                multispeaker_table[escaped_npc_name][voicename] = {}

            multispeaker_table[escaped_npc_name][voicename][escaped_text] = gossip_hash

        with open(output_file, 'w', encoding='UTF-8') as f:
            f.write(DATAMODULE_TABLE_GUARD_CLAUSE + '\n')
            f.write(f'{module_name}.{filename} = ')
            f.write(lua.encode(multispeaker_table))
            f.write('\n')

        print(f'Finished writing {filename}.lua')

    # --- End multi-speaker table writers ---

    def write_gossip_file_lookups_table(self, df, module_name, type, table, filename):
        output_file = self.output_folder + f"/{filename}.lua"
        gossip_table = {}

        accept_df = df[(df['quest'] == '') & (df['type'] == type)]

        for i, row in tqdm(accept_df.iterrows()):
            if row['id'] not in gossip_table:
                gossip_table[row['id']] = {}

            escapedText = row['text'].replace('"', '\'').replace('\r', ' ').replace('\n', ' ')

            gossip_table[row['id']][escapedText] = row['templateText_race_gender_hash']

        with open(output_file, "w", encoding="UTF-8") as f:
            f.write(DATAMODULE_TABLE_GUARD_CLAUSE + "\n")
            f.write(f"{module_name}.{table} = ")
            f.write(lua.encode(gossip_table))
            f.write("\n")

        print(f"Finished writing {filename}.lua")

    def write_questlog_npc_lookups_table(self, df, module_name, type, table, filename):
        output_file = self.output_folder + f"/{filename}.lua"
        questlog_table = {}

        accept_df = df[(df['source'] == 'accept') & (df['type'] == type)]

        for i, row in tqdm(accept_df.iterrows()):
            questlog_table[int(row['quest'])] = row['id']

        with open(output_file, "w", encoding="UTF-8") as f:
            f.write(DATAMODULE_TABLE_GUARD_CLAUSE + "\n")
            f.write(f"{module_name}.{table} = ")
            f.write(lua.encode(questlog_table))
            f.write("\n")

        print(f"Finished writing {filename}.lua")

    def write_npc_name_lookup_table(self, df, module_name, type, table, filename):
        output_file = self.output_folder + f"/{filename}.lua"
        npc_name_table = {}

        accept_df = df[df['type'] == type]

        for i, row in tqdm(accept_df.iterrows()):
            npc_name_table[row['id']] = row['name']

        with open(output_file, "w", encoding="UTF-8") as f:
            f.write(DATAMODULE_TABLE_GUARD_CLAUSE + "\n")
            f.write(f"{module_name}.{table} = ")
            f.write(lua.encode(npc_name_table))
            f.write("\n")

        print(f"Finished writing {filename}.lua")

    def write_quest_id_lookup(self, df, module_name):
        output_file = self.output_folder + "/quest_id_lookups.lua"
        quest_id_table = {}

        quest_df = df[df['quest'] != '']

        for i, row in tqdm(quest_df.iterrows()):
            quest_source = row['source']
            quest_id = int(row['quest'])
            quest_title = row['quest_title']
            quest_text = get_first_n_words(row['text'], 25) + ' ' + get_last_n_words(row['text'], 25)
            escaped_quest_text = replace_dollar_bs_with_space(
                quest_text.replace('"', '\'').replace('\r', ' ').replace('\n', ' '))
            escaped_quest_title = quest_title.replace('"', '\'').replace('\r', ' ').replace('\n', ' ')
            npc_name = row['name']
            escaped_npc_name = npc_name.replace('"', '\'').replace('\r', ' ').replace('\n', ' ')

            # table[source][title][npcName][text]
            if quest_source not in quest_id_table:
                quest_id_table[quest_source] = {}

            if escaped_quest_title not in quest_id_table[quest_source]:
                quest_id_table[quest_source][escaped_quest_title] = {}

            if escaped_npc_name not in quest_id_table[quest_source][escaped_quest_title]:
                quest_id_table[quest_source][escaped_quest_title][escaped_npc_name] = {}

            if quest_text not in quest_id_table[quest_source][escaped_quest_title][escaped_npc_name]:
                quest_id_table[quest_source][escaped_quest_title][escaped_npc_name][escaped_quest_text] = quest_id

        pruned_quest_id_table = prune_quest_id_table(quest_id_table)

        # UTF-8 Encoding is important for other languages!
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(DATAMODULE_TABLE_GUARD_CLAUSE + "\n")
            f.write(f"{module_name}.QuestIDLookup = ")
            f.write(lua.encode(pruned_quest_id_table))
            f.write("\n")

    def write_npc_name_gossip_file_lookups_table(self, df, module_name, type, table, filename):
        output_file = self.output_folder + f"/{filename}.lua"
        gossip_table = {}

        accept_df = df[(df['quest'] == '') & (df['type'] == type)]

        for i, row in tqdm(accept_df.iterrows()):
            npc_name = row['name']
            escaped_npc_name = npc_name.replace('"', '\'').replace('\r', ' ').replace('\n', ' ')

            if escaped_npc_name not in gossip_table:
                gossip_table[escaped_npc_name] = {}

            escapedText = row['text'].replace('"', '\'').replace('\r', ' ').replace('\n', ' ')

            gossip_table[escaped_npc_name][escapedText] = row['templateText_race_gender_hash']

        with open(output_file, "w", encoding="UTF-8") as f:
            f.write(DATAMODULE_TABLE_GUARD_CLAUSE + "\n")
            f.write(f"{module_name}.{table} = ")
            f.write(lua.encode(gossip_table))
            f.write("\n")

        print(f"Finished writing {filename}.lua")

    def tts_dataframe(self, df, selected_voices, expansions=[0, 1, 2], module_name=None,
                      temperature=0.75, length_penalty=1.0, repetition_penalty=10.0,
                      top_k=1, top_p=1.0, speed=1.0, cps=15, f0_up_key=0, f0_method="rmvpe",
                      index_rate=0.70, filter_radius=3, resample_sr=0, rms_mix_rate=1,
                      protect=0.25):
        # Add specific codes for quests that need to be in the module search table so the audio for certain quests with the same name is found properly.
        exp_for_audio = expansions

        if 0 in expansions:
            expansions.append(-77)
        if 1 in expansions:
            expansions.append(-88)
        if 2 in expansions:
            expansions.append(-99)

        df = df[df['expansion'].isin(exp_for_audio)]
        df.sort_values("voice_name")

        if module_name:
            self.module_name = module_name
            self.output_folder = os.path.join(self.module_name, 'generated')
            self.sound_output_folder = os.path.join(self.output_folder, 'sounds')

        self.create_output_dirs()
        self.process_rows_in_parallel(df, self.process_row, selected_voices, max_workers=1,
                                      temperature=temperature, length_penalty=length_penalty,
                                      repetition_penalty=repetition_penalty, top_k=top_k, top_p=top_p, speed=speed, cps=cps,
                                      f0_up_key=f0_up_key, f0_method=f0_method, index_rate=index_rate,
                                      filter_radius=filter_radius,
                                      resample_sr=resample_sr, rms_mix_rate=rms_mix_rate, protect=protect)
        print("Audio finished generating.")

    def export_module_full(self, df, module_name=None, volume_fix=False, volume_value=None):
        if not module_name:
            print(f"Module not set!")
            return
        
        if module_name:
            self.module_name_export = module_name
            self.output_folder_export = os.path.join(ADDON_PATH_EXPORT, self.module_name_export)
            self.output_folder_export = os.path.join(self.output_folder_export, 'generated')
            self.sound_output_folder_export = os.path.join(self.output_folder_export, 'sounds')

        # replace output folder for export
        old_output_folder = self.output_folder
        old_sound_output_folder = self.sound_output_folder
        self.output_folder = self.output_folder_export
        self.sound_output_folder = self.sound_output_folder_export
        
        self.create_export_dirs()

        def get_module_expansion_code(module_name):
            mapping = {
                'AI_VoiceOverData_Vanilla': 0,
                'AI_VoiceOverData_VanillaExtra': 0,
                'AI_VoiceOverData_TBC': 1,
                'AI_VoiceOverData_WoTLK': 2,
                'AI_VoiceOverData_Turtle': 3
            }
            return mapping.get(module_name, module_name)

        df = df[df['expansion'] == get_module_expansion_code(module_name)]

        base_works = Path(ADDON_PATH_WORKS) / module_name
        export_path = Path(ADDON_PATH_EXPORT) / module_name
        module_path = Path(module_name)
        
        print(f"📁 Merging files for module: {module_name}")
        print(f"   Base source: {base_works}")
        print(f"   Module source: {module_path}")
        print(f"   Export destination: {export_path}")

        # Step 1: Copy all files from ADDON_PATH_WORKS to export folder
        if base_works.exists():
            print(f"📋 Copying base files (skipping existing)...")
            copied_count = 0
            skipped_count = 0
            for item in base_works.rglob('*'):
                if item.is_file():
                    # Calculate relative path
                    rel_path = item.relative_to(base_works)
                    dest_path = export_path / rel_path
                    
                    # Skip if file already exists
                    if dest_path.exists():
                        skipped_count += 1
                        continue

                    # Create parent directories if needed
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    # Copy file
                    shutil.copy2(item, dest_path)
                    copied_count += 1

            print(f"   ✅ Copied {copied_count} new files, skipped {skipped_count} existing")
        else:
            print(f"   ⚠️ Base works folder not found: {base_works}")
            return f"   ⚠️ Base works folder not found: {base_works}"

        # Step 2: Copy from module_name (overwrite always)
        if module_path.exists():
            print(f"📋 Copying module files (overwriting)...")
            overwritten_count = 0
            
            for item in module_path.rglob('*'):
                if item.is_file() and 'mp3' in item.name:
                    rel_path = item.relative_to(module_path)
                    dest_path = export_path / rel_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, dest_path)
                    overwritten_count += 1

                    # fix volume
                    if volume_fix and volume_value:
                        self.match_peak_volume(None, False, dest_path, dest_path, volume_value)

                    if not self.is_already_correct_format(dest_path):
                        print(f"Correcting formal to 64kbs 44100hz for {dest_path}")
                        self.correct_format(dest_path)
                        
            
            print(f"   ✅ Copied {overwritten_count} files (overwriting existing)")
        else:
            print(f"   ⚠️ Module folder not found: {module_path}")
            return f"   ⚠️ Module folder not found: {module_path}"

        print(f"✅ Merge complete! Files are in: {export_path}")
    
        total_files = sum(1 for _ in export_path.rglob('*') if _.is_file())
        print(f"   Total files in export: {total_files}")

        self.write_gossip_file_lookups_table(df, module_name, 'creature', 'GossipLookupByNPCID',
                                             'npc_gossip_file_lookups')
        self.write_gossip_file_lookups_table(df, module_name, 'gameobject', 'GossipLookupByObjectID',
                                             'object_gossip_file_lookups')

        self.write_quest_id_lookup(df, module_name)
        print("Finished writing quest_id_lookups.lua")

        self.write_npc_name_gossip_file_lookups_table(df, module_name, 'creature', 'GossipLookupByNPCName',
                                                      'npc_name_gossip_file_lookups')
        self.write_npc_name_gossip_file_lookups_table(df, module_name, 'gameobject', 'GossipLookupByObjectName',
                                                      'object_name_gossip_file_lookups')

        self.write_questlog_npc_lookups_table(df, module_name, 'creature', 'NPCIDLookupByQuestID',
                                              'questlog_npc_lookups')
        self.write_questlog_npc_lookups_table(df, module_name, 'gameobject', 'ObjectIDLookupByQuestID',
                                              'questlog_object_lookups')
        self.write_questlog_npc_lookups_table(df, module_name, 'item', 'ItemIDLookupByQuestID', 'questlog_item_lookups')

        self.write_npc_name_lookup_table(df, module_name, 'creature', 'NPCNameLookupByNPCID', 'npc_name_lookups')
        self.write_npc_name_lookup_table(df, module_name, 'gameobject', 'ObjectNameLookupByObjectID',
                                         'object_name_lookups')
        self.write_npc_name_lookup_table(df, module_name, 'item', 'ItemNameLookupByItemID', 'item_name_lookups')

        # --- Multi-speaker tables ---
        for source in ['accept', 'complete', 'progress']:
            self.write_multispeaker_quest_table(df, module_name, source, f'multispeaker_{source}')
        self.write_multispeaker_questlog_table(df, module_name, 'multispeaker_questlog')
        self.write_multispeaker_gossip_table(df, module_name, 'multispeaker_gossip')

        write_sound_length_table_lua(module_name, self.sound_output_folder, self.output_folder)
        print("Updated sound_length_table.lua")

        self.output_folder = old_output_folder
        self.sound_output_folder = old_sound_output_folder

        return "Lookup tables complete."
        

    def generate_lookup_tables(self, df, expansions=[0, 1, 2], module_name=None):
        if module_name:
            self.module_name = module_name
            self.output_folder = os.path.join(self.module_name, 'generated')
            self.sound_output_folder = os.path.join(self.output_folder, 'sounds')

        self.create_output_dirs()

        if 0 in expansions:
            expansions.append(-77)
        if 1 in expansions:
            expansions.append(-88)
        if 2 in expansions:
            expansions.append(-99)

        df = df[df['expansion'].isin(expansions)]
        self.create_output_dirs()
        self.write_gossip_file_lookups_table(df, module_name, 'creature', 'GossipLookupByNPCID',
                                             'npc_gossip_file_lookups')
        self.write_gossip_file_lookups_table(df, module_name, 'gameobject', 'GossipLookupByObjectID',
                                             'object_gossip_file_lookups')

        self.write_quest_id_lookup(df, module_name)
        print("Finished writing quest_id_lookups.lua")

        self.write_npc_name_gossip_file_lookups_table(df, module_name, 'creature', 'GossipLookupByNPCName',
                                                      'npc_name_gossip_file_lookups')
        self.write_npc_name_gossip_file_lookups_table(df, module_name, 'gameobject', 'GossipLookupByObjectName',
                                                      'object_name_gossip_file_lookups')

        self.write_questlog_npc_lookups_table(df, module_name, 'creature', 'NPCIDLookupByQuestID',
                                              'questlog_npc_lookups')
        self.write_questlog_npc_lookups_table(df, module_name, 'gameobject', 'ObjectIDLookupByQuestID',
                                              'questlog_object_lookups')
        self.write_questlog_npc_lookups_table(df, module_name, 'item', 'ItemIDLookupByQuestID', 'questlog_item_lookups')

        self.write_npc_name_lookup_table(df, module_name, 'creature', 'NPCNameLookupByNPCID', 'npc_name_lookups')
        self.write_npc_name_lookup_table(df, module_name, 'gameobject', 'ObjectNameLookupByObjectID',
                                         'object_name_lookups')
        self.write_npc_name_lookup_table(df, module_name, 'item', 'ItemNameLookupByItemID', 'item_name_lookups')

        # --- Multi-speaker tables ---
        for source in ['accept', 'complete', 'progress']:
            self.write_multispeaker_quest_table(df, module_name, source, f'multispeaker_{source}')
        self.write_multispeaker_questlog_table(df, module_name, 'multispeaker_questlog')
        self.write_multispeaker_gossip_table(df, module_name, 'multispeaker_gossip')

        write_sound_length_table_lua(module_name, self.sound_output_folder, self.output_folder)
        print("Updated sound_length_table.lua")

        return "Lookup tables complete."

    def tts_quest_auto(self, df, quest_id: str, spec=None, module_name="AI_VoiceOverData_TBC", voice_to_clone=None, volume_match_sample=False):

        if spec is None:
            spec = ['accept', 'complete', 'progress']

        quest_id = str(quest_id)

        df_filtered = df[(df['quest'] == quest_id) & (df['expansion'] >= 0) & df['source'].isin(spec)]

        df_filtered['voice_type'] = df_filtered['voice_type'].apply(extract_voice_type)

        generated_ok = False
        a_path = None
        generate_check = None

        for _, row in df_filtered.iterrows():
            voice_name = row['voice_name']
            voice_type = row['voice_type']
            npc_id = row['id']
            voice_sample = voice_to_clone

            default_seed = None
            randomize_seed = True

            default_seed = self.seed_manager.get_default_seed(voice_name, voice_type)
            if default_seed:
                randomize_seed = False

            if not voice_sample:
                voice_sample = get_default_sample(voice_name, voice_type)
                if not voice_sample:
                    if voice_name == 'custom_custom':
                        voice_sample_extracted = get_game_sample_for_npc_id(npc_id)
                        if not voice_sample_extracted:
                            voice_sample_extracted = generate_game_sample_for_npc_id(npc_id, module_name)
                            if voice_sample_extracted:
                                voice_sample = voice_sample_extracted
                        else:
                            voice_sample = voice_sample_extracted

            if not voice_sample:
                print(f"tts_quest_auto: voice for {voice_name} ({voice_type}) not found!")
                continue

            if row['has_narrator']:
                narrator_sample = get_default_sample('narrator')
            else:
                narrator_sample = None

            print(f"generating auto for quest {quest_id}")

            # check if is problematic
            params = {}
            if self.should_regenerate_quest(quest_id, row['source']):
                    params = {
                    "temperature": 5.0,
                    "class_temperature": 0.0,
                    "cfg_scale": 2.0,
                    "speed": 1.0
                }

            generate_check, a_exp, a_path, _ = self.tts_quest(df, quest_id, module_name, row['source'], force_gen=False, free_memory=False, voice_to_clone=voice_sample, voice_to_clone_2=narrator_sample, detect_emotions=False, seed=default_seed, randomize_seed=randomize_seed, volume_match_sample=volume_match_sample, voice_name=voice_name, voice_type=voice_type, **params)
            if generate_check != None:
                generated_ok = True
        
        return generated_ok, a_path, generate_check

    def tts_quest(self, df, quest_id: str, module_name="AI_VoiceOverData_TBC",
                  spec=None, voice_to_clone=None, voice_to_clone_2=None, text_to_use = None,
                  model_name="VibeVoice-1.5B", seed: int = 42, randomize_seed: bool = False, temperature=5.0, class_temperature=0.3, speed=1.0, cps=15,
                  max_words_per_chunk=75, attention_type="auto",
                  quantize_llm="full precision", free_memory=True, steps: int = 32,
                  cfg_scale: float = 2.0, use_sampling: bool = True, force_gen = False, volume_match = True, volume_match_sample = False, apply_effects = True, apply_effects_override = None, detect_emotions = False, voice_name = None, voice_type = None):

        if spec is None:
            spec = ['accept', 'complete', 'progress']
        quest_id = str(quest_id)

        if isinstance(spec, str):
            spec = [spec]
        """
        Regenerate audio for a specific quest id (overwrites existing audio).
        """
        if module_name:
            self.module_name = module_name
            self.output_folder = os.path.join(self.module_name, 'generated')
            self.sound_output_folder = os.path.join(self.output_folder, 'sounds')

        self.create_output_dirs()

        quest_id = str(quest_id)

        df_filtered = df[(df['quest'] == quest_id) & (df['expansion'] >= 0) & df['source'].isin(spec)]

        # Добавь прямо перед генерацией:
        print("\n=== DEBUG DUPLICATES ===")
        # Покажи все строки для этого квеста с source='accept'
        dupes = df[(df['quest'] == quest_id) & (df['source'] == spec[0])]
        print(f"Found {len(dupes)} rows with source='accept' for quest {quest_id}:")
        for idx, row in dupes.iterrows():
            print(f"  Row {idx}: id={row.get('id', 'N/A')}, name={row.get('name', 'N/A')}")
            print(f"    text: {row.get('cleanedText', '')[:50]}...")

        if len(df_filtered) > 1 and len(spec) == 1:
            print(f"⚠️ WARNING: Found {len(df_filtered)} rows, taking first one!")
            df_filtered = df_filtered.iloc[[0]]  # Берем только первую

        expansions = df_filtered['expansion'].drop_duplicates().tolist()
        if 0 in expansions:
            expansions.append(-77)
        if 1 in expansions:
            expansions.append(-88)
        if 2 in expansions:
            expansions.append(-99)

        if df_filtered.empty:
            print(f"No rows found for quest {quest_id}", flush=True)
            return None

        two_genders = False
        narrator_voice = False
        vanilla_different = False


        if self.get_module_name() != 'AI_VoiceOverData_Turtle':
            # extend rows with vanilla text before adding gender text
            rows = []
            for _, row in df_filtered.iterrows():

                if row['vanilla_text'] != 0:
                    vanilla_different = True
                    print(f"tts_quest: different vanilla_text")
                
                if text_to_use is not None and len(spec) == 1 and (text_to_use != row['cleanedText'] and text_to_use != row['vanilla_text']):
                    rows.append(row)
                elif row['vanilla_text'] != 0:

                    audio_original = self.find_quest_audio(quest_id, row['source'], MODULE_NAMES[0])
                    if audio_original and audio_original.get(MODULE_NAMES[0]):
                        row_og = row.copy()
                        # remove marking here
                        row_og['vanilla_text'] = 0
                        rows.append(row_og)
                    else:
                        print(f"tts_quest: adding vanilla_text")
                        row_vanilla = row.copy()
                        row_vanilla['cleanedText'] = row_vanilla['vanilla_text']
                        row_og = row.copy()
                        # remove marking here
                        row_og['vanilla_text'] = 0
                        if module_name == MODULE_NAMES[1]:
                            rows.extend([row_og, row_vanilla])
                        else:
                            rows.extend([row_vanilla, row_og])
                else:
                    rows.append(row)

            df_filtered = pd.DataFrame(rows)

        df_filtered['player_gender'] = None
        df_filtered['multi_speaker'] = None  # track which multi_name this row is for
        rows = []
        for _, row in df_filtered.iterrows():

            if self.get_module_name() != 'AI_VoiceOverData_Turtle':
                if text_to_use is not None and len(spec) == 1 and (text_to_use != row['cleanedText'] and (not vanilla_different and text_to_use != row['vanilla_text'])):
                    print(f"tts_quest: replacing text from UI")
                    row['cleanedText'] = text_to_use
            else:
                if text_to_use is not None and len(spec) == 1 and text_to_use != row['cleanedText']:
                    print(f"tts_quest: replacing text from UI")
                    row['cleanedText'] = text_to_use

            if re.search(r'\$[Gg]', row['cleanedText']):
                print(f"Gender tag found in quest {quest_id}...", flush=True)
                two_genders = True
                male_text, female_text = self.handle_gender_options(row['cleanedText'])

                row_female = row.copy()
                row_female['cleanedText'] = female_text
                row_female['player_gender'] = 'f'

                row_male = row.copy()
                row_male['cleanedText'] = male_text
                row_male['player_gender'] = 'm'

                rows.extend([row_female, row_male])
            else:
                rows.append(row)

        df_filtered = pd.DataFrame(rows)

        # Multi-speaker splitting: for rows with multi_name, spawn one row per
        # distinct multi_name value (like gender splitting but per-speaker).
        # Each spawned row uses that speaker's voice_name for audio selection.
        multi_rows = []
        for _, row in df_filtered.iterrows():
            multi_name = row.get('multi_name')
            if multi_name is not None and multi_name != '':
                # Find all rows with the same multi_name (same dialogue, different speakers)
                matching = df_filtered[df_filtered['multi_name'] == multi_name]
                # Only split if there are actually multiple different voice_names
                # sharing this multi_name (i.e. true multi-speaker)
                distinct_voices = matching['voice_name'].unique()
                if len(distinct_voices) > 1:
                    # Spawn one row per distinct voice_name for this multi_name
                    for vname in distinct_voices:
                        spawned = matching[matching['voice_name'] == vname].iloc[0].copy()
                        spawned['multi_speaker'] = vname  # use original voice_name as speaker key
                        multi_rows.append(spawned)
                else:
                    # Single voice for this multi_name — keep as-is
                    multi_rows.append(row)
            else:
                multi_rows.append(row)

        df_filtered = pd.DataFrame(multi_rows)

        print(f"Regenerating {len(df_filtered)} rows for quest {quest_id}...", flush=True)

        final_seed = random.randint(0, 2147483647) if randomize_seed else int(seed)
        print(f"Generating with seed: {final_seed}")

        for _, row in df_filtered.iterrows():
            file_name = f'{row["quest"]}-{row["source"]}' if row['quest'] else f'{row["templateText_race_gender_hash"]}'
            if row['player_gender'] is not None and isinstance(row['player_gender'], str):
                file_name = row['player_gender'] + '-' + file_name
                print(f"Generating {row['player_gender']} voice...")
            # Multi-speaker: use multi_speaker as prefix for quest audio
            if row.get('multi_speaker') is not None and row['multi_speaker'] != '':
                file_name = row['multi_speaker'] + '-' + file_name
                print(f"Generating multi-speaker audio for: {row['multi_speaker']}...")
            file_name = file_name + '.mp3'

            subfolder = 'quests' if row['quest'] else 'gossip'
            tts_text = row['cleanedText']
            # Use multi_speaker voice_name for audio sample selection
            current_voice_name = row.get('multi_speaker') or row['voice_name']
            questgiver_id = row['id']

            if voice_to_clone_2:
                narrator_voice = True

            narrator_voice = voice_to_clone_2

            # check if really has narrator
            print(f"Checking Narrator again: narrator found: {narrator_voice}")
            def check_narrator_string(text):
                return any(target in text for target in ["[1]", "[2]"])
            
            if not check_narrator_string(tts_text):
                narrator_voice = None

            if self.get_module_name() != 'AI_VoiceOverData_Turtle':
                # put vanilla texts to vanilla & wotlk
                if row['vanilla_text'] != 0:
                    self.module_name = MODULE_NAMES[1]
                    self.output_folder = os.path.join(self.module_name, 'generated')
                    self.sound_output_folder = os.path.join(self.output_folder, 'sounds')
                else:
                    self.module_name = module_name if not vanilla_different else MODULE_NAMES[3]
                    self.output_folder = os.path.join(self.module_name, 'generated')
                    self.sound_output_folder = os.path.join(self.output_folder, 'sounds')

            self.tts(tts_text, current_voice_name, voice_type, voice_to_clone, narrator_voice, file_name, subfolder, forceGen=force_gen, volume_match = volume_match, volume_match_sample = volume_match_sample,
                     questgiver_id=questgiver_id, model_name=model_name, seed=final_seed, temperature=temperature,
                     class_temperature=class_temperature, speed=speed, cps=cps,
                     max_words_per_chunk=max_words_per_chunk, attention_type=attention_type,
                     quantize_llm=quantize_llm, free_memory=free_memory, steps=steps,
                     cfg_scale=cfg_scale, use_sampling=use_sampling, apply_effects=apply_effects, apply_effects_override=apply_effects_override, detect_emotions=detect_emotions)

        return f"Generation complete{f' for 2 genders' if two_genders else ''}{f' with narrator voice' if narrator_voice else ''}", expansions, self.output_path, gradio.update(value=final_seed)  # Return both info message and expansions

    def tts_gossip_auto(self, df, npc_name = None, module_name="AI_VoiceOverData_TBC", file_hash=None, voice_to_clone=None, volume_match_sample=False):

        if npc_name:
            spec = str(npc_name)
        else:
            return False, False, "No specification given to produce audio on."

        df_filtered = df[(df['name'] == spec) & (df['expansion'] >= 0) & (df['source'] == 'gossip')]

        df_filtered['voice_type'] = df_filtered['voice_type'].apply(extract_voice_type)

        generate_check = None
        generated_ok = False
        a_path = None

        print(f"tts_gossip_auto: files: {len(df_filtered)}, hash set: {file_hash}")

        for _, row in df_filtered.iterrows():
            if row['templateText_race_gender_hash'] != file_hash:
                continue

            voice_name = row['voice_name']
            voice_type = row['voice_type']
            npc_id = row['id']
            voice_sample = voice_to_clone
            default_seed = None
            randomize_seed = True

            default_seed = self.seed_manager.get_default_seed(voice_name, voice_type)
            if default_seed:
                randomize_seed = False

            if not voice_sample:
                voice_sample = get_default_sample(voice_name, voice_type)
                if not voice_sample:
                    if voice_name == 'custom_custom':
                        voice_sample_extracted = get_game_sample_for_npc_id(npc_id)
                        if not voice_sample_extracted:
                            voice_sample_extracted = generate_game_sample_for_npc_id(npc_id, module_name)
                            if voice_sample_extracted:
                                voice_sample = voice_sample_extracted
                        else:
                            voice_sample = voice_sample_extracted

            if not voice_sample:
                print(f"tts_quest_auto: voice for {voice_name} ({voice_type}) not found!")
                continue

            if row['has_narrator']:
                narrator_sample = get_default_sample('narrator')
            else:
                narrator_sample = None

            print(f"generating auto for npc {npc_name}")

            # check if is problematic
            params = {}
            if self.should_regenerate_gossip(npc_name, row['templateText_race_gender_hash']):
                params = {
                "temperature": 5.0,
                "class_temperature": 0.0,
                "cfg_scale": 2.0,
                "speed": 1.0
                }

            generate_check, a_exp, a_path, _ = self.tts_gossip(df, module_name, npc_name, force_gen=False, free_memory=False, voice_to_clone=voice_sample, voice_to_clone_2=narrator_sample, detect_emotions=False, seed=default_seed, randomize_seed=randomize_seed, volume_match_sample=volume_match_sample, text_to_use=row['cleanedText'], cleaned_text_og=row['cleanedText'], voice_name=voice_name, voice_type=voice_type, **params)
            if generate_check != None:
                generated_ok = True
        
        return generated_ok, a_path, generate_check
    
    def tts_gossip(self, df, module_name="AI_VoiceOverData_TBC", npc_name = None, voice_to_clone=None, voice_to_clone_2=None, cleaned_text_og=None, text_to_use=None,
                  model_name="VibeVoice-1.5B", seed: int = 42, randomize_seed: bool = False, temperature=5.0, class_temperature=0.3, speed=1.0, cps=15,
                  max_words_per_chunk=75, attention_type="auto",
                  quantize_llm="full precision", free_memory=True, steps: int = 32,
                  cfg_scale: float = 2.0, use_sampling: bool = True, force_gen = False, volume_match = True, volume_match_sample = False, apply_effects = True, apply_effects_override = None, detect_emotions = False, voice_name = None, voice_type = None):

        """
        Regenerate audio for a specific gossip, either specific npc or race_gender (overwrites existing audio).
        """
        if module_name:
            self.module_name = module_name
            self.output_folder = os.path.join(self.module_name, 'generated')
            self.sound_output_folder = os.path.join(self.output_folder, 'sounds')

        self.create_output_dirs()

        if npc_name:
            spec = str(npc_name)
        else:
            return "No specification given to produce audio on."

        df_filtered = df[(df['name'] == spec) & (df['expansion'] >= 0) & (df['source'] == 'gossip')]

        expansions = df['expansion'].drop_duplicates().tolist()
        if 0 in expansions:
            expansions.append(-77)
        if 1 in expansions:
            expansions.append(-88)
        if 2 in expansions:
            expansions.append(-99)

        if df_filtered.empty:
            print(f"No rows found for NPC {npc_name}", flush=True)
            return None

        two_genders = False
        narrator_voice = False
        df_filtered['player_gender'] = None
        df_filtered['multi_speaker'] = None
        rows = []
        for _, row in df_filtered.iterrows():
            # use only selected text
            if cleaned_text_og and text_to_use:
                if row['cleanedText'] != cleaned_text_og:
                    continue
                row['cleanedText'] = text_to_use

            if re.search(r'\$[Gg]', row['cleanedText']):
                two_genders = True
                print(f"Gender tag found in gossip for {npc_name}...", flush=True)
                male_text, female_text = self.handle_gender_options(row['cleanedText'])

                row_female = row.copy()
                row_female['cleanedText'] = female_text
                row_female['player_gender'] = 'f'

                row_male = row.copy()
                row_male['cleanedText'] = male_text
                row_male['player_gender'] = 'm'

                rows.extend([row_male, row_female])
            else:
                rows.append(row)

        df_filtered = pd.DataFrame(rows)

        # Multi-speaker splitting for gossip: spawn one row per distinct
        # voice_name sharing the same multi_name.
        multi_rows = []
        for _, row in df_filtered.iterrows():
            multi_name = row.get('multi_name')
            if multi_name is not None and multi_name != '':
                matching = df_filtered[df_filtered['multi_name'] == multi_name]
                distinct_voices = matching['voice_name'].unique()
                if len(distinct_voices) > 1:
                    for vname in distinct_voices:
                        spawned = matching[matching['voice_name'] == vname].iloc[0].copy()
                        spawned['multi_speaker'] = vname
                        multi_rows.append(spawned)
                else:
                    multi_rows.append(row)
            else:
                multi_rows.append(row)

        df_filtered = pd.DataFrame(multi_rows)

        print(f"Regenerating {len(df_filtered)} rows for NPC {npc_name}...", flush=True)

        final_seed = random.randint(0, 2147483647) if randomize_seed else int(seed)
        print(f"Generating with seed: {final_seed}")

        for _, row in df_filtered.iterrows():
            file_name = f'{row["quest"]}-{row["source"]}' if row['quest'] else f'{row["templateText_race_gender_hash"]}'
            if row['player_gender'] is not None and isinstance(row['player_gender'], str):
                file_name = row['player_gender'] + '-' + file_name
            # Multi-speaker: use multi_speaker as prefix
            if row.get('multi_speaker') is not None and row['multi_speaker'] != '':
                file_name = row['multi_speaker'] + '-' + file_name
                print(f"Generating multi-speaker gossip for: {row['multi_speaker']}...")
            file_name = file_name + '.mp3'

            subfolder = 'quests' if row['quest'] else 'gossip'
            tts_text = row['cleanedText']
            current_voice_name = row.get('multi_speaker') or row['voice_name']
            questgiver_id = row['id']

            if voice_to_clone_2:
                narrator_voice = True

            # check if really has narrator
            print(f"Checking Narrator again: narrator found: {narrator_voice}")
            def check_narrator_string(text):
                return any(target in text for target in ["[1]", "[2]"])
            
            if not check_narrator_string(tts_text):
                narrator_voice = None


            self.tts(tts_text, current_voice_name, voice_type, voice_to_clone, voice_to_clone_2, file_name, subfolder, forceGen=force_gen, volume_match = volume_match, volume_match_sample = volume_match_sample,
                     questgiver_id=questgiver_id, model_name=model_name, seed=final_seed, temperature=temperature,
                     class_temperature=class_temperature, speed=speed, cps=cps,
                     max_words_per_chunk=max_words_per_chunk, attention_type=attention_type,
                     quantize_llm=quantize_llm, free_memory=free_memory, steps=steps,
                     cfg_scale=cfg_scale, use_sampling=use_sampling, apply_effects=apply_effects, apply_effects_override=apply_effects_override, detect_emotions=detect_emotions)

        return f"Generation complete{f' for 2 genders' if two_genders else ''}{f' with narrator voice' if narrator_voice else ''}", expansions, self.output_path, gradio.update(
            value=final_seed)  # Return both info message and expansions


    def find_quest_audio(self, q_id, q_type, module_name=None):
        regular_filename = f"{q_id}-{q_type}.mp3"
        fallback_filename = f"m-{q_id}-{q_type}.mp3"
        results = {}

        def check_file(filename):
            for module in MODULE_NAMES:
                full_path = Path(ADDON_PATH) / module / "generated/sounds/quests" / filename
                if full_path.exists():
                    results[module] = str(full_path)

            for module in MODULE_NAMES:
                if module_name and module != module_name:
                    continue

                out_path = Path(module) / "generated/sounds/quests" / filename
                if out_path.exists():
                    results['Output'] = str(out_path)

            # check if custom module
            if self.get_module_name() != 'AI_VoiceOverData_TBC':
                if self.get_module_name() == module_name:
                    out_path = Path(self.get_module_name()) / "generated/sounds/quests" / filename
                    if out_path.exists():
                        results['Output'] = str(out_path)


            test_path = Path('Testing') / "generated/sounds/quests" / filename
            if test_path.exists():
                results['Testing'] = str(test_path)

            game_path = Path(VOICE_SAMPLE_PATH) / "GameVoice" / filename
            if game_path.exists():
                results['FromGame'] = str(game_path)

        check_file(fallback_filename)
        check_file(regular_filename)

        return results

    def find_gossip_audio(self, npc_race_gender_hash, module_name=None, gender='male'):
        gender_prefix = 'm-' if gender == 'male' else 'f-'
        regular_filename = f"{npc_race_gender_hash}.mp3"
        fallback_filename = f"{gender_prefix}{npc_race_gender_hash}.mp3"
        results = {}

        def check_file(filename):
            for module in MODULE_NAMES:
                full_path = Path(ADDON_PATH) / module / "generated/sounds/gossip" / filename
                if full_path.exists():
                    results[module] = str(full_path)

            for module in MODULE_NAMES:
                if module_name and module != module_name:
                    continue

                out_path = Path(module) / "generated/sounds/gossip" / filename
                if out_path.exists():
                    results['Output'] = str(out_path)

                works_path = Path(ADDON_PATH_WORKS) / module / "generated/sounds/gossip" / filename
                if works_path.exists():
                    results['Works'] = str(works_path)

            # check if custom module
            if self.get_module_name() != 'AI_VoiceOverData_TBC':
                if self.get_module_name() == module_name:
                    out_path = Path(self.get_module_name()) / "generated/sounds/gossip" / filename
                    if out_path.exists():
                        results['Output'] = str(out_path)

            out_path = Path('Testing') / "generated/sounds/gossip" / filename
            if out_path.exists():
                results['Testing'] = str(out_path)

            if not results.get('FromGame'):
                out_path = Path(VOICE_SAMPLE_PATH) / "GameVoice" / filename
                if out_path.exists():
                    results['FromGame'] = str(out_path)

        check_file(fallback_filename)
        check_file(regular_filename)

        return results