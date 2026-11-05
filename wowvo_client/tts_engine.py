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
from pydub import AudioSegment
from scipy.io import wavfile
from scipy import signal
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

from data_prep.env_vars import MODELS_DIR, ADDON_PATH, VOICE_SAMPLE_PATH, MODULE_NAMES, ADDON_PATH_EXPORT, ADDON_PATH_WORKS
from tts.vvembed.modular.modeling_vibevoice_inference import (
    VibeVoiceForConditionalGenerationInference,
)
import wowvo_client.audio_effects as ae
from wowvo_client.consts import RACE_DICT, GENDER_DICT, VOICE_MODEL_MAP, NPC_EFFECTS
from wowvo_client.length_table import write_sound_length_table_lua
from wowvo_client.utils import get_first_n_words, get_last_n_words, replace_dollar_bs_with_space
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
# logging.basicConfig(
#     level=logging.INFO
# )
logger = logging.getLogger("VibeVoice")
logger.setLevel(logging.INFO)

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
            return []

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
        npcsoundid_column = 'NPCSoundID'
        creature_di_table = 'creaturedisplayinfo'
    else:
        display_column = 'CreatureDisplayID as display_id1'
        creature_table = 'creature_template_model'
        npcsoundid_column = 'NPCSoundID'
        creature_di_table = 'creaturedisplayinfo2'

    def get_npc_sound_files(npc_id: int, db_connection) -> List[Dict[str, str]]:
        result_files = []
        
        try:
            with db_connection.cursor() as cursor:
                # 1. display_id1
                cursor.execute("""
                    SELECT %s FROM %s WHERE entry = %s
                """, (display_column, creature_table, npc_id,))
                creature = get_row_as_dict(cursor)
                if not creature:
                    return []
                display_id = creature['display_id1']
                
                # 2. NPCSoundID
                cursor.execute("""
                    SELECT NPCSoundID FROM %s WHERE ID = %s
                """, (creature_di_table, display_id,))
                display_info = get_row_as_dict(cursor)
                if not display_info:
                    return []
                npc_sound_id = display_info['NPCSoundID']
                
                # 3. SoundID_1-4
                cursor.execute("""
                    SELECT SoundID_1, SoundID_2
                    FROM dbc_npcsounds WHERE ID = %s
                """, (npc_sound_id,))
                npc_sounds = get_row_as_dict(cursor)
            
                if not npc_sounds:
                    print(f"❌ NPC {npc_id}: dbc_npcsounds entry not found (id={npc_sound_id})")
                    return []
            
                # 4. Обрабатываем каждый SoundID
                for i in range(1, 3):
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
        
        return result_files
    
    sound_files  = get_npc_sound_files(npc_id, db_con)

    if not sound_files:
        print(f"❌ NPC {npc_id} sound files not found")
        return None
    
    # Объединяем все найденные звуки в один сэмпл
    combined = AudioSegment.empty()
    
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
    
    # Сохраняем
    output_subdir = os.path.join(VOICE_SAMPLE_PATH, "GameVoice")
    os.makedirs(output_subdir, exist_ok=True)
    output_path = os.path.join(output_subdir, f"npc_{npc_id}.mp3")
    
    combined.export(output_path, format="mp3", bitrate="64k")
    print(f"✅ Sample created: {output_path} (duration: {len(combined)/1000:.1f} sec)")
    
    return output_path

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
                return {'temperature': 0.75, 'top_p': 0.9, 'cfg_scale': 1.3, 'voice_speed_factor': 1.0, 'label': 'Default', 'score': 0}

         # Detect emotion
        try:
            result = self.emotion_classifier(text, top_k=1)[0]
            emotion = result['label']
            score = result['score']
        except Exception as e:
            print(f"⚠️ Emotion detection failed: {e}")
            return {'temperature': 0.75, 'top_p': 0.9, 'cfg_scale': 1.3, 'voice_speed_factor': 1.0, 'label': 'Default', 'score': 0}

        # Emotion to TTS parameters mapping
        # emotion_map = {
        #     'anger': {'temperature': 0.9, 'top_p': 0.95, 'label': 'Angry'},
        #     'joy': {'temperature': 0.85, 'top_p': 0.95, 'label': 'Joyful'},
        #     'sadness': {'temperature': 0.7, 'top_p': 0.85, 'label': 'Sad'},
        #     'fear': {'temperature': 0.8, 'top_p': 0.9, 'label': 'Frightened'},
        #     'surprise': {'temperature': 0.85, 'top_p': 0.92, 'label': 'Surprised'},
        #     'love': {'temperature': 0.75, 'top_p': 0.88, 'label': 'Loving'},
        #     'disgust': {'temperature': 0.7, 'top_p': 0.88, 'label': 'Disgusted'},
        #     'neutral': {'temperature': 0.75, 'top_p': 0.9, 'label': 'Neutral'}
        # }
        # VibeVoice Stable Emotion Map
        # Note: cfg_scale in VibeVoice is usually 1.0 to 2.0
        # speed_factor: < 1.0 is slower, > 1.0 is faster
        emotion_map = {
            'anger':    {'temperature': 0.82, 'top_p': 0.90, 'cfg_scale': 1.3, 'speed_factor': 1.10, 'label': 'Angry'},
            'joy':      {'temperature': 0.80, 'top_p': 0.92, 'cfg_scale': 1.3, 'speed_factor': 1.05, 'label': 'Joyful'},
            'surprise': {'temperature': 0.80, 'top_p': 0.95, 'cfg_scale': 1.3, 'speed_factor': 1.15, 'label': 'Surprised'},
            'fear':     {'temperature': 0.75, 'top_p': 0.85, 'cfg_scale': 1.3, 'speed_factor': 1.15, 'label': 'Frightened'},
            'love':     {'temperature': 0.70, 'top_p': 0.88, 'cfg_scale': 1.2, 'speed_factor': 0.90, 'label': 'Loving'},
            'sadness':  {'temperature': 0.60, 'top_p': 0.80, 'cfg_scale': 1.1, 'speed_factor': 0.85, 'label': 'Sad'},
            'disgust':  {'temperature': 0.75, 'top_p': 0.88, 'cfg_scale': 1.3, 'speed_factor': 0.95, 'label': 'Disgusted'},
            'neutral':  {'temperature': 0.75, 'top_p': 0.90, 'cfg_scale': 1.3, 'speed_factor': 1.00, 'label': 'Neutral'}
        }
        
        # # Get parameters (default to neutral if emotion not found)
        # params = emotion_map.get(emotion, emotion_map['neutral'])
        
        # # Adjust intensity based on confidence
        # if score > 0.8:
        #     # High confidence = more pronounced emotion
        #     if emotion == 'anger':
        #         params = {**params, 'temperature': min(0.95, params['temperature'] + 0.05)}
        #     elif emotion == 'joy':
        #         params = {**params, 'temperature': min(0.95, params['temperature'] + 0.05)}
        #     elif emotion == 'sadness':
        #         params = {**params, 'temperature': max(0.6, params['temperature'] - 0.05)}
        
        # params['score'] = score

        # 1. Get base parameters (fallback to neutral)
        params = emotion_map.get(emotion, emotion_map['neutral']).copy()
        
        if score > 0.5:
            params = emotion_map.get(emotion, emotion_map['neutral']).copy()
            
            # 3. Extra intensity for very high confidence
            if score > 0.8:
                if emotion in ['sadness', 'love']:
                    params['temperature'] = max(0.65, params['temperature'] - 0.05)

        # 3. SAFETY CLAMP: Never let Temperature go into the "Hallucination Zone"
        # Even if other logic tries to push it, we hard-cap it at 0.85
        params['temperature'] = min(0.85, params['temperature'])
        params['score'] = score

        # 4. THE LEGIBILITY GUARD
        # If the temp is too low, the AI "mumbles." Let's keep it readable.
        if params['temperature'] < 0.7:
            params['temperature'] = 0.7 # Hard floor for word clarity
        
        print(f"🎭 Emotion detected: {params['label']} (conf: {score:.2f})")
        print(f"   → Temperature: {params['temperature']}, Top_p: {params['top_p']}, Cfg_scale: {params['cfg_scale']}")
        
        return params
    
    def generate_speech(self, text: str = "", model: str = "VibeVoice-1.5B",
                       attention_type: str = "auto", quantize_llm: str = "full precision",
                       free_memory_after_generate: bool = True,
                       diffusion_steps: int = 20, seed: int = 42, cfg_scale: float = 1.3,
                       use_sampling: bool = False, voice_to_clone=None, voice_to_clone_2=None, lora=None,
                       temperature: float = 0.95, top_p: float = 0.95,
                       max_words_per_chunk: int = 75, voice_speed_factor: float = 1.0,
                       voice_key = None, questgiver_id = None, apply_effects: bool = True, apply_effects_override: str = None, detect_emotions: bool = False):
        """Generate speech from text using VibeVoice"""

        if detect_emotions:
            original_temp = temperature
            original_top_p = top_p
            
            # Detect emotion from text
            emotion_params = self.get_emotion_preset(text)
            
            # Override temperature and top_p with detected values
            temperature = emotion_params['temperature']
            top_p = emotion_params['top_p']
            cfg_scale = emotion_params['cfg_scale']
            # voice_speed_factor = emotion_params['speed_factor']

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
            
        # use stricter cfg_scale when using custom voices
        if voice_key == 'custom_custom':
            print(f"custom voice: cfg_scale changed {cfg_scale} -> 1.5")
            cfg_scale = 1.5

        print("\n" + "=" * 50)
        print("TTS GENERATION DEBUG".center(50))
        print("=" * 50)
        print(f" TEXT LENGTH: {len(text)} chars")
        print(f" SEED       : {seed}")
        print(f" SAMPLING   : {use_sampling}")
        print(f" TEMP       : {temperature}")
        print(f" TOP_P      : {top_p}")
        print(f" CFG_SCALE  : {cfg_scale}")
        print(f" SPEED      : {voice_speed_factor}")
        print(f" STEPS      : {diffusion_steps}")
        print(f" CHUNK SIZE : {max_words_per_chunk}")
        print(f" SPEED      : {voice_speed_factor}")
        print(f" ATTENTION  : {attention_type}")
        print(f" EFFECTS    : {apply_effects}")
        print(f" FX OVERRIDE : {apply_effects_override}")
        print(f" DETECT EMOTIONS: {detect_emotions}")
        print("=" * 50 + "\n")
        
        try:
            # Check text input
            if not text or not text.strip():
                raise Exception(
                    "No text provided. Please enter text with speaker labels (e.g., '[1]: Hello' or '[2]: Hi')")

            # First detect how many speakers are in the text
            bracket_pattern = r'\[(\d+)\]\s*:'
            speakers_numbers = sorted(list(set([int(m) for m in re.findall(bracket_pattern, text)])))

            # Limit to 1-2 speakers
            if not speakers_numbers:
                num_speakers = 1  # Default to 1 if no speaker format found
            else:
                num_speakers = min(max(speakers_numbers), 2)  # Max speaker number, capped at 4
                if max(speakers_numbers) > 2:
                    print(f"[VibeVoice] Warning: Found {max(speakers_numbers)} speakers, limiting to 2")

            if not voice_to_clone_2:
                num_speakers = 1  # Default to 1 if no speaker format found

            # Direct conversion from [N]: to Speaker (N-1): for VibeVoice processor
            # This avoids multiple conversion steps
            converted_text = text

            # Find all [N]: patterns in the text
            speakers_in_text = sorted(list(set([int(m) for m in re.findall(bracket_pattern, text)])))

            if voice_to_clone_2:
                if not speakers_in_text:
                    # No [N]: format found, try Speaker N: format
                    speaker_pattern = r'Speaker\s+(\d+)\s*:'
                    speakers_in_text = sorted(list(set([int(m) for m in re.findall(speaker_pattern, text)])))

                    if speakers_in_text:
                        # Text already in Speaker N format, convert to 0-based
                        for speaker_num in sorted(speakers_in_text, reverse=True):
                            pattern = f'Speaker\\s+{speaker_num}\\s*:'
                            replacement = f'Speaker {speaker_num - 1}:'
                            converted_text = re.sub(pattern, replacement, converted_text)
                    else:
                        # No speaker format found
                        speakers_in_text = [1]

                        # Parse pause keywords even for single speaker
                        pause_segments = self._parse_pause_keywords(text)

                        # Store speaker segments for pause processing
                        speaker_segments_with_pauses = []
                        segments = []

                        for seg_type, seg_content in pause_segments:
                            if seg_type == 'pause':
                                speaker_segments_with_pauses.append(('pause', seg_content, None))
                            else:
                                # Clean up newlines
                                text_clean = seg_content.replace('\n', ' ').replace('\r', ' ')
                                text_clean = ' '.join(text_clean.split())

                                if text_clean:
                                    speaker_segments_with_pauses.append(('text', text_clean, 1))
                                    segments.append(f"Speaker 0: {text_clean}")

                        # Join all segments for fallback
                        converted_text = '\n'.join(segments) if segments else f"Speaker 0: {text}"
                else:
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
                                    segments.append(f'Speaker {speaker_num - 1}: {text_clean}')

                    # Join all segments with newlines (required for multi-speaker format) - for fallback
                    converted_text = '\n'.join(segments) if segments else ""
            else:
                speakers_in_text = [1]


            # Build speaker names list - these are just for logging, not used by processor
            # The processor uses the speaker labels in the text itself
            speakers = [f"Speaker {i}" for i in range(len(speakers_in_text))]

            # Get the actual folder path for the selected model
            available_models = get_available_models()
            model_path = None
            for folder, display_name in available_models:
                if display_name == model:
                    model_path = folder
                    break

            if not model_path:
                raise Exception(f"Model '{model}' not found in models/")

            # Extract LoRA configuration if provided
            lora_path = None
            llm_lora_strength = 1.0
            if lora and isinstance(lora, dict):
                lora_path = lora.get("path", None)
                llm_lora_strength = lora.get("llm_strength", 1.0)

                # Set LoRA component flags based on configuration
                self.use_llm_lora = lora.get("use_llm", True)
                self.use_diffusion_head_lora = lora.get("use_diffusion_head", True)
                self.use_acoustic_connector_lora = lora.get("use_acoustic_connector", True)
                self.use_semantic_connector_lora = lora.get("use_semantic_connector", True)

                if lora_path:
                    logger.info(f"Using LoRA from: {lora_path}")

            # Load model with optional LoRA
            self.load_model(model, model_path, attention_type, quantize_llm=quantize_llm, lora_path=lora_path)
            
            # For single speaker, we just use ["Speaker 1"]
            voice_inputs = [voice_to_clone, voice_to_clone_2]
            # Prepare voice samples in order of appearance
            voice_samples = []
            for i, speaker_num in enumerate(speakers_in_text):
                idx = speaker_num - 1  # Convert to 0-based for voice array

                # Try to use provided voice sample
                if idx < len(voice_inputs) and voice_inputs[idx] is not None:
                    voice_sample = self.prepare_audio_from_path_sf(voice_inputs[idx], 24000, voice_speed_factor)
                    if voice_sample is None:
                        # Use the actual speaker index for consistent synthetic voice
                        voice_sample = self._create_synthetic_voice_sample(idx)
                else:
                    # Use the actual speaker index for consistent synthetic voice
                    voice_sample = self._create_synthetic_voice_sample(idx)

                voice_samples.append(voice_sample)

            # speakers = ["Speaker 1"]

            # Ensure voice_samples count matches detected speakers
            if len(voice_samples) != len(speakers_in_text):
                logger.error(f"Mismatch: {len(speakers_in_text)} speakers but {len(voice_samples)} voice samples!")
                raise Exception(
                    f"Voice sample count mismatch: expected {len(speakers_in_text)}, got {len(voice_samples)}")

            # Parse pause keywords from text
            # segments = self._parse_pause_keywords(final_text)
            
            # # Process segments
            # all_audio_segments = []
            # voice_samples = None  # Will be created on first text segment
            # sample_rate = 24000  # VibeVoice uses 24kHz
            #
            # for seg_idx, (seg_type, seg_content) in enumerate(segments):
            #     if seg_type == 'pause':
            #         # Generate silence for pause
            #         duration_ms = seg_content
            #         logger.info(f"Adding {duration_ms}ms pause")
            #         silence_audio = self._generate_silence(duration_ms, sample_rate)
            #         all_audio_segments.append(silence_audio)
            #
            #     elif seg_type == 'text':
            #         # Process text segment (with chunking if needed)
            #         word_count = len(seg_content.split())
            #
            #         if word_count > max_words_per_chunk:
            #             # Split long text into chunks
            #             logger.info(f"Text segment {seg_idx+1} has {word_count} words, splitting into chunks...")
            #             text_chunks = self._split_text_into_chunks(seg_content, max_words_per_chunk)
            #
            #             for chunk_idx, chunk in enumerate(text_chunks):
            #                 logger.info(f"Processing chunk {chunk_idx+1}/{len(text_chunks)} of segment {seg_idx+1}...")
            #
            #                 # Format chunk for VibeVoice
            #                 formatted_text = self._format_text_for_vibevoice(chunk, speakers)
            #
            #                 # Create voice samples on first text segment
            #                 if voice_samples is None:
            #                     voice_samples = [self.prepare_audio_from_path_sf(voice_to_clone, 24000)]
            #
            #                 # Generate audio for this chunk
            #                 chunk_audio = self._generate_with_vibevoice(
            #                     formatted_text, voice_samples, cfg_scale,
            #                     seed,  # Use same seed for voice consistency
            #                     diffusion_steps, use_sampling, temperature, top_p,
            #                     llm_lora_strength=llm_lora_strength
            #                 )
            #
            #                 all_audio_segments.append(chunk_audio)
            #         else:
            #             # Process as single chunk
            #             logger.info(f"Processing text segment {seg_idx+1} ({word_count} words)")
            #
            #             # Format text for VibeVoice
            #             formatted_text = self._format_text_for_vibevoice(seg_content, speakers)
            #
            #             # Create voice samples on first text segment
            #             if voice_samples is None:
            #                 voice_samples = [self.prepare_audio_from_path_sf(voice_to_clone, 24000)]
            #
            #             # Generate audio
            #             segment_audio = self._generate_with_vibevoice(
            #                 formatted_text, voice_samples, cfg_scale, seed, diffusion_steps,
            #                 use_sampling, temperature, top_p, llm_lora_strength=llm_lora_strength
            #             )
            #
            #             all_audio_segments.append(segment_audio)
            #
            # # Concatenate all audio segments (including pauses)
            # if all_audio_segments:
            #     logger.info(f"Concatenating {len(all_audio_segments)} audio segments (including pauses)...")
            #
            #     # Extract waveforms from all segments
            #     waveforms = []
            #     for audio_segment in all_audio_segments:
            #         if isinstance(audio_segment, dict) and "waveform" in audio_segment:
            #             waveforms.append(audio_segment["waveform"])
            #
            #     if waveforms:
            #         # Filter out None values if any
            #         valid_waveforms = [w for w in waveforms if w is not None]
            #
            #         if valid_waveforms:
            #             # Concatenate along the time dimension (last dimension)
            #             combined_waveform = torch.cat(valid_waveforms, dim=-1)
            #
            #             # Create final audio dict
            #             audio_dict = {
            #                 "waveform": combined_waveform,
            #                 "sample_rate": sample_rate
            #             }
            #             logger.info(f"Successfully generated audio with {len(segments)} segments")
            #         else:
            #             raise Exception("No valid audio waveforms generated")
            #     else:
            #         raise Exception("Failed to extract waveforms from audio segments")
            # else:
            #     raise Exception("No audio segments generated")

            # Check if we have pause segments to process
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
                            formatted_text = f"Speaker {speaker_num - 1}: {combined_text}"

                            # Get voice sample for this speaker
                            speaker_idx = speakers_in_text.index(speaker_num)
                            speaker_voice_samples = [voice_samples[speaker_idx]]

                            logger.info(
                                f"Generating audio for Speaker {speaker_num}: {len(combined_text.split())} words")

                            # fix cfg scale for short texts
                            cfg_scale_temp = fix_cfg_scale_by_word_count(combined_text, cfg_scale)
                            if speaker_idx == 1 and cfg_scale_temp > 1.2:
                                cfg_scale_temp = 1.2

                            # Generate audio for this speaker's text
                            segment_audio = self._generate_with_vibevoice(
                                formatted_text, speaker_voice_samples, cfg_scale_temp, seed,
                                diffusion_steps, use_sampling, temperature, top_p,
                                llm_lora_strength=llm_lora_strength
                            )

                            # print(f"Before effects: {segment_audio['waveform'].shape}")
                            # apply effects if not narrator
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
                                    print(f"Wretched creature post-processing")
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
                                        print(f"Doing small effects", flush=True)
                                        segment_audio = ae.small_effects_on_waveform(segment_audio)
                                    elif effect_type == "comms":
                                        print(f"Doing telephone effects", flush=True)
                                        segment_audio = ae.comms_effects_on_waveform(segment_audio)
                                    elif effect_type == "wretched":
                                        print(f"Doing wretched effects", flush=True)
                                        segment_audio = ae.wretched_effect_on_waveform(segment_audio)

                            if speaker_idx != 1 and apply_effects and apply_effects_override:
                                effect_type = apply_effects_override
                                if ae.is_sox_command(effect_type):
                                    print(f"Doing custom effects", flush=True)
                                    effect_type_clean = effect_type.strip('"')
                                    args = re.findall(r"'([^']*)'", effect_type_clean)
                                    segment_audio = ae._apply_sox_effects_to_waveform(segment_audio, args)
                                else:
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
                                        print(f"Doing small effects", flush=True)
                                        segment_audio = ae.small_effects_on_waveform(segment_audio)
                                    elif effect_type == "comms":
                                        print(f"Doing telephone effects", flush=True)
                                        segment_audio = ae.comms_effects_on_waveform(segment_audio)
                                    elif effect_type == "letter":
                                        print(f"Doing letter effects", flush=True)
                                        segment_audio = ae.letter_effect_on_waveform(segment_audio)
                                    elif effect_type == "wretched":
                                        print(f"Doing wretched effects", flush=True)
                                        segment_audio = ae.wretched_effect_on_waveform(segment_audio)

                            # print(f"After effects: {segment_audio['waveform'].shape}")

                            all_audio_segments.append(segment_audio)

                    # Concatenate all audio segments
                    if all_audio_segments:
                        logger.info(f"Concatenating {len(all_audio_segments)} audio segments (including pauses)...")

                        # Extract waveforms
                        waveforms = []
                        for audio_segment in all_audio_segments:
                            if isinstance(audio_segment, dict) and "waveform" in audio_segment:
                                waveforms.append(audio_segment["waveform"])

                        if waveforms:
                            # Filter out None values if any
                            valid_waveforms = [w for w in waveforms if w is not None]

                            if valid_waveforms:
                                # Concatenate along time dimension
                                combined_waveform = torch.cat(valid_waveforms, dim=-1)

                                audio_dict = {
                                    "waveform": combined_waveform,
                                    "sample_rate": sample_rate
                                }
                                logger.info(f"Successfully generated multi-speaker audio with pauses")
                            else:
                                raise Exception("No valid audio waveforms generated")
                        else:
                            raise Exception("Failed to extract waveforms from audio segments")
                    else:
                        raise Exception("No audio segments generated")
                else:
                    # Fallback to original method without pause support
                    logger.info("Processing without pause support (no pause keywords found)")
                    # fix cfg scale for short texts
                    cfg_scale_temp = fix_cfg_scale_by_word_count(text, cfg_scale)
                    audio_dict = self._generate_with_vibevoice(
                        converted_text, voice_samples, cfg_scale_temp, seed, diffusion_steps,
                        use_sampling, temperature, top_p, llm_lora_strength=llm_lora_strength
                    )
            else:
                # Parse pause keywords from text
                segments = self._parse_pause_keywords(converted_text)

                # Process segments
                all_audio_segments = []
                voice_samples = None  # Will be created on first text segment
                sample_rate = 24000  # VibeVoice uses 24kHz

                for seg_idx, (seg_type, seg_content) in enumerate(segments):
                    if seg_type == 'pause':
                        # Generate silence for pause
                        duration_ms = seg_content
                        logger.info(f"Adding {duration_ms}ms pause")
                        silence_audio = self._generate_silence(duration_ms, sample_rate)
                        all_audio_segments.append(silence_audio)

                    elif seg_type == 'text':
                        # Process text segment (with chunking if needed)
                        word_count = len(seg_content.split())

                        if word_count > max_words_per_chunk:
                            # Split long text into chunks
                            logger.info(f"Text segment {seg_idx+1} has {word_count} words, splitting into chunks...")
                            text_chunks = self._split_text_into_chunks(seg_content, max_words_per_chunk)
                                
                            for chunk_idx, chunk in enumerate(text_chunks):
                                logger.info(f"Processing chunk {chunk_idx+1}/{len(text_chunks)} of segment {seg_idx+1}...")

                                # Format chunk for VibeVoice
                                formatted_text = self._format_text_for_vibevoice(chunk, speakers)

                                # Create voice samples on first text segment
                                if voice_samples is None:
                                    voice_samples = [self.prepare_audio_from_path_sf(voice_to_clone, 24000, voice_speed_factor)]

                                # Generate audio for this chunk
                                # fix cfg scale for short texts
                                cfg_scale_temp = fix_cfg_scale_by_word_count(chunk, cfg_scale)
                                if voice_key == 'narrator' and cfg_scale_temp > 1.2:
                                    cfg_scale_temp = 1.2

                                chunk_audio = self._generate_with_vibevoice(
                                    formatted_text, voice_samples, cfg_scale_temp,
                                    seed,  # Use same seed for voice consistency
                                    diffusion_steps, use_sampling, temperature, top_p,
                                    llm_lora_strength=llm_lora_strength
                                )

                                # apply effects if not narrator
                                if apply_effects and not apply_effects_override:
                                    # after voice generation, apply sfx based on certain race-gender combos
                                    if voice_key.endswith("_dk"):
                                        print(f"DK post-processing")
                                        chunk_audio = ae.dk_effects_on_waveform(chunk_audio)
                                    elif voice_key in ("sylvanas", "forsaken_male", "forsaken_female"):
                                        print(f"Undead post-processing")
                                        chunk_audio = ae.undead_effects_on_waveform(chunk_audio)
                                    elif voice_key in ("mechanical", "titan_male"):
                                        print(f"Robot post-processing")
                                        chunk_audio = ae.robot_effects_on_waveform(chunk_audio)
                                    elif voice_key in ("demon_male", "keeper", "dragon_male", "dragon_female"):
                                        print(f"Demon post-processing")
                                        chunk_audio = ae.demon_effects_on_waveform(chunk_audio)
                                    elif voice_key in ("giant_male", "ogre_male", "ogrila_ogre", "ancient"):
                                        print(f"Giant post-processing")
                                        chunk_audio = ae.giant_effects_on_waveform(chunk_audio, voice_key)
                                    elif voice_key in ("wolvar_male", "gorloc_male"):
                                        print(f"Small creature post-processing")
                                        chunk_audio = ae.small_effects_on_waveform(chunk_audio)
                                    elif voice_key in ("earthen"):
                                        print(f"Earthen creature post-processing")
                                        chunk_audio = ae.earthen_effects_on_waveform(chunk_audio)
                                    elif voice_key in ("naaru"):
                                        print(f"Naaru creature post-processing")
                                        chunk_audio = ae.naaru_effects_on_waveform(chunk_audio)
                                    elif voice_key in ("ethereal_male"):
                                        print(f"Ethereal creature post-processing")
                                        chunk_audio = ae.ethereal_effects_on_waveform(chunk_audio)
                                    elif voice_key in ("wretched"):
                                        print(f"Wretched creature post-processing")
                                        chunk_audio = ae.wretched_effect_on_waveform(chunk_audio)
                                    # Apply questgiver-specific effects that don't conform to race_gender categories
                                    if questgiver_id and questgiver_id in NPC_EFFECTS:
                                        effect_type = NPC_EFFECTS[questgiver_id]
                                        if effect_type == "ghost":
                                            print("Doing ghost effects", flush=True)
                                            chunk_audio = ae.ghost_effects_on_waveform(chunk_audio)
                                        elif effect_type == "demon":
                                            print("Doing demon effects", flush=True)
                                            chunk_audio = ae.demon_effects_on_waveform(chunk_audio)
                                        elif effect_type == "giant":
                                            print("Doing giant effects", flush=True)
                                            chunk_audio = ae.giant_effects_on_waveform(chunk_audio, voice_key)
                                        elif effect_type == "ancient":
                                            print("Doing ancient effects", flush=True)
                                            chunk_audio = ae.giant_effects_on_waveform(chunk_audio, effect_type)
                                        elif effect_type == "undead":
                                            print("Doing undead effects", flush=True)
                                            chunk_audio = ae.undead_effects_on_waveform(chunk_audio)
                                        elif effect_type == "small":
                                            print(f"Doing small effects", flush=True)
                                            chunk_audio = ae.small_effects_on_waveform(chunk_audio)
                                        elif effect_type == "comms":
                                            print(f"Doing telephone effects", flush=True)
                                            chunk_audio = ae.comms_effects_on_waveform(chunk_audio)
                                        elif effect_type == "wretcged":
                                            print(f"Doing wretched effects", flush=True)
                                            chunk_audio = ae.wretched_effect_on_waveform(chunk_audio)

                                if apply_effects and apply_effects_override:
                                    effect_type = apply_effects_override
                                    if ae.is_sox_command(effect_type):
                                        print(f"Doing custom effects", flush=True)
                                        effect_type_clean = effect_type.strip('"')
                                        args = re.findall(r"'([^']*)'", effect_type_clean)
                                        chunk_audio = ae._apply_sox_effects_to_waveform(chunk_audio, args)
                                    else:
                                        if effect_type == "ghost":
                                            print("Doing ghost effects", flush=True)
                                            chunk_audio = ae.ghost_effects_on_waveform(chunk_audio)
                                        elif effect_type == "demon":
                                            print("Doing demon effects", flush=True)
                                            chunk_audio = ae.demon_effects_on_waveform(chunk_audio)
                                        elif effect_type == "giant":
                                            print("Doing giant effects", flush=True)
                                            chunk_audio = ae.giant_effects_on_waveform(chunk_audio, voice_key)
                                        elif effect_type == "ancient":
                                            print("Doing ancient effects", flush=True)
                                            chunk_audio = ae.giant_effects_on_waveform(chunk_audio, effect_type)
                                        elif effect_type == "undead":
                                            print("Doing undead effects", flush=True)
                                            chunk_audio = ae.undead_effects_on_waveform(chunk_audio)
                                        elif effect_type == "small":
                                            print(f"Doing small effects", flush=True)
                                            chunk_audio = ae.small_effects_on_waveform(chunk_audio)
                                        elif effect_type == "comms":
                                            print(f"Doing telephone effects", flush=True)
                                            chunk_audio = ae.comms_effects_on_waveform(chunk_audio)
                                        elif effect_type == "dk":
                                            print(f"Doing dk effects", flush=True)
                                            chunk_audio = ae.dk_effects_on_waveform(chunk_audio)
                                        elif effect_type == "robot":
                                            print(f"Doing robot effects", flush=True)
                                            chunk_audio = ae.robot_effects_on_waveform(chunk_audio)
                                        elif effect_type == "ethereal":
                                            print(f"Doing ethereal effects", flush=True)
                                            chunk_audio = ae.ethereal_effects_on_waveform(chunk_audio)
                                        elif effect_type == "naaru":
                                            print(f"Doing naaru effects", flush=True)
                                            chunk_audio = ae.naaru_effects_on_waveform(chunk_audio)
                                        elif effect_type == "earthen":
                                            print(f"Doing earthen effects", flush=True)
                                            chunk_audio = ae.earthen_effects_on_waveform(chunk_audio)
                                        elif effect_type == "letter":
                                            print(f"Doing letter effects", flush=True)
                                            chunk_audio = ae.letter_effect_on_waveform(chunk_audio)
                                        elif effect_type == "wretched":
                                            print(f"Doing wretched effects", flush=True)
                                            chunk_audio = ae.wretched_effect_on_waveform(chunk_audio)

                                all_audio_segments.append(chunk_audio)
                        else:
                            # Process as single chunk
                            logger.info(f"Processing text segment {seg_idx+1} ({word_count} words)")

                            # Format text for VibeVoice
                            formatted_text = self._format_text_for_vibevoice(seg_content, speakers)

                            # Create voice samples on first text segment
                            if voice_samples is None:
                                voice_samples = [self.prepare_audio_from_path_sf(voice_to_clone, 24000, voice_speed_factor)]

                            # Generate audio
                            # fix cfg scale for short texts
                            cfg_scale_temp = fix_cfg_scale_by_word_count(seg_content, cfg_scale)
                            if voice_key == 'narrator' and cfg_scale_temp > 1.2:
                                cfg_scale_temp = 1.2
                                    
                            segment_audio = self._generate_with_vibevoice(
                                formatted_text, voice_samples, cfg_scale_temp, seed, diffusion_steps,
                                use_sampling, temperature, top_p, llm_lora_strength=llm_lora_strength
                            )

                            # apply effects if not narrator
                            if apply_effects and not apply_effects_override:
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

                            if apply_effects and apply_effects_override:
                                effect_type = apply_effects_override
                                if ae.is_sox_command(effect_type):
                                    print(f"Doing custom effects", flush=True)
                                    effect_type_clean = effect_type.strip('"')
                                    args = re.findall(r"'([^']*)'", effect_type_clean)
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

                            all_audio_segments.append(segment_audio)

                # Concatenate all audio segments (including pauses)
                if all_audio_segments:
                    logger.info(f"Concatenating {len(all_audio_segments)} audio segments (including pauses)...")

                    # Extract waveforms from all segments
                    waveforms = []
                    for audio_segment in all_audio_segments:
                        if isinstance(audio_segment, dict) and "waveform" in audio_segment:
                            waveforms.append(audio_segment["waveform"])

                    if waveforms:
                        # Filter out None values if any
                        valid_waveforms = [w for w in waveforms if w is not None]

                        if valid_waveforms:
                            # Concatenate along the time dimension (last dimension)
                            combined_waveform = torch.cat(valid_waveforms, dim=-1)

                            # Create final audio dict
                            audio_dict = {
                                "waveform": combined_waveform,
                                "sample_rate": sample_rate
                            }
                            logger.info(f"Successfully generated audio with {len(segments)} segments")
                        else:
                            raise Exception("No valid audio waveforms generated")
                    else:
                        raise Exception("Failed to extract waveforms from audio segments")
                else:
                    raise Exception("No audio segments generated")


            # Free memory if requested
            if free_memory_after_generate:
                self.free_memory()

            waveform = audio_dict["waveform"]
            sr = audio_dict["sample_rate"]
            if isinstance(waveform, torch.Tensor):
                audio_np = waveform.cpu().float().numpy()
            else:
                audio_np = np.array(waveform)

            output_path = self.make_output_path(f"{uuid.uuid4()}.wav")

            audio_np = np.squeeze(audio_np)
            audio_int16 = np.clip(audio_np, -1.0, 1.0)
            audio_int16 = (audio_int16 * 32767).astype(np.int16)

            full_abs_path = os.path.abspath(output_path)
            print(f"--- DEBUG INFO ---")
            print(f"Current Working Directory: {os.getcwd()}")
            print(f"Output Path (raw): {output_path}")
            print(f"Full Absolute Path: {full_abs_path}")
            print(f"Directory exists: {os.path.exists(os.path.dirname(full_abs_path))}")
            print(f"------------------")
            os.makedirs(os.path.dirname(full_abs_path), exist_ok=True)

            wavfile.write(output_path, sr, audio_int16)
            return output_path
            
            return (audio_dict,)
                    
        except Exception as e:
            raise Exception(f"Error generating speech: {str(e)}")

    def _apply_lora(self, lora_path: str):
        """Apply LoRA adapters to the model"""
        try:
            logger.info(f"Starting LoRA application from path: {lora_path}")

            # Check component flags
            use_llm = getattr(self, 'use_llm_lora', True)
            use_diffusion = getattr(self, 'use_diffusion_head_lora', True)
            use_acoustic = getattr(self, 'use_acoustic_connector_lora', True)
            use_semantic = getattr(self, 'use_semantic_connector_lora', True)

            logger.info(f"LoRA component flags - LLM: {use_llm}, Diffusion: {use_diffusion}, Acoustic: {use_acoustic}, Semantic: {use_semantic}")

            if not any([use_llm, use_diffusion, use_acoustic, use_semantic]):
                logger.info("All LoRA components disabled, skipping LoRA application")
                return

            # Apply LLM LoRA adapter if requested
            if use_llm:
                # Check if adapter files exist
                adapter_model_path = os.path.join(lora_path, "adapter_model.safetensors")
                adapter_bin_path = os.path.join(lora_path, "adapter_model.bin")
                adapter_config = os.path.join(lora_path, "adapter_config.json")

                has_adapter = os.path.exists(adapter_model_path) or os.path.exists(adapter_bin_path)

                if has_adapter and os.path.exists(adapter_config):
                    try:
                        from peft import PeftModel
                        base_lm = getattr(self.model.model, 'language_model', None)
                        if base_lm is not None:
                            logger.info(f"Applying LLM LoRA adapter from: {lora_path}")
                            lora_wrapped = PeftModel.from_pretrained(base_lm, lora_path, is_trainable=False)
                            device = next(self.model.parameters()).device
                            dtype = next(self.model.parameters()).dtype
                            lora_wrapped = lora_wrapped.to(device=device, dtype=dtype)
                            self.model.model.language_model = lora_wrapped
                            logger.info("LLM LoRA adapter successfully applied")
                    except ImportError:
                        logger.warning("PEFT library not available for LLM LoRA")
                    except Exception as e:
                        logger.warning(f"Failed to apply LLM LoRA: {e}")
                else:
                    logger.info(f"No LLM LoRA adapter files found in {lora_path}, skipping LLM LoRA")

            # Helper function to load state dict into module
            def _load_state_dict_into(module, folder):
                if module is None:
                    logger.warning(f"Module is None, cannot load state dict from {folder}")
                    return False
                if not os.path.isdir(folder):
                    logger.warning(f"Folder does not exist: {folder}")
                    return False

                try:
                    # Try safetensors first
                    safetensor_path = os.path.join(folder, "model.safetensors")
                    if os.path.exists(safetensor_path):
                        try:
                            import safetensors.torch as st
                            logger.info(f"Loading safetensor from: {safetensor_path}")
                            state_dict = st.load_file(safetensor_path)
                            logger.info(f"Loaded state dict with {len(state_dict)} keys")

                            # Get device and dtype from module
                            device = next(module.parameters()).device
                            dtype = next(module.parameters()).dtype
                            logger.info(f"Module device: {device}, dtype: {dtype}")

                            # Convert state dict to correct device and dtype
                            for key in state_dict:
                                state_dict[key] = state_dict[key].to(device=device, dtype=dtype)

                            # Load with strict=False to allow partial loading
                            missing_keys, unexpected_keys = module.load_state_dict(state_dict, strict=False)

                            if missing_keys:
                                logger.warning(f"Missing keys when loading state dict ({len(missing_keys)} total): {missing_keys[:5]}..." if len(missing_keys) > 5 else f"Missing keys: {missing_keys}")
                            if unexpected_keys:
                                logger.warning(f"Unexpected keys when loading state dict ({len(unexpected_keys)} total): {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"Unexpected keys: {unexpected_keys}")

                            # Log success even with missing keys if most were loaded
                            total_keys = len(state_dict)
                            if missing_keys:
                                logger.info(f"Loaded {total_keys} keys from LoRA, {len(missing_keys)} keys not found in model")

                            logger.info("Successfully loaded state dict into module")
                            return True
                        except Exception as e:
                            logger.warning(f"Failed to load safetensors: {e}")
                            import traceback
                            logger.debug(f"Traceback: {traceback.format_exc()}")

                    # Fallback to PyTorch format
                    pytorch_path = os.path.join(folder, "pytorch_model.bin")
                    if os.path.exists(pytorch_path):
                        logger.info(f"Loading pytorch model from: {pytorch_path}")
                        state_dict = torch.load(pytorch_path, map_location="cpu")
                        logger.info(f"Loaded state dict with {len(state_dict)} keys")

                        # Get device and dtype from module
                        device = next(module.parameters()).device
                        dtype = next(module.parameters()).dtype

                        # Convert state dict to correct device and dtype
                        for key in state_dict:
                            state_dict[key] = state_dict[key].to(device=device, dtype=dtype)

                        missing_keys, unexpected_keys = module.load_state_dict(state_dict, strict=False)

                        if missing_keys:
                            logger.warning(f"Missing keys: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"Missing keys: {missing_keys}")
                        if unexpected_keys:
                            logger.warning(f"Unexpected keys: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"Unexpected keys: {unexpected_keys}")

                        logger.info("Successfully loaded pytorch model into module")
                        return True
                    else:
                        logger.warning(f"No model file found in {folder}")
                        logger.warning(f"Looked for: {safetensor_path} and {pytorch_path}")

                except Exception as e:
                    logger.error(f"Failed to load state dict from {folder}: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                return False

            # Load diffusion head if requested
            if use_diffusion:
                diffusion_path = os.path.join(lora_path, "diffusion_head")
                if os.path.exists(diffusion_path):
                    logger.info(f"Found diffusion_head directory at: {diffusion_path}")

                    # The diffusion head is called 'prediction_head' in VibeVoice
                    module = getattr(self.model.model, 'prediction_head', None)
                    if module:
                        logger.info("Found prediction_head module in model")

                        # Check model compatibility by looking at dimensions
                        skip_loading = False
                        try:
                            # Get hidden size from the module
                            if hasattr(module, 'cond_proj') and hasattr(module.cond_proj, 'weight'):
                                model_hidden_size = module.cond_proj.weight.shape[0]
                                logger.info(f"Current model prediction_head hidden size: {model_hidden_size}")

                                # Check LoRA dimensions
                                safetensor_path = os.path.join(diffusion_path, "model.safetensors")
                                if os.path.exists(safetensor_path):
                                    import safetensors.torch as st
                                    lora_state = st.load_file(safetensor_path)
                                    if 'cond_proj.weight' in lora_state:
                                        lora_hidden_size = lora_state['cond_proj.weight'].shape[0]
                                        logger.info(f"LoRA diffusion head hidden size: {lora_hidden_size}")

                                        if model_hidden_size != lora_hidden_size:
                                            skip_loading = True
                                            if lora_hidden_size == 3584:
                                                logger.error("="*60)
                                                logger.error("LoRA MODEL MISMATCH!")
                                                logger.error(f"This LoRA was trained on VibeVoice-Large (hidden_size=3584)")
                                                if model_hidden_size == 1536:
                                                    logger.error(f"You are using VibeVoice-1.5B (hidden_size=1536)")
                                                    logger.error("Please switch to 'VibeVoice-Large' model to use this LoRA")
                                                else:
                                                    logger.error(f"Your model has hidden_size={model_hidden_size}")
                                                    logger.error("Please use VibeVoice-Large (non-quantized) model")
                                                logger.error("="*60)
                                                logger.error("Skipping LoRA loading due to incompatible model")
                                            elif lora_hidden_size == 1536:
                                                logger.error("="*60)
                                                logger.error("LoRA MODEL MISMATCH!")
                                                logger.error(f"This LoRA was trained on VibeVoice-1.5B (hidden_size=1536)")
                                                logger.error(f"You are using a model with hidden_size={model_hidden_size}")
                                                logger.error("Please switch to 'VibeVoice-1.5B' model to use this LoRA")
                                                logger.error("="*60)
                                                logger.error("Skipping LoRA loading due to incompatible model")
                        except Exception as e:
                            logger.debug(f"Could not check model compatibility: {e}")

                        # Only attempt to load if compatible
                        if not skip_loading:
                            if _load_state_dict_into(module, diffusion_path):
                                logger.info("Diffusion head LoRA loaded successfully into prediction_head")
                            else:
                                logger.warning("Failed to load diffusion head LoRA")
                        else:
                            logger.info("Diffusion head LoRA loading skipped due to model mismatch")
                    else:
                        logger.warning("Model does not have prediction_head attribute")
                        # Debug: list available attributes
                        attrs = [a for a in dir(self.model.model) if not a.startswith('_')]
                        logger.debug(f"Available model.model attributes: {attrs[:15]}...")
                else:
                    logger.info(f"No diffusion_head directory found at: {diffusion_path}")

            # Load acoustic connector if requested
            if use_acoustic:
                acoustic_path = os.path.join(lora_path, "acoustic_connector")
                if os.path.exists(acoustic_path):
                    module = getattr(self.model.model, 'acoustic_connector', None)
                    if module and _load_state_dict_into(module, acoustic_path):
                        logger.info("Acoustic connector LoRA loaded")

            # Load semantic connector if requested
            if use_semantic:
                semantic_path = os.path.join(lora_path, "semantic_connector")
                if os.path.exists(semantic_path):
                    module = getattr(self.model.model, 'semantic_connector', None)
                    if module and _load_state_dict_into(module, semantic_path):
                        logger.info("Semantic connector LoRA loaded")


            # Log summary of what was loaded
            logger.info("LoRA application completed")

        except Exception as e:
            logger.error(f"Error applying LoRA: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Don't fail the entire load, just log the error

    def _verify_quantization(self, expected_mode: str):
        """Verify that quantization was actually applied correctly"""
        try:
            quantized_layers = []
            fp_layers = []

            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    module_type = type(module).__name__

                    if 'Linear8bitLt' in module_type or '8bit' in module_type.lower():
                        quantized_layers.append((name, '8bit'))
                    elif 'Linear4bit' in module_type or '4bit' in module_type.lower():
                        quantized_layers.append((name, '4bit'))
                    else:
                        fp_layers.append(name)

            # Concise summary
            total_linear = len(quantized_layers) + len(fp_layers)

            if len(quantized_layers) > 0:
                pct = 100 * len(quantized_layers) / total_linear
                logger.info(f"✅ {expected_mode} quantization: {len(quantized_layers)}/{total_linear} layers ({pct:.1f}%)")
            else:
                logger.warning(f"⚠️ No {expected_mode} quantization detected")

        except Exception as e:
            logger.debug(f"Could not verify quantization: {e}")

    def _apply_sage_attention(self):
        """Apply SageAttention to the loaded model by monkey-patching attention layers"""
        try:
            from sageattention import sageattn
            import torch.nn.functional as F
            
            # Counter for patched layers
            patched_count = 0
            
            def patch_attention_forward(module):
                """Recursively patch attention layers to use SageAttention"""
                nonlocal patched_count
                
                # Check if this module has scaled_dot_product_attention
                if hasattr(module, 'forward'):
                    original_forward = module.forward
                    
                    # Create wrapper that replaces F.scaled_dot_product_attention with sageattn
                    def sage_forward(*args, **kwargs):
                        # Temporarily replace F.scaled_dot_product_attention
                        original_sdpa = F.scaled_dot_product_attention
                        
                        def sage_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
                            """Wrapper that converts sdpa calls to sageattn"""
                            # Log any unexpected parameters for debugging
                            if kwargs:
                                unexpected_params = list(kwargs.keys())
                                logger.debug(f"SageAttention: Ignoring unsupported parameters: {unexpected_params}")
                            
                            try:
                                # SageAttention expects tensors in specific format
                                # Transformers typically use (batch, heads, seq_len, head_dim)
                                
                                # Check tensor dimensions to determine layout
                                if query.dim() == 4:
                                    # 4D tensor: (batch, heads, seq, dim)
                                    batch_size = query.shape[0]
                                    num_heads = query.shape[1]
                                    seq_len_q = query.shape[2]
                                    seq_len_k = key.shape[2]
                                    head_dim = query.shape[3]
                                    
                                    # Reshape to (batch*heads, seq, dim) for HND layout
                                    query_reshaped = query.reshape(batch_size * num_heads, seq_len_q, head_dim)
                                    key_reshaped = key.reshape(batch_size * num_heads, seq_len_k, head_dim)
                                    value_reshaped = value.reshape(batch_size * num_heads, seq_len_k, head_dim)
                                    
                                    # Call sageattn with HND layout
                                    output = sageattn(
                                        query_reshaped, key_reshaped, value_reshaped,
                                        is_causal=is_causal,
                                        tensor_layout="HND"  # Heads*batch, seqN, Dim
                                    )
                                    
                                    # Output should be (batch*heads, seq_len_q, head_dim)
                                    # Reshape back to (batch, heads, seq, dim)
                                    if output.dim() == 3:
                                        output = output.reshape(batch_size, num_heads, seq_len_q, head_dim)
                                    
                                    return output
                                else:
                                    # For 3D tensors, assume they're already in HND format
                                    output = sageattn(
                                        query, key, value,
                                        is_causal=is_causal,
                                        tensor_layout="HND"
                                    )
                                    return output
                                    
                            except Exception as e:
                                # If SageAttention fails, fall back to original implementation
                                logger.debug(f"SageAttention failed, using original: {e}")
                                # Call with proper arguments - scale is a keyword argument in PyTorch 2.0+
                                # Pass through any additional kwargs that the original sdpa might support
                                if scale is not None:
                                    return original_sdpa(query, key, value, attn_mask=attn_mask, 
                                                       dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)
                                else:
                                    return original_sdpa(query, key, value, attn_mask=attn_mask, 
                                                       dropout_p=dropout_p, is_causal=is_causal, **kwargs)
                        
                        # Replace the function
                        F.scaled_dot_product_attention = sage_sdpa
                        
                        try:
                            # Call original forward with patched attention
                            result = original_forward(*args, **kwargs)
                        finally:
                            # Restore original function
                            F.scaled_dot_product_attention = original_sdpa
                        
                        return result
                    
                    # Check if this module likely uses attention
                    # Look for common attention module names
                    module_name = module.__class__.__name__.lower()
                    if any(name in module_name for name in ['attention', 'attn', 'multihead']):
                        module.forward = sage_forward
                        patched_count += 1
                
                # Recursively patch child modules
                for child in module.children():
                    patch_attention_forward(child)
            
            # Apply patching to the entire model
            patch_attention_forward(self.model)
            
            logger.info(f"Patched {patched_count} attention layers with SageAttention")
            
            if patched_count == 0:
                logger.warning("No attention layers found to patch - SageAttention may not be applied")
                
        except Exception as e:
            logger.error(f"Failed to apply SageAttention: {e}")
            logger.warning("Continuing with standard attention implementation")
    
    def load_model(self, model_name: str, model_folder: str, attention_type: str = "auto", quantize_llm: str = "full precision", lora_path: str = None):
        """Load VibeVoice model with specified attention implementation and optional LoRA

        Args:
            model_name: The display name of the model (e.g., "VibeVoice-Large")
            model_folder: The folder name in models/vibevoice/ containing the model
            attention_type: The attention implementation to use
            quantize_llm: LLM quantization mode ("full precision", "8bit", or "4bit")
            lora_path: Optional path to LoRA adapter directory
        """
        # Check if we need to reload model due to attention type, quantization, or LoRA change
        current_attention = getattr(self, 'current_attention_type', None)
        current_quantize_llm = getattr(self, 'current_quantize_llm', 'full precision')
        current_lora = getattr(self, 'current_lora_path', None)
        lora_changed = (current_lora or "") != (lora_path or "")
        quantize_changed = current_quantize_llm != quantize_llm

        if (self.model is None or
            getattr(self, 'current_model_folder', None) != model_folder or
            current_attention != attention_type or
            quantize_changed or
            lora_changed):
            
            # Free existing model before loading new one (important for attention type, quantization, or LoRA changes)
            if self.model is not None and (current_attention != attention_type or quantize_changed or getattr(self, 'current_model_folder', None) != model_folder or lora_changed):
                logger.info(f"Freeing existing model before loading with new settings (attention: {current_attention} -> {attention_type}, quantize: {current_quantize_llm} -> {quantize_llm}, LoRA: {current_lora} -> {lora_path})")
                self.free_memory()
            
            try:
                # from tts.vvembed.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
                # vibevoice, VibeVoiceInferenceModel = self._check_dependencies()

                # from tts.vvembed.modular.configuration_vibevoice import VibeVoiceConfig
                # from tts.vvembed.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
                # from tts.vvembed.processor.vibevoice_processor import VibeVoiceProcessor
                # from tts.vvembed.modular.streamer import AudioStreamer
                
                # Set ComfyUI models directory
                # import folder_paths
                # models_dir = folder_paths.get_folder_paths("checkpoints")[0]
                # comfyui_models_dir = os.path.join(os.path.dirname(models_dir), "vibevoice")
                # os.makedirs(comfyui_models_dir, exist_ok=True)
                
                # Import time for timing
                import time
                start_time = time.time()
                
                # Suppress verbose logs
                # import transformers
                # import warnings
                # transformers.logging.set_verbosity_error()
                # warnings.filterwarnings("ignore", category=UserWarning)
                
                # Get the actual model path using our discovery function
                model_full_path = os.path.join(MODELS_DIR, model_folder)

                # Find where the actual model files are
                model_files_path = find_model_files_path(model_full_path)

                if not model_files_path:
                    raise Exception(f"No valid model files found in {model_full_path}. Please ensure the model is properly downloaded.")

                logger.info(f"Found model files at: {model_files_path}")

                # Check if model files are in a 4bit subfolder
                use_4bit_subfolder = False
                actual_model_path = model_files_path
                if model_files_path.endswith(os.sep + "4bit"):
                    # If files are in 4bit subfolder, use parent path and set subfolder
                    actual_model_path = os.path.dirname(model_files_path)
                    use_4bit_subfolder = True
                    logger.info(f"Model uses 4bit subfolder structure")

                # Detect if model is quantized
                quantization = detect_model_quantization(model_files_path)
                if quantization:
                    logger.info(f"Detected {quantization} quantization")
                
                # Check if this is a quantized model
                is_quantized_4bit = quantization == "4bit"
                is_quantized_8bit = quantization == "8bit"
                is_quantized = is_quantized_4bit or is_quantized_8bit
                
                # Prepare attention implementation kwargs
                model_kwargs = {
                    "cache_dir": MODELS_DIR,
                    "trust_remote_code": True,
                    "dtype": torch.bfloat16,
                    "device_map": get_device_map(),
                }
                
                # Handle quantized model loading
                if is_quantized_4bit or is_quantized_8bit:
                    # Check if CUDA is available (required for quantization)
                    if not torch.cuda.is_available():
                        raise Exception("Quantized models require a CUDA GPU. Please use standard models on CPU/MPS.")

                    # Try to import bitsandbytes
                    try:
                        from transformers import BitsAndBytesConfig

                        if is_quantized_4bit:
                            logger.info("Loading 4-bit quantized model with bitsandbytes...")
                            # Configure 4-bit quantization
                            bnb_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type='nf4'
                            )
                            if use_4bit_subfolder:
                                model_kwargs["subfolder"] = "4bit"
                                logger.info("Using subfolder='4bit' for loading")
                        else:  # 8-bit
                            logger.info("Loading 8-bit quantized model with bitsandbytes...")
                            # Configure 8-bit quantization
                            bnb_config = BitsAndBytesConfig(
                                load_in_8bit=True,
                                bnb_8bit_compute_dtype=torch.bfloat16
                            )

                        model_kwargs["quantization_config"] = bnb_config
                        model_kwargs["device_map"] = "cuda"  # Force CUDA for quantized models

                    except ImportError:
                        raise Exception(
                            "Quantized models require 'bitsandbytes' library.\n"
                            "Please install it with: pip install bitsandbytes\n"
                            "Or use the standard VibeVoice models instead."
                        )

                # Handle LLM-only 8-bit quantization (for non-quantized models) - EXPERIMENTAL
                elif quantize_llm == "8bit" and not is_quantized:
                    # Check if CUDA is available (required for quantization)
                    if not torch.cuda.is_available():
                        raise Exception("LLM quantization requires a CUDA GPU. Please use 'full precision' on CPU/MPS.")

                    # Try to import bitsandbytes
                    try:
                        from transformers import BitsAndBytesConfig

                        logger.info("Quantizing LLM component to 8-bit...")
                        # Configure 8-bit quantization for LLM only
                        # CRITICAL: Must skip all audio-related components to prevent noise
                        bnb_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            bnb_8bit_compute_dtype=torch.bfloat16,
                            # Skip ALL audio-critical components (same as 4bit + more conservative)
                            llm_int8_skip_modules=[
                                "lm_head",              # Output projection
                                "prediction_head",      # Diffusion head - CRITICAL for audio quality
                                "acoustic_connector",   # Audio->LLM projection - CRITICAL
                                "semantic_connector",   # Semantic->LLM projection - CRITICAL
                                "acoustic_tokenizer",   # VAE encoder/decoder for audio
                                "semantic_tokenizer",   # VAE encoder for semantics
                            ],
                            # Ultra-conservative outlier threshold (lower = more fp16 processing)
                            # Default is 6.0, but audio/diffusion models need 3.0-4.0 for stability
                            llm_int8_threshold=3.0,
                            # Disable fp16 weights (use int8 storage)
                            llm_int8_has_fp16_weight=False,
                        )

                        model_kwargs["quantization_config"] = bnb_config
                        model_kwargs["device_map"] = "auto"

                        # Flag for post-load verification
                        model_kwargs["_quantization_mode"] = "8bit"

                    except ImportError:
                        raise Exception(
                            "LLM quantization requires 'bitsandbytes' library.\n"
                            "Please install it with: pip install bitsandbytes\n"
                            "Or use 'full precision' mode instead."
                        )

                # Handle LLM-only 4-bit quantization (for non-quantized models)
                elif quantize_llm == "4bit" and not is_quantized:
                    # Check if CUDA is available (required for quantization)
                    if not torch.cuda.is_available():
                        raise Exception("LLM quantization requires a CUDA GPU. Please use 'full precision' on CPU/MPS.")

                    # Try to import bitsandbytes
                    try:
                        from transformers import BitsAndBytesConfig

                        logger.info("Quantizing LLM component to 4-bit...")
                        # Configure 4-bit quantization for LLM only
                        # Note: lm_head must be skipped to avoid bitsandbytes assertion errors
                        bnb_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type='nf4',
                            # Skip lm_head and non-LLM components to avoid errors
                            llm_int8_skip_modules=["lm_head", "prediction_head", "acoustic_connector", "semantic_connector", "diffusion_head"]
                        )

                        model_kwargs["quantization_config"] = bnb_config
                        model_kwargs["device_map"] = "auto"
                        logger.info("LLM will be quantized to 4-bit, diffusion head and connectors remain at full precision")

                        # Flag for post-load verification
                        model_kwargs["_quantization_mode"] = "4bit"

                    except ImportError:
                        raise Exception(
                            "LLM quantization requires 'bitsandbytes' library.\n"
                            "Please install it with: pip install bitsandbytes\n"
                            "Or use 'full precision' mode instead."
                        )

                # Set attention implementation based on user selection
                use_sage_attention = False
                if attention_type == "sage":
                    # SageAttention requires special handling - can't be set via attn_implementation
                    if not SAGE_AVAILABLE:
                        logger.warning("SageAttention not installed, falling back to sdpa")
                        logger.warning("Install with: pip install sageattention")
                        model_kwargs["attn_implementation"] = "sdpa"
                    elif not torch.cuda.is_available():
                        logger.warning("SageAttention requires CUDA GPU, falling back to sdpa")
                        model_kwargs["attn_implementation"] = "sdpa"
                    else:
                        # Don't set attn_implementation for sage, will apply after loading
                        use_sage_attention = True
                        logger.info("Will apply SageAttention after model loading")
                elif attention_type != "auto":
                    model_kwargs["attn_implementation"] = attention_type
                    logger.info(f"Using {attention_type} attention implementation")
                else:
                    # Auto mode - let transformers decide the best implementation
                    logger.info("Using auto attention implementation selection")
                
                # Load the model from local path only
                model_kwargs["local_files_only"] = True

                # Extract quantization mode flag before loading (it's not a model parameter)
                quant_mode = model_kwargs.pop("_quantization_mode", None)

                try:
                    # Use the correct path (parent if 4bit subfolder is used)
                    logger.info(f"Loading model from: {actual_model_path}")
                    if is_quantized:
                        logger.info(f"Loading {quantization} quantized model...")
                        if use_4bit_subfolder:
                            logger.info(f"Using parent path with subfolder='4bit'")

                    # config = VibeVoiceAcousticTokenizerConfig.from_pretrained(actual_model_path)
                    #
                    # # 2. Маппинг (перевод) имен параметров
                    # # Если модель ищет num_filters, отдаем ей значение из decoder_n_filters
                    # if hasattr(config, "decoder_n_filters"):
                    #     config.num_filters = config.decoder_n_filters
                    #     print(f"✅ Synced: num_filters set to {config.num_filters}")

                    # 3. Решаем вопрос с decoder_config
                    # Если модель паникует из-за отсутствия decoder_config,
                    # мы скажем ей, что основной конфиг и есть конфиг декодера (частая практика)
                    # if not hasattr(config, "decoder_config") or config.decoder_config is None:
                    #     config.decoder_config = config
                    #
                    # # Теперь загружаем модель с исправленным конфигом
                    # self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    #     actual_model_path,
                    #     config=config,
                    #     **model_kwargs
                    # )

                    self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        actual_model_path,
                        **model_kwargs
                    )
                except Exception as e:
                    logger.error(f"Failed to load model from {model_files_path}: {e}")
                    raise Exception(
                        f"Failed to load model from {model_files_path}.\n"
                        f"Please ensure the model files are complete and properly downloaded.\n"
                        f"Required files: config.json, pytorch_model.bin or model safetensors\n"
                        f"Error: {str(e)}"
                    )

                elapsed = time.time() - start_time
                logger.info(f"Model loaded in {elapsed:.2f} seconds")

                # Verify quantization if requested (quant_mode was extracted earlier)
                if quant_mode:
                    self._verify_quantization(quant_mode)

                # Verify model was loaded
                if self.model is None:
                    raise Exception("Model failed to load - model is None after loading")

                # Load processor with proper error handling
                from tts.vvembed.processor.vibevoice_processor import VibeVoiceProcessor

                logger.info("Loading VibeVoice processor...")
                processor_kwargs = {
                    "trust_remote_code": True,
                    "cache_dir": MODELS_DIR,
                    "local_files_only": True
                }

                # Add subfolder if needed
                if use_4bit_subfolder:
                    processor_kwargs["subfolder"] = "4bit"

                # Pre-check for Qwen tokenizer - REQUIRED
                tokenizer_path = find_qwen_tokenizer_path(MODELS_DIR)
                if not tokenizer_path:
                    # Tokenizer is required - fail early with clear instructions
                    logger.error("="*60)
                    logger.error("QWEN TOKENIZER NOT FOUND!")
                    logger.error("The VibeVoice processor requires the Qwen2.5-1.5B tokenizer.")
                    logger.error("")
                    logger.error("To fix this, please download the tokenizer:")
                    logger.error("1. Download from: https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/main")
                    logger.error("   Required files: tokenizer_config.json, vocab.json, merges.txt, tokenizer.json")
                    logger.error("2. Place files in ONE of these locations (in order of priority):")
                    logger.error(f"   - {os.path.join(MODELS_DIR, 'tokenizer')}/ (RECOMMENDED)")
                    logger.error(f"   - {os.path.join(MODELS_DIR, 'models--Qwen--Qwen2.5-1.5B')}/snapshots/[hash]/")
                    logger.error("3. Restart ComfyUI and try again")
                    logger.error("="*60)
                    raise Exception(
                        "Qwen tokenizer not found. Please download it manually.\n"
                        "Download from: https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/main\n"
                        "Required files: tokenizer_config.json, vocab.json, merges.txt, tokenizer.json\n"
                        f"Place files in: {os.path.join(MODELS_DIR, 'tokenizer')}/"
                    )

                # Validate that all required tokenizer files exist
                required_files = ["tokenizer_config.json", "vocab.json", "merges.txt"]
                missing_files = []
                for file_name in required_files:
                    file_path = os.path.join(tokenizer_path, file_name)
                    if not os.path.exists(file_path):
                        missing_files.append(file_name)

                if missing_files:
                    logger.error("="*60)
                    logger.error(f"TOKENIZER IS INCOMPLETE!")
                    logger.error(f"Tokenizer folder found at: {tokenizer_path}")
                    logger.error(f"But missing required files: {', '.join(missing_files)}")
                    logger.error("")
                    logger.error("Please download ALL required files from:")
                    logger.error("https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/main")
                    logger.error("Required files:")
                    logger.error("  - tokenizer_config.json")
                    logger.error("  - vocab.json")
                    logger.error("  - merges.txt")
                    logger.error("  - tokenizer.json (optional but recommended)")
                    logger.error("="*60)
                    raise Exception(
                        f"Tokenizer is incomplete. Missing files: {', '.join(missing_files)}\n"
                        "Please download ALL required files from: https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/main\n"
                        "Required files: tokenizer_config.json, vocab.json, merges.txt, tokenizer.json\n"
                        f"Place them in: {tokenizer_path}/"
                    )

                logger.info(f"Found complete tokenizer at: {tokenizer_path}")
                # Override the language model path to use local tokenizer
                processor_kwargs["language_model_pretrained_name"] = tokenizer_path
                # Remove cache_dir to avoid HuggingFace cache interference
                processor_kwargs.pop('cache_dir', None)

                try:
                    # Load processor from same path as model
                    self.processor = VibeVoiceProcessor.from_pretrained(
                        actual_model_path,
                        **processor_kwargs
                    )
                except Exception as proc_error:
                    logger.warning(f"Failed to load processor from {model_files_path}: {proc_error}")

                    # Check if error is about missing Qwen tokenizer
                    if ("Qwen" in str(proc_error) or "tokenizer" in str(proc_error).lower()):
                        logger.info("Processor needs Qwen tokenizer. Searching for tokenizer...")

                        # Try to find tokenizer using priority system
                        tokenizer_path = find_qwen_tokenizer_path(MODELS_DIR)

                        if tokenizer_path:
                            logger.info(f"Found tokenizer at: {tokenizer_path}")
                            # Try to load processor with tokenizer path hint
                            try:
                                from transformers import AutoTokenizer
                                # Load tokenizer from the found path
                                tokenizer = AutoTokenizer.from_pretrained(
                                    tokenizer_path,
                                    trust_remote_code=True,
                                    local_files_only=True
                                )
                                logger.info("Qwen tokenizer loaded successfully from local path")
                                # Store for later use if needed
                                self._temp_tokenizer = tokenizer
                            except Exception as tok_error:
                                logger.warning(f"Failed to load tokenizer from {tokenizer_path}: {tok_error}")
                        else:
                            logger.error("="*60)
                            logger.error("QWEN TOKENIZER NOT FOUND!")
                            logger.error("The VibeVoice processor requires the Qwen2.5-1.5B tokenizer.")
                            logger.error("")
                            logger.error("To fix this, please download the tokenizer:")
                            logger.error("1. Download from: https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/main")
                            logger.error("   Required files: tokenizer_config.json, vocab.json, merges.txt, tokenizer.json")
                            logger.error("2. Place files in ONE of these locations:")
                            logger.error(f"   - {os.path.join(MODELS_DIR, 'tokenizer')}/")
                            logger.error(f"   - {os.path.join(MODELS_DIR, 'models--Qwen--Qwen2.5-1.5B')}/snapshots/[hash]/")
                            logger.error("3. Restart ComfyUI and try again")
                            logger.error("="*60)
                            raise Exception(
                                "Qwen tokenizer not found. Please download it manually.\n"
                                "Download from: https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/main\n"
                                "Required files: tokenizer_config.json, vocab.json, merges.txt, tokenizer.json\n"
                                f"Place tokenizer files in: {os.path.join(MODELS_DIR, 'tokenizer')}/"
                            )
                    
                    logger.info("Attempting to load processor with fallback method...")
                    
                    # Fallback: try loading without subfolder
                    try:
                        if "subfolder" in processor_kwargs:
                            del processor_kwargs["subfolder"]
                        self.processor = VibeVoiceProcessor.from_pretrained(
                            model_files_path,
                            **processor_kwargs
                        )
                    except Exception as fallback_error:
                        logger.error(f"Processor loading failed completely: {fallback_error}")
                        # Check if it's still about Qwen tokenizer
                        if "Qwen" in str(fallback_error):
                            tokenizer_path = find_qwen_tokenizer_path(MODELS_DIR)
                            if not tokenizer_path:
                                raise Exception(
                                    f"Failed to load VibeVoice processor: Missing Qwen tokenizer.\n"
                                    f"Download from: https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/main\n"
                                    f"Required files: tokenizer_config.json, vocab.json, merges.txt, tokenizer.json\n"
                                    f"Place files in: {os.path.join(MODELS_DIR, 'tokenizer')}/"
                                )

                        raise Exception(
                            f"Failed to load VibeVoice processor. Error: {fallback_error}\n"
                            f"Please ensure transformers>=4.51.3 is installed."
                        )
                
                # Move to appropriate device (skip for quantized models as they use device_map)
                # Skip device movement for both pre-quantized models and LLM-quantized models
                is_llm_quantized = quantize_llm != "full precision"
                if not is_quantized and not is_llm_quantized:
                    device = get_optimal_device()
                    if device == "cuda":
                        self.model = self.model.cuda()
                    elif device == "mps":
                        self.model = self.model.to("mps")
                else:
                    logger.info("Quantized model already mapped to device via device_map")
                
                # Apply SageAttention if requested and available
                if use_sage_attention and SAGE_AVAILABLE:
                    self._apply_sage_attention()
                    logger.info("SageAttention successfully applied to model")

                # Apply LoRA if provided and path exists
                if lora_path and os.path.isdir(lora_path):
                    self._apply_lora(lora_path)

                self.current_model_folder = model_folder
                self.current_attention_type = attention_type
                self.current_quantize_llm = quantize_llm
                self.current_lora_path = lora_path
                
            except Exception as e:
                logger.error(f"Failed to load VibeVoice model: {str(e)}")
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
                processed.append(f"[2]: {narrator_text}")
        
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
                result.append(f"[2]: {content}")
        else:
            # This is speaker 1
            content = part.strip()
            if content:  # Only add if not empty
                result.append(f"[1]: {content}")

    return ' '.join(result), True

class TTSProcessor(TTSEngine):
    def __init__(self, module_name='AI_VoiceOverData_TBC'):
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
        super().__init__()

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
            
            # Check if it matches exactly: 44100 Hz and 64000 bps (64 kbps)
            if sample_rate == 44100 and bit_rate == 64000:
                return True
                
            return False

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
            farewell_pattern = os.path.join(target_folder, "*[Ff][Aa][Rr][Ee][Ww][Ee][Ll][Ll]*.wav")

            greeting_files = glob.glob(greeting_pattern)
            farewell_files = glob.glob(farewell_pattern)

            # Combine both lists
            all_files = farewell_files + greeting_files

            if not all_files:
                continue

            # Sort so 01 comes before 02
            all_files.sort()

            # 4. Merge
            combined = AudioSegment.empty()
            for file in all_files:
                audio = AudioSegment.from_wav(file)
                combined += audio + AudioSegment.silent(duration=400)  # Gap for clarity

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
            model_name="VibeVoice-1.5B", seed: int = 42, temperature=0.75, top_p=0.90, speed=1.00,
            max_words_per_chunk=75, attention_type="auto",
            quantize_llm="full precision", free_memory=True, steps: int = 20,
            cfg_scale: float = 1.3, use_sampling: bool = False, apply_effects: bool = True, apply_effects_override: str = None, detect_emotions: bool = False
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
            top_p=top_p,
            max_words_per_chunk=max_words_per_chunk,
            voice_speed_factor=speed,
            voice_key=voice_key,
            questgiver_id=questgiver_id,
            apply_effects=apply_effects,
            apply_effects_override=apply_effects_override,
            detect_emotions=detect_emotions
        )

        # Convert response.content (which is WAV) into an MP3
        audio = AudioSegment.from_file(response, format="wav")

        audio.export(outpath, format="mp3", bitrate="64k")

        if volume_match:
            volume_diff = self.match_peak_volume(voice_to_clone, volume_match_sample, outpath, outpath)

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

        # add 2nd speaker (narrator) if needed
        speakers_check = df[column_name].apply(split_by_speakers_brackets)
        df[column_name] = speakers_check.apply(lambda x: x[0])
        df['two_speakers'] = df['two_speakers'] | speakers_check.apply(lambda x: x[1])
        speakers_check_quotes = df[column_name].apply(split_by_speakers_quotes)
        df[column_name] = speakers_check_quotes.apply(lambda x: x[0])
        df['two_speakers'] = df['two_speakers'] | speakers_check_quotes.apply(lambda x: x[1])

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
        
        # Add 2nd speaker (narrator) from brackets
        speakers_check = split_by_speakers_brackets(text)
        text = speakers_check[0]
        two_speakers = speakers_check[1]
        
        # Add 2nd speaker from quotes
        speakers_check_quotes = split_by_speakers_quotes(text)
        text = speakers_check_quotes[0]
        two_speakers = two_speakers or speakers_check_quotes[1]
        
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

        # add 2nd speaker (narrator) if needed
        df['two_speakers'] = False
        speakers_check = df['cleanedText'].apply(split_by_speakers_brackets)
        df['cleanedText'] = speakers_check.apply(lambda x: x[0])
        df['two_speakers'] = df['two_speakers'] | speakers_check.apply(lambda x: x[1])
        speakers_check_quotes = df['cleanedText'].apply(split_by_speakers_quotes)
        df['cleanedText'] = speakers_check_quotes.apply(lambda x: x[0])
        df['two_speakers'] = df['two_speakers'] | speakers_check_quotes.apply(lambda x: x[1])

        # deal with text in brackets: if its all the text for that NPC, keep the text, otherwise remove it
        df['cleanedText'] = df['cleanedText'].apply(clean_brackets)

        # Remove quotes
        df['cleanedText'] = df['cleanedText'].str.replace(r'(?<!^)"(?!$)', '', regex=True)

        # Removes commas
        df['cleanedText'] = df['cleanedText'].str.replace(
            r',\s+(and|but|you|he|she|they|we|i)\b',
            r' \1',
            regex=True
        )

        # Remove appositive commas (optional)
        # df['cleanedText'] = df['cleanedText'].str.replace(
        #     r',\s*([^,]{1,60}?)\s*,',
        #     r' \1',
        #     regex=True
        # )

        # no commas for short sentences
        df['cleanedText'] = df['cleanedText'].apply(lambda x: x.replace(",", "") if len(x) <= 50 else x)

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
            speed=1.05,
            f0_up_key=0,
            f0_method="rmvpe",
            index_rate=0.70,
            filter_radius=3,
            resample_sr=0,
            rms_mix_rate=1,
            protect=0.25
    ):
        row = pd.Series(row_tuple[1:], index=row_tuple._fields[1:])
        # if row['voice_name'].endswith("_dk"):
        #    voice_name = f'{row["voice_name"][:-3]}'
        # else:
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
                         repetition_penalty=repetition_penalty, top_k=top_k, top_p=top_p, speed=speed,
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
                      top_k=1, top_p=1.0, speed=1.05, f0_up_key=0, f0_method="rmvpe",
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
                                      repetition_penalty=repetition_penalty, top_k=top_k, top_p=top_p, speed=speed,
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
                if item.is_file():
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

            if row['two_speakers']:
                narrator_sample = get_default_sample('narrator')
            else:
                narrator_sample = None

            print(f"generating auto for quest {quest_id}")

            generate_check, a_exp, a_path, _ = self.tts_quest(df, quest_id, module_name, row['source'], force_gen=False, free_memory=False, voice_to_clone=voice_sample, voice_to_clone_2=narrator_sample, detect_emotions=True, seed=default_seed, randomize_seed=randomize_seed, volume_match_sample=volume_match_sample, voice_name=voice_name, voice_type=voice_type)
            if generate_check != None:
                generated_ok = True
        
        return generated_ok, a_path, generate_check

    def tts_quest(self, df, quest_id: str, module_name="AI_VoiceOverData_TBC",
                  spec=None, voice_to_clone=None, voice_to_clone_2=None, text_to_use = None,
                  model_name="VibeVoice-1.5B", seed: int = 42, randomize_seed: bool = False, temperature=0.75, top_p=0.90, speed=1.00,
                  max_words_per_chunk=75, attention_type="auto",
                  quantize_llm="full precision", free_memory=True, steps: int = 20,
                  cfg_scale: float = 1.3, use_sampling: bool = True, force_gen = False, volume_match = True, volume_match_sample = False, apply_effects = True, apply_effects_override = None, detect_emotions = False, voice_name = None, voice_type = None):

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

        print(f"Regenerating {len(df_filtered)} rows for quest {quest_id}...", flush=True)

        final_seed = random.randint(0, 2147483647) if randomize_seed else int(seed)
        print(f"Generating with seed: {final_seed}")

        for _, row in df_filtered.iterrows():
            file_name = f'{row["quest"]}-{row["source"]}' if row['quest'] else f'{row["templateText_race_gender_hash"]}'
            if row['player_gender'] is not None and isinstance(row['player_gender'], str):
                file_name = row['player_gender'] + '-' + file_name
                print(f"Generating {row['player_gender']} voice...")
            file_name = file_name + '.mp3'

            subfolder = 'quests' if row['quest'] else 'gossip'
            tts_text = row['cleanedText']
            # voice_name = row['voice_name']
            questgiver_id = row['id']

            if voice_to_clone_2:
                narrator_voice = True

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

            self.tts(tts_text, voice_name, voice_type, voice_to_clone, voice_to_clone_2, file_name, subfolder, forceGen=force_gen, volume_match = volume_match, volume_match_sample = volume_match_sample,
                     questgiver_id=questgiver_id, model_name=model_name, seed=final_seed, temperature=temperature,
                     top_p=top_p, speed=speed,
                     max_words_per_chunk=max_words_per_chunk, attention_type=attention_type,
                     quantize_llm=quantize_llm, free_memory=free_memory, steps=steps,
                     cfg_scale=cfg_scale, use_sampling=use_sampling, apply_effects=apply_effects, apply_effects_override=apply_effects_override, detect_emotions=detect_emotions)

        return f"Generation complete{f' for 2 genders' if two_genders else ''}{f' with narrator voice' if narrator_voice else ''}", expansions, self.output_path, gradio.update(value=final_seed)  # Return both info message and expansions

    def tts_gossip_auto(self, df, npc_name = None, module_name="AI_VoiceOverData_TBC", file_hash=None, voice_to_clone=None, volume_match_sample=False):

        if npc_name:
            spec = str(npc_name)
        else:
            return "No specification given to produce audio on."

        df_filtered = df[(df['name'] == spec) & (df['expansion'] >= 0) & (df['source'] == 'gossip')]

        df_filtered['voice_type'] = df_filtered['voice_type'].apply(extract_voice_type)

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

            if row['two_speakers']:
                narrator_sample = get_default_sample('narrator')
            else:
                narrator_sample = None

            print(f"generating auto for npc {npc_name}")

            generate_check, a_exp, a_path, _ = self.tts_gossip(df, module_name, npc_name, force_gen=False, free_memory=False, voice_to_clone=voice_sample, voice_to_clone_2=narrator_sample, detect_emotions=True, seed=default_seed, randomize_seed=randomize_seed, volume_match_sample=volume_match_sample, text_to_use=row['cleanedText'], cleaned_text_og=row['cleanedText'], voice_name=voice_name, voice_type=voice_type)
            if generate_check != None:
                generated_ok = True
        
        return generated_ok, a_path, generate_check
    
    def tts_gossip(self, df, module_name="AI_VoiceOverData_TBC", npc_name = None, voice_to_clone=None, voice_to_clone_2=None, cleaned_text_og=None, text_to_use=None,
                  model_name="VibeVoice-1.5B", seed: int = 42, randomize_seed: bool = False, temperature=0.75, top_p=0.90, speed=1.00,
                  max_words_per_chunk=75, attention_type="auto",
                  quantize_llm="full precision", free_memory=True, steps: int = 20,
                  cfg_scale: float = 1.3, use_sampling: bool = True, force_gen = False, volume_match = True, volume_match_sample = False, apply_effects = True, apply_effects_override = None, detect_emotions = False, voice_name = None, voice_type = None):

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

        print(f"Regenerating {len(df_filtered)} rows for NPC {npc_name}...", flush=True)

        final_seed = random.randint(0, 2147483647) if randomize_seed else int(seed)
        print(f"Generating with seed: {final_seed}")

        for _, row in df_filtered.iterrows():
            file_name = f'{row["quest"]}-{row["source"]}' if row['quest'] else f'{row["templateText_race_gender_hash"]}'
            if row['player_gender'] is not None and isinstance(row['player_gender'], str):
                file_name = row['player_gender'] + '-' + file_name
            file_name = file_name + '.mp3'

            subfolder = 'quests' if row['quest'] else 'gossip'
            tts_text = row['cleanedText']
            voice_name = row['voice_name']
            questgiver_id = row['id']

            if voice_to_clone_2:
                narrator_voice = True

            self.tts(tts_text, voice_name, voice_type, voice_to_clone, voice_to_clone_2, file_name, subfolder, forceGen=force_gen, volume_match = volume_match, volume_match_sample = volume_match_sample,
                     questgiver_id=questgiver_id, model_name=model_name, seed=final_seed, temperature=temperature,
                     top_p=top_p, speed=speed,
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