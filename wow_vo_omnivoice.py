import os
import sys
import re
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from wowvo_client_omnivoice.tts_engine import TTSProcessor, find_voice_sample, get_default_sample, generate_game_sample_for_npc_id, get_game_sample_for_npc_id, get_quests_by_npc_model_id_lk
from clean_data import clean_quest_data, clean_quest_data_turtle
from data_prep.sql_queries import is_vanilla_broadcast_text, is_tbc_broadcast_text, is_wotlk_broadcast_text
from data_prep.init_db import init_db
import gradio as gr
from pathlib import Path
import random
import shutil
import hashlib

from data_prep.env_vars import MODELS_DIR, ADDON_PATH, ADDON_PATH_WORKS, VOICE_SAMPLE_PATH, MODULE_NAMES
from wowvo_client_omnivoice.consts import NPC_EFFECTS


# 1. Настройка путей, чтобы Python видел папку modular
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # Убедись, что этот путь ведет ровно туда, где лежит папка modular
# vvembed_path = os.path.join(current_dir, 'tts', 'vvembed')
#
# if vvembed_path not in sys.path:
#     sys.path.insert(0, 'tts/vvembed')
#
# # 2. Импортируем и модель, и её конфиг
# try:
#     from modular.configuration_vibevoice import VibeVoiceAcousticTokenizerConfig
#     from modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
#     print("✅ VibeVoice modules imported successfully")
# except ImportError as e:
#     print(f"❌ Critical Import Error: {e}")
#     # Если упало тут, значит vvembed_path указан неверно
#     raise

copy_to_clipboard_js = """
(text) => {
    navigator.clipboard.writeText(text).then(() => {
        console.log('Copied: ' + text);
    });
    return text;
}
"""

css = """
.my-column {
    margin-top: -6px !important;
}
/* Vanilla (0) - Orange/Yellow */
td[data-column="Expansion"][data-value="0"] {
    color: #FF8C00 !important;
    font-weight: 600 !important;
}

/* TBC (1) - Green */
td[data-column="Expansion"][data-value="1"] {
    color: #2E8B57 !important;
    font-weight: 600 !important;
}

/* WotLK (2) - Blue */
td[data-column="Expansion"][data-value="2"] {
    color: #1E90FF !important;
    font-weight: 600 !important;
}
/* Vanilla - Orange/Yellow */
td[data-column="Expansion"][data-value="Vanilla"] {
    color: #FF8C00 !important;
    font-weight: 600 !important;
}

/* TBC - Green */
td[data-column="Expansion"][data-value="TBC"] {
    color: #2E8B57 !important;
    font-weight: 600 !important;
}

/* WotLK - Blue */
td[data-column="Expansion"][data-value="WotLK"] {
    color: #1E90FF !important;
    font-weight: 600 !important;
}

.fix-button-cell {
    background-color: #4CAF50;
    color: white;
    padding: 4px 12px;
    border-radius: 4px;
    cursor: pointer;
    display: inline-block;
    font-size: 12px;
    text-align: center;
    transition: background-color 0.3s;
}
.fix-button-cell:hover {
    background-color: #45a049;
}
.fix-button-cell-grey {
    background-color: #535355;
    color: black;
    padding: 4px 12px;
    border-radius: 4px;
    cursor: pointer;
    display: inline-block;
    font-size: 12px;
    text-align: center;
    transition: background-color 0.3s;
}
.seed-normal input {
    border: 2px solid #cccccc !important;
    background-color: #2a2a2a !important;
}

.seed-approved input {
    border: 2px solid #00ff00 !important;
    background-color: #1a3a1a !important;
    box-shadow: 0 0 5px rgba(0, 255, 0, 0.3) !important;
}

.seed-approved-with-icon {
    position: relative;
}

.seed-approved-with-icon input {
    border: 2px solid #00ff00 !important;
    background-color: #1a3a1a !important;
    padding-right: 30px !important;
}

.seed-approved-with-icon::after {
    content: "✓";
    position: absolute;
    right: 10px;
    top: 50%;
    transform: translateY(-50%);
    color: #00ff00;
    font-size: 18px;
    font-weight: bold;
    pointer-events: none;
    z-index: 10;
}

.text-normal input {
    border: none;
}

.text-vanilla input {
    border: 2px solid #ffa500 !important;  // changed from green to orange
    background-color: #1a3a1a !important;
    box-shadow: 0 0 5px rgba(255, 165, 0, 0.3) !important;  // updated color for shadow too
}

.text-vanilla-with-icon {
    position: relative;
}

.text-vanilla-with-icon input {
    border: 2px solid #ffa500 !important;  // changed from green to orange
    background-color: #1a3a1a !important;
    padding-right: 30px !important;
}

.text-vanilla-with-icon::after {
    content: "V";
    position: absolute;
    right: 10px;
    top: 50%;
    transform: translateY(-50%);
    color: #ffa500;
    font-size: 18px;
    font-weight: bold;
    pointer-events: none;
    z-index: 10;
}
"""

now_dir = os.getcwd()
sys.path.append(now_dir)
os.makedirs(os.path.join(now_dir, "webui_output"), exist_ok=True)

tts_processor = TTSProcessor()

f0_up_key = 0

selected_voice_names = []

def get_module_short_name(module_name):
    """Convert full module name to short display name"""
    mapping = {
        'AI_VoiceOverData_Vanilla': 'Vanilla',
        'AI_VoiceOverData_VanillaExtra': 'VanillaExtra',
        'AI_VoiceOverData_TBC': 'TBC',
        'AI_VoiceOverData_WoTLK': 'WoTLK',
        'AI_VoiceOverData_Turtle': 'Turtle'
    }
    return mapping.get(module_name, module_name)

def get_module_long_name(short_name):
    """Convert short display name to full module name"""
    reverse_mapping = {
        'Vanilla': 'AI_VoiceOverData_Vanilla',
        'VanillaExtra': 'AI_VoiceOverData_VanillaExtra',
        'TBC': 'AI_VoiceOverData_TBC',
        'WoTLK': 'AI_VoiceOverData_WoTLK',
        'Turtle': 'AI_VoiceOverData_Turtle'
    }
    return reverse_mapping.get(short_name, short_name)

def get_expansion_display(exp_value):
    """Return colored HTML text based on expansion value"""

    # Define colors and text for each expansion
    expansion_config = {
        0: {'color': '#FF8C00', 'text': 'Vanilla'},  # Orange
        1: {'color': '#2E8B57', 'text': 'TBC'},  # Green
        2: {'color': '#1E90FF', 'text': 'WotLK'},  # Blue
        3: {'color': '#32CD32', 'text': 'Turtle'},  # Lime Green
        4: {'color': '#9370DB', 'text': 'Custom'}  # Purple
    }

    # Get config or default
    config = expansion_config.get(exp_value, {'color': '#666666', 'text': f'Unknown ({exp_value})'})

    # Create HTML span with color
    html = f'<span style="color: {config["color"]}; font-weight: bold;">   {config["text"]}</span>'

    return html

def is_turtle(tts_processor):
    """Check if Turtle module is present in MODULE_NAMES"""
    return tts_processor.get_module_name() == 'AI_VoiceOverData_Turtle'
# def get_default_sample(voice_name, voice_type='standard'):
#     """
#     Get default sample path for a given voice name
    
#     Args:
#         audio_name: Name of the voice (e.g., 'human_male', 'orc_female')
    
#     Returns:
#         Path to the default sample file if exists, None otherwise
#     """
#     game_samples = find_voice_sample(voice_name, voice_type)
#     if game_samples.get('FromGame'):
#         return str(game_samples.get('FromGame'))
    
#     voice_sample_default_path = os.path.join(VOICE_SAMPLE_PATH, "default")
#     default_file = Path(voice_sample_default_path) / f"{voice_name}.mp3"
    
#     if default_file.exists():
#         return str(default_file)
    
#     return None

def get_existing_sample_for_npc(df, npc_name, npc_id = 0):
    
    # df_filtered = df[(df['name'] == npc_name) & (df['expansion'] >= 0)]
    
    # find vanilla quest samples
    df_filtered = df[(df['name'] == npc_name) & (df['expansion'] >= 0) & (df['source'].isin(['accept', 'complete']))]
    for _, row in df_filtered.iterrows():

        if row['name'] != npc_name:
                continue
        if row['voice_name'] in ['troll_male', 'boy', 'girl', 'narrator']:
            continue

        src_type = row['source']
        voice_sample = find_quest_audio(row['quest'], src_type)
        if voice_sample:
            if voice_sample.get(get_module_long_name('Vanilla')):
                print(f"found sample {voice_sample.get(get_module_long_name('Vanilla'))} sample")
                return voice_sample.get(get_module_long_name('Vanilla'))


    # find vanilla gossip samples
    df_filtered = df[(df['name'] == npc_name) & (df['expansion'] >= 0) & (df['source'].isin(['gossip']))]
    for _, row in df_filtered.iterrows():

        if row['name'] != npc_name:
            continue
        if row['voice_name'] in ['troll_male', 'boy', 'girl', 'narrator']:
                continue
        
        voice_sample = find_gossip_audio(row['templateText_race_gender_hash'])
        if voice_sample:
            if voice_sample.get(get_module_long_name('Vanilla')):
                print(f"found sample {voice_sample.get(get_module_long_name('Vanilla'))} sample")
                return voice_sample.get(get_module_long_name('Vanilla'))
            
    # find existing in updated modules
    df_filtered = df[(df['name'] == npc_name) & (df['expansion'] >= 0) & (df['source'].isin(['accept', 'complete']))]
    for _, row in df_filtered.iterrows():

        if row['name'] != npc_name:
                continue
        
        src_type = row['source']
        voice_sample = find_quest_audio(row['quest'], src_type)
        if voice_sample:
            for mod_n in MODULE_NAMES:

                # skip OG Vanilla
                if mod_n == MODULE_NAMES[0]:
                    continue

                if voice_sample.get(mod_n):
                    print(f"found sample {voice_sample.get(mod_n)} sample")
                    return voice_sample.get(mod_n)
                    
    # find existing in updated modules
    df_filtered = df[(df['name'] == npc_name) & (df['expansion'] >= 0) & (df['source'].isin(['gossip']))]
    for _, row in df_filtered.iterrows():
        if row['name'] != npc_name:
                continue
        
        voice_sample = find_gossip_audio(row['templateText_race_gender_hash'])
        if voice_sample:
            for mod_n in MODULE_NAMES:
                # skip OG Vanilla
                if mod_n == MODULE_NAMES[0]:
                    continue

                if voice_sample.get(mod_n):
                    print(f"found sample {voice_sample.get(mod_n)} sample")
                    return voice_sample.get(mod_n)
            
    # fall back to game samples
    df_filtered = df[(df['name'] == npc_name) & (df['expansion'] >= 0)]
    df_filtered['voice_type'] = df_filtered['voice_type'].apply(extract_voice_type)
    base_row = df_filtered.iloc[0]
    game_samples = find_voice_sample(base_row['voice_name'], base_row['voice_type'])
    if game_samples.get('FromGame'):
        return str(game_samples.get('FromGame'))
    
    # Try check other modules
    if is_turtle(tts_processor):
        starters, enders = get_quests_by_npc_model_id_lk(npc_id, tts_processor.get_module_name())
        refer_voice_file = find_model_voice_sample(starters, enders)
        if refer_voice_file:
            return refer_voice_file
            
    if base_row['voice_type'] == 'unknown':            
        return None
    
    voice_sample_default_path = os.path.join(VOICE_SAMPLE_PATH, "default")
    default_file = Path(voice_sample_default_path) / f"{base_row['voice_name']}.mp3"
    
    if default_file.exists():
        return str(default_file)
    
    return None

def find_model_voice_sample(starters_list: list, enders_list: list) -> str | None:
    """
    Пробегается по квестам-стартерам и эндерам, пытается найти первый доступный
    аудиосэмпл и возвращает путь к нему.
    """
    # Собираем пары (quest_id, quest_type) в один список для итерации.
    # Для стартеров тип 'accept', для эндеров — 'complete'
    quests_to_check = []
    
    for q_id in starters_list:
        quests_to_check.append((q_id, 'accept'))
        
    for q_id in enders_list:
        quests_to_check.append((q_id, 'complete'))

    # Теперь перебираем их по очереди
    for quest_id, quest_type in quests_to_check:
        
        # Вызываем твою функцию поиска аудио
        voice_sample = find_quest_audio(quest_id, quest_type)
        
        if voice_sample:
            # Твоя логика перебора модулей
            for mod_n in MODULE_NAMES:
                
                # Пропускаем ванилу (MODULE_NAMES[0])
                if mod_n == MODULE_NAMES[0]:
                    continue

                # Если в модуле есть путь к файлу — возвращаем его
                sample_path = voice_sample.get(mod_n)
                if sample_path:
                    print(f"🎯 Найдено совпадение по ModelID! Квест {quest_id} ({quest_type}), модуль: {mod_n}")
                    print(f"Путь: {sample_path}")
                    return sample_path

    print("❌ Ни один квест для этой модели не имеет сгенерированного аудио.")
    return None

import json
import os
from pathlib import Path

class VoiceSampleManager:
    all_btns_sample = {'sample': []}
    players_sample = []
    audio_samples = {}
    sample_voice_current = gr.State(None)

    def __init__(self, max_samples=20):
        self.max_samples = max_samples
        self.components = []
        self.sample_voice_current = gr.State(None)

    def create_voice_card_sample(self, label):
        with gr.Group():
            b_sample = gr.Button("Mark as Default ♻️", size="sm", variant="secondary")

            audio_player = gr.Audio(
                label=label,
                type="filepath",
                interactive=True,
                buttons=[b_sample],
                min_width=100
            )

            self.all_btns_sample['sample'].append(b_sample)
            self.players_sample.append(audio_player)

        return audio_player

    def save_as_default_sample(self, audio_path, audio_name):
        print(f"voice path set to {audio_path}")
        gr.Info(f"Voice to clone set to: {audio_path}")
        voice_sample_default_path = os.path.join(VOICE_SAMPLE_PATH, "default")
        os.makedirs(voice_sample_default_path, exist_ok=True)
        dest_file = Path(voice_sample_default_path) / f"{audio_name}.mp3"
        shutil.copy2(audio_path, dest_file)
        return audio_path

    def create_ui_samples(self, df_state):
        """Create fixed number of samples rows"""
        with gr.Group():
            with gr.Row():
                self.sample_voice_current = self.create_voice_card_sample(label=f"Currenf Default Sample")
        for i in range(self.max_samples):
            with gr.Group():
                gr.Markdown(f"#### Sample {i + 1}", min_height=30)
                with gr.Row():
                    self.audio_samples[i] = self.create_voice_card_sample(label=f"Gossip {i + 1}")

                self.components.append({
                    'sample': self.audio_samples[i]
                })


        for i in range(len(self.all_btns_sample['sample'])):
            self.all_btns_sample['sample'][i].click(
                fn=self.save_as_default_sample,
                inputs=[self.players_sample[i], selected_voice_names_sample],
                outputs=[self.sample_voice_current]
            )

    def update_samples(self, df, selected_voice):

        def get_hash(text):
            return hashlib.md5(text.encode()).hexdigest()

        working_df = df.copy()
        voice_data = working_df[working_df['voice_name'] == selected_voice]
        voice_sample_default_path = os.path.join(VOICE_SAMPLE_PATH, "default")
        voice_samples_list = []
        updates = []

        default_file = Path(voice_sample_default_path) / f"{selected_voice}.mp3"

        updates.append(f"### 🎵 Voice: {selected_voice}")
        updates.append(gr.update(visible=True))
        updates.append(gr.update(value=str(default_file) if default_file.exists() else None))

        for voice_path in MODULE_NAMES:

            # Skip game for now
            if voice_path == "FromGame":
                continue

            copied_count = 0
            # print(f"Searching: {voice_name}")

            found_file = None
            best_fallback = None

            # output_subdir = os.path.join(voice_sample_default_path, voice_path)
            # os.makedirs(output_subdir, exist_ok=True)

            # --- ШАГ 1: Ищем в квестах (по ID) ---
            quest_dir = Path(ADDON_PATH) / voice_path / "generated/sounds/quests"
            if quest_dir.exists():
                # Пробуем найти любой accept или complete для этого NPC
                for _, row in voice_data.iterrows():
                    quest_id = row['quest']
                    source_type = row['source']  # accept, complete и т.д.

                    if quest_id == "": continue

                    # Формируем имя: 12345-accept.wav
                    target_name = f"{int(quest_id)}-{source_type}.mp3"
                    full_path = quest_dir / target_name

                    if full_path.exists():
                        size = os.path.getsize(full_path)
                        if (50 * 1024 <= size <= 250 * 1024):
                            voice_samples_list.append((size, full_path, row['source'], {row['quest'] or row['name']}, voice_path))

            copied_count = 0
            gossip_dir = Path(ADDON_PATH) / voice_path / "generated/sounds/gossip"
            if gossip_dir.exists():
                for _, row in voice_data.iterrows():
                    target_hash_name = f"{row['templateText_race_gender_hash']}.mp3"
                    full_path = gossip_dir / target_hash_name
                    if full_path.exists():
                        size = os.path.getsize(full_path)
                        if (50 * 1024 <= size <= 250 * 1024):
                            voice_samples_list.append((size, full_path, row['source'], {row['quest'] or row['name']}, voice_path))

        voice_samples_list.sort(reverse=True)  # Largest first
        voice_samples_list = voice_samples_list[:50]
        filtered_samples = []
        # for size, sample_path, src, who, module_n in voice_samples_list:
        #     if sample_path.exists():
        #         if not tts_processor.check_for_audio_artifacts(sample_path):
        #             filtered_samples.append((size, sample_path, src, who, module_n))
        #             print(f"found {len(filtered_samples)}/20 samples")
        #         if len(filtered_samples) == 20:
        #             break
        filtered_samples = tts_processor.filter_good_samples_multithread(voice_samples_list)

        voice_samples_list = filtered_samples
        voice_samples_list.sort(reverse=True)
        voice_samples_list = voice_samples_list[:20]

        for i, comp in enumerate(self.components):
            # Extract just the path (second element of tuple)
            path_value = voice_samples_list[i][1] if i < len(voice_samples_list) else None
            # Ensure it's a string, not Path object
            if path_value and not isinstance(path_value, str):
                path_value = str(path_value)

            if not path_value:
                updates.append(gr.skip())
                continue

            # updates.append(gr.update(value=voice_name))
            updates.append(gr.update(value=path_value, label=f"{voice_samples_list[i][2]} {voice_samples_list[i][3]} ({voice_samples_list[i][4]})"))

        return updates

class GossipManager:
    all_btns_gossip = {'npc': []}
    all_btns_gossip_gen = {'npc': []}
    all_btns_global_gossip = {'narrator': []}
    players_gossip = []
    players_global_gossip = []
    audio_gossip_og = {}
    audio_gossip_vanilla = {}
    audio_gossip_tbc = {}
    audio_gossip_wotlk = {}
    audio_gossip_output = {}
    audio_gossip_gen = {}
    audio_gossip_gen_progress = {}
    gossip_texts = {}
    gossip_texts_og = {}
    gossip_texts_cleaned_og = {}
    gossip_names = {}
    gossip_texts_state = gr.State([])
    gossip_current_gen_text = gr.State(None)
    selected_clone_paths_gossip = {
        'npc': gr.State(None)
    }
    selected_clone_paths_global_gossip = {
        'narrator': gr.State(None)
    }
    gossip_groups = {}
    narrator_voices_gossip = gr.State(None)
    monitors_gossip = {
        'npc': gr.State(None)
    }
    monitors_global_gossip = {
        'narrator': gr.State(None)
    }

    def __init__(self, max_gossip=40):
        self.max_gossip = max_gossip
        self.components = []
        self.gossip_texts_state = gr.State([])

    def create_voice_card_gossip(self, label):
        with gr.Group():
            b_gossip = gr.Button("Use Sample ♻️", size="sm", variant="secondary")

            audio_player = gr.Audio(
                label=label,
                type="filepath",
                interactive=True,
                buttons=[b_gossip],
                min_width=100
            )

            self.all_btns_gossip['npc'].append(b_gossip)
            self.players_gossip.append(audio_player)

        return audio_player

    def set_clone_source_exclusive_gossip(self, audio_path):
        print(f"voice path set to {audio_path}")
        gr.Info(f"Voice to clone set to: {audio_path}")
        return audio_path

    def create_voice_card_narrator_gossip(self, label):
        with gr.Group():
            # Создаем кнопки

            b_n_gossip = gr.Button("Use Sample ♻️", size="sm", variant="secondary")

            # Создаем плеер и вставляем кнопки в Toolbar
            audio_player = gr.Audio(
                label=label,
                type="filepath",
                interactive=True,
                buttons=[b_n_gossip],
                min_width=100
            )

            # Сохраняем кнопки в списки для массового управления (например, ресета)
            self.all_btns_global_gossip['narrator'].append(b_n_gossip)
            self.players_global_gossip.append(audio_player)

        return audio_player

    def set_clone_source_narrator_gossip(self, audio_path):
        # type_label is 'accept', 'progress', etc.
        # Logic: return 'A ✅' for clicked index, and 'A ❌' for all others in that list
        print(f"Narrator voice path set to {audio_path}")
        gr.Info(f"Narrator Voice to clone set to: {audio_path}")
        return audio_path

    def update_single_gossip(self, idx, new_value):
        """Update the state when a single gossip textbox changes"""
        # print(f"📝 update_single_gossip called: idx={idx}, new_value='{new_value}'")
        # Get current state (or initialize empty list)
        current = self.gossip_texts_state.value or [""] * self.max_gossip
        
        # Create a new list (don't modify the original)
        updated = list(current)
        
        # Update the specific index
        updated[idx] = new_value
        
        # print(f"Updated list: {updated}")

        # Return the updated list
        return updated

    def update_gossip_state(self, idx, textbox):
        """Update state with current text value"""
        current = self.gossip_texts_state.value or [""] * 20
        current[idx] = textbox.value
        return current
    
    def update_voice_sample_gossip(self, df, module_n, npc_name, voice_name, voice_type):
        try:
            voice_name_used = voice_name
            voice_type_used = voice_type

            if voice_name_used == 'custom_custom' or voice_type_used == 'unknown':
                working_df = df.copy()
                gossip_rows = working_df[(df['name'] == npc_name) & (df['expansion'] >= 0) & (df['source'] == 'gossip')]
                if not gossip_rows.empty:
                    base_row = gossip_rows.iloc[0]
                    npc_id = base_row['id']
                    npc_name = base_row['name']
                    voice_sample_extracted = get_game_sample_for_npc_id(npc_id)
                    if not voice_sample_extracted:
                        voice_sample_extracted = generate_game_sample_for_npc_id(npc_id, module_n)
                        if not voice_sample_extracted and voice_type_used != 'unknown':
                            voice_name_used = None


            if not voice_name_used:
                print(f"No voice found!")
                updates = []
                updates.extend([
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip()
                ])
                return updates

            if (voice_name_used == 'custom_custom' or voice_type_used == 'unknown') and voice_sample_extracted:
                voices_used_for_defaults = voice_sample_extracted
                possible_voice_types = [npc_name]
                voice_map = {}
                voice_map['FromGame'] = voice_sample_extracted
            else:
                voice_map = find_voice_sample(voice_name_used, voice_type_used)
                voices_used_for_defaults = get_default_sample(voice_name_used, voice_type_used)
                possible_voice_types = get_possible_voice_types(voice_name_used)

            if not possible_voice_types:
                update_voice_types = gr.update(value=None, choices=possible_voice_types, interactive=False)
            else:
                update_voice_types = gr.update(choices=possible_voice_types, interactive=True)
                if voice_type_used not in possible_voice_types:
                    update_voice_types = gr.update(choices=possible_voice_types, value=possible_voice_types[0], interactive=True)
                if len(possible_voice_types) == 1:
                    update_voice_types = gr.update(choices=possible_voice_types, value=possible_voice_types[0], interactive=False)

                if voice_name_used == 'custom_custom' and voice_sample_extracted:
                    update_voice_types = gr.update(choices=[npc_name], value=npc_name, interactive=False)

            default_seed = tts_processor.seed_manager.get_default_seed(voice_name_used, voice_type_used) if voice_type_used else None
            
            return {
                # seed_gossip: gr.skip(),
                seed_gossip: gr.update(value=default_seed, elem_classes="seed-approved-with-icon") if default_seed is not None else gr.update(elem_classes="seed-normal"),
                gossip_voice_type: update_voice_types,
                manager.v_sample_og_gossip: voice_map.get('AI_VoiceOverData_Vanilla'),
                manager.v_sample_vanilla_gossip: voice_map.get('AI_VoiceOverData_VanillaExtra'),
                manager.v_sample_tbc_gossip: voice_map.get('AI_VoiceOverData_TBC'),
                manager.v_sample_wotlk_gossip: voice_map.get('AI_VoiceOverData_WoTLK'),
                manager.v_sample_game_gossip: voice_map.get('FromGame'),
                manager.monitors_gossip['npc']: gr.update(value=voices_used_for_defaults) if voices_used_for_defaults is not None else gr.skip(),
            }

        except Exception as e:
            import traceback
            traceback.print_exc()  # Выведет полную ошибку в консоль
            return {f"### ❗ Error: {str(e)}"}

    def create_ui(self, df_state):

        # monitors_gossip = {
        #     'npc': gr.State(None)
        # }
        """Create fixed number of gossip rows"""
        for i in range(self.max_gossip):
            with gr.Group(visible=False) as group:
                self.gossip_names[i] = gr.Markdown(f"#### • Gossip {i + 1}", min_height=30)
                with gr.Row():
                    self.gossip_texts[i] = gr.Textbox(label=f"Gossip {i + 1}", lines=1, interactive=True)
                    self.gossip_texts_og[i] = gr.Textbox(label=f"Gossip {i + 1} Original", lines=1, interactive=False)
                    self.gossip_texts_cleaned_og[i] = gr.Textbox(label=f"Gossip {i + 1} Cleaned Original", visible=False, interactive=False)
                with gr.Row():
                    self.audio_gossip_og[i] = self.create_voice_card_gossip("Vanilla")
                    self.audio_gossip_vanilla[i] = self.create_voice_card_gossip("Vanilla Extra")
                with gr.Row():
                    self.audio_gossip_tbc[i] = self.create_voice_card_gossip("TBC")
                    self.audio_gossip_wotlk[i] = self.create_voice_card_gossip("WotLK")
                with gr.Row():
                    self.audio_gossip_output[i] = self.create_voice_card_gossip("Generated Before")
                    self.audio_gossip_gen[i] = gr.Audio(label="Synthesised Audio", interactive=False, autoplay=True)
                with gr.Row():
                    btn = gr.Button("Generate Audio", variant="primary")
                    self.all_btns_gossip_gen['npc'].append(btn)

                self.gossip_groups[i] = group
                self.components.append({
                    'group': self.gossip_groups[i],
                    'name': self.gossip_names[i],
                    'text': self.gossip_texts[i],
                    'original_text': self.gossip_texts_og[i],
                    'original_cleaned_text': self.gossip_texts_cleaned_og[i],
                    'audio_og': self.audio_gossip_og[i],
                    'audio_vanilla': self.audio_gossip_vanilla[i],
                    'audio_tbc': self.audio_gossip_tbc[i],
                    'audio_wotlk': self.audio_gossip_wotlk[i],
                    'audio_output': self.audio_gossip_output[i],
                    'btn': btn
                })


        with gr.Group():
            monitor_g = gr.Audio(label="Selected NPC Voice", type="filepath", interactive=False)
            self.monitors_gossip = {
                'npc': monitor_g
            }

            gr.Markdown("#### 🎙️ LONG VOICE SAMPLES")
            with gr.Row():
                self.v_sample_og_gossip = self.create_voice_card_gossip("Vanilla")
            #with gr.Row():
                self.v_sample_vanilla_gossip = self.create_voice_card_gossip("VanillaExtra")
            with gr.Row():
                self.v_sample_tbc_gossip = self.create_voice_card_gossip("TBC")
            #with gr.Row():
                self.v_sample_wotlk_gossip = self.create_voice_card_gossip("WotLK")
            with gr.Row():
                self.v_sample_game_gossip = self.create_voice_card_gossip("FromGame")

        with gr.Group(visible=False) as narrator_voices_gossip:
            self.narrator_voices_gossip = narrator_voices_gossip
            gr.Markdown("#### 🎙️ NARRATOR VOICE SAMPLES")
            monitor_n_gossip = gr.Audio(label="Selected Narrator Voice", type="filepath", interactive=False,
                                 value=get_default_sample('narrator'))
            self.monitors_global_gossip = {
                'narrator': monitor_n_gossip
            }
            with gr.Row():
                self.n_sample_og_gossip = self.create_voice_card_narrator_gossip("Vanilla")
                self.n_sample_vanilla_gossip = self.create_voice_card_narrator_gossip("VanillaExtra")
            with gr.Row():
                self.n_sample_tbc_gossip = self.create_voice_card_narrator_gossip("TBC")
                self.n_sample_wotlk_gossip = self.create_voice_card_narrator_gossip("WotLK")


        for i in range(len(self.all_btns_gossip['npc'])):
            self.all_btns_gossip['npc'][i].click(
                fn=self.set_clone_source_exclusive_gossip,
                inputs=[self.players_gossip[i]],
                outputs=[self.monitors_gossip['npc']]
            )

        for i in range(len(self.all_btns_global_gossip['narrator'])):
            self.all_btns_global_gossip['narrator'][i].click(
                fn=self.set_clone_source_narrator_gossip,
                inputs=[self.players_global_gossip[i]],
                outputs=[self.monitors_global_gossip['narrator']]
            )

        def make_update_func(idx):
            def update_func(value):
                # value is passed from the textbox's change event
                return self.update_single_gossip(idx, value)
            return update_func

        for i in range(self.max_gossip):
            self.gossip_texts[i].change(
                make_update_func(i),
                inputs=self.gossip_texts[i],  # Pass the textbox to get its value
                outputs=self.gossip_texts_state
            )

        unused_output = gr.State()
        for ib in range(len(self.all_btns_gossip_gen['npc'])):
            self.all_btns_gossip_gen['npc'][ib].click(
                fn=tts_processor.tts_gossip,
                inputs=[
                    df_state,
                    module_name_gossip,
                    npc_name,
                    self.monitors_gossip['npc'],
                    self.monitors_global_gossip['narrator'],
                    self.gossip_texts_cleaned_og[ib],
                    self.gossip_texts[ib],
                    model_gossip,
                    seed_gossip,
                    randomize_seed_gossip,
                    temperature_gossip,
                    temperature_class_gossip,
                    voice_speed_factor_gossip,
                    voice_cps_gossip,
                    max_words_per_chunk_gossip,
                    attention_type_gossip,
                    quantize_llm_gossip,
                    free_memory_gossip,
                    diffusion_steps_gossip,
                    cfg_scale_gossip,
                    use_sampling_gossip,
                    force_gen_gossip,
                    volume_match_gossip,
                    volume_match_sample_gossip,
                    apply_effects_gossip,
                    apply_effects_override_gossip,
                    use_emotions_gossip_text,
                    gossip_voice,
                    gossip_voice_type
                ],
                outputs=[unused_output, expansions_gossip, self.audio_gossip_gen[ib], seed_gossip],
            )

        # def bind_events(self, external_input):
        # for i, comp in enumerate(self.components):
        #     if i < len(gossip_df):
        #         row = gossip_df.iloc[i]
        #
        #         voices_in_addon = find_gossip_audio(row["templateText_race_gender_hash"])
        #         updates.append(gr.update(visible=True))  # group
        #         updates.append(gr.update(
        #             visible=True,
        #             value=row['cleanedText']
        #         ))  # text
        #         updates.append(gr.update(
        #             visible=True,
        #             value=row['text']
        #         ))  # og text
        #         updates.append(gr.update(value=voices_in_addon.get('AI_VoiceOverData_Vanilla')))  # audio
        #         updates.append(gr.update(value=voices_in_addon.get('AI_VoiceOverData_VanillaExtra')))  # audio
        #         updates.append(gr.update(value=voices_in_addon.get('AI_VoiceOverData_TBC')))  # audio
        #         updates.append(gr.update(value=voices_in_addon.get('AI_VoiceOverData_WoTLK')))  # audio
        #         updates.append(gr.update(value=voices_in_addon.get('Output')))  # audio
        #         updates.append(gr.update(visible=True))  # button
        #
        #         self.all_btns_gossip_gen['npc'][i].click(
        #             fn=tts_processor.tts_gossip,
        #             # inputs=[self.players_gossip[ib], gr.State('npc')],
        #             inputs=[
        #                 df_filtered,
        #                 module_name_gossip,
        #                 npc_name,
        #                 row['cleanedText'],
        #                 selected_clone_paths_gossip['npc'],
        #                 model_gossip,
        #                 seed_gossip,
        #                 randomize_seed_gossip,
        #                 temperature_gossip,
        #                 top_p_gossip,
        #                 voice_speed_factor_gossip,
        #                 max_words_per_chunk_gossip,
        #                 attention_type_gossip,
        #                 quantize_llm_gossip,
        #                 free_memory_gossip,
        #                 diffusion_steps_gossip,
        #                 cfg_scale_gossip,
        #                 use_sampling_gossip,
        #                 force_gen_gossip,
        #                 volume_match_gossip
        #             ],
        #             outputs=[unused_output, expansions_gossip, self.audio_gossip_gen[i], seed_gossip],
        #         )
        

    def update_from_df(self, df, npc_search_name, current_name, current_gossip_text_states, use_modded_gossip):
        """Update visibility and values based on dataframe"""
        df_filtered = df[(df['name'] == npc_search_name) & (df['expansion'] >= 0) & (df['source'] == 'gossip')]
        gossip_df = df_filtered
        gossip_df['voice_type'] = gossip_df['voice_type'].apply(extract_voice_type)
        updates = []
        two_genders_gossip = False

        gossip_df = gossip_df.sort_values('id')

        expansions_used = set()
        for _, row in gossip_df.iterrows():
            expansions_used.add(get_expansion_display(row['expansion']))

        zones_used = set()
        for _, row in gossip_df.iterrows():
            zones_used.add(row['zone'])

        ids_used = set()
        for _, row in gossip_df.iterrows():
            ids_used.add(str(row['id']))

        zones_string = ", ".join(zones_used)
        expansions_string = ", ".join(expansions_used)
        ids_string = ", ".join(ids_used)

        if len(gossip_df) < 1:
            return
        
        voices_list = get_possible_voice_names(df)

        row_first = gossip_df.iloc[0]
        default_voice_sample = get_default_sample(row_first['voice_name'], row_first['voice_type'])
        voice_effect = NPC_EFFECTS[row_first['id']] if row_first['id'] in NPC_EFFECTS else None
        default_seed = tts_processor.seed_manager.get_default_seed(row_first['voice_name'], row_first['voice_type'])
        updates.append(gr.update(value=default_seed, elem_classes="seed-approved-with-icon") if default_seed is not None else gr.update(elem_classes="seed-normal"))
        updates.append(f"### 🧐 NPC: {npc_search_name} (ID: {ids_string}) | Expansion: {expansions_string} | Zone: {zones_string} | Voice: {row_first['voice_name']} {'('+row_first['voice_type']+')' if 'male' in row_first['voice_name'] else ''} {'effect: <b>'+voice_effect+'</b>' if voice_effect else ''}")
        voice_map = find_voice_sample(row_first['voice_name'], row_first['voice_type'])
        voice_map_narrator_gossip = {}
        
        # update current npc name
        updates.append(gr.update(value=npc_search_name))

        # disable modified voice and emotions
        updates.append(gr.update(value=False))
        updates.append(gr.update(value=False))
        updates.append(gr.update(value=None))

         # Debug prints
        # print("=" * 50)
        # print("DEBUG: update_from_df called")
        # print(f"Type of current_gossip_texts_state: {type(current_gossip_text_states)}")
        # print(f"Value: {current_gossip_text_states}")
        
        # # If it's a list, print its contents
        # if isinstance(current_gossip_text_states, list):
        #     print(f"Length: {len(current_gossip_text_states)}")
        #     for i, item in enumerate(current_gossip_text_states):
        #         print(f"  [{i}]: {item} (type: {type(item)})")
        # elif isinstance(current_gossip_text_states, dict):
        #     print(f"Dict keys: {list(current_gossip_text_states.keys())}")
        #     for key, value in current_gossip_text_states.items():
        #         print(f"  {key}: {value}")
        # else:
        #     print(f"Not a list or dict: {current_gossip_text_states}")

        # if current_gossip_text_states:
        #     print("=" * 50)
        #     print("GOSSIP STATE CONTENTS:")
        #     for idx in range(20):
        #         if idx < len(current_gossip_text_states):
        #             val = current_gossip_text_states[idx]
        #             # Show the actual representation with quotes to see empty strings
        #             print(f"  [{idx}]: '{val}' (type: {type(val).__name__}, len: {len(val) if val else 0})")
        #         else:
        #             print(f"  [{idx}]: OUT OF RANGE")
        #     print("=" * 50)

        for i, comp in enumerate(self.components):
            if i < len(gossip_df):
                row = gossip_df.iloc[i]

                # try check existing quest
                if use_modded_gossip and current_name and current_name == npc_search_name:
                    print(f"Using modded gossip texts")
                    if current_gossip_text_states and current_gossip_text_states[i]:
                        print(f"{current_gossip_text_states[i]}")
                        row['cleanedText'] = current_gossip_text_states[i]
                        if re.search(r'\[1]|\[2]', row['cleanedText']):
                            print(f"Narrator text detected in modified text")
                            voice_map_narrator_gossip = find_voice_sample('narrator')

                if re.search(r'\$[Gg]', row['cleanedText']):
                    two_genders_gossip = True
                if not voice_map_narrator_gossip and row['has_narrator']:
                    print(f"Narrator text detected")
                    voice_map_narrator_gossip = find_voice_sample('narrator')

                if row['multi_name'] is not None and row['multi_name'] != '':
                    print(f"Multi speaker detected")

                is_vanilla_text = is_vanilla_broadcast_text(row['broadcast_text_id'], row['text'])
                is_tbc_text = not is_vanilla_text and is_tbc_broadcast_text(row['broadcast_text_id'], row['text'])
                is_wotlk_text = not is_vanilla_text and not is_tbc_text

                expansion_check = 0 if is_vanilla_text else 1 if is_tbc_text else 2 if is_wotlk_text else -1  

                voices_in_addon = find_gossip_audio(row["templateText_race_gender_hash"])
                updates.append(gr.update(visible=True))  # group
                updates.append(f"#### • Gossip {i + 1} • ID: {row['id']} • Expansion: {get_expansion_display(row['expansion'])} {'(Text is '+get_expansion_display(expansion_check)+')' if expansion_check >= 0 else ''} • Zone: {row['zone']}")
                updates.append(gr.update(
                    visible=True,
                    value=row['cleanedText']
                ))  # text
                updates.append(gr.update(
                    visible=True,
                    value=row['text']
                ))  # og text
                updates.append(gr.update(
                    visible=True,
                    value=row['cleanedText']
                ))  # og text hidden
                updates.append(gr.update(visible=True, value=voices_in_addon.get('AI_VoiceOverData_Vanilla')))  # audio
                updates.append(gr.update(visible=True, value=voices_in_addon.get('AI_VoiceOverData_VanillaExtra')))  # audio
                updates.append(gr.update(visible=True, value=voices_in_addon.get('AI_VoiceOverData_TBC')))  # audio
                updates.append(gr.update(visible=True, value=voices_in_addon.get('AI_VoiceOverData_WoTLK')))  # audio
                updates.append(gr.update(visible=True, value=voices_in_addon.get('Output')))  # audio
                updates.append(gr.update(visible=True))  # button

                # self.all_btns_gossip_gen['npc'][i].click(
                #     fn=tts_processor.tts_gossip,
                #     #inputs=[self.players_gossip[ib], gr.State('npc')],
                #     inputs=[
                #         df_filtered,
                #         module_name_gossip,
                #         npc_name,
                #         row['cleanedText'],
                #         selected_clone_paths_gossip['npc'],
                #         model_gossip,
                #         seed_gossip,
                #         randomize_seed_gossip,
                #         temperature_gossip,
                #         top_p_gossip,
                #         voice_speed_factor_gossip,
                #         max_words_per_chunk_gossip,
                #         attention_type_gossip,
                #         quantize_llm_gossip,
                #         free_memory_gossip,
                #         diffusion_steps_gossip,
                #         cfg_scale_gossip,
                #         use_sampling_gossip,
                #         force_gen_gossip,
                #         volume_match_gossip
                #     ],
                #     outputs=[unused_output, expansions_gossip, self.audio_gossip_gen[i], seed_gossip],
                # )
            else:
                updates.extend([
                    # gr.skip(),
                    gr.update(visible=False),  # group
                    gr.skip(),  # name
                    gr.update(visible=False),  # text
                    gr.update(visible=False),  # og text
                    gr.update(visible=False),  # og cleaned text
                    gr.update(visible=False),  # audio
                    gr.update(visible=False),  # button
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                ])

        # Also update container visibility
        # updates.insert(0, gr.update(visible=len(gossip_df) > 0))

        # Update voice samples
        updates.append(gr.update(value=voice_map.get('AI_VoiceOverData_Vanilla')))
        updates.append(gr.update(value=voice_map.get('AI_VoiceOverData_VanillaExtra')))
        updates.append(gr.update(value=voice_map.get('AI_VoiceOverData_TBC')))
        updates.append(gr.update(value=voice_map.get('AI_VoiceOverData_WoTLK')))
        updates.append(gr.update(value=voice_map.get('FromGame')))

        # Update narrator voice samples
        updates.append(gr.update(visible=bool(voice_map_narrator_gossip)))
        updates.append(gr.update(value=voice_map_narrator_gossip.get('AI_VoiceOverData_Vanilla')))
        updates.append(gr.update(value=voice_map_narrator_gossip.get('AI_VoiceOverData_VanillaExtra')))
        updates.append(gr.update(value=voice_map_narrator_gossip.get('AI_VoiceOverData_TBC')))
        updates.append(gr.update(value=voice_map_narrator_gossip.get('AI_VoiceOverData_WoTLK')))
        updates.append(gr.update(value=get_default_sample('narrator') if voice_map_narrator_gossip else None))
        updates.append(gr.update(visible=bool(voice_map_narrator_gossip)))
        updates.append(gr.update(visible=bool(two_genders_gossip)))
        updates.append(gr.update(value=default_voice_sample if default_voice_sample else None))

        updates.append(gr.update(choices=voices_list, value=row_first['voice_name']))
        updates.append(gr.update(value=row_first['voice_type'] if row_first['voice_type'] else None, choices=get_possible_voice_types(row_first['voice_name'])))
                        
        return updates

def load_voices(df):
    selected_voice_names = df['voice_name'].drop_duplicates().tolist()
    return gr.CheckboxGroup(choices=selected_voice_names, value=[])

def load_voices_radio(df):
    selected_voice_names = df['voice_name'].drop_duplicates().tolist()
    return [gr.Radio(choices=selected_voice_names, value=None), gr.update(visible=False)]

def load_zones(df):
    selected_zones_ttszone = df['zone'].drop_duplicates().tolist()
    return [gr.Radio(choices=selected_zones_ttszone, value=None), gr.update(visible=False)]

def hide_stuff():
    return gr.update(open=False)

def get_possible_voice_types(voice_name=None):
    voice_types = ['standard', 'official', 'noble', 'military', 'nerdy', 'happy', 'young', 'grim', 'maternal', 'guard', 'sentinel', 'priestess', 'shady', 'warrior', 'shaman', 'elder', 'magic', 'dark', 'old', 'zany', 'gruff', 'highelf']
    if not voice_name:
        return []
    
    voice_types_possible = ['standard']
    for voice_type in voice_types:
        if voice_type != 'standard':
            sample_path = Path("voices") / "GameVoice" / f"{voice_name}_{voice_type}.mp3"
            if sample_path.exists():
                voice_types_possible.append(voice_type)

    return voice_types_possible

def get_possible_voice_types_dropdown(voice_name=None):
    voice_types = ['standard', 'official', 'noble', 'military', 'nerdy', 'happy', 'young', 'grim', 'maternal', 'guard', 'sentinel', 'priestess', 'shady', 'warrior', 'shaman', 'elder', 'magic', 'dark', 'old', 'zany', 'gruff', 'highelf']
    if not voice_name:
        return []
    
    voice_types_possible = ['standard']
    for voice_type in voice_types:
        if voice_type != 'standard':
            sample_path = Path("voices") / "GameVoice" / f"{voice_name}_{voice_type}.mp3"
            if sample_path.exists():
                voice_types_possible.append(voice_type)

    return gr.update(choices=voice_types_possible, value='standard')

def get_possible_voice_names(df):
    selected_voice_names = df['voice_name'].drop_duplicates().tolist()
    voice_types_possible = []
    for voice_name in selected_voice_names:
        voice_list = find_voice_sample(voice_name)
        if voice_list:
            voice_types_possible.append(voice_name)

    return voice_types_possible

def extract_voice_type(voice_type):
    if not voice_type:
        return 'standard'
    
    voice_types = ['standard', 'official', 'noble', 'military', 'nerdy', 'happy', 'young', 'grim', 'maternal', 'guard', 'sentinel', 'priestess', 'shady', 'warrior', 'shaman', 'elder', 'magic', 'dark', 'old', 'zany', 'gruff', 'highelf']
    pattern = re.compile(r'(' + '|'.join(voice_types) + r')', re.IGNORECASE)
    match = pattern.search(str(voice_type))
    return match.group(1).lower() if match else 'unknown'

def add_duplicate_marker(df):
    """Add duplicate marker for quests appearing in multiple sources"""
    
    # Check for duplicates (keeping all duplicates)
    duplicates = df.duplicated(subset=['quest', 'source'], keep=False)
    
    # Check if any duplicates exist
    has_duplicates = duplicates.any()
    
    if has_duplicates:
        # Add marker column
        df['2ROW'] = duplicates.map({True: '⚠️', False: ''})
        print(f"⚠️ Found {duplicates.sum()} duplicate rows")
        
        # Optional: Show which rows are duplicates
        # duplicate_rows = df[duplicates]
        # print(f"Duplicate examples:")
        # print(duplicate_rows[['quest', 'source']].head())
    # else:
        # print("✅ No duplicate rows found")
        # No column added
    
    return df, has_duplicates

def add_vanilla_different_marker(df):    
    # Check if 'vanilla_text' column is not equal to 0
    vanilla_condition = df['vanilla_text'] != 0

    # Count the number of rows that meet the condition
    has_vanilla_different = vanilla_condition.any()
    if has_vanilla_different:
        # Add marker column
        df['VD'] = vanilla_condition.map({True: '⚠️', False: ''})
    
    return df, has_vanilla_different

def add_duplicate_marker_gossip(df):
    check_columns = ['ORG', 'VAN', 'TBC', 'WLK']
    
    # Check if any row has at least 2 checks
    has_duplicates = False
    for _, row in df.iterrows():
        count = sum(1 for col in check_columns if '✅' in row.get(col) or '☑️' in row.get(col))
        if count >= 2:
            has_duplicates = True
            break
    
    # Only add column if duplicates exist
    if has_duplicates:
        def has_two_checks(row):
            count = sum(1 for col in check_columns if '✅' in row.get(col) or '☑️' in row.get(col))
            return count >= 2
        
        df['DUPL'] = df.apply(
            lambda row: '⚠️' if has_two_checks(row) else '', 
            axis=1
        )
    
    return df, has_duplicates

def select_columns(df):
    display_df = df[['source', 'quest', 'quest_title', 'name', 'id', 'voice_name', 'voice_type', 'DisplayRaceID', 'DisplaySexID', 'cleanedText', 'expansion', 'zone']].copy()
    # Map numeric to display text
    expansion_names = {
        0: 'Vanilla',
        1: 'TBC',
        2: 'WoTLK',
        3: 'Turtle'
    }
    expansion_names_turtle = {
        0: 'Vanilla',
        1: 'Turtle'
    }
    display_df = display_df.rename(columns={
        'quest_title': 'title',
        'name': 'npc_name',
        'id': 'npc',
        'DisplayRaceID': 'Race',
        'DisplaySexID': 'Sex',
        'cleanedText': 'Text',
        'expansion': 'Exp',
    })

    # load voices for later use
    load_voices(df)
    
    display_df['voice_type'] = display_df['voice_type'].apply(extract_voice_type)

    display_df['Exp'] = display_df['Exp'].map(expansion_names)
    return display_df

def select_columns_ttszone(df, selected_zone, remove_bad_rows=False, check_bad_rows=False, progress=gr.Progress()):
    if is_turtle(tts_processor):
        display_df = df[['source', 'quest', 'quest_title', 'name', 'id', 'voice_name', 'voice_type', 'cleanedText', 'expansion', 'zone']].copy()
    else:
        display_df = df[['source', 'quest', 'quest_title', 'name', 'id', 'voice_name', 'voice_type', 'cleanedText', 'expansion', 'zone', 'vanilla_text']].copy()
    text_types = ['accept', 'complete', 'progress']
    display_df = display_df[display_df['source'].isin(text_types)]

    display_df = display_df[display_df['zone'] == selected_zone]

    display_df = display_df.rename(columns={
        'quest_title': 'title',
        'name': 'npc_name',
        'id': 'npc',
        'cleanedText': 'Text',
        'expansion': 'Exp',
    })
    expansion_names = {
        0: 'Vanilla',
        1: 'TBC',
        2: 'WoTLK',
        3: 'Turtle'
    }
    display_df['Exp'] = display_df['Exp'].map(expansion_names)

    display_df = display_df.sort_values('source')

    display_df['voice_type'] = display_df['voice_type'].apply(extract_voice_type)

    total_rows = len(display_df)
    total_steps = total_rows * 4
    progress_state = {'step': 0}

    def check_module(module_name, remove_bad_rows):
        """Return function to check if audio exists for a module"""

        def check(row):

            progress_state['step'] += 1
            progress(progress_state['step'] / total_steps)

            audio = find_quest_audio(row['quest'], row['source'], module_name)
            if not audio:
                return '❌'
            
            # two_genders = re.search(r'\$[Gg]', str(row['Text']))

            testing_module_match = False
            if module_name == MODULE_NAMES[1] and row['Exp'] == 'Vanilla':
                testing_module_match = True
            if module_name == MODULE_NAMES[2] and row['Exp'] == 'TBC':
                testing_module_match = True
            if module_name == MODULE_NAMES[3] and row['Exp'] == 'WotLK':
                testing_module_match = True

            if audio and audio.get(module_name):
                if module_name == MODULE_NAMES[0]:
                    return '✅'
    
                print(f'Checking silence in quest {row['quest']} ({row['source']})')
                if audio.get('Output'):
                    result = None
                    if check_bad_rows:
                        okcheck, result = tts_processor.evaluate_and_register_quest(row['quest'], row['source'], audio.get('Output'), row['Text'], remove_bad_rows)
                        if result['decision'] == "REGENERATE":
                            print(f"   Status: [{result['decision']}] | WER: {result['wer']*100}% | Reason: {result['reason']}")

                        if remove_bad_rows and result['decision'] == "REGENERATE":
                            return f'🗑️ 👎'
                    
                    return f'☑️{f"♻️" if audio.get('Testing') and testing_module_match else ''}{f"👎" if result and result['decision'] == "REGENERATE" else ''}'
                elif tts_processor.check_for_long_silence_ffmpeg(audio.get(module_name)):
                    return f'✅🔇{f"♻️" if audio.get('Testing') and testing_module_match else ''}'
                else:
                    result = None
                    if check_bad_rows:
                        okcheck, result = tts_processor.evaluate_and_register_quest(row['quest'], row['source'], audio.get(module_name), row['Text'], False) # Do not remove downloaded files
                        if result['decision'] == "REGENERATE":
                            print(f"   Status: [{result['decision']}] | WER: {result['wer']*100}% | Reason: {result['reason']}")

                        # if remove_bad_rows and result['decision'] == "REGENERATE":
                        #     return f'🗑️ 👎'
                        
                    return f'✅{f"👎" if result and result['decision'] == "REGENERATE" else ''}'

            # accept aready generated
            if audio.get('Output'):
                return f'☑️{f"♻️" if audio.get('Testing') and testing_module_match else ''}'
            elif audio.get('Testing') and testing_module_match:
                return f'♻️'

            return '❌'

        return check
    
    def check_fix_button(has_vanilla_different):

        def check(row):

            if row['Exp'] == 'Vanilla':
                if row['ORG'] == '✅':
                    if has_vanilla_different and row['VD'] == '⚠️' and row['WLK'] != '✅' and row['WLK'] != '☑️':
                        print(f"wotlk different text missing! quest {row['quest']} ({row['source']})")
                        return row['Fix']
                    return ''
                elif row['VAN'] == '✅':
                    if has_vanilla_different and row['VD'] == '⚠️' and row['WLK'] != '✅' and row['WLK'] != '☑️':
                        print("wotlk different text missing!")
                        print(f"wotlk different text missing! quest {row['quest']} ({row['source']})")
                        return row['Fix']
                    return ''
                elif row['VAN'] == '☑️':
                    return ''
                
            if row['Exp'] == 'TBC':
                if row['TBC'] == '✅':
                    return ''
                if row['TBC'] == '☑️':
                    return ''
                
            if row['Exp'] == 'WoTLK':
                if row['WLK'] == '✅':
                    return ''
                if row['WLK'] == '☑️':
                    return ''
                
            # # if narrator is voice and narrator in text...no auto fix
            # if row['NAR'] == '✅':
            #     if row['Fix'] != '':
            #         return f'<span class="fix-button-cell-grey">MANUAL</span>'
                
            return row['Fix']
        
        return check
    
    def check_module_turtle(module_name, remove_bad_rows):
        """Return function to check if audio exists for a module"""

        def check(row):

            progress_state['step'] += 1
            progress(progress_state['step'] / total_steps * 4)

            audio = find_quest_audio(row['quest'], row['source'], module_name)
            if not audio:
                return '❌'
            
            # two_genders = re.search(r'\$[Gg]', str(row['Text']))

            testing_module_match = False
            if module_name == MODULE_NAMES[1] and row['Exp'] == 'Vanilla':
                testing_module_match = True
            if module_name == 'AI_VoiceOverData_Turtle' and row['Exp'] == 'Turtle':
                testing_module_match = True

            if audio and audio.get(module_name):
                if module_name == MODULE_NAMES[0]:
                    return '✅'
    
                print(f'Checking silence in quest {row['quest']} ({row['source']})')
                if audio.get('Output'):
                    return f'☑️{f"♻️" if audio.get('Testing') and testing_module_match else ''}'
                elif tts_processor.check_for_long_silence_ffmpeg(audio.get(module_name)):
                    return f'✅🔇{f"♻️" if audio.get('Testing') and testing_module_match else ''}'
                else:
                    # result = tts_processor.evaluate_audio_quality(audio.get(module_name), row['Text'])
                    # print(f"Status: [{result['decision']}] | WER: {result['wer']*100}% | Reason: {result['reason']}")
                    return f'✅'

            # accept aready generated
            if audio.get('Output'):
                result = None
                if check_bad_rows:
                    okcheck, result = tts_processor.evaluate_and_register_quest(row['quest'], row['source'], audio.get('Output'), row['Text'], remove_bad_rows)
                    if result['decision'] == "REGENERATE":
                        print(f"   Status: [{result['decision']}] | WER: {result['wer']*100}% | Reason: {result['reason']}")

                    if remove_bad_rows and result['decision'] == "REGENERATE":
                        return f'🗑️ 👎'
                
                return f'☑️{f"♻️" if audio.get('Testing') and testing_module_match else ''}{f"👎" if result and result['decision'] == "REGENERATE" else ''}'
            elif audio.get('Testing') and testing_module_match:
                return f'♻️'

            return '❌'

        return check
    
    def check_fix_button_turtle():

        def check(row):

            if row['Exp'] == 'Vanilla':
                if row['ORG'] == '✅':
                    return ''
                elif row['VAN'] == '✅':
                    return ''
                elif row['VAN'] == '☑️':
                    return ''
                
            if row['Exp'] == 'Turtle':
                if row['TRT'] == '✅':
                    return ''
                if row['TRT'] == '☑️':
                    return ''
                
            # if narrator is voice and narrator in text...no auto fix
            # if row['NAR'] == '✅':
            #     if row['Fix'] != '':
            #         return f'<span class="fix-button-cell-grey">MANUAL</span>'
                
            return row['Fix']
        
        return check

    display_df['ORG'] = display_df.apply(check_module(MODULE_NAMES[0], remove_bad_rows), axis=1)
    display_df['VAN'] = display_df.apply(check_module(MODULE_NAMES[1], remove_bad_rows), axis=1)
    if is_turtle(tts_processor):
        display_df['TRT'] = display_df.apply(check_module_turtle(tts_processor.get_module_name(), remove_bad_rows), axis=1)
    else:
        display_df['TBC'] = display_df.apply(check_module(MODULE_NAMES[2], remove_bad_rows), axis=1)
        display_df['WLK'] = display_df.apply(check_module(MODULE_NAMES[3], remove_bad_rows), axis=1)
    # display_df['Text'] = display_df['Text'].str[:50] + \
    #                             (display_df['Text'].str.len() > 50).map({True: '...', False: ''})
    display_df['NAR'] = display_df['Text'].apply(
        lambda x: '✅' if re.search(r'\[1]|\[2]', str(x)) else ''
    )
    display_df['2GE'] = display_df['Text'].apply(
        lambda x: '✅' if re.search(r'\$[Gg]', str(x)) else ''
    )
    
    display_df['Fix'] = f'<span class="fix-button-cell">Fix</span>'
    # display_df['Gen'] = f'<span class="fix-button-cell">Gen</span>'

    display_df, has_duplicates = add_duplicate_marker(display_df)
    if has_duplicates:
        # display_df['Fix'] = display_df['2ROW'].apply(
        #     lambda x: '' if '⚠️' in str(x) else display_df['Fix']
        # )
        display_df.loc[display_df['2ROW'].str.contains('⚠️', na=False), 'Fix'] = ''
        # mask = display_df['2ROW'] == '⚠️'
        # display_df.loc[mask, 'Fix'] = ''
        

    if is_turtle(tts_processor):
        display_df['Fix'] = display_df.apply(check_fix_button_turtle(), axis=1)
    else:
        display_df, has_van_diff = add_vanilla_different_marker(display_df)
        display_df['Fix'] = display_df.apply(check_fix_button(has_van_diff), axis=1)
        display_df.drop(columns=['vanilla_text'], inplace=True)

    return display_df, f"### 🗺️ Quests in: {selected_zone}", gr.update(visible=True), f"📦 ZIP Quests for {selected_zone}", gr.update(visible=True)

def zip_generated_ttszone(df, selected_zone, zip_prefix):
    import zipfile

    display_df = df.copy()
    text_types = ['accept', 'complete', 'progress']
    display_df = display_df[display_df['source'].isin(text_types)]
    display_df = display_df[display_df['zone'] == selected_zone]
    quest_ids = set(display_df['quest'].astype(str))

    # Create zip filename
    safe_zone_name = selected_zone.replace(" ", "_").replace("/", "_")
    zip_filename = f"archives/{safe_zone_name}-quests{zip_prefix}.zip"
    zip_path = Path(zip_filename)
    
    print(f"📦 Creating zip: {zip_path}")

    # Track files to remove after zipping
    files_to_remove = []
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        files_added = 0
        base_dir = Path.cwd()  # Current working directory
        
        for module in MODULE_NAMES:
            module_path = Path(module)
            quests_path = module_path / "generated/sounds/quests"
            
            if quests_path.exists():
                print(f"  Scanning: {quests_path}")
                
                for file_path in quests_path.rglob("*.mp3"):  # Only mp3 files
                    if file_path.is_file():
                        filename = file_path.name
                        # Check if filename contains any quest ID from this zone
                        # Filename format examples: "12345-accept.mp3", "m-12345-accept.mp3", "12345-complete.mp3"
                        matched = False
                        for quest_id in quest_ids:
                            if f"{quest_id}-" in filename or f"-{quest_id}-" in filename or filename.startswith(f"{quest_id}-"):
                                matched = True
                                break
                        
                        if matched:
                            relative_path = os.path.relpath(file_path, start=base_dir)
                            zipf.write(file_path, relative_path)
                            files_added += 1
                            print(f"    Added: {relative_path}")
                        # else:
                        #     print(f"    Skipped (not in zone): {filename}")

    
    print(f"✅ Zip created: {zip_path} with {files_added} files")
    # print(f"🗑️ Removed {removed_count} files")
    gr.Info(f"Zone zip created: {zip_filename} with {files_added} files")
    
    return str(zip_path), gr.update(visible=True) 

def select_columns_ttszone_gossip(df, selected_zone, remove_bad_rows=False, check_bad_rows=False, progress=gr.Progress()):
    display_df = df[['source', 'name', 'id', 'voice_name', 'voice_type', 'text', 'broadcast_text_id', 'cleanedText', 'expansion', 'zone', 'templateText_race_gender_hash']].copy()
    text_types = ['gossip']
    display_df = display_df[display_df['source'].isin(text_types)]
    display_df = display_df[display_df['zone'] == selected_zone]

    if display_df.empty:
        return display_df, f"### ❌ NO Gossips in: {selected_zone}", gr.update(visible=True), f"📦 ZIP Gossips for {selected_zone}", gr.update(visible=False)

    display_df = display_df.rename(columns={
        'name': 'npc_name',
        'id': 'npc',
        'text': 'Text_OG',
        'cleanedText': 'Text',
        'expansion': 'Exp',
        'templateText_race_gender_hash': 'hash'
    })
    expansion_names = {
        0: 'Vanilla',
        1: 'TBC',
        2: 'WoTLK',
        3: 'Turtle'
    }
    display_df['Exp'] = display_df['Exp'].map(expansion_names)

    display_df = display_df.sort_values('npc')

    total_rows = len(display_df)
    total_steps = total_rows * 4
    progress_state = {'step': 0}

    progress(0, desc=f"Checking {total_rows} gossips...")

    def has_two_checks(row):
        """Check if row has at least 2 columns with '✅' among ORG, VAN, TBC, WLK"""
        check_columns = ['ORG', 'VAN', 'TBC', 'WLK']
        count = sum(1 for col in check_columns if row.get(col) == '✅')
        return count >= 2

    def check_module(module_name, remove_bad_rows):
        """Return function to check if audio exists for a module"""

        def check(row):

            progress_state['step'] += 1
            progress(progress_state['step'] / total_steps)

            is_vanilla_text = is_vanilla_broadcast_text(row['broadcast_text_id'], row['Text_OG'])
            is_tbc_text = not is_vanilla_text and is_tbc_broadcast_text(row['broadcast_text_id'], row['Text_OG'])
            is_wotlk_text = not is_vanilla_text and not is_tbc_text
            expansion_check = 0 if is_vanilla_text else 1 if is_tbc_text else 2 if is_wotlk_text else -1
            db_module_match = False
            if expansion_check == 0:
                db_module_match = module_name == MODULE_NAMES[1] or module_name == MODULE_NAMES[0]
            if expansion_check == 1:
                db_module_match = module_name == MODULE_NAMES[2]
            if expansion_check == 2:
                db_module_match = module_name == MODULE_NAMES[3]

            audio = find_gossip_audio(row['hash'], module_name)
            if not audio:                
                return '❌' if db_module_match else '🚫'
            
            # two_genders = re.search(r'\$[Gg]', str(row['cleanedText']))

            testing_module_match = False
            if module_name == MODULE_NAMES[1] and row['Exp'] == 'Vanilla':
                testing_module_match = True
            if module_name == MODULE_NAMES[2] and row['Exp'] == 'TBC':
                testing_module_match = True
            if module_name == MODULE_NAMES[3] and row['Exp'] == 'WotLK':
                testing_module_match = True

            if audio and audio.get(module_name):
                if module_name == MODULE_NAMES[0]:
                    return '✅'
                
                # removed in works
                if not audio.get('Works'):
                    return '✖️'
    
                print(f'Checking silence in gossip {row['npc_name']} ({row['npc']})')
                if audio.get('Output'):
                    result = None
                    if check_bad_rows:
                        okcheck, result = tts_processor.evaluate_and_register_gossip(row['npc_name'], row['hash'], audio.get('Output'), row['Text'], remove_bad_rows)
                        if result['decision'] == "REGENERATE":
                            print(f"   Status: [{result['decision']}] | WER: {result['wer']*100}% | Reason: {result['reason']}")

                        if remove_bad_rows and result['decision'] == "REGENERATE":
                            return f'🗑️ 👎'
                        
                    return f'☑️{f"♻️" if audio.get('Testing') and testing_module_match else ''}{f"👎" if result and result['decision'] == "REGENERATE" else ''}'
                elif tts_processor.check_for_long_silence_ffmpeg(audio.get(module_name)):
                    return f'✅🔇{f"♻️" if audio.get('Testing') and testing_module_match else ''}'
                else:
                    result = None
                    if check_bad_rows:
                        okcheck, result = tts_processor.evaluate_and_register_gossip(row['npc_name'], row['hash'], audio.get(module_name), row['Text'], False) # Do not remove downloaded files
                        if result['decision'] == "REGENERATE":
                            print(f"   Status: [{result['decision']}] | WER: {result['wer']*100}% | Reason: {result['reason']}")

                    return f'✅{f"👎" if result and result['decision'] == "REGENERATE" else ''}'

            # accept aready generated
            if audio.get('Output'):
                return f'☑️{f"♻️" if audio.get('Testing') and testing_module_match else ''}'
            elif audio.get('Testing') and testing_module_match:
                return f'♻️'

            return '❌' if db_module_match else '🚫'

        return check
    
    def check_fix_button():

        def check(row):

            def no_mark(marking):
                return marking =='❌' or marking == '🚫'

            is_vanilla_text = is_vanilla_broadcast_text(row['broadcast_text_id'], row['Text_OG'])
            is_tbc_text = not is_vanilla_text and is_tbc_broadcast_text(row['broadcast_text_id'], row['Text_OG'])
            is_wotlk_text = not is_vanilla_text and not is_tbc_text
            expansion_check = 0 if is_vanilla_text else 1 if is_tbc_text else 2 if is_wotlk_text else -1
            
            if row['Exp'] == 'Vanilla':
                if row['ORG'] == '✅' and ('✅' in row['TBC'] or '✅' in row['WLK']):
                     return '<span class="fix-button-cell">REM DUP</span>'
                elif row['ORG'] == '✅':
                    return ''
                elif row['VAN'] == '✅':
                    return ''
                elif row['VAN'] == '☑️':
                    return ''
                elif row['VAN'] == '☑️♻️':
                    return ''
                
            if row['Exp'] == 'TBC':
                if (row['TBC'] == '✅' or row['TBC'] == '☑️') and ('✅' in row['WLK'] or '✅' in row['ORG'] or '✅' in row['VAN']):
                     return '<span class="fix-button-cell">REM DUP</span>'
                if row['TBC'] == '✅':
                    return ''
                if row['TBC'] == '☑️':
                    return ''
                if row['TBC'] == '☑️♻️':
                    return ''
                
            if row['Exp'] == 'WoTLK':
                if (row['WLK'] == '✅' or row['WLK'] == '☑️') and ('✅' in row['TBC'] or '✅' in row['ORG'] or '✅' in row['VAN']):
                     return '<span class="fix-button-cell">REM DUP</span>'
                if row['WLK'] == '✅':
                    return ''
                if row['WLK'] == '☑️':
                    return ''
                if row['WLK'] == '☑️♻️':
                    return ''
                
            # if narrator is voice and narrator in text...no auto fix
            # if row['NAR'] == '✅':
            #     if row['Fix'] != '':
            #         return f'<span class="fix-button-cell-grey">MANUAL</span>'

            # if vanilla text doesn't match other modules (if we want to copy missing)
            if row['Exp'] == 'Vanilla':
                if no_mark(row['ORG']) and no_mark(row['VAN']):
                    if row['TBC'] == '✅' or row['WLK'] == '✅':
                        if row['Fix'] != '<span class="fix-button-cell">REM DUP</span>':
                            if not is_vanilla_text:
                                return f'<span class="fix-button-cell-grey">NOT VAN</span>'
                        
            if row['Exp'] == 'TBC':
                if no_mark(row['TBC']):
                    if row['WLK'] == '✅':
                        if row['Fix'] != '<span class="fix-button-cell">REM DUP</span>':
                            if not is_tbc_text:
                                return f'<span class="fix-button-cell-grey">NOT TBC</span>'
                            
            if row['Exp'] == 'Vanilla' and not is_vanilla_text:
                if no_mark(row['ORG']) and no_mark(row['VAN']) and no_mark(row['TBC']) and no_mark(row['WLK']):
                    return f'<span class="fix-button-cell">WRN EXP</span>'
                return f'<span class="fix-button-cell-grey">WRN EXP</span>'
            if row['Exp'] == 'TBC' and not is_tbc_text:
                if no_mark(row['ORG']) and no_mark(row['VAN']) and no_mark(row['TBC']) and no_mark(row['WLK']):
                    return f'<span class="fix-button-cell">WRN EXP</span>'
                return f'<span class="fix-button-cell-grey">WRN EXP</span>'
            if row['Exp'] == 'WoTLK' and not is_wotlk_text:
                if no_mark(row['ORG']) and no_mark(row['VAN']) and no_mark(row['TBC']) and no_mark(row['WLK']):
                    return f'<span class="fix-button-cell">WRN EXP</span>'
                return f'<span class="fix-button-cell-grey">WRN EXP</span>'
            
            return row['Fix']
        
        return check
    
    def check_module_turtle(module_name, remove_bad_rows):

        def check(row):

            progress_state['step'] += 1
            progress(progress_state['step'] / total_steps * 4)

            is_vanilla_text = is_vanilla_broadcast_text(row['broadcast_text_id'], row['Text_OG'])
            is_turtle_text = not is_vanilla_text
            expansion_check = 0 if is_vanilla_text else 1 if is_turtle_text else -1
            db_module_match = False
            if expansion_check == 0:
                db_module_match = module_name == MODULE_NAMES[1] or module_name == MODULE_NAMES[0]
            if expansion_check == 1:
                db_module_match = module_name == 'AI_VoiceOverData_Turtle'

            # mark as turtle for turtle only NPCS
            if row['source'] == 'gossip' and row['Exp'] == 'Turtle' and not db_module_match:
                db_module_match = True

            if module_name == 'AI_VoiceOverData_Turtle' and row['Exp'] == 'Turtle':
                testing_module_match = True

            audio = find_gossip_audio(row['hash'], module_name)
            if not audio:           
                return '❌' if db_module_match else '🚫'
            
            # two_genders = re.search(r'\$[Gg]', str(row['cleanedText']))

            testing_module_match = False
            if module_name == MODULE_NAMES[1] and row['Exp'] == 'Vanilla':
                testing_module_match = True
            if module_name == 'AI_VoiceOverData_Turtle' and row['Exp'] == 'Turtle':
                testing_module_match = True

            if audio and audio.get(module_name):
                if module_name == MODULE_NAMES[0]:
                    return '✅'
                
                # removed in works
                if not audio.get('Works'):
                    return '✖️'
    
                print(f'Checking silence in gossip {row['npc_name']} ({row['npc']})')
                if audio.get('Output'):
                    return f'☑️{f"♻️" if audio.get('Testing') and testing_module_match else ''}'
                elif tts_processor.check_for_long_silence_ffmpeg(audio.get(module_name)):
                    return f'✅🔇{f"♻️" if audio.get('Testing') and testing_module_match else ''}'
                else:
                    return f'✅'

            # accept aready generated
            if audio.get('Output'):
                result = None
                if check_bad_rows:
                    okcheck, result = tts_processor.evaluate_and_register_gossip(row['npc_name'], row['hash'], audio.get('Output'), row['Text'], remove_bad_rows)
                    if result['decision'] == "REGENERATE":
                        print(f"   Status: [{result['decision']}] | WER: {result['wer']*100}% | Reason: {result['reason']}")

                    if remove_bad_rows and result['decision'] == "REGENERATE":
                        return f'🗑️ 👎'
                    
                return f'☑️{f"♻️" if audio.get('Testing') and testing_module_match else ''}{f"👎" if result and result['decision'] == "REGENERATE" else ''}'
            elif audio.get('Testing') and testing_module_match:
                return f'♻️'

            return '❌' if db_module_match else '🚫'

        return check
    

    def check_fix_button_turtle():

        def check(row):

            def no_mark(marking):
                return marking =='❌' or marking == '🚫'

            is_vanilla_text = is_vanilla_broadcast_text(row['broadcast_text_id'], row['Text_OG'])
            is_turtle_text = not is_vanilla_text
            
            if row['Exp'] == 'Vanilla':
                if row['ORG'] == '✅':
                    return ''
                elif row['VAN'] == '✅':
                    return ''
                elif row['VAN'] == '☑️':
                    return ''
                elif row['VAN'] == '☑️♻️':
                    return ''
                
            if row['Exp'] == 'Turtle':
                if row['TRT'] == '✅':
                    return ''
                if row['TRT'] == '☑️':
                    return ''
                if row['TRT'] == '☑️♻️':
                    return ''
                
            # if narrator is voice and narrator in text...no auto fix
            # if row['NAR'] == '✅':
            #     if row['Fix'] != '':
            #         return f'<span class="fix-button-cell-grey">MANUAL</span>'
            
            return row['Fix']
        
        return check
    

    display_df['ORG'] = display_df.apply(check_module(MODULE_NAMES[0], remove_bad_rows), axis=1)
    display_df['VAN'] = display_df.apply(check_module(MODULE_NAMES[1], remove_bad_rows), axis=1)
    if is_turtle(tts_processor):
        display_df['TRT'] = display_df.apply(check_module_turtle(tts_processor.get_module_name(), remove_bad_rows), axis=1)
    else:
        display_df['TBC'] = display_df.apply(check_module(MODULE_NAMES[2], remove_bad_rows), axis=1)
        display_df['WLK'] = display_df.apply(check_module(MODULE_NAMES[3], remove_bad_rows), axis=1)
    # display_df['Text'] = display_df['Text'].str[:50] + \
    #                             (display_df['Text'].str.len() > 50).map({True: '...', False: ''})
    display_df['NAR'] = display_df['Text'].apply(
        lambda x: '✅' if re.search(r'\[1]|\[2]', str(x)) else ''
    )
    display_df['2GE'] = display_df['Text'].apply(
        lambda x: '✅' if re.search(r'\$[Gg]', str(x)) else ''
    )
    
    display_df['Fix'] = f'<span class="fix-button-cell">Fix</span>'

    if not is_turtle(tts_processor):
        display_df, has_duplicates = add_duplicate_marker_gossip(display_df)
        if has_duplicates:
            display_df.loc[display_df['DUPL'].str.contains('⚠️', na=False), 'Fix'] = f'<span class="fix-button-cell">REM DUP</span>'

    if is_turtle(tts_processor):
        display_df['Fix'] = display_df.apply(check_fix_button_turtle(), axis=1)
    else:
        display_df['Fix'] = display_df.apply(check_fix_button(), axis=1)
    
    display_df['voice_type'] = display_df['voice_type'].apply(extract_voice_type)

    if is_turtle(tts_processor):
        display_df = display_df[['npc_name', 'npc', 'voice_name', 'voice_type', 'Text', 'hash', 'Exp', 'zone', 'ORG', 'VAN', 'TRT', 'NAR', '2GE', 'Fix']]
    else:
        if has_duplicates:
            display_df = display_df[['npc_name', 'npc', 'voice_name', 'voice_type', 'Text', 'hash', 'Exp', 'zone', 'ORG', 'VAN', 'TBC', 'WLK', 'NAR', '2GE', 'Fix', 'DUPL']]
        else:
            display_df = display_df[['npc_name', 'npc', 'voice_name', 'voice_type', 'Text', 'hash', 'Exp', 'zone', 'ORG', 'VAN', 'TBC', 'WLK', 'NAR', '2GE', 'Fix']]

    progress(1.0, desc="All gossips loaded!")
    
    return display_df, f"### 🗺️ Gossips in: {selected_zone}", gr.update(visible=True), f"📦 ZIP Gossips for {selected_zone}", gr.update(visible=True)

def zip_generated_ttszone_gossip(df, selected_zone, zip_prefix):
    import zipfile

    display_df = df.copy()
    text_types = ['gossip']
    display_df = display_df[display_df['source'].isin(text_types)]
    display_df = display_df[display_df['zone'] == selected_zone]
    gossip_names = set(display_df['templateText_race_gender_hash'].astype(str))

    # Create zip filename
    safe_zone_name = selected_zone.replace(" ", "_").replace("/", "_")
    zip_filename = f"archives/{safe_zone_name}-gossip{zip_prefix}.zip"
    zip_path = Path(zip_filename)
    
    print(f"📦 Creating zip: {zip_path}")

    # Track files to remove after zipping
    files_to_remove = []
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        files_added = 0
        base_dir = Path.cwd()  # Current working directory
        
        for module in MODULE_NAMES:
            module_path = Path(module)
            quests_path = module_path / "generated/sounds/gossip"
            
            if quests_path.exists():
                print(f"  Scanning: {quests_path}")
                
                for file_path in quests_path.rglob("*.mp3"):  # Only mp3 files
                    if file_path.is_file():
                        filename = file_path.name
                        # Check if filename contains any quest ID from this zone
                        # Filename format examples: "12345-accept.mp3", "m-12345-accept.mp3", "12345-complete.mp3"
                        matched = False
                        for gossip_name in gossip_names:
                            if f"{gossip_name}-" in filename or f"-{gossip_name}-" in filename or filename.startswith(f"{gossip_name}"):
                                matched = True
                                break
                        
                        if matched:
                            relative_path = os.path.relpath(file_path, start=base_dir)
                            zipf.write(file_path, relative_path)
                            files_added += 1
                            print(f"    Added: {relative_path}")

    
    print(f"✅ Zip created: {zip_path} with {files_added} files")
    # print(f"🗑️ Removed {removed_count} files")
    gr.Info(f"Zone zip created: {zip_filename} with {files_added} files")
    
    return str(zip_path), gr.update(visible=True)

def select_columns_ttszone_full(df, selected_zone, remove_bad='', progress=gr.Progress()):

    def generate_text_progress_bar(pct, length=10):
        """Генерирует визуальный прогресс-бар из символов на основе процента."""
        if pct == 0:
            return "░" * length
        
        # Считаем, сколько блоков должно быть закрашено
        filled_length = int(round(length * pct / 100))
        # Защита, чтобы при 99.9% не округлялось до полного бара, если еще есть хвосты
        if filled_length == length and pct < 100:
            filled_length = length - 1
            
        bar = "█" * filled_length + "░" * (length - filled_length)
        return bar

    def calculate_quest_progress(display_df):
        import pandas as pd
        """
        Считает прогресс модуля, фильтруя строки датафрейма 
        по соответствующему значению в колонке 'Exp'.
        """
        all_modules = ["ORG", "VAN", "VNL", "TRT", "TBC", "WLK"]
        summary_data = []
        
        # Карта соответствия: какое значение в 'Exp' относится к какому модулю
        # Для Ваниллы проверяем обе колонки (ORG и VAN) на основе строк 'Vanilla'
        exp_filters = {
            'ORG': 'Vanilla',
            'VAN': 'Vanilla',
            'TRT': 'Turtle',
            'TBC': 'TBC',
            'WLK': 'WoTLK'
        }
        
        for mod in all_modules:
            if "Exp" not in display_df.columns:
                summary_data.append({
                    "Module": mod,
                    "Total": 0,
                    "Done": 0,
                    "Missing": 0,
                    "Removed": 0,
                    "Progress_%": 0.0,
                    "Status": "🚫 Not needed"
                })
                continue
                
            # # 1. Берем целевое значение Exp для текущего модуля
            # target_exp = exp_filters[mod]
            
            # # 2. Фильтруем датафрейм: берем ТОЛЬКО те строки, которые относятся к этому Exp
            # sub_df = display_df[display_df["Exp"].str.strip() == target_exp]
            # 1. Фильтруем строки датафрейма под конкретный Exp
            if mod == "VNL":
                sub_df = display_df[display_df["Exp"].str.strip() == "Vanilla"]
                target_cols = [c for c in ["ORG", "VAN"] if c in display_df.columns]
            elif mod == "TRT":
                sub_df = display_df[display_df["Exp"].str.strip() == "Turtle"]
                target_cols = ["TRT"] if "TRT" in display_df.columns else []
            elif mod == "ORG":
                sub_df = display_df[display_df["Exp"].str.strip() == "Vanilla"]
                target_cols = ["ORG"] if "ORG" in display_df.columns else []
            elif mod == "VAN":
                sub_df = display_df[display_df["Exp"].str.strip() == "Vanilla"]
                target_cols = ["VAN"] if "VAN" in display_df.columns else []
            elif mod == "TBC":
                sub_df = display_df[display_df["Exp"].str.strip() == "TBC"]
                target_cols = ["TBC"] if "TBC" in display_df.columns else []
            elif mod == "WLK":
                sub_df = display_df[display_df["Exp"].str.strip() == "WoTLK"]
                target_cols = ["WLK"] if "WLK" in display_df.columns else []
            
            # Если в этой зоне вообще нет квестов для этого дополнения
            if sub_df.empty or not target_cols:
                summary_data.append({
                    "Module": mod,
                    "Total": 0,
                    "Done": 0,
                    "Missing": 0,
                    "Removed": 0,
                    "Progress_%": 0.0,
                    "Status": "🚫 Not needed"
                })
                continue
                
            # 3. Считаем иконки только в отфильтрованных строках
            # counts = sub_df[mod].value_counts()
            
            # done_global = counts.get("✅", 0)
            # done_by_me = counts.get("☑️", 0)
            # missing_vo = counts.get("❌", 0)

            done_global = 0
            done_by_me = 0
            missing_vo = 0
            removed_vo = 0

            for col in target_cols:
                counts = sub_df[col].value_counts()
                done_global += counts.get("✅", 0)
                done_by_me += counts.get("☑️", 0)
                missing_vo += counts.get("❌", 0)
                removed_vo += counts.get("🗑️ 👎", 0)

            
            # Общее количество строк этого дополнения в зоне (игнорируя внутренние 🚫)
            total_valid = done_global + done_by_me + missing_vo + removed_vo
            total_done = done_global + done_by_me
            
            progress_pct = (total_done / total_valid * 100) if total_valid > 0 else 0.0
            
            summary_data.append({
                "Module": mod,
                "Total": total_valid,
                "Done": total_done,
                "Missing": missing_vo,
                "Removed": removed_vo,
                "Progress_%": round(progress_pct, 2),
                "Status": "Active"
            })
            
        return pd.DataFrame(summary_data)
    
    def create_combined_ui_table(quest_progress_df, gossip_progress_df):
        import pandas as pd
        modules = ["ORG", "VAN", "VNL", "TRT", "TBC", "WLK"]
        combined_rows = []
        
        for df_type, df in [("Quests", quest_progress_df), ("Gossips", gossip_progress_df)]:
            row_data = {"Тип": df_type}
            
            for mod in modules:
                mod_data = df[df["Module"] == mod]
                
                if not mod_data.empty:
                    status = mod_data.iloc[0].get("Status", "Active")
                    
                    if status == "🚫 Not needed":
                        row_data[mod] = "🚫 Not needed"
                    else:
                        total = mod_data.iloc[0]["Total"]
                        done = mod_data.iloc[0]["Done"]
                        pct = mod_data.iloc[0]["Progress_%"]
                        rmv = mod_data.iloc[0]["Removed"]

                        if total > 0:
                            # Генерируем мини-бар (длиной в 8 символов, чтобы не раздувать таблицу)
                            bar = generate_text_progress_bar(pct, length=8)
                            # Собираем красивую строку: [████░░░░] 50% (50/100)
                            row_data[mod] = f"[{bar}] {pct}%\n({done}/{total}{f' Removed: {rmv}' if rmv else ''})"
                        else:
                            row_data[mod] = "[░░░░░░░░] 0%\n(0/0)"
                else:
                    row_data[mod] = "🚫 No data"
                    
            combined_rows.append(row_data)
            
        return pd.DataFrame(combined_rows)
    
    remove_bad_rows = False
    if remove_bad == 'remove_bad':
        remove_bad_rows = True
    
    display_df_quests = select_columns_ttszone(df, selected_zone, remove_bad_rows, True)[0]
    prog_df_quests = calculate_quest_progress(display_df_quests)
    display_df_gossip = select_columns_ttszone_gossip(df, selected_zone, remove_bad_rows, True)[0]
    prog_df_gossip = calculate_quest_progress(display_df_gossip)

    display_df = create_combined_ui_table(prog_df_quests, prog_df_gossip)

    return display_df, f"### 🗺️ VoiceOver in: {selected_zone}"

def on_cell_select(evt: gr.SelectData, df):
    copied_val = str(evt.value)
    row_idx = evt.index[0]
    col_idx = evt.index[1]
    col_name = df.columns[col_idx] if col_idx < len(df.columns) else None
    gr.Info(f"Copied: {copied_val}")

    if col_name == 'quest':
        # Возвращаем:
        # - значение для копирования
        # - само значение в поле quest_id
        # - переключение вкладки на "tab_editor"
        return copied_val, gr.update(value=evt.value), gr.skip(), gr.Tabs(selected="ttsquest")

    if col_name == 'npc_name':
        # Возвращаем:
        # - значение для копирования
        # - само значение в поле quest_id
        # - переключение вкладки на "tab_editor"
        return copied_val, gr.skip(), gr.update(value=evt.value), gr.Tabs(selected="ttsgossip")

    # Если кликнули на другую колонку, просто копируем, не переключая вкладку
    return copied_val, gr.skip(), gr.skip(), gr.skip()

def process_row_ttszone_quest(df_orig, df, row_idx):
    quest = df.iloc[row_idx]
    quest_id = quest['quest']
    need_fix = False
    silence_found = False
    generate_fix = False

    # check columns to see what to fix
    quest_expansion = quest['Exp']

    def yes_mark(marking):
        return marking =='✅' or marking == '☑️'

    # Vanilla
    if quest_expansion == 'Vanilla':
        if df.at[row_idx, 'ORG'] != '✅':
            if not yes_mark(df.at[row_idx, 'VAN']):
                print(f"issue found in {quest_expansion} quest #{quest_id}")
                if '🔇' in df.at[row_idx, 'VAN']:
                    print("silence detected!")
                    need_fix = True
                    silence_found = True
                if df.at[row_idx, 'VAN'] == '❌':
                    print("voice missing!")
                    need_fix = True
        if df.at[row_idx, 'ORG'] == '✅' or yes_mark(df.at[row_idx, 'VAN']):
            if 'VD' in df.columns and df.at[row_idx, 'VD'] == '⚠️' and not yes_mark(df.at[row_idx, 'WLK']):
                print(f"issue found in {quest_expansion} quest #{quest_id}")
                print("wotlk different text missing!")
                need_fix = True
                # fake expansion
                quest_expansion = 'WoTLK'
    
    if quest_expansion == 'TBC':
        if not yes_mark(df.at[row_idx, 'TBC']):
            print(f"issue found in {quest_expansion} quest #{quest_id}")
            if '🔇' in df.at[row_idx, 'TBC']:
                print("silence detected!")
                need_fix = True
                silence_found = True
            if df.at[row_idx, 'TBC'] == '❌':
                print("voice missing!")
                need_fix = True

    if quest_expansion == 'WoTLK':
        if not yes_mark(df.at[row_idx, 'WLK']):
            print(f"issue found in {quest_expansion} quest #{quest_id}")
            if '🔇' in df.at[row_idx, 'WLK']:
                print("silence detected!")
                need_fix = True
                silence_found = True
            if df.at[row_idx, 'WLK'] == '❌':
                print("voice missing!")
                need_fix = True

    # Turtle TODO any custom
    if tts_processor.get_module_name() not in MODULE_NAMES:
        if quest_expansion == get_module_short_name(tts_processor.get_module_name()):
            if not yes_mark(df.at[row_idx, 'TRT']):
                print(f"issue found in {quest_expansion} quest #{quest_id}")
                if '🔇' in df.at[row_idx, 'TRT']:
                    print("silence detected!")
                    need_fix = True
                    silence_found = True
                if df.at[row_idx, 'TRT'] == '❌':
                    print("voice missing!")
                    need_fix = True

    # Handle different vanilla vs wotlk texts
    # Vanilla
    if quest_expansion == 'Vanilla':
        if df.at[row_idx, 'ORG'] == '✅' or yes_mark(df.at[row_idx, 'VAN']):
            if 'VD' in df.columns and df.at[row_idx, 'VD'] == '⚠️' and not yes_mark(df.at[row_idx, 'WLK']):
                print(f"issue found in {quest_expansion} quest #{quest_id}")
                print("wotlk different text missing!")
                need_fix = True
                # fake expansion
                quest_expansion = 'WoTLK'
                
    if need_fix:
        # replace to use Extra module for gen
        if quest_expansion == 'Vanilla':
            quest_expansion = 'VanillaExtra'

        # skip narrator for omni voice FIXED
        # if df.at[row_idx, 'NAR'] != "":
        #     return df, True, False

        # try use game samples only for turtle for consistent results
        # if is_turtle() and quest['expansion'] > 0:
        #     voice_sample_game = find_voice_sample(quest['voice_name'], quest['voice_type'])
        #     if voice_sample_game.get('FromGame'):
        #         voice_sample = voice_sample_game.get('FromGame')

        voice_sample = get_existing_sample_for_npc(df_orig, quest['npc_name'], quest['npc'])
        # if silence_found and not voice_sample:
        #     audio = find_quest_audio(quest['quest'], quest['source'], get_module_long_name(quest_expansion))
        #     if audio and audio.get(get_module_long_name(quest_expansion)):
        #         trimmed_silence = tts_processor.trim_long_silence(audio.get(get_module_long_name(quest_expansion)))
        #         voice_sample = trimmed_silence

        if silence_found:
            audio = find_quest_audio(quest['quest'], quest['source'], get_module_long_name(quest_expansion))
            output_folder = os.path.join(get_module_long_name(quest_expansion), 'generated/sounds/')
            sound_output_folder = os.path.join(output_folder, 'quests')
            os.makedirs(sound_output_folder, exist_ok=True)
            filename = os.path.basename(audio.get(get_module_long_name(quest_expansion)))
            dest_file = Path(sound_output_folder) / f"{filename}"
            trimmed_silence = tts_processor.remove_long_silence_ffmpeg(audio.get(get_module_long_name(quest_expansion)), dest_file)
            check_silence = tts_processor.check_for_long_silence_ffmpeg(audio.get(get_module_long_name(quest_expansion)))
            if not check_silence:
                generate_fix = True
                msg = f"Silence removed from file {audio.get(get_module_long_name(quest_expansion))}"
            else:
                audio = find_quest_audio(quest['quest'], quest['source'], get_module_long_name(quest_expansion))
                if audio and audio.get('get_module_long_name(quest_expansion)'):
                    trimmed_silence = tts_processor.trim_long_silence(audio.get(get_module_long_name(quest_expansion)))
                    voice_sample = trimmed_silence

        if not generate_fix:
            generate_fix, path, msg = tts_processor.tts_quest_auto(df_orig, quest_id, [quest['source']], get_module_long_name(quest_expansion), voice_sample)

        if generate_fix == True:
            if quest_expansion == 'VanillaExtra':
                df.at[row_idx, 'VAN'] = f"✅"
            elif quest_expansion == 'TBC':
                df.at[row_idx, 'TBC'] = f"✅"
            elif quest_expansion == 'WoTLK':
                df.at[row_idx, 'WLK'] = f"✅"
            else:
                df.at[row_idx, 'TRT'] = f"✅"

            df.at[row_idx, 'Fix'] = ''

            gr.Info(msg)

            # Remove bad VO marking if it exists
            key = f"{quest['quest']}_{quest['source']}"
            if tts_processor.should_regenerate_quest(quest['quest'], quest['source']):
                if key in tts_processor.problematic_quests:
                    tts_processor.problematic_quests[key]["attempts"] += 1
                # tts_processor.mark_as_fixed(quest['quest'], quest['source'])

    return df, need_fix, generate_fix

def process_row_ttszone_gossip(df_orig, df, row_idx):
    npc = df.iloc[row_idx]
    npc_name = npc['npc_name']
    text_hash = npc['hash']
    need_fix = False
    silence_found = False
    generate_fix = False

    # check columns to see what to fix
    npc_expansion = npc['Exp']

    df_filtered_check = df_orig[(df_orig['name'] == npc_name) & (df_orig['expansion'] >= 0) & (df_orig['source'] == 'gossip') & (df_orig['templateText_race_gender_hash'] == text_hash)]
    base_row = df_filtered_check.iloc[0]
    is_vanilla_text = is_vanilla_broadcast_text(base_row["broadcast_text_id"], base_row["text"])
    is_tbc_text = not is_vanilla_text and is_tbc_broadcast_text(base_row["broadcast_text_id"], base_row["text"])
    is_wotlk_text = not is_vanilla_text and not is_tbc_text

    def get_short_name_from_module(module_name):
        mapping = {
            'AI_VoiceOverData_Vanilla': 'ORG',
            'AI_VoiceOverData_VanillaExtra': 'VAN',
            'AI_VoiceOverData_TBC': 'TBC',
            'AI_VoiceOverData_WoTLK': 'WLK'
        }
        return mapping.get(module_name, module_name)
    
    def no_mark(marking):
        return marking =='❌' or marking == '🚫'
    
    def yes_mark(marking):
        return marking =='✅' or marking == '☑️'
    
    def check_fix_button():

        def check(row):

            df_filtered_check = df_orig[(df_orig['name'] == npc_name) & (df_orig['expansion'] >= 0) & (df_orig['source'] == 'gossip') & (df_orig['templateText_race_gender_hash'] == text_hash)]
            base_row = df_filtered_check.iloc[0]
            is_vanilla_text = is_vanilla_broadcast_text(base_row["broadcast_text_id"], base_row["text"])
            is_tbc_text = not is_vanilla_text and is_tbc_broadcast_text(base_row["broadcast_text_id"], base_row["text"])
            is_wotlk_text = not is_vanilla_text and not is_tbc_text

            if row['Exp'] == 'Vanilla':
                if row['ORG'] == '✅' and '✅' in row['TBC'] or '✅' in row['WLK']:
                     return '<span class="fix-button-cell">REM DUP</span>'
                elif row['ORG'] == '✅':
                    return ''
                elif row['VAN'] == '✅':
                    return ''
                elif row['VAN'] == '☑️':
                    return ''
                elif row['VAN'] == '☑️♻️':
                    return ''
                
            if row['Exp'] == 'TBC':
                if row['TBC'] == '✅' and '✅' in row['WLK'] or '✅' in row['ORG'] or '✅' in row['VAN']:
                     return '<span class="fix-button-cell">REM DUP</span>'
                if row['TBC'] == '✅':
                    return ''
                if row['TBC'] == '☑️':
                    return ''
                if row['TBC'] == '☑️♻️':
                    return ''
                
            if row['Exp'] == 'WoTLK':
                if row['WLK'] == '✅' and '✅' in row['TBC'] or '✅' in row['ORG'] or '✅' in row['VAN']:
                     return '<span class="fix-button-cell">REM DUP</span>'
                if row['WLK'] == '✅':
                    return ''
                if row['WLK'] == '☑️':
                    return ''
                if row['WLK'] == '☑️♻️':
                    return ''
                
            # if narrator is voice and narrator in text...no auto fix
            # if row['NAR'] == '✅':
            #     if row['Fix'] != '':
            #         return f'<span class="fix-button-cell-grey">MANUAL</span>'

            # if vanilla text doesn't match other modules (if we want to copy missing)
            if row['Exp'] == 'Vanilla':
                if no_mark(row['ORG']) and no_mark(row['VAN']):
                    if row['TBC'] == '✅' or row['WLK'] == '✅':
                        if row['Fix'] != '<span class="fix-button-cell">REM DUP</span>':
                            if not is_vanilla_text:
                                return f'<span class="fix-button-cell-grey">NOT VAN</span>'
                        
            if row['Exp'] == 'TBC':
                if no_mark(row['TBC']):
                    if row['WLK'] == '✅':
                        if row['Fix'] != '<span class="fix-button-cell">REM DUP</span>':
                            if not is_tbc_text:
                                return f'<span class="fix-button-cell-grey">NOT TBC</span>'
                
            return row['Fix']
        
        return check

    if df.at[row_idx, 'Fix'] == '<span class="fix-button-cell">REM DUP</span>':
        # remove dupes first
        module_del_from = None

        # remove VanillaExtra if exists original
        if npc['ORG'] == '✅' and npc['VAN'] == '✅':
            module_del_from = get_module_long_name('VanillaExtra')

        # Remove wotlk if exists original
        if npc['ORG'] == '✅' and npc['WLK'] == '✅':
            if is_vanilla_text:
                module_del_from = get_module_long_name('WoTLK')
        
        # Remove tbc if exists original
        if npc['ORG'] == '✅' and npc['TBC'] == '✅':
            if is_vanilla_text:
                module_del_from = get_module_long_name('TBC')
        
        # Remove wotlk if exists tbc
        if (npc['TBC'] == '✅' or npc['TBC'] == '☑️') and '✅' in npc['WLK']:
            if is_tbc_text and '🔇' not in npc['TBC']:
                module_del_from = get_module_long_name('WoTLK')
            elif not is_tbc_text and not is_vanilla_text:
                if '🔇' in npc['WLK']:
                    module_del_from = get_module_long_name('TBC')

            
        if module_del_from:
            audio = find_gossip_audio(npc['hash'], module_del_from)
            file_src = audio.get(module_del_from)
            filename = os.path.basename(file_src)
            file_to_remove = Path(ADDON_PATH_WORKS) / module_del_from / "generated/sounds/gossip" / filename
            if file_to_remove.exists():
                file_to_remove.unlink()
                print(f"removed {file_to_remove}")
                msg = f"Removed duplicate file: {filename} from {module_del_from}"
                df.at[row_idx, get_short_name_from_module(module_del_from)] = f"✖️"
                df['Fix'] = df.apply(check_fix_button(), axis=1)
                gr.Info(msg)
                return df
            
        gr.Info("failed to remove duplicate, use manual fix!")
        return df

    # Vanilla
    if npc_expansion == 'Vanilla':
        if df.at[row_idx, 'ORG'] != '✅':
            if not yes_mark(df.at[row_idx, 'VAN']):
                print(f"issue found in {npc_expansion} npc <{npc_name}>")
                if '🔇' in df.at[row_idx, 'VAN']:
                    print("silence detected!")
                    need_fix = True
                    silence_found = True
                if no_mark(df.at[row_idx, 'VAN']):
                    print("voice missing!")
                    need_fix = True
    
    if npc_expansion == 'TBC':
        if not yes_mark(df.at[row_idx, 'TBC']):
            print(f"issue found in {npc_expansion} npc <{npc_name}>")
            if '🔇' in df.at[row_idx, 'TBC']:
                print("silence detected!")
                need_fix = True
                silence_found = True
            if no_mark(df.at[row_idx, 'TBC']):
                print("voice missing!")
                need_fix = True

    if npc_expansion == 'WoTLK':
        if not yes_mark(df.at[row_idx, 'WLK']):
            print(f"issue found in {npc_expansion} npc <{npc_name}>")
            if '🔇' in df.at[row_idx, 'WLK']:
                print("silence detected!")
                need_fix = True
                silence_found = True
            if no_mark(df.at[row_idx, 'WLK']):
                print("voice missing!")
                need_fix = True

    # Turtle TODO any custom
    if tts_processor.get_module_name() not in MODULE_NAMES:
        if npc_expansion == get_module_short_name(tts_processor.get_module_name()):
            if not yes_mark(df.at[row_idx, 'TRT']):
                print(f"issue found in {npc_expansion} npc <{npc_name}>")
                if '🔇' in df.at[row_idx, 'TRT']:
                    print("silence detected!")
                    need_fix = True
                    silence_found = True
                if df.at[row_idx, 'TRT'] == '❌':
                    print("voice missing!")
                    need_fix = True
                # let generate custom voices even if text is vanilla
                if df.at[row_idx, 'TRT'] == '🚫':
                    if df.at[row_idx, 'VAN'] == '❌' or df.at[row_idx, 'ORG'] == '❌':
                        print("voice missing!")
                        need_fix = True
                
    if need_fix:
        generate_fix = False
        # replace to use Extra module for gen
        if npc_expansion == 'Vanilla':
            npc_expansion = 'VanillaExtra'

        # skip narrator for omni voice FIXED
        # if df.at[row_idx, "NAR"] != "":
        #     return df, True, False

        # let generate if guessed expansion and no voice lines
        if df.at[row_idx, 'Fix'] == '<span class="fix-button-cell">WRN EXP</span>':
            if is_vanilla_text:
                npc_expansion = 'VanillaExtra'
            elif is_tbc_text:
                npc_expansion = 'TBC'
            elif is_wotlk_text:
                npc_expansion = "WoTLK"

        voice_sample = get_existing_sample_for_npc(df_orig, npc_name, npc['npc'])
        if silence_found and not voice_sample:
            audio = find_gossip_audio(npc['hash'], get_module_long_name(npc_expansion))
            if audio and audio.get(get_module_long_name(npc_expansion)):
                trimmed_silence = tts_processor.trim_long_silence(audio.get(get_module_long_name(npc_expansion)))
                voice_sample = trimmed_silence

        if silence_found:
            audio = find_gossip_audio(npc['hash'], get_module_long_name(npc_expansion))
            output_folder = os.path.join(get_module_long_name(npc_expansion), 'generated/sounds/')
            sound_output_folder = os.path.join(output_folder, 'gossip')
            os.makedirs(sound_output_folder, exist_ok=True)
            filename = os.path.basename(audio.get(get_module_long_name(npc_expansion)))
            dest_file = Path(sound_output_folder) / f"{filename}"
            trimmed_silence = tts_processor.remove_long_silence_ffmpeg(find_gossip_audio(npc['hash'], get_module_long_name(npc_expansion)), dest_file)
            check_silence = tts_processor.check_for_long_silence_ffmpeg(dest_file)
            if not check_silence:
                generate_fix = True
                msg = f"Silence removed from file {audio.get(get_module_long_name(npc_expansion))}"
            else:
                msg = f"Failed to remove silence from file {audio.get(get_module_long_name(npc_expansion))}"
                audio = find_gossip_audio(npc['hash'], get_module_long_name(npc_expansion))
                if audio and audio.get(get_module_long_name(npc_expansion)):
                    trimmed_silence = tts_processor.trim_long_silence(audio.get(get_module_long_name(npc_expansion)))
                    voice_sample = trimmed_silence

        # check if exists in other expansion
        if not generate_fix and (npc_expansion == 'VanillaExtra' or npc_expansion == 'TBC'):
            module_copy_from = None
            proper_module = False

            if npc_expansion == 'VanillaExtra':
                proper_module = is_vanilla_text
            if npc_expansion == 'TBC':
                proper_module = is_tbc_text

            if proper_module:
                if npc_expansion == 'VanillaExtra' and df.at[row_idx, 'TBC'] == '✅':
                    if is_vanilla_text:
                        module_copy_from = get_module_long_name('TBC')
                elif df.at[row_idx, 'WLK'] == '✅':
                    module_copy_from = get_module_long_name('WoTLK')

            if module_copy_from:
                audio = find_gossip_audio(npc['hash'], module_copy_from)
                file_src = audio.get(module_copy_from)
                output_folder = os.path.join(get_module_long_name(npc_expansion), 'generated/sounds/')
                sound_output_folder = os.path.join(output_folder, 'gossip')
                os.makedirs(sound_output_folder, exist_ok=True)
                filename = os.path.basename(file_src)
                dest_file = Path(sound_output_folder) / f"{filename}"
                shutil.copy2(file_src, dest_file)
                file_to_remove = Path(ADDON_PATH_WORKS) / module_copy_from / "generated/sounds/gossip" / filename
                if file_to_remove.exists():
                    file_to_remove.unlink()
                    df.at[row_idx, get_short_name_from_module(module_copy_from)] = '✖️'
                    print(f"removed {file_to_remove}")

                if npc['2GE'] == '✅':
                    audio = find_gossip_audio(npc['hash'], module_copy_from, 'female')
                    file_src = audio.get(module_copy_from)
                    filename = os.path.basename(file_src)
                    dest_file = Path(sound_output_folder) / f"{filename}"
                    shutil.copy2(file_src, dest_file)
                    file_to_remove = Path(ADDON_PATH_WORKS) / module_copy_from / "generated/sounds/gossip" / filename
                    if file_to_remove.exists():
                        file_to_remove.unlink()
                        print(f"removed {file_to_remove}")

                generate_fix = True
                msg = f"Copied to correct module: {filename} to {npc_expansion} from {module_copy_from}"

        if not generate_fix:
            generate_fix, path, msg = tts_processor.tts_gossip_auto(df_orig, npc_name, get_module_long_name(npc_expansion), npc['hash'], voice_sample)

        if generate_fix == True:
            if npc_expansion == 'VanillaExtra':
                df.at[row_idx, 'VAN'] = f"☑️"
            if npc_expansion == 'TBC':
                df.at[row_idx, 'TBC'] = f"☑️"
            if npc_expansion == 'WoTLK':
                df.at[row_idx, 'WLK'] = f"☑️"

            df['Fix'] = df.apply(check_fix_button(), axis=1)

            gr.Info(msg)

            key = f"{npc_name}_{text_hash}"
            if tts_processor.should_regenerate_gossip(npc_name, text_hash):
                if key in tts_processor.problematic_gossip:
                    tts_processor.problematic_gossip[key]["attempts"] += 1

            # gr.Info(play_audio_with_controls(path))

    
    return df, need_fix, generate_fix

def process_row_ttszone_gossip_turtle(df_orig, df, row_idx):
    npc = df.iloc[row_idx]
    npc_name = npc['npc_name']
    text_hash = npc['hash']
    need_fix = False
    silence_found = False
    generate_fix = False

    # check columns to see what to fix
    npc_expansion = npc['Exp']

    df_filtered_check = df_orig[(df_orig['name'] == npc_name) & (df_orig['expansion'] >= 0) & (df_orig['source'] == 'gossip') & (df_orig['templateText_race_gender_hash'] == text_hash)]
    base_row = df_filtered_check.iloc[0]
    is_vanilla_text = is_vanilla_broadcast_text(base_row["broadcast_text_id"], base_row["text"])
    is_turtle_text = not is_vanilla_text
    
    def no_mark(marking):
        return marking =='❌' or marking == '🚫'
    
    def check_fix_button():

        def check(row):

            df_filtered_check = df_orig[(df_orig['name'] == npc_name) & (df_orig['expansion'] >= 0) & (df_orig['source'] == 'gossip') & (df_orig['templateText_race_gender_hash'] == text_hash)]
            base_row = df_filtered_check.iloc[0]
            is_vanilla_text = is_vanilla_broadcast_text(base_row["broadcast_text_id"], base_row["text"])
            is_turtle_text = not is_vanilla_text

            if row['Exp'] == 'Vanilla':
                if row['ORG'] == '✅':
                    return ''
                elif row['VAN'] == '✅':
                    return ''
                elif row['VAN'] == '☑️':
                    return ''
                elif row['VAN'] == '☑️♻️':
                    return ''
                
            if row['Exp'] == 'Turtle':
                if row['TRT'] == '✅':
                    return ''
                if row['TRT'] == '☑️':
                    return ''
                if row['TRT'] == '☑️♻️':
                    return ''
                
            # if narrator is voice and narrator in text...no auto fix
            # if row['NAR'] == '✅':
            #     if row['Fix'] != '':
            #         return f'<span class="fix-button-cell-grey">MANUAL</span>'
                
            return row['Fix']
        
        return check

    # Vanilla
    if npc_expansion == 'Vanilla':
        if df.at[row_idx, 'ORG'] != '✅':
            if df.at[row_idx, 'VAN'] != '✅':
                print(f"issue found in {npc_expansion} npc <{npc_name}>")
                if '🔇' in df.at[row_idx, 'VAN']:
                    print("silence detected!")
                    need_fix = True
                    silence_found = True
                if no_mark(df.at[row_idx, 'VAN']):
                    print("voice missing!")
                    need_fix = True

    # Turtle TODO any custom
    if tts_processor.get_module_name() not in MODULE_NAMES:
        if npc_expansion == get_module_short_name(tts_processor.get_module_name()):
            if df.at[row_idx, 'TRT'] != '✅' and df.at[row_idx, 'TRT'] != '☑️':
                print(f"issue found in {npc_expansion} npc <{npc_name}>")
                if '🔇' in df.at[row_idx, 'TRT']:
                    print("silence detected!")
                    need_fix = True
                    silence_found = True
                if df.at[row_idx, 'TRT'] == '❌':
                    print("voice missing!")
                    need_fix = True
                # let generate custom voices even if text is vanilla
                if df.at[row_idx, 'TRT'] == '🚫':
                    if df.at[row_idx, 'VAN'] == '❌' or df.at[row_idx, 'ORG'] == '❌':
                        print("voice missing!")
                        need_fix = True
                
    if need_fix:
        generate_fix = False
        # replace to use Extra module for gen
        if npc_expansion == 'Vanilla':
            npc_expansion = 'VanillaExtra'

        voice_sample = get_existing_sample_for_npc(df_orig, npc_name, npc['npc'])
        if silence_found and not voice_sample:
            audio = find_gossip_audio(npc['hash'], get_module_long_name(npc_expansion))
            output_folder = os.path.join(get_module_long_name(npc_expansion), 'generated/sounds/')
            sound_output_folder = os.path.join(output_folder, 'gossip')
            os.makedirs(sound_output_folder, exist_ok=True)
            filename = os.path.basename(audio.get(get_module_long_name(npc_expansion)))
            dest_file = Path(sound_output_folder) / f"{filename}"
            trimmed_silence = tts_processor.remove_long_silence_ffmpeg(find_gossip_audio(npc['hash'], get_module_long_name(npc_expansion)), dest_file)
            check_silence = tts_processor.check_for_long_silence_ffmpeg(dest_file)
            if not check_silence:
                generate_fix = True
                msg = f"Silence removed from file {audio.get(get_module_long_name(npc_expansion))}"
            else:
                msg = f"Failed to remove silence from file {audio.get(get_module_long_name(npc_expansion))}"
                audio = find_gossip_audio(npc['hash'], get_module_long_name(npc_expansion))
                if audio and audio.get(get_module_long_name(npc_expansion)):
                    trimmed_silence = tts_processor.trim_long_silence(audio.get(get_module_long_name(npc_expansion)))
                    voice_sample = trimmed_silence

        if not generate_fix:
            generate_fix, path, msg = tts_processor.tts_gossip_auto(df_orig, npc_name, get_module_long_name(npc_expansion), npc['hash'], voice_sample)

        if generate_fix == True:
            if npc_expansion == 'VanillaExtra':
                df.at[row_idx, 'VAN'] = f"☑️"
            if npc_expansion == 'Turtle':
                df.at[row_idx, 'TRT'] = f"☑️"

            df['Fix'] = df.apply(check_fix_button(), axis=1)

            gr.Info(msg)

            key = f"{npc_name}_{text_hash}"
            if tts_processor.should_regenerate_gossip(npc_name, text_hash):
                if key in tts_processor.problematic_gossip:
                    tts_processor.problematic_gossip[key]["attempts"] += 1

    
    return df, need_fix, generate_fix

def on_cell_select_ttszone(evt: gr.SelectData, df_orig, df):
    """
    If clicked on a "Generate" column, trigger generation for that row
    """
    row_idx = evt.index[0]
    col_idx = evt.index[1]
    col_name = df.columns[col_idx] if col_idx < len(df.columns) else None
    print(f"Clicked column: {col_name}")

    # def get_module_name_from_column(short_name):
    #     reverse_mapping = {
    #         'ORG': 'AI_VoiceOverData_Vanilla',
    #         'VAN': 'AI_VoiceOverData_VanillaExtra',
    #         'TBC': 'AI_VoiceOverData_TBC',
    #         'WLK': 'AI_VoiceOverData_WoTLK',
    #         'TRT': 'AI_VoiceOverData_Turtle'
    #     }
    #     return reverse_mapping.get(short_name, short_name)
    
    # def play_audio_with_controls(audio_path):
    #     """
    #     Play audio with visible controls
    #     """
    #     abs_path = os.path.abspath(audio_path)
    #     url_path = abs_path.replace('\\', '/')
    #     html = f"""
    #     <div style="margin: 10px 0;">
    #         <audio controls autoplay style="width: 100%;">
    #             <source src="file:///{url_path}" type="audio/mpeg">
    #             Your browser does not support the audio element.
    #         </audio>
    #     </div>
    #     """
    #     return html
    
    # Different actions based on which column was clicked
    if col_name == "Fix":
        return process_row_ttszone_quest(df_orig, df, row_idx)[0], (gr.update(choices=
        ['AI_VoiceOverData_Turtle', 'AI_VoiceOverData_VanillaExtra',
        'Testing'],
        value='AI_VoiceOverData_Turtle') if is_turtle(tts_processor) else gr.skip()), gr.update(choices=
                                ['AI_VoiceOverData_Turtle', 'AI_VoiceOverData_VanillaExtra',
                                'Testing'],
                                value='AI_VoiceOverData_Turtle') if is_turtle(tts_processor) else gr.skip()
    
    return df, (gr.update(choices=
        ['AI_VoiceOverData_Turtle', 'AI_VoiceOverData_VanillaExtra',
        'Testing'],
        value='AI_VoiceOverData_Turtle') if is_turtle(tts_processor) else gr.skip()), gr.update(choices=
                                ['AI_VoiceOverData_Turtle', 'AI_VoiceOverData_VanillaExtra',
                                'Testing'],
                                value='AI_VoiceOverData_Turtle') if is_turtle(tts_processor) else gr.skip()

def on_cell_select_ttszone_gossip(evt: gr.SelectData, df_orig, df):
    """
    If clicked on a "Generate" column, trigger generation for that row
    """
    row_idx = evt.index[0]
    col_idx = evt.index[1]
    col_name = df.columns[col_idx] if col_idx < len(df.columns) else None
    print(f"Clicked column: {col_name}")

    if col_name == "Fix":
        if is_turtle(tts_processor):
            return process_row_ttszone_gossip_turtle(df_orig, df, row_idx)[0], (gr.update(choices=
            ['AI_VoiceOverData_Turtle', 'AI_VoiceOverData_VanillaExtra',
            'Testing'],
            value='AI_VoiceOverData_Turtle' if is_turtle(tts_processor) else gr.skip())), gr.update(choices=
                                    ['AI_VoiceOverData_Turtle', 'AI_VoiceOverData_VanillaExtra',
                                    'Testing'],
                                    value='AI_VoiceOverData_Turtle' if is_turtle(tts_processor) else gr.skip())
        else:
            return process_row_ttszone_gossip(df_orig, df, row_idx)[0], (gr.update(choices=
            ['AI_VoiceOverData_Turtle', 'AI_VoiceOverData_VanillaExtra',
            'Testing'],
            value='AI_VoiceOverData_Turtle' if is_turtle(tts_processor) else gr.skip())), gr.update(choices=
                                    ['AI_VoiceOverData_Turtle', 'AI_VoiceOverData_VanillaExtra',
                                    'Testing'],
                                    value='AI_VoiceOverData_Turtle' if is_turtle(tts_processor) else gr.skip())
    
    return df, (gr.update(choices=
        ['AI_VoiceOverData_Turtle', 'AI_VoiceOverData_VanillaExtra',
        'Testing'],
        value='AI_VoiceOverData_Turtle' if is_turtle(tts_processor) else gr.skip())), gr.update(choices=
                                ['AI_VoiceOverData_Turtle', 'AI_VoiceOverData_VanillaExtra',
                                'Testing'],
                                value='AI_VoiceOverData_Turtle' if is_turtle(tts_processor) else gr.skip())

    def get_module_name_from_column(short_name):
        reverse_mapping = {
            'ORG': 'AI_VoiceOverData_Vanilla',
            'VAN': 'AI_VoiceOverData_VanillaExtra',
            'TBC': 'AI_VoiceOverData_TBC',
            'WLK': 'AI_VoiceOverData_WoTLK'
        }
        return reverse_mapping.get(short_name, short_name)
    
    def get_short_name_from_module(module_name):
        mapping = {
            'AI_VoiceOverData_Vanilla': 'ORG',
            'AI_VoiceOverData_VanillaExtra': 'VAN',
            'AI_VoiceOverData_TBC': 'TBC',
            'AI_VoiceOverData_WoTLK': 'WLK'
        }
        return mapping.get(module_name, module_name)
    
    def no_mark(marking):
        return marking =='❌' or marking == '🚫'
    
    def check_fix_button():

        def check(row):

            df_filtered_check = df_orig[(df_orig['name'] == npc_name) & (df_orig['expansion'] >= 0) & (df_orig['source'] == 'gossip') & (df_orig['templateText_race_gender_hash'] == text_hash)]
            base_row = df_filtered_check.iloc[0]
            is_vanilla_text = is_vanilla_broadcast_text(base_row["broadcast_text_id"], base_row["text"])
            is_tbc_text = not is_vanilla_text and is_tbc_broadcast_text(base_row["broadcast_text_id"], base_row["text"])
            is_wotlk_text = not is_vanilla_text and not is_tbc_text

            if row['Exp'] == 'Vanilla':
                if row['ORG'] == '✅' and '✅' in row['TBC'] or '✅' in row['WLK']:
                     return '<span class="fix-button-cell">REM DUP</span>'
                elif row['ORG'] == '✅':
                    return ''
                elif row['VAN'] == '✅':
                    return ''
                elif row['VAN'] == '☑️':
                    return ''
                elif row['VAN'] == '☑️♻️':
                    return ''
                
            if row['Exp'] == 'TBC':
                if row['TBC'] == '✅' and '✅' in row['WLK'] or '✅' in row['ORG'] or '✅' in row['VAN']:
                     return '<span class="fix-button-cell">REM DUP</span>'
                if row['TBC'] == '✅':
                    return ''
                if row['TBC'] == '☑️':
                    return ''
                if row['TBC'] == '☑️♻️':
                    return ''
                
            if row['Exp'] == 'WoTLK':
                if row['WLK'] == '✅' and '✅' in row['TBC'] or '✅' in row['ORG'] or '✅' in row['VAN']:
                     return '<span class="fix-button-cell">REM DUP</span>'
                if row['WLK'] == '✅':
                    return ''
                if row['WLK'] == '☑️':
                    return ''
                if row['WLK'] == '☑️♻️':
                    return ''
                
            # if narrator is voice and narrator in text...no auto fix
            if row['NAR'] == '✅':
                if row['Fix'] != '':
                    return f'<span class="fix-button-cell-grey">MANUAL</span>'

            # if vanilla text doesn't match other modules (if we want to copy missing)
            if row['Exp'] == 'Vanilla':
                if no_mark(row['ORG']) and no_mark(row['VAN']):
                    if row['TBC'] == '✅' or row['WLK'] == '✅':
                        if row['Fix'] != '<span class="fix-button-cell">REM DUP</span>':
                            if not is_vanilla_text:
                                return f'<span class="fix-button-cell-grey">NOT VAN</span>'
                        
            if row['Exp'] == 'TBC':
                if no_mark(row['TBC']):
                    if row['WLK'] == '✅':
                        if row['Fix'] != '<span class="fix-button-cell">REM DUP</span>':
                            if not is_tbc_text:
                                return f'<span class="fix-button-cell-grey">NOT TBC</span>'
                
            return row['Fix']
        
        return check
    
    
    # Different actions based on which column was clicked
    if col_name == "Fix":
        # Generate for this row
        npc = df.iloc[row_idx]
        npc_name = npc['npc_name']
        text_hash = npc['hash']
        need_fix = False
        silence_found = False

        # check columns to see what to fix
        npc_expansion = npc['Exp']

        df_filtered_check = df_orig[(df_orig['name'] == npc_name) & (df_orig['expansion'] >= 0) & (df_orig['source'] == 'gossip') & (df_orig['templateText_race_gender_hash'] == text_hash)]
        base_row = df_filtered_check.iloc[0]
        is_vanilla_text = is_vanilla_broadcast_text(base_row["broadcast_text_id"], base_row["text"])
        is_tbc_text = not is_vanilla_text and is_tbc_broadcast_text(base_row["broadcast_text_id"], base_row["text"])
        is_wotlk_text = not is_vanilla_text and not is_tbc_text

        if df.at[row_idx, 'Fix'] == '<span class="fix-button-cell">REM DUP</span>':
            # remove dupes first
            module_del_from = None

            # remove VanillaExtra if exists original
            if npc['ORG'] == '✅' and npc['VAN'] == '✅':
                module_del_from = get_module_long_name('VanillaExtra')

            # Remove wotlk if exists original
            if npc['ORG'] == '✅' and npc['WLK'] == '✅':
                if is_vanilla_text:
                    module_del_from = get_module_long_name('WoTLK')
            
            # Remove tbc if exists original
            if npc['ORG'] == '✅' and npc['TBC'] == '✅':
                if is_vanilla_text:
                    module_del_from = get_module_long_name('TBC')
            
            # Remove wotlk if exists tbc
            if (npc['TBC'] == '✅' or npc['TBC'] == '☑️') and '✅' in npc['WLK']:
                if is_tbc_text and '🔇' not in npc['TBC']:
                    module_del_from = get_module_long_name('WoTLK')
                elif not is_tbc_text and not is_vanilla_text:
                    if '🔇' in npc['WLK']:
                        module_del_from = get_module_long_name('TBC')

                
            if module_del_from:
                audio = find_gossip_audio(npc['hash'], module_del_from)
                file_src = audio.get(module_del_from)
                filename = os.path.basename(file_src)
                file_to_remove = Path(ADDON_PATH_WORKS) / module_del_from / "generated/sounds/gossip" / filename
                if file_to_remove.exists():
                    file_to_remove.unlink()
                    print(f"removed {file_to_remove}")
                    msg = f"Removed duplicate file: {filename} from {module_del_from}"
                    df.at[row_idx, get_short_name_from_module(module_del_from)] = f"✖️"
                    df['Fix'] = df.apply(check_fix_button(), axis=1)
                    gr.Info(msg)
                    return df
                
            gr.Info("failed to remove duplicate, use manual fix!")
            return df

        # Vanilla
        if npc_expansion == 'Vanilla':
            if df.at[row_idx, 'ORG'] != '✅':
                if df.at[row_idx, 'VAN'] != '✅':
                    print(f"issue found in {npc_expansion} npc <{npc_name}>")
                    if '🔇' in df.at[row_idx, 'VAN']:
                        print("silence detected!")
                        need_fix = True
                        silence_found = True
                    if no_mark(df.at[row_idx, 'VAN']):
                        print("voice missing!")
                        need_fix = True
        
        if npc_expansion == 'TBC':
            if df.at[row_idx, 'TBC'] != '✅':
                print(f"issue found in {npc_expansion} npc <{npc_name}>")
                if '🔇' in df.at[row_idx, 'TBC']:
                    print("silence detected!")
                    need_fix = True
                    silence_found = True
                if no_mark(df.at[row_idx, 'TBC']):
                    print("voice missing!")
                    need_fix = True

        if npc_expansion == 'WoTLK':
            if df.at[row_idx, 'WLK'] != '✅':
                print(f"issue found in {npc_expansion} npc <{npc_name}>")
                if '🔇' in df.at[row_idx, 'WLK']:
                    print("silence detected!")
                    need_fix = True
                    silence_found = True
                if no_mark(df.at[row_idx, 'WLK']):
                    print("voice missing!")
                    need_fix = True
                    
        if need_fix:
            generate_fix = False
            # replace to use Extra module for gen
            if npc_expansion == 'Vanilla':
                npc_expansion = 'VanillaExtra'

            # let generate if guessed expansion and no voice lines
            if df.at[row_idx, 'Fix'] == '<span class="fix-button-cell">WRN EXP</span>':
                if is_vanilla_text:
                    npc_expansion = 'VanillaExtra'
                elif is_tbc_text:
                    npc_expansion = 'TBC'
                elif is_wotlk_text:
                    npc_expansion = "WoTLK"

            voice_sample = get_existing_sample_for_npc(df_orig, npc_name, npc['npc'])
            if silence_found and not voice_sample:
                audio = find_gossip_audio(npc['hash'], get_module_long_name(npc_expansion))
                if audio and audio.get(get_module_long_name(npc_expansion)):
                    trimmed_silence = tts_processor.trim_long_silence(audio.get(get_module_long_name(npc_expansion)))
                    voice_sample = trimmed_silence

            # check if exists in other expansion
            if npc_expansion == 'VanillaExtra' or npc_expansion == 'TBC':
                module_copy_from = None
                proper_module = False

                if npc_expansion == 'VanillaExtra':
                    proper_module = is_vanilla_text
                if npc_expansion == 'TBC':
                    proper_module = is_tbc_text

                if proper_module:
                    if npc_expansion == 'VanillaExtra' and df.at[row_idx, 'TBC'] == '✅':
                        if is_vanilla_text:
                            module_copy_from = get_module_long_name('TBC')
                    elif df.at[row_idx, 'WLK'] == '✅':
                        module_copy_from = get_module_long_name('WoTLK')

                if module_copy_from:
                    audio = find_gossip_audio(npc['hash'], module_copy_from)
                    file_src = audio.get(module_copy_from)
                    output_folder = os.path.join(get_module_long_name(npc_expansion), 'generated/sounds/')
                    sound_output_folder = os.path.join(output_folder, 'gossip')
                    os.makedirs(sound_output_folder, exist_ok=True)
                    filename = os.path.basename(file_src)
                    dest_file = Path(sound_output_folder) / f"{filename}"
                    shutil.copy2(file_src, dest_file)
                    file_to_remove = Path(ADDON_PATH_WORKS) / module_copy_from / "generated/sounds/gossip" / filename
                    if file_to_remove.exists():
                        file_to_remove.unlink()
                        df.at[row_idx, get_short_name_from_module(module_copy_from)] = '✖️'
                        print(f"removed {file_to_remove}")

                    if npc['2GE'] == '✅':
                        audio = find_gossip_audio(npc['hash'], module_copy_from, 'female')
                        file_src = audio.get(module_copy_from)
                        filename = os.path.basename(file_src)
                        dest_file = Path(sound_output_folder) / f"{filename}"
                        shutil.copy2(file_src, dest_file)
                        file_to_remove = Path(ADDON_PATH_WORKS) / module_copy_from / "generated/sounds/gossip" / filename
                        if file_to_remove.exists():
                            file_to_remove.unlink()
                            print(f"removed {file_to_remove}")

                    generate_fix = True
                    msg = f"Copied to correct module: {filename} to {npc_expansion} from {module_copy_from}"


            if not generate_fix:
                generate_fix, path, msg = tts_processor.tts_gossip_auto(df_orig, npc_name, get_module_long_name(npc_expansion), npc['hash'], voice_sample)

            if generate_fix == True:
                if npc_expansion == 'VanillaExtra':
                    df.at[row_idx, 'VAN'] = f"☑️"
                if npc_expansion == 'TBC':
                    df.at[row_idx, 'TBC'] = f"☑️"
                if npc_expansion == 'WoTLK':
                    df.at[row_idx, 'WLK'] = f"☑️"

                df['Fix'] = df.apply(check_fix_button(), axis=1)

                gr.Info(msg)

                # gr.Info(play_audio_with_controls(path))

        
        return df
    
    return df

def auto_fix_ttszone(df_orig, df, selected_zone, quest_text_type, progress=gr.Progress()):

    df_filtered = df_orig[['source', 'quest', 'quest_title', 'name', 'id', 'voice_name', 'voice_type', 'cleanedText', 'expansion', 'zone']].copy()
    df_filtered = df_filtered[df_filtered['source'].isin([quest_text_type])]
    df_filtered = df_filtered[df_filtered['zone'] == selected_zone]

    needed_fix = 0
    fix_oks = 0

    df_filtered = df_filtered.rename(columns={
        'quest_title': 'title',
        'name': 'npc_name',
        'id': 'npc',
        'cleanedText': 'Text',
        'expansion': 'Exp',
    })
    expansion_names = {
        0: 'Vanilla',
        1: 'TBC',
        2: 'WoTLK',
        3: 'Turtle'
    }
    df_filtered['Exp'] = df_filtered['Exp'].map(expansion_names)

    df_filtered = df_filtered.sort_values('source')

    if df_filtered.empty:
        return df, gr.skip()
    
    # filter turtle to avoid generating vanilla missing stuff
    if is_turtle(tts_processor):
        df_filtered = df_filtered[df_filtered['Exp'] == 'Turtle']

    total_rows = len(df_filtered)
    total_steps = total_rows
    progress_state = {'step': 0}

    progress(0, desc=f"Generating {progress_state['step'] + 1}/{total_steps} {quest_text_type} rows... {'\nNeed:'+str(needed_fix) if needed_fix else ''} {'\nFixed:'+str(fix_oks) if fix_oks else ''}")
    
    # Create a dictionary mapping (quest, source) to index in df
    index_map = { (row['quest'], row['source']): idx for idx, row in df.iterrows() }
    
    # Iterate through df_filtered
    for _, row in df_filtered.iterrows():
        key = (row['quest'], row['source'])
        
        if key in index_map:
            original_index = index_map[key]
            # Call your function with df and the index
            df, need_fix, fix_ok = process_row_ttszone_quest(df_orig, df, original_index)
            if need_fix:
                needed_fix += 1
            if fix_ok:
                fix_oks += 1

        progress_state['step'] += 1
        progress(progress_state['step'] / total_steps, desc=f"Generating {progress_state['step']}/{total_steps} {quest_text_type} rows... {'\nNeed:'+str(needed_fix) if needed_fix else ''} {'\nFixed:'+str(fix_oks) if fix_oks else ''}")
    
    import gc
    import torch
    gc.collect() # Чистим мусор в Python
    torch.cuda.empty_cache() # Освобождаем неиспользуемую память в CUDA
    torch.cuda.ipc_collect()

    return df, gr.update(value=f"📥 Fix all {str.capitalize(quest_text_type)} texts {'\nNeed:'+str(needed_fix) if needed_fix else ''} {'\nFixed:'+str(fix_oks) if fix_oks else ''}")

def auto_fix_ttszone_gossip(df_orig, df, selected_zone, progress=gr.Progress()):

    df_filtered = df_orig[['source', 'name', 'id', 'voice_name', 'voice_type', 'cleanedText', 'templateText_race_gender_hash', 'expansion', 'zone']].copy()
    df_filtered = df_filtered[df_filtered['source'].isin(['gossip'])]
    df_filtered = df_filtered[df_filtered['zone'] == selected_zone]

    needed_fix = 0
    fix_oks = 0

    df_filtered = df_filtered.rename(columns={
        'name': 'npc_name',
        'id': 'npc',
        'cleanedText': 'Text',
        'expansion': 'Exp',
        'templateText_race_gender_hash': 'hash'
    })
    expansion_names = {
        0: 'Vanilla',
        1: 'TBC',
        2: 'WoTLK',
        3: 'Turtle'
    }
    df_filtered['Exp'] = df_filtered['Exp'].map(expansion_names)

    df_filtered = df_filtered.sort_values('npc')

    if df_filtered.empty:
        return df_filtered
    
    # filter turtle to avoid generating vanilla missing stuff
    if is_turtle(tts_processor):
        df_filtered = df_filtered[df_filtered['Exp'] == 'Turtle']

    total_rows = len(df_filtered)
    total_steps = total_rows
    progress_state = {'step': 0}

    progress(0, desc=f"Generating {progress_state['step'] + 1}/{total_steps} gossip rows... {'\nNeed:'+str(needed_fix) if needed_fix else ''} {'\nFixed:'+str(fix_oks) if fix_oks else ''}")
    
    # Create a dictionary mapping (quest, source) to index in df
    index_map = { (row['npc'], row['hash']): idx for idx, row in df.iterrows() }
    
    # Iterate through df_filtered
    for _, row in df_filtered.iterrows():
        key = (row['npc'], row['hash'])
        
        if key in index_map:
            original_index = index_map[key]
            # Call your function with df and the index
            if is_turtle(tts_processor):
                df, need_fix, fix_ok = process_row_ttszone_gossip_turtle(df_orig, df, original_index)
            else:
                df, need_fix, fix_ok = process_row_ttszone_gossip(df_orig, df, original_index)
            if need_fix:
                needed_fix += 1
            if fix_ok:
                fix_oks += 1

        progress_state['step'] += 1
        progress(progress_state['step'] / total_steps, desc=f"Generating {progress_state['step']}/{total_steps} gossip rows... {'\nNeed:'+str(needed_fix) if needed_fix else ''} {'\nFixed:'+str(fix_oks) if fix_oks else ''}")
    
    import gc
    import torch
    gc.collect() # Чистим мусор в Python
    torch.cuda.empty_cache() # Освобождаем неиспользуемую память в CUDA
    torch.cuda.ipc_collect()

    return df, gr.update(value=f"📥 Fix all {str.capitalize('gossip')} texts {'\nNeed:'+str(needed_fix) if needed_fix else ''} {'\nFixed:'+str(fix_oks) if fix_oks else ''}")

def auto_fix_ttszone_full(df_orig, selected_zone, progress=gr.Progress()):

    # Clear whisper memory
    tts_processor.free_memory_whisper()

    # display_df_quests = select_columns_ttszone(df, selected_zone)[0]
    # prog_df_quests = calculate_quest_progress(display_df_quests)
    # display_df_gossip = select_columns_ttszone_gossip(df, selected_zone)[0]
    # prog_df_gossip = calculate_quest_progress(display_df_gossip)

    display_df_quests = select_columns_ttszone(df_orig, selected_zone)[0].reset_index(drop=True)

    # do all quests
    auto_fix_ttszone(df_orig, display_df_quests, selected_zone, 'accept', progress)
    auto_fix_ttszone(df_orig, display_df_quests, selected_zone, 'complete', progress)
    auto_fix_ttszone(df_orig, display_df_quests, selected_zone, 'progress', progress)

    display_df_gossip = select_columns_ttszone_gossip(df_orig, selected_zone)[0].reset_index(drop=True)

    # do all gossips
    auto_fix_ttszone_gossip(df_orig, display_df_gossip, selected_zone, progress)

    return select_columns_ttszone_full(df_orig, selected_zone)[0]

def get_quest_data(q_id):
    # Тут логика поиска в твоем DataFrame
    # filtered_df = df_quests[df_quests['id'] == q_id]
    # Возвращаем данные или None
    pass

def find_quest_audio(q_id, q_type, module_name=None):
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
        if tts_processor.get_module_name() != 'AI_VoiceOverData_TBC':
            if tts_processor.get_module_name() == module_name:
                out_path = Path(tts_processor.get_module_name()) / "generated/sounds/quests" / filename
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

def find_gossip_audio(npc_race_gender_hash, module_name=None, gender='male'):
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
        if tts_processor.get_module_name() != 'AI_VoiceOverData_TBC':
            if tts_processor.get_module_name() == module_name:
                out_path = Path(tts_processor.get_module_name()) / "generated/sounds/gossip" / filename
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

# def find_voice_sample(voice_name, voice_type='standard'):
#     samples = {}
#     for module in MODULE_NAMES:
#         sample_path = Path("voices") / module / f"{voice_name}.mp3"
#         if sample_path.exists():
#             samples[module] = str(sample_path)

#     sample_path = Path("voices") / "GameVoice" / f"{voice_name}_{voice_type}.mp3"
#     if voice_type != 'standard' and sample_path.exists():
#         samples["FromGame"] = str(sample_path)
#     else:
#         sample_path = Path("voices") / "GameVoice" / f"{voice_name}.mp3"
#         if sample_path.exists():
#             samples["FromGame"] = str(sample_path)

#     return samples


def get_audio_with_label(audio_accept, module_name, update_label=True, gender=None):
    """
    Get audio value and determine appropriate label based on gender files

    Args:
        audio_accept: The audio_accept dict containing file paths
        module_name: The module name (e.g., 'AI_VoiceOverData_Vanilla')
        gender: Optional gender 'm', 'f', or None

    Returns:
        gr.update with value and label
    """

    # Debug prints
    # print("=" * 50)
    # print(f"📢 get_audio_with_label called")
    # print(f"   audio_accept: {audio_accept}")
    # print(f"   audio_accept type: {type(audio_accept)}")
    # print(f"   module_name: {module_name}")
    # print(f"   update_label: {update_label}")
    # print(f"   gender: {gender}")
    # Get the audio file path
    audio_path = audio_accept.get(module_name)

    if not audio_path:
        return gr.update(value=None)

    # Check for gender prefix
    base_label = get_module_short_name(module_name)

    # Check if file has "m-" prefix
    has_male = 'm-' in str(audio_path) or audio_path.startswith('m-')
    # Check if female version exists in the same module
    if has_male:
        female_path = audio_accept.get(module_name).replace('m-', 'f-')
        has_female = female_path and os.path.exists(female_path)

    # Build label
    label = base_label

    if has_male:
        label = f"{label} (m)"
        if has_female:
            label = f"{label} (f exists)"
    elif gender == 'm':
        label = f"{label} (m)"
    elif gender == 'f':
        label = f"{label} (f)"

    if update_label:
        return gr.update(value=audio_path, label=label)
    return gr.update(value=audio_path)

with gr.Blocks(title="WoW Voiceover WebUI") as app:
    df_state = gr.State(value=None)
    df_state_turtle = gr.State(value=None)
    current_quest_id = gr.State(None)
    current_npc_name = gr.State(None)
    selector_piece = gr.State(value=None)
    gr.Markdown("## WoW Voiceover WebUI")
    gr.Markdown(
        value = "Voiceover for quests and gossip from WoW, configured for TBC and WoTLK."

    )
    with gr.Tabs() as tabs_container:
        with gr.TabItem("Data preparation", id="datapreparation"):
            with gr.Group(visible=False):
                with gr.Row():
                    but0 = gr.Button("Initialize Database", variant="primary")
                    info0 = gr.Textbox(label="Output", value="")
                    but0.click(
                        init_db,
                        inputs=[],
                        outputs=info0,
                    )
            with gr.Group():
                with gr.Row():
                    but1 = gr.Button("Create gossip & quest data", variant="primary")
                    info1 = gr.Textbox(label="Output", value="")
                    but1.click(
                        lambda: clean_quest_data(tts_processor),
                        inputs=[],
                        outputs=[info1,df_state],
                    )
            with gr.Group():
                with gr.Row():
                    but_turtle_load = gr.Button("Load gossip & text TURTLE", variant="primary")
                    info_turtle_load = gr.Textbox(label="Output", value="")
            with gr.Group(visible=True):
                with gr.Row():
                    but1 = gr.Button("Find best voice samples", variant="primary")
                    info2 = gr.Textbox(label="Output", value="")
                    but1.click(
                        fn=tts_processor.collect_best_references, 
                        inputs=[df_state],
                        outputs=info2
                    )
            with gr.Group():
                with gr.Row():
                        dataframe_display = gr.Dataframe(
                            show_search = 'search',
                            wrap = True,
                            interactive=False,
                            column_widths = ["90px","70px","90px","90px","60px","150px", "150px", "40px","40px","700px", "70px", "100px"]
                        )
                copy_trigger = gr.Textbox(visible=False)

            df_state.change(
                select_columns,
                inputs=[df_state],
                outputs=[dataframe_display]
            )

            # with gr.Group():
            #     with gr.Row():
            #             dataframe_display_turtle = gr.Dataframe(
            #                 show_search = 'search',
            #                 wrap = True,
            #                 interactive=False,
            #                 column_widths = ["90px","70px","90px","90px","60px","150px","40px","40px","700px", "70px", "100px"]
            #             )
                # copy_trigger = gr.Textbox(visible=False)

            # df_state_turtle.change(
            #     select_columns,
            #     inputs=[df_state_turtle],
            #     outputs=[dataframe_display_turtle]
            # )

            # dataframe_display.select(fn=on_cell_select, inputs=None, outputs=[copy_trigger, quest_id, tabs_container])
            # copy_trigger.change(None, inputs=[copy_trigger], outputs=None, js=copy_to_clipboard_js)


        with gr.TabItem("TTS by quest", id="ttsquest"):
            selected_clone_paths = {
                'accept': gr.State(None),
                'progress': gr.State(None),
                'complete': gr.State(None)
            }
            selected_clone_paths_global = {
                'narrator': gr.State(None)
            }
            expansions_state = gr.State()
            with gr.Accordion("⚙️ Extended VibeVoice parameters", open=False):
                with gr.Group():
                    gr.Markdown("### 🧠 Choose model")
                    with gr.Row():
                        model = gr.Dropdown(
                            label="Model",
                            choices=tts_processor.available_models,
                            value=tts_processor.default_model
                        )
                        # seed = gr.Number(label="Seed", value=42, precision=0)

                with gr.Group():
                    gr.Markdown("### ⚙️ Generation options")
                    with gr.Row():
                        # use_sampling = gr.Checkbox(
                        #     label="Use sampling",
                        #     info="Enable sampling mode. When False (default), uses deterministic generation like official examples",
                        #     value=False)
                        temperature = gr.Slider(
                            label="Position Temperature", 
                            minimum=0.1, maximum=10.0, value=5.0, step=0.05,
                            info="Higher values increase emotion and variety; lower values improve stability and clarity.",
                            interactive=True
                        )
                        temperature_class = gr.Slider(
                            label="Class Temperature", 
                            minimum=0.0, maximum=1.0, value=0.3, step=0.05,
                            info="Higher values increase emotion and variety; lower values improve stability and clarity.",
                            interactive=True
                        )
                    
                    with gr.Row():
                        voice_speed_factor = gr.Slider(
                            label="Voice Speed Factor", 
                            minimum=0.8, maximum=1.5, value=1.1, step=0.01,
                            info="1.0 = normal speed, <1.0 = slower speed, >1.0 = faster speed."
                        )
                        voice_cps = gr.Slider(
                            minimum=8.0,
                            maximum=18.0,
                            value=15,
                            step=0.5,
                            label="CPS (Characters Per Second) — speech speed"
                        )
                        max_words_per_chunk = gr.Number(
                            label="Max Words Per Chunk", 
                            value=75, precision=0,
                            info="Maximum words per chunk for long texts. Lower values prevent speed issues but create more chunks."
                        )

                with gr.Group():
                    gr.Markdown("### ⚡ VRAM optimisations")
                    with gr.Row():
                        attention_type = gr.Dropdown(
                            label="Attention Type",
                            choices=["auto", "eager", "sdpa", "flash_attention_2", "sage"],
                            value="auto",
                            info="Attention implementation. Auto selects the best available, eager is standard, sdpa is optimized PyTorch, flash_attention_2 requires compatible GPU, sage uses quantized attention for speedup (CUDA only)."
                        )
                        quantize_llm = gr.Radio(
                            label="Quantize LLM",
                            choices=["full precision", "4bit", "8bit"],
                            value="full precision",
                            info="Dynamically quantize only the LLM component for non-quantized models. 4bit: major VRAM savings with minimal quality loss. 8bit: good balance of quality and memory usage. Full precision: original quality. Note: ignored for pre-quantized models. Requires CUDA GPU."
                        )
                    free_memory = gr.Checkbox(label="Free memory after generation", value=False)

                with gr.Group():
                    gr.Markdown("### 🎨 Diffusion options")
                    with gr.Row():
                        diffusion_steps = gr.Slider(
                            label="Steps", minimum=1, maximum=100, value=32, step=1,
                            info="Number of denoising steps. More steps = theoretically better quality but slower. Default: 20"
                        )
                        cfg_scale = gr.Slider(
                            label="CFG Scale", minimum=0.5, maximum=3.5, value=2.0, step=0.05,
                            info="Classifier-free guidance scale (official default: 1.3)"
                        )
                with gr.Group():
                    gr.Markdown("### 🧠 Settings")
                    with gr.Row():
                        use_sampling = gr.Checkbox(
                            label="Use sampling",
                            info="Enable sampling mode. When False (default), uses deterministic generation like official examples",
                            value=True)


                        def toggle_sampling(is_checked):
                            return gr.update(interactive=is_checked), gr.update(interactive=is_checked)


                        use_sampling.change(
                            fn=toggle_sampling,
                            inputs=[use_sampling],
                            outputs=[temperature, temperature_class]
                        )

                        randomize_seed = gr.Checkbox(label="Auto-Random", value=False)
                        volume_match = gr.Checkbox(label="Match Volume", value=True)
                        volume_match_sample = gr.Checkbox(label="Volume From sample", value=False)

            # "♻️✅❌"

            # with gr.Group():
            #     with gr.Row():
            #         with gr.Column():
            #             expansion_label = gr.Markdown("Choose whether you are generating the accept, complete or both parts of the quest audio")
            #             spec_single = gr.Dropdown(
            #                 label="",
            #                 choices=(
            #                     ['accept','complete','progress']
            #                 ),
            #                 value = ['accept','complete','progress'],
            #                 multiselect = True,
            #                 interactive = True
            #                 )
            #         # with gr.Column():
            #         #     module_label = gr.Markdown("Module to export audio to")
            #         #     module_name_single = gr.Dropdown(
            #         #         label="",
            #         #         choices=(
            #         #             ['AI_VoiceOverData_VanillaExtra', 'AI_VoiceOverData_TBC', 'AI_VoiceOverData_WoTLK','Testing']
            #         #         ),
            #         #         value = 'Testing',
            #         #         allow_custom_value = True,
            #         #         interactive = True
            #         #         )
            #         #     force_gen = gr.Checkbox(
            #         #     label="Replace existing",
            #         #     value=False)
            #         # with gr.Column():
            #         #     questid_label = gr.Markdown("Write the quest ID of the quest you will be regenerating")
            #         #     quest_id = gr.Textbox(
            #         #         interactive = True
            #         #         )
            with gr.Row():
                with gr.Column():
                    quest_id = gr.Textbox(label="Quest ID", placeholder="Enter quest ID and press Enter")
                    btn_load = gr.Button("📥 Load Quest Info", variant="primary")
                with gr.Column():
                    # module_label = gr.Markdown("Module to export audio to")
                    module_name_single = gr.Dropdown(
                        label="Module to export audio to",
                        choices=(
                            ['AI_VoiceOverData_VanillaExtra', 'AI_VoiceOverData_TBC', 'AI_VoiceOverData_WoTLK',
                             'Testing']
                        ),
                        value='Testing',
                        allow_custom_value=True,
                        interactive=True
                    )
                    force_gen = gr.Checkbox(
                        label="Replace existing",
                        value=True)
                    # apply_effects_quest = gr.Checkbox(
                    #     label="Apply Voice effects",
                    #     value=True)
                    # apply_effects_override_quest = gr.Textbox(
                    #     label="Voice Effect override",
                    #     placeholder="ghost/undead/demon/giant/ancient/small/comms")

                with gr.Column():
                    seed = gr.Number(label="Seed", value=random.randint(0, 2147483647), precision=0, elem_classes="seed-normal")
                    btn_random_seed = gr.Button("🎲")

                    def get_random_seed():
                        return random.randint(0, 2147483647)

                    btn_random_seed.click(
                        fn=get_random_seed,
                        inputs=[],
                        outputs=seed
                    )

                    def toggle_seed_interactive(is_auto):
                        return gr.update(interactive=not is_auto)

                    randomize_seed.change(
                        fn=toggle_seed_interactive,
                        inputs=[randomize_seed],
                        outputs=seed
                    )

            with gr.Row():
                with gr.Column():
                    apply_effects_quest = gr.Checkbox(
                        label="Apply Voice effects",
                        value=True)
                    use_emotions_quest_text = gr.Checkbox(label="Detect emotions", value=False)
                with gr.Column(elem_classes="my-column"):
                    apply_effects_override_quest = gr.Dropdown(
                        label='Voice effect ovveride',
                        choices=['ghost', 'robot', 'undead', 'demon', 'giant', 'small', 'letter', 'comms', 'dk', 'ancient', 'ethereal', 'earthen', 'naaru', 'wretched'],
                        value=None,
                        allow_custom_value=True)
                with gr.Column():
                    use_narrator_quest = gr.Checkbox(
                        label="Add Narrator voice",
                        value=True,
                        interactive=False)
                    use_modified_quest_text = gr.Checkbox(label="Use modified Quest Text", value=False)

            # --- ИНФО О КВЕСТЕ ---
            with gr.Group() as main_panel:
                with gr.Row():
                    q_info_display = gr.Markdown("### 📜 Quest: [Choose ID]")

                with gr.Row(visible=False) as narrator_indicator:
                    q_info_display_two_speakers = gr.Markdown("### 🧑‍‍🧑 <span style='color: red'>Narrator text detected</span>")

                with gr.Row(visible=False) as two_genders_indicator:
                    q_info_display_two_genders = gr.Markdown("### ️♂️♀️ <span style='color: red'>Two genders text detected</span>")

                with gr.Row(visible=False) as different_vanilla_indicator:
                    q_info_display_different_vanilla = gr.Markdown("### 🔰‼️ <span style='color: red'>Different Vanilla text detected</span>")

                with gr.Row(visible=False) as quest_expansion_indicator:
                    q_info_display_expansion = gr.Markdown("### <span style='color: red'>Unknown Expansion</span>")

            def set_clone_source(audio_path, type_label):
                if not audio_path:
                    return gr.update()  # Do nothing if player is empty

                # Marks the selection visually (e.g., A ✅)
                return audio_path, f"{type_label[0].upper()} ✅"


            # 2. THE UI BUILDER LOOP
            all_btns = {'accept': [], 'progress': [], 'complete': []}
            players = []

            all_btns_global = {'narrator': []}
            players_global = []

            all_btns_testing = {'accept': [], 'progress': [], 'complete': []}
            players_testing = {'accept': [], 'progress': [], 'complete': []}


            def create_voice_card(label):
                with gr.Group():
                    # Создаем кнопки

                    b_a = gr.Button("A ♻️", size="sm", variant="secondary")
                    b_p = gr.Button("P ♻️", size="sm", variant="secondary")
                    b_c = gr.Button("C ♻️", size="sm", variant="secondary")

                    # Создаем плеер и вставляем кнопки в Toolbar
                    audio_player = gr.Audio(
                        label=label,
                        type="filepath",
                        interactive=True,
                        buttons=[b_a, b_p, b_c],
                        min_width=100
                    )

                    # Логика для кнопок: при клике вызываем функцию выбора
                    # Мы передаем сам плеер как input
                    # current_idx = len(all_btns['accept'])

                    # b_a.click(fn=mark_as_source, inputs=[audio_player, gr.State("accept"), len(all_btns['accept'])],
                    #           outputs=[selected_clone_paths['accept'], b_a])
                    # b_p.click(fn=mark_as_source, inputs=[audio_player, gr.State("progress"), len(all_btns['accept'])],
                    #           outputs=[selected_clone_paths['progress'], b_p])
                    # b_c.click(fn=mark_as_source, inputs=[audio_player, gr.State("complete"), len(all_btns['accept'])],
                    #           outputs=[selected_clone_paths['complete'], b_c])

                    # Сохраняем кнопки в списки для массового управления (например, ресета)
                    all_btns['accept'].append(b_a)
                    all_btns['progress'].append(b_p)
                    all_btns['complete'].append(b_c)
                    players.append(audio_player)

                return audio_player

            def create_voice_card_narrator(label):
                with gr.Group():
                    # Создаем кнопки

                    b_n = gr.Button("Use Sample ♻️", size="sm", variant="secondary")

                    # Создаем плеер и вставляем кнопки в Toolbar
                    audio_player = gr.Audio(
                        label=label,
                        type="filepath",
                        interactive=True,
                        buttons=[b_n],
                        min_width=100
                    )

                    # Сохраняем кнопки в списки для массового управления (например, ресета)
                    all_btns_global['narrator'].append(b_n)
                    players_global.append(audio_player)

                return audio_player

            def create_voice_card_testing(label, t_type):
                with gr.Group():

                    use_this_audio = gr.Button("Use this ♻️", size="sm", variant="secondary")

                    audio_player = gr.Audio(
                        label=label,
                        type="filepath",
                        interactive=True,
                        buttons=[use_this_audio],
                        min_width=100
                    )
                    
                    all_btns_testing[t_type].append(use_this_audio)
                    players_testing[t_type].append(audio_player)

                return audio_player


            def set_clone_source_exclusive(audio_path, type_label):
                # type_label is 'accept', 'progress', etc.
                # Logic: return 'A ✅' for clicked index, and 'A ❌' for all others in that list
                print(f"{type_label} voice path set to {audio_path}")
                gr.Info(f"Voice to clone for {type_label} set to: {audio_path}")
                return audio_path, audio_path# , gr.update(value=f"A ✅", variant="primary")

            def set_clone_source_narrator(audio_path):
                # type_label is 'accept', 'progress', etc.
                # Logic: return 'A ✅' for clicked index, and 'A ❌' for all others in that list
                print(f"Narrator voice path set to {audio_path}")
                gr.Info(f"Narrator Voice to clone set to: {audio_path}")
                return audio_path, audio_path

            def set_audio_from_testing(audio_path, type_name, type_text, module_n):
                if module_n == "Testing":
                    print(f"Current module is Testing!")
                    gr.Info(f"Current module is Testing!")
                    return [gr.update(), gr.update(), gr.update()]

                if audio_path is None:
                    # print("⚠️ No audio selected yet")
                    # gr.Info("Please select an audio file first")
                    return [gr.update(), gr.update(), gr.update()]

                output_folder = os.path.join(module_n, 'generated/sounds/')
                sound_output_folder = os.path.join(output_folder, type_name)
                os.makedirs(sound_output_folder, exist_ok=True)
                filename = os.path.basename(audio_path)
                dest_file = Path(sound_output_folder) / f"{filename}"
                shutil.copy2(audio_path, dest_file)
                print(f"Testing voice copied to {module_n}/generated/sounds/{type_name}/{filename}")
                gr.Info(f"Testing voice copied to {module_n}/generated/sounds/{type_name}/{filename}")

                print(f"path to file is {dest_file}")
                print(f"filename is {filename}")

                updates = [gr.update(), gr.update(), gr.update()]  # No change for others
                # Update only the matching output
                if type_text == "accept":
                    updates[0] = gr.update(value=audio_path)
                elif type_text == "complete":
                    updates[1] = gr.update(value=audio_path)
                elif type_text == "progress":
                    updates[2] = gr.update(value=audio_path)

                return updates

            # --- ГРУППА ACCEPT ---
            with gr.Group():
                gr.Markdown("#### 🟢 ACCEPT", min_height=30)
                with gr.Row():
                    accept_npc_info_display = gr.Markdown(" 🧐 NPC: ---  |  Voice: ---", min_height=30)
                with gr.Row():
                    with gr.Column():
                        # noinspection PyPackageRequirements
                        accept_clean = gr.Textbox(label="Text to generate", lines=1, interactive=True)
                    with gr.Column():
                        accept_raw = gr.Textbox(label="Original text", lines=1)
                with gr.Row():
                    accept_voice = gr.Dropdown(
                            show_label=False,
                            choices=selected_voice_names,
                            value='',
                            interactive=True,
                            allow_custom_value=True
                        )
                    accept_voice_type = gr.Dropdown(
                            show_label=False,
                            choices=get_possible_voice_types(),
                            value='standard',
                            interactive=True,
                            allow_custom_value=True
                        )
                    accept_voice_change = gr.Button("Change Accept Voice")
                with gr.Row():
                    gr.Markdown("🔍 Voice Samples:", min_height=30)
                with gr.Row():
                    a_p_og = create_voice_card("Vanilla")
                    a_p_vanilla = create_voice_card("VanillaExtra")
                with gr.Row():
                    a_p_tbc = create_voice_card("TBC")
                    a_p_wotlk = create_voice_card("WotLK")
                with gr.Row():
                    a_p_out = create_voice_card(f"Generated in selected module")
                    a_p_testing = create_voice_card_testing("Testing", "accept")
                with gr.Row():
                    monitor_a = gr.Audio(label="Selected Accept Voice", type="filepath", interactive=False)
                    audio_gen_accept = gr.Audio(label="Synthesised Audio", interactive=False, autoplay=True)
                with gr.Row():
                    but_gen_accept = gr.Button("Generate Accept Audio", variant="primary")
                    info_gen_accept = gr.Textbox(label="Output", value="")

            with gr.Group():
                gr.Markdown("#### 🟢 COMPLETE", min_height=30)
                with gr.Row():
                    complete_npc_info_display = gr.Markdown(" 🧐 NPC: ---  |  Voice: ---", min_height=30)
                with gr.Row():
                    with gr.Column(scale=2):
                        complete_clean = gr.Textbox(label="Text to generate", lines=1, interactive=True)
                    with gr.Column(scale=2):
                        complete_raw = gr.Textbox(label="Original text", lines=1)
                with gr.Row():
                    complete_voice = gr.Dropdown(
                            show_label=False,
                            choices=selected_voice_names,
                            value='',
                            interactive=True,
                            allow_custom_value=True
                        )
                    complete_voice_type = gr.Dropdown(
                            show_label=False,
                            choices=get_possible_voice_types(),
                            value='standard',
                            interactive=True,
                            allow_custom_value=True
                        )
                    complete_voice_change = gr.Button("Change Complete Voice")
                with gr.Row():
                    gr.Markdown("🔍 Voice Samples:", min_height=30)
                with gr.Row():
                    c_p_og = create_voice_card("Vanilla")
                    c_p_vanilla = create_voice_card("VanillaExtra")
                with gr.Row():
                    c_p_tbc = create_voice_card("TBC")
                    c_p_wotlk = create_voice_card("WotLK")
                with gr.Row():
                    c_p_out = create_voice_card(f"Generated in selected module")
                    c_p_testing = create_voice_card_testing("Testing", "complete")
                with gr.Row():
                    monitor_c = gr.Audio(label="Selected Complete Voice", type="filepath", interactive=False)
                    audio_gen_complete = gr.Audio(label="Synthesised Audio", interactive=False, autoplay=True)
                with gr.Row():
                    but_gen_complete = gr.Button("Generate Complete Audio", variant="primary")
                    info_gen_complete = gr.Textbox(label="Output", value="")

            with gr.Group():
                gr.Markdown(" #### 🟢 PROGRESS", min_height=30)
                with gr.Row():
                    progress_npc_info_display = gr.Markdown(" 🧐 NPC: ---  |  Voice: ---", min_height=30)
                with gr.Row():
                    with gr.Column(scale=2):
                        progress_clean = gr.Textbox(label="Text to generate", lines=1, interactive=True)
                    with gr.Column(scale=2):
                        progress_raw = gr.Textbox(label="Original text", lines=1)
                with gr.Row():
                    progress_voice = gr.Dropdown(
                            show_label=False,
                            choices=selected_voice_names,
                            value='',
                            interactive=True,
                            allow_custom_value=True
                        )
                    progress_voice_type = gr.Dropdown(
                            show_label=False,
                            choices=get_possible_voice_types(),
                            value='standard',
                            interactive=True,
                            allow_custom_value=True
                        )
                    progress_voice_change = gr.Button("Change Progress Voice", )
                with gr.Row():
                    gr.Markdown("🔍 Voice Samples:", min_height=30)
                with gr.Row():
                    p_p_og = create_voice_card("Vanilla")
                    p_p_vanilla = create_voice_card("VanillaExtra")
                with gr.Row():
                    p_p_tbc = create_voice_card("TBC")
                    p_p_wotlk = create_voice_card("WotLK")
                with gr.Row():
                    p_p_out = create_voice_card(f"Generated in selected module")
                    p_p_testing = create_voice_card_testing("Testing", "progress")
                with gr.Row():
                    monitor_p = gr.Audio(label="Selected Progress Voice", type="filepath", interactive=False)
                    audio_gen_progress = gr.Audio(label="Synthesised Audio", interactive=False, autoplay=True)
                with gr.Row():
                    but_gen_progress = gr.Button("Generate Progress Audio", variant="primary")
                    info_gen_progress = gr.Textbox(label="Output", value="")

                    but_gen_accept.click(
                        tts_processor.tts_quest,
                        inputs=[
                            df_state,
                            quest_id,
                            module_name_single,
                            gr.State('accept'),
                            # selected_clone_paths['accept'],
                            monitor_a,
                            selected_clone_paths_global['narrator'],
                            accept_clean,
                            model,
                            seed,
                            randomize_seed,
                            temperature,
                            temperature_class,
                            voice_speed_factor,
                            voice_cps,
                            max_words_per_chunk,
                            attention_type,
                            quantize_llm,
                            free_memory,
                            diffusion_steps,
                            cfg_scale,
                            use_sampling,
                            force_gen,
                            volume_match,
                            volume_match_sample,
                            apply_effects_quest,
                            apply_effects_override_quest,
                            use_emotions_quest_text,
                            accept_voice,
                            accept_voice_type
                        ],
                        outputs=[info_gen_accept, expansions_state, audio_gen_accept, seed],
                    )

                    but_gen_progress.click(
                        tts_processor.tts_quest,
                        inputs=[
                            df_state,
                            quest_id,
                            module_name_single,
                            gr.State('progress'),
                            # selected_clone_paths['progress'],
                            monitor_p,
                            selected_clone_paths_global['narrator'],
                            progress_clean,
                            model,
                            seed,
                            randomize_seed,
                            temperature,
                            temperature_class,
                            voice_speed_factor,
                            voice_cps,
                            max_words_per_chunk,
                            attention_type,
                            quantize_llm,
                            free_memory,
                            diffusion_steps,
                            cfg_scale,
                            use_sampling,
                            force_gen,
                            volume_match,
                            volume_match_sample,
                            apply_effects_quest,
                            apply_effects_override_quest,
                            use_emotions_quest_text,
                            progress_voice,
                            progress_voice_type
                        ],
                        outputs=[info_gen_progress, expansions_state, audio_gen_progress, seed],
                    )

                    but_gen_complete.click(
                        tts_processor.tts_quest,
                        inputs=[
                            df_state,
                            quest_id,
                            module_name_single,
                            gr.State('complete'),
                            # selected_clone_paths['complete'],
                            monitor_c,
                            selected_clone_paths_global['narrator'],
                            complete_clean,
                            model,
                            seed,
                            randomize_seed,
                            temperature,
                            temperature_class,
                            voice_speed_factor,
                            voice_cps,
                            max_words_per_chunk,
                            attention_type,
                            quantize_llm,
                            free_memory,
                            diffusion_steps,
                            cfg_scale,
                            use_sampling,
                            force_gen,
                            volume_match,
                            volume_match_sample,
                            apply_effects_quest,
                            apply_effects_override_quest,
                            use_emotions_quest_text,
                            complete_voice,
                            complete_voice_type
                        ],
                        outputs=[info_gen_complete, expansions_state, audio_gen_complete, seed],
                    )

            print(f"DEBUG: Плееров: {len(players)}")
            print(f"DEBUG: Кнопок Accept: {len(all_btns['accept'])}")
            print(f"DEBUG: Кнопок Progress: {len(all_btns['progress'])}")
            print(f"DEBUG: Кнопок Complete: {len(all_btns['complete'])}")

            monitors = {
                'accept': monitor_a,
                'progress': monitor_p,
                'complete': monitor_c
            }

            # --- ГРУППА ГОЛОСА ---
            gr.Markdown("#### 🎙️ LONG VOICE SAMPLES")
            with gr.Row():
                v_sample_og = create_voice_card("Vanilla")
                v_sample_og_alt = create_voice_card("Vanilla 2nd voice")
            with gr.Row():
                v_sample_vanilla = create_voice_card("VanillaExtra")
                v_sample_vanilla_alt = create_voice_card("VanillaExtra 2nd voice")
            with gr.Row():
                v_sample_tbc = create_voice_card("TBC")
                v_sample_tbc_alt = create_voice_card("TBC 2nd voice")
            with gr.Row():
                v_sample_wotlk = create_voice_card("WotLK")
                v_sample_wotlk_alt = create_voice_card("WotLK 2nd voice")
            with gr.Row():
                v_sample_game = create_voice_card("FromGame")
                v_sample_game_alt = create_voice_card("FromGame 2nd voice")
                # Тут можно добавить больше, если нужно видеть все модули сразу

            with gr.Group(visible=False) as narrator_voices:
                gr.Markdown("#### 🎙️ NARRATOR VOICE SAMPLES")
                monitor_n = gr.Audio(label="Selected Narrator Voice", type="filepath", interactive=False, value=r"C:\soft\wow_voice_over\script\voices\AI_VoiceOverData_VanillaExtra\narrator.mp3")
                with gr.Row():
                    n_sample_og = create_voice_card_narrator("Vanilla")
                    n_sample_vanilla = create_voice_card_narrator("VanillaExtra")
                with gr.Row():
                    n_sample_tbc = create_voice_card_narrator("TBC")
                    n_sample_wotlk = create_voice_card_narrator("WotLK")

            monitors_global = {
                'narrator': monitor_n,
            }

            for key in ['accept', 'progress', 'complete']:
                for i in range(len(all_btns[key])):
                    all_btns[key][i].click(
                        fn=set_clone_source_exclusive,
                        inputs=[players[i], gr.State(key)],
                        outputs=[selected_clone_paths[key], monitors[key]]
                    )

            for key in ['narrator']:
                for i in range(len(all_btns_global[key])):
                    all_btns_global[key][i].click(
                        fn=set_clone_source_narrator,
                        inputs=[players_global[i]],
                        outputs=[selected_clone_paths_global[key], monitors_global[key]]
                    )

            # for key in ['accept', 'complete', 'progress']:
            #     for i in range(len(players_testing)):
            #         all_btns_testing[key][0].click(
            #             fn=set_audio_from_testing,
            #             inputs=[players_testing[i], gr.State('quests'), gr.State(key), module_name_single],
            #             outputs=[a_p_out, c_p_out, p_p_out]
            #         )
            
            for key in ['accept', 'complete', 'progress']:
                if all_btns_testing[key]:  # Check if there's a button for this key
                    all_btns_testing[key][0].click(
                        fn=set_audio_from_testing,
                        inputs=[players_testing[key][0], gr.State('quests'), gr.State(key), module_name_single],
                        outputs=[a_p_out, c_p_out, p_p_out]
        )

            def update_ui_from_id(df, module_n, q_id, current_q_id, a_clean, c_clean, p_clean, use_modded_quest_texts):

                try:
                    voices_used = set()
                    voices_types_used = set()
                    voices_used_for_defaults = {}
                    two_genders = False
                    vanilla_different = False
                    # 1. Очищаем ввод: убираем лишние пробелы
                    search_val = str(q_id).strip()

                    voices_list = get_possible_voice_names(df)

                    working_df = df.copy()
                    # 2. Ищем по колонке 'quest'.
                    # Проверяем и как строку, и как число, чтобы наверняка.
                    quest_rows = working_df[working_df['quest'].astype(str) == search_val]

                    if quest_rows.empty:
                        # Дебаг принт в консоль, чтобы ты видел, что пришло
                        print(
                            f"DEBUG: Поиск ID {search_val} не дал результатов. В базе есть: {working_df['quest'].unique()[:5]}")
                        return {q_info_display: f"### ❌ Квест ID {search_val} не найден!"}

                    # 3. Берем данные (используем названия колонок со скриншота)
                    quest_rows['voice_type'] = quest_rows['voice_type'].apply(extract_voice_type)

                    base_row = quest_rows.iloc[0]
                    title = base_row.get('quest_title', 'Unknown Quest')
                    # npc_name_found = base_row.get('name', 'Unknown NPC')
                    # voice = base_row.get('voice_name', 'Unknown')

                    # Ищем аудио для всех трех типов (в аддоне и output)
                    audio_accept = find_quest_audio(q_id, "accept", module_n)
                    audio_progress = find_quest_audio(q_id, "progress", module_n)
                    audio_complete = find_quest_audio(q_id, "complete", module_n)

                    voice_map = find_voice_sample(base_row['voice_name'], base_row['voice_type'])
                    voice_map_alt = {}
                    voice_map_narrator = {}

                    # 4. Раскидываем тексты по типам (accept, progress, complete)
                    # На скриншоте колонка 'source' содержит тип.
                    texts = {'accept': '', 'progress': '', 'complete': '', 'accept_original': '',
                             'progress_original': '', 'complete_original': '', 'accept_npc': '', 'progress_npc': '',
                             'complete_npc': '', 'accept_voice': '', 'progress_voice': '', 'complete_voice': '', 'accept_voice_type': '', 'progress_voice_type': '', 'complete_voice_type': '',}
                    for _, row in quest_rows.iterrows():
                        s_type = str(row['source']).lower().strip()
                        if re.search(r'\$[Gg]', row['cleanedText']):
                            two_genders = True
                        if 'vanilla_text' in df.columns and row['vanilla_text'] != 0:
                            vanilla_different = True

                        if s_type in texts:
                            texts[s_type] = row['cleanedText'] if not vanilla_different or module_n != MODULE_NAMES[1] else row['vanilla_text']
                            texts[s_type + '_original'] = row['text']
                            texts[s_type + '_npc'] = row['name']
                            texts[s_type + '_npc_id'] = row['id']
                            texts[s_type + '_voice'] = row['voice_name']
                            texts[s_type + '_voice_type'] = row['voice_type']
                            texts[s_type + '_voice_effect'] = NPC_EFFECTS[row['id']] if row['id'] in NPC_EFFECTS else None
                            voices_used.add(row['voice_name'])
                            voices_types_used.add(row['voice_type'])
                            voices_used_for_defaults[s_type] = get_default_sample(row['voice_name'], row['voice_type'])

                    if len(voices_used) > 1:
                        print(f"More than 1 voice detected")
                        voice_list = list(voices_used)
                        voice_types_list = list(voices_types_used)
                        alt_voice = voice_list[1] if len(voice_list) > 1 else voice_list[0]
                        alt_voice_type = voice_types_list[1] if len(voice_types_list) > 1 else voice_types_list[0]
                        if alt_voice == base_row['voice_name']:
                            alt_voice = voice_list[0]
                            alt_voice_type = voice_types_list[0]
                        print(f"2nd voice: {alt_voice}")
                        voice_map_alt = find_voice_sample(alt_voice, alt_voice_type)

                    for _, row in quest_rows.iterrows():
                        if row['has_narrator']:
                            print(f"Narrator text detected")
                            voice_map_narrator = find_voice_sample('narrator')

                    if row['multi_name'] is not None and row['multi_name'] != '':
                        print(f"Multi speaker detected")

                    # try check existing quest
                    if use_modded_quest_texts and current_q_id and current_q_id == q_id:
                        print(f"Using modded quest texts")
                        vanilla_different = False
                        texts['accept'] = a_clean
                        texts['complete'] = c_clean
                        texts['progress'] = p_clean

                        if not voice_map_narrator:
                            if re.search(r'\[1]|\[2]', texts['accept']):
                                print(f"Narrator text detected in modified text")
                                voice_map_narrator = find_voice_sample('narrator')
                            elif re.search(r'\[1]|\[2]', texts['complete']):
                                print(f"Narrator text detected in modified text")
                                voice_map_narrator = find_voice_sample('narrator')
                            elif re.search(r'\[1]|\[2]', texts['progress']):
                                print(f"Narrator text detected in modified text")
                                voice_map_narrator = find_voice_sample('narrator')

                        if re.search(r'\$[Gg]', texts['accept']):
                            two_genders = True
                        elif re.search(r'\$[Gg]', texts['complete']):
                            two_genders = True
                        elif re.search(r'\$[Gg]', texts['progress']):
                            two_genders = True
                    
                    default_seed = tts_processor.seed_manager.get_default_seed(texts['accept_voice'], texts['accept_voice_type'])

                    # try extract game voice if custom voice
                    # if texts['accept_voice'] == 'custom_custom' and not voice_map.get('FromGame'):
                    #     voice_sample_extracted = generate_game_sample_for_npc_id(texts['accept_npc_id'], module_n)
                    # skip and let update_sample function do it

                    # 5. Возвращаем данные в UI
                    # gr.Markdown(" NPC: --- | Voice: ---")
                    return {
                        seed: gr.update(value=default_seed, elem_classes="seed-approved-with-icon") if default_seed is not None else gr.update(elem_classes="seed-normal"),
                        current_quest_id: q_id,
                        use_modified_quest_text: gr.update(value=False),
                        use_emotions_quest_text: gr.update(value=False),
                        apply_effects_override_quest: gr.update(value=None),
                        q_info_display: f"### 📜 {title} (ID: {search_val}), Expansion: {get_expansion_display(base_row['expansion'])}, Zone: {base_row['zone']}",
                        accept_clean: gr.update(value=texts['accept'], elem_classes = "text-vanilla-with-icon" if vanilla_different and module_n == MODULE_NAMES[1] else "text-normal"),
                        accept_raw: texts['accept_original'] or '',
                        accept_npc_info_display: f" 🧐 NPC: {texts['accept_npc']} ({texts['accept_npc_id']}) | Voice: {texts['accept_voice']} {'('+texts['accept_voice_type']+')' if 'male' in texts['accept_voice'] else ''} {'effect: <b>'+texts['accept_voice_effect']+'</b>' if texts['accept_voice_effect'] else ''}",
                        complete_clean: gr.update(value=texts['complete'] or '', elem_classes = "text-vanilla-with-icon" if vanilla_different and module_n == MODULE_NAMES[1] else "text-normal"), 
                        complete_raw: texts['complete_original'] or '',
                        complete_npc_info_display: f" 🧐 NPC: {texts['complete_npc']} ({texts['complete_npc_id']}) | Voice: {texts['complete_voice']} {'('+texts['complete_voice_type']+')' if 'male' in texts['complete_voice'] else ''} {'effect: <b>'+texts['complete_voice_effect']+'</b>' if texts['complete_voice_effect'] else ''}",
                        progress_clean: gr.update(value=texts['progress'] or '', elem_classes = "text-vanilla-with-icon" if vanilla_different and module_n == MODULE_NAMES[1] else "text-normal"),
                        progress_raw: texts['progress_original'] or '',
                        progress_npc_info_display: f" 🧐 NPC: {texts['progress_npc']} ({texts['progress_npc_id']}) | Voice: {texts['progress_voice']} {'('+texts['progress_voice_type']+')' if 'male' in texts['progress_voice'] else ''} {'effect: <b>'+texts['progress_voice_effect']+'</b>' if texts['progress_voice_effect'] else ''}",
                        a_p_og: get_audio_with_label(audio_accept, 'AI_VoiceOverData_Vanilla'),
                        a_p_vanilla: get_audio_with_label(audio_accept, 'AI_VoiceOverData_VanillaExtra'),
                        a_p_tbc: get_audio_with_label(audio_accept, 'AI_VoiceOverData_TBC'),
                        a_p_wotlk: get_audio_with_label(audio_accept, 'AI_VoiceOverData_WoTLK'),
                        a_p_out: get_audio_with_label(audio_accept, 'Output', False),
                        c_p_og: get_audio_with_label(audio_complete, 'AI_VoiceOverData_Vanilla'),
                        c_p_vanilla: get_audio_with_label(audio_complete, 'AI_VoiceOverData_VanillaExtra'),
                        c_p_tbc: get_audio_with_label(audio_complete, 'AI_VoiceOverData_TBC'),
                        c_p_wotlk: get_audio_with_label(audio_complete, 'AI_VoiceOverData_WoTLK'),
                        c_p_out: get_audio_with_label(audio_complete, 'Output', False),
                        p_p_og: get_audio_with_label(audio_progress, 'AI_VoiceOverData_Vanilla'),
                        p_p_vanilla: get_audio_with_label(audio_progress, 'AI_VoiceOverData_VanillaExtra'),
                        p_p_tbc: get_audio_with_label(audio_progress, 'AI_VoiceOverData_TBC'),
                        p_p_wotlk: get_audio_with_label(audio_progress, 'AI_VoiceOverData_WoTLK'),
                        p_p_out: get_audio_with_label(audio_progress, 'Output', False),
                        v_sample_og: voice_map.get('AI_VoiceOverData_Vanilla'),
                        v_sample_vanilla: voice_map.get('AI_VoiceOverData_VanillaExtra'),
                        v_sample_tbc: voice_map.get('AI_VoiceOverData_TBC'),
                        v_sample_wotlk: voice_map.get('AI_VoiceOverData_WoTLK'),
                        v_sample_og_alt: voice_map_alt.get('AI_VoiceOverData_Vanilla'),
                        v_sample_vanilla_alt: voice_map_alt.get('AI_VoiceOverData_VanillaExtra'),
                        v_sample_tbc_alt: voice_map_alt.get('AI_VoiceOverData_TBC'),
                        v_sample_wotlk_alt: voice_map_alt.get('AI_VoiceOverData_WoTLK'),
                        v_sample_game: voice_map.get('FromGame'),
                        v_sample_game_alt: voice_map_alt.get('FromGame'),
                        narrator_voices: gr.update(visible=bool(voice_map_narrator)),
                        n_sample_og: voice_map_narrator.get('AI_VoiceOverData_Vanilla'),
                        n_sample_vanilla: voice_map_narrator.get('AI_VoiceOverData_VanillaExtra'),
                        n_sample_tbc: voice_map_narrator.get('AI_VoiceOverData_TBC'),
                        n_sample_wotlk: voice_map_narrator.get('AI_VoiceOverData_WoTLK'),
                        selected_clone_paths_global['narrator']: (
                            get_default_sample('narrator') if voice_map_narrator else None
                        ),
                        narrator_indicator: gr.update(visible=bool(voice_map_narrator)),
                        two_genders_indicator: gr.update(visible=bool(two_genders)),
                        different_vanilla_indicator: gr.update(visible=bool(vanilla_different)),
                        quest_expansion_indicator: gr.update(visible=False),
                        q_info_display_expansion: gr.update(value=get_expansion_display(base_row['expansion'])),
                        monitor_a: gr.update(value=voices_used_for_defaults.get('accept')) if voices_used_for_defaults.get('accept') is not None else gr.update(),
                        monitor_c: gr.update(value=voices_used_for_defaults.get('complete')) if voices_used_for_defaults.get('complete') is not None else gr.update(),
                        monitor_p: gr.update(value=voices_used_for_defaults.get('progress')) if voices_used_for_defaults.get('progress') is not None else gr.update(),
                        a_p_testing: get_audio_with_label(audio_accept, 'Testing', False),
                        c_p_testing: get_audio_with_label(audio_complete, 'Testing', False),
                        p_p_testing: get_audio_with_label(audio_progress, 'Testing', False),
                        accept_voice: gr.update(choices=voices_list, value=texts['accept_voice']),
                        accept_voice_type: gr.update(value=texts['accept_voice_type'] if texts['accept_voice_type'] else None, choices=get_possible_voice_types(texts['accept_voice'])),
                        complete_voice: gr.update(choices=voices_list, value=texts['complete_voice']),
                        complete_voice_type: gr.update(value=texts['complete_voice_type'] if texts['complete_voice_type'] else None, choices=get_possible_voice_types(texts['complete_voice'])),
                        progress_voice: gr.update(choices=voices_list, value=texts['progress_voice']),
                        progress_voice_type: gr.update(value=texts['progress_voice_type'] if texts['progress_voice_type'] else None, choices=get_possible_voice_types(texts['progress_voice']))
                    }

                except Exception as e:
                    import traceback
                    traceback.print_exc()  # Выведет полную ошибку в консоль
                    return {q_info_display: f"### ❗ Ошибка: {str(e)}"}
                

            def update_voice_sample_quest(df, module_n, q_id, a_voice_name, a_voice_type, c_voice_name, c_voice_type, p_voice_name, p_voice_type, text_type):

                # # --- DEBUG SECTION START ---
                # print("\n" + "="*50)
                # print(f"🔍 DEBUG: update_voice_sample_quest")
                # print("="*50)
                
                # debug_args = {
                #     "Module Name": module_n,
                #     "Quest ID": q_id,
                #     "Accept Name": a_voice_name,
                #     "Accept Type": a_voice_type,
                #     "Complete Name": c_voice_name,
                #     "Complete Type": c_voice_type,
                #     "Progress Name": p_voice_name,
                #     "Progress Type": p_voice_type,
                #     "Text Type Trigger": text_type
                # }
                
                # for label, val in debug_args.items():
                #     # Printing both value and type to catch 'None' vs '' or '42' vs 42
                #     print(f"{label:20} : {repr(val)} ({type(val).__name__})")

                try:
                    voice_name_used = None
                    voice_type_used = None
                    if text_type == 'accept':
                        voice_name_used = a_voice_name
                        voice_type_used = a_voice_type
                    if text_type == 'complete':
                        voice_name_used = c_voice_name
                        voice_type_used = c_voice_type
                    if text_type == 'progress':
                        voice_name_used = p_voice_name
                        voice_type_used = p_voice_type

                    if voice_name_used == 'custom_custom' or voice_type_used == 'unknown':
                        working_df = df.copy()
                        quest_rows = working_df[working_df['quest'].astype(str) == q_id]
                        quest_rows = quest_rows[quest_rows['source'] == text_type]
                        if not quest_rows.empty:
                            base_row = quest_rows.iloc[0]
                            npc_id = base_row['id']
                            npc_name = base_row['name']
                            voice_sample_extracted = get_game_sample_for_npc_id(npc_id)
                            if not voice_sample_extracted:
                                voice_sample_extracted = generate_game_sample_for_npc_id(npc_id, module_n)
                                if not voice_sample_extracted and voice_type_used != 'unknown':
                                    voice_name_used = None


                    if not voice_name_used:
                        print(f"No voice found!")
                        updates = []
                        updates.extend([
                            gr.skip(),
                            gr.skip(),
                            gr.skip(),
                            gr.skip(),
                            gr.skip(),
                            gr.skip(),
                            gr.skip(),
                            gr.skip(),
                            gr.skip(),
                            gr.skip(),
                            gr.skip(),
                            gr.skip(),
                        ])
                        return updates

                    if (voice_name_used == 'custom_custom' or voice_type_used == 'unknown') and voice_sample_extracted:
                        voices_used_for_defaults = voice_sample_extracted
                        possible_voice_types = [npc_name]
                        voice_map = {}
                        voice_map['FromGame'] = voice_sample_extracted
                    else:
                        voice_map = find_voice_sample(voice_name_used, voice_type_used)
                        voices_used_for_defaults = get_default_sample(voice_name_used, voice_type_used)
                        possible_voice_types = get_possible_voice_types(voice_name_used)

                    if not possible_voice_types:
                        update_voice_types = gr.update(value=None, choices=possible_voice_types, interactive=False)
                    else:
                        update_voice_types = gr.update(choices=possible_voice_types, interactive=True)
                        if voice_type_used not in possible_voice_types:
                            update_voice_types = gr.update(choices=possible_voice_types, value=possible_voice_types[0], interactive=True)
                        if len(possible_voice_types) == 1:
                            update_voice_types = gr.update(choices=possible_voice_types, value=possible_voice_types[0], interactive=False)

                        if voice_name_used == 'custom_custom' and voice_sample_extracted:
                            update_voice_types = gr.update(choices=[npc_name], value=npc_name, interactive=False)

                    default_seed = tts_processor.seed_manager.get_default_seed(voice_name_used, voice_type_used) if voice_type_used else None
                    
                    return {
                        # seed: gr.skip(),
                        seed: gr.update(value=default_seed, elem_classes="seed-approved-with-icon") if default_seed is not None else gr.update(elem_classes="seed-normal"),
                        accept_voice_type: update_voice_types if text_type == 'accept' else gr.skip(),
                        complete_voice_type: update_voice_types if text_type == 'complete' else gr.skip(),
                        progress_voice_type: update_voice_types if text_type == 'progress' else gr.skip(),
                        v_sample_og: voice_map.get('AI_VoiceOverData_Vanilla'),
                        v_sample_vanilla: voice_map.get('AI_VoiceOverData_VanillaExtra'),
                        v_sample_tbc: voice_map.get('AI_VoiceOverData_TBC'),
                        v_sample_wotlk: voice_map.get('AI_VoiceOverData_WoTLK'),
                        v_sample_game: voice_map.get('FromGame'),
                        monitor_a: gr.update(value=voices_used_for_defaults) if voices_used_for_defaults is not None and text_type == 'accept' else gr.skip(),
                        monitor_c: gr.update(value=voices_used_for_defaults) if voices_used_for_defaults is not None and text_type == 'complete' else gr.skip(),
                        monitor_p: gr.update(value=voices_used_for_defaults) if voices_used_for_defaults is not None and text_type == 'progress' else gr.skip()
                    }

                except Exception as e:
                    import traceback
                    traceback.print_exc()  # Выведет полную ошибку в консоль
                    return {f"### ❗ Error: {str(e)}"}


            # Привязываем событие
            btn_load.click(
                fn=update_ui_from_id,
                inputs=[df_state, module_name_single, quest_id, current_quest_id, accept_clean, complete_clean, progress_clean, use_modified_quest_text],
                outputs=[seed, current_quest_id, use_modified_quest_text, use_emotions_quest_text, apply_effects_override_quest, q_info_display, accept_npc_info_display, complete_npc_info_display, progress_npc_info_display,
                         accept_clean,
                         complete_clean, progress_clean, accept_raw, complete_raw, progress_raw, a_p_og, a_p_vanilla,
                         a_p_tbc, a_p_wotlk, a_p_out, c_p_og, c_p_vanilla,
                         c_p_tbc, c_p_wotlk, c_p_out, p_p_og, p_p_vanilla, p_p_tbc, p_p_wotlk, p_p_out,
                         v_sample_og, v_sample_vanilla, v_sample_tbc, v_sample_wotlk, v_sample_og_alt, v_sample_vanilla_alt, v_sample_tbc_alt, v_sample_wotlk_alt, v_sample_game, v_sample_game_alt, narrator_voices, n_sample_og, n_sample_vanilla, n_sample_tbc, n_sample_wotlk, selected_clone_paths_global['narrator'], narrator_indicator, two_genders_indicator, different_vanilla_indicator, quest_expansion_indicator, q_info_display_expansion, monitor_a, monitor_c, monitor_p, a_p_testing, c_p_testing, p_p_testing, accept_voice, accept_voice_type, complete_voice, complete_voice_type, progress_voice, progress_voice_type]
            )
            quest_id.submit(
                fn=update_ui_from_id,
                inputs=[df_state, module_name_single, quest_id, current_quest_id, accept_clean, complete_clean, progress_clean, use_modified_quest_text],
                outputs=[seed, current_quest_id, use_modified_quest_text, use_emotions_quest_text, apply_effects_override_quest, q_info_display, accept_npc_info_display, complete_npc_info_display, progress_npc_info_display,
                         accept_clean,
                         complete_clean, progress_clean, accept_raw, complete_raw, progress_raw, a_p_og, a_p_vanilla,
                         a_p_tbc, a_p_wotlk, a_p_out, c_p_og, c_p_vanilla,
                         c_p_tbc, c_p_wotlk, c_p_out, p_p_og, p_p_vanilla, p_p_tbc, p_p_wotlk, p_p_out,
                         v_sample_og, v_sample_vanilla, v_sample_tbc, v_sample_wotlk, v_sample_og_alt, v_sample_vanilla_alt, v_sample_tbc_alt, v_sample_wotlk_alt, v_sample_game, v_sample_game_alt, narrator_voices, n_sample_og, n_sample_vanilla, n_sample_tbc, n_sample_wotlk, selected_clone_paths_global['narrator'], narrator_indicator, two_genders_indicator, different_vanilla_indicator, quest_expansion_indicator, q_info_display_expansion, monitor_a, monitor_c, monitor_p, a_p_testing, c_p_testing, p_p_testing, accept_voice, accept_voice_type, complete_voice, complete_voice_type, progress_voice, progress_voice_type]
            )
            accept_voice_change.click(
                fn=update_voice_sample_quest,
                inputs=[df_state, module_name_single, quest_id, accept_voice, accept_voice_type, complete_voice, complete_voice_type, progress_voice, progress_voice_type, gr.State('accept')],
                outputs=[seed, accept_voice_type, complete_voice_type, progress_voice_type, v_sample_og, v_sample_vanilla, v_sample_tbc, v_sample_wotlk, v_sample_game, monitor_a, monitor_c, monitor_p]
            )
            complete_voice_change.click(
                fn=update_voice_sample_quest,
                inputs=[df_state, module_name_single, quest_id, accept_voice, accept_voice_type, complete_voice, complete_voice_type, progress_voice, progress_voice_type, gr.State('complete')],
                outputs=[seed, accept_voice_type, complete_voice_type, progress_voice_type, v_sample_og, v_sample_vanilla, v_sample_tbc, v_sample_wotlk, v_sample_game, monitor_a, monitor_c, monitor_p]
            )
            progress_voice_change.click(
                fn=update_voice_sample_quest,
                inputs=[df_state, module_name_single, quest_id, accept_voice, accept_voice_type, complete_voice, complete_voice_type, progress_voice, progress_voice_type, gr.State('progress')],
                outputs=[seed, accept_voice_type, complete_voice_type, progress_voice_type, v_sample_og, v_sample_vanilla, v_sample_tbc, v_sample_wotlk, v_sample_game, monitor_a, monitor_c, monitor_p]
            )
            accept_voice.change(
                fn=update_voice_sample_quest,
                inputs=[df_state, module_name_single, quest_id, accept_voice, accept_voice_type, complete_voice, complete_voice_type, progress_voice, progress_voice_type, gr.State('accept')],
                outputs=[seed, accept_voice_type, complete_voice_type, progress_voice_type, v_sample_og, v_sample_vanilla, v_sample_tbc, v_sample_wotlk, v_sample_game, monitor_a, monitor_c, monitor_p]
            )
            complete_voice.change(
                fn=update_voice_sample_quest,
                inputs=[df_state, module_name_single, quest_id, accept_voice, accept_voice_type, complete_voice, complete_voice_type, progress_voice, progress_voice_type, gr.State('complete')],
                outputs=[seed, accept_voice_type, complete_voice_type, progress_voice_type, v_sample_og, v_sample_vanilla, v_sample_tbc, v_sample_wotlk, v_sample_game, monitor_a, monitor_c, monitor_p]
            )
            progress_voice.change(
                fn=update_voice_sample_quest,
                inputs=[df_state, module_name_single, quest_id, accept_voice, accept_voice_type, complete_voice, complete_voice_type, progress_voice, progress_voice_type, gr.State('progress')],
                outputs=[seed, accept_voice_type, complete_voice_type, progress_voice_type, v_sample_og, v_sample_vanilla, v_sample_tbc, v_sample_wotlk, v_sample_game, monitor_a, monitor_c, monitor_p]
            )

            with gr.Group():
                with gr.Row():
                    but1 = gr.Button("Generate lookup tables", variant="primary")
                    info1 = gr.Textbox(label="Output", value="")

                    but1.click(
                        tts_processor.generate_lookup_tables,
                        inputs=[
                            df_state,
                            expansions_state,
                            module_name_single,
                        ],
                        outputs=info1,
                    )
        with gr.TabItem("TTS Gossip", id="ttsgossip"):
            expansions_gossip = gr.State()  # Hidden state to store expansions
            with gr.Accordion("⚙️ Extended VibeVoice parameters", open=False):
                with gr.Group():
                    gr.Markdown("### 🧠 Choose model")
                    with gr.Row():
                        model_gossip = gr.Dropdown(
                            label="Model",
                            choices=tts_processor.available_models,
                            value=tts_processor.default_model
                        )
                        # seed = gr.Number(label="Seed", value=42, precision=0)

                with gr.Group():
                    gr.Markdown("### ⚙️ Generation options")
                    with gr.Row():
                        # use_sampling = gr.Checkbox(
                        #     label="Use sampling",
                        #     info="Enable sampling mode. When False (default), uses deterministic generation like official examples",
                        #     value=False)
                        temperature_gossip = gr.Slider(
                            label="Temperature",
                            minimum=0.1, maximum=10.0, value=5.0, step=0.05,
                            info="Higher values increase emotion and variety; lower values improve stability and clarity.",
                            interactive=True
                        )
                        temperature_class_gossip = gr.Slider(
                            label="Class Temperature", 
                            minimum=0.0, maximum=1.0, value=0.3, step=0.05,
                            info="Higher values increase emotion and variety; lower values improve stability and clarity.",
                            interactive=True
                        )

                    with gr.Row():
                        voice_speed_factor_gossip = gr.Slider(
                            label="Voice Speed Factor",
                            minimum=0.8, maximum=1.5, value=1.1, step=0.01,
                            info="1.0 = normal speed, <1.0 = slower speed, >1.0 = faster speed."
                        )
                        voice_cps_gossip = gr.Slider(
                            minimum=8.0,
                            maximum=18.0,
                            value=15,
                            step=0.5,
                            label="CPS (Characters Per Second) — speech speed"
                        )
                        max_words_per_chunk_gossip = gr.Number(
                            label="Max Words Per Chunk",
                            value=75, precision=0,
                            info="Maximum words per chunk for long texts. Lower values prevent speed issues but create more chunks."
                        )

                with gr.Group():
                    gr.Markdown("### ⚡ VRAM optimisations")
                    with gr.Row():
                        attention_type_gossip = gr.Dropdown(
                            label="Attention Type",
                            choices=["auto", "eager", "sdpa", "flash_attention_2", "sage"],
                            value="auto",
                            info="Attention implementation. Auto selects the best available, eager is standard, sdpa is optimized PyTorch, flash_attention_2 requires compatible GPU, sage uses quantized attention for speedup (CUDA only)."
                        )
                        quantize_llm_gossip = gr.Radio(
                            label="Quantize LLM",
                            choices=["full precision", "4bit", "8bit"],
                            value="full precision",
                            info="Dynamically quantize only the LLM component for non-quantized models. 4bit: major VRAM savings with minimal quality loss. 8bit: good balance of quality and memory usage. Full precision: original quality. Note: ignored for pre-quantized models. Requires CUDA GPU."
                        )
                    free_memory_gossip = gr.Checkbox(label="Free memory after generation", value=False)

                with gr.Group():
                    gr.Markdown("### 🎨 Diffusion options")
                    with gr.Row():
                        diffusion_steps_gossip = gr.Slider(
                            label="Steps", minimum=1, maximum=100, value=32, step=1,
                            info="Number of denoising steps. More steps = theoretically better quality but slower. Default: 20"
                        )
                        cfg_scale_gossip = gr.Slider(
                            label="CFG Scale", minimum=0.5, maximum=3.5, value=2.0, step=0.05,
                            info="Classifier-free guidance scale (official default: 1.3)"
                        )
                with gr.Group():
                    gr.Markdown("### 🧠 Settings")
                    with gr.Row():
                        use_sampling_gossip = gr.Checkbox(
                            label="Use sampling",
                            info="Enable sampling mode. When False (default), uses deterministic generation like official examples",
                            value=True)


                        def toggle_sampling_gossip(is_checked):
                            return gr.update(interactive=is_checked), gr.update(interactive=is_checked)


                        use_sampling_gossip.change(
                            fn=toggle_sampling_gossip,
                            inputs=[use_sampling_gossip],
                            outputs=[temperature_gossip, temperature_class_gossip]
                        )

                        randomize_seed_gossip = gr.Checkbox(label="Auto-Random", value=False)
                        volume_match_gossip = gr.Checkbox(label="Match Volume", value=True)
                        volume_match_sample_gossip = gr.Checkbox(label="Volume Match Sample", value=False)


            with gr.Row():
                with gr.Column():
                    npc_name = gr.Textbox(
                        label="NPC Name",
                        interactive = True,
                        placeholder="Enter NPC Name and press Enter"
                    )
                    btn_load_gossip = gr.Button("📥 Load Gossip Info", variant="primary")
                with gr.Column():
                    module_name_gossip = gr.Dropdown(
                        label="Module to export audio to",
                        choices=(
                            ['AI_VoiceOverData_VanillaExtra', 'AI_VoiceOverData_TBC', 'AI_VoiceOverData_WoTLK','Testing']
                        ),
                        value = 'Testing',
                        allow_custom_value = True,
                        interactive = True
                    )
                    force_gen_gossip = gr.Checkbox(
                        label="Replace existing",
                        value=True)
                    # apply_effects_gossip = gr.Checkbox(
                    #     label="Apply Voice effects",
                    #     value=True)
                    # apply_effects_override_gossip = gr.Textbox(
                    #     label="Voice Effect override",
                    #     placeholder="ghost/undead/demon/giant/ancient/small/comms")
                with gr.Column():
                    seed_gossip = gr.Number(label="Seed", value=random.randint(0, 2147483647), precision=0)
                    btn_random_seed_gossip = gr.Button("🎲", scale=1)


                    def get_random_seed_gossip():
                        return random.randint(0, 2147483647)


                    btn_random_seed_gossip.click(
                        fn=get_random_seed_gossip,
                        inputs=[],
                        outputs=seed_gossip
                    )


                    def toggle_seed_interactive_gossip(is_auto):
                        return gr.update(interactive=not is_auto)


                    randomize_seed_gossip.change(
                        fn=toggle_seed_interactive_gossip,
                        inputs=[randomize_seed_gossip],
                        outputs=seed_gossip
                    )

            with gr.Row():
                with gr.Column():
                    apply_effects_gossip = gr.Checkbox(
                        label="Apply Voice effects",
                        value=True)
                    use_emotions_gossip_text = gr.Checkbox(label="Detect emotions", value=False)
                with gr.Column(elem_classes="my-column"):
                    apply_effects_override_gossip = gr.Dropdown(
                        label='Voice effect ovveride',
                        choices=['ghost', 'robot', 'undead', 'demon', 'giant', 'small', 'letter', 'comms', 'dk', 'ancient', 'ethereal', 'earthen', 'naaru', 'wretched'],
                        value=None,
                        allow_custom_value=True)
                with gr.Column():
                    use_narrator_gossip = gr.Checkbox(
                        label="Add Narrator voice",
                        value=True,
                        interactive=False)
                    use_modified_gossip_text = gr.Checkbox(label="Use modified Gossip Text", value=False)


            with gr.Group():
                with gr.Row():
                    gossip_voice = gr.Dropdown(
                            show_label=False,
                            choices=selected_voice_names,
                            value='',
                            interactive=True,
                            allow_custom_value=True
                        )
                    gossip_voice_type = gr.Dropdown(
                            show_label=False,
                            choices=get_possible_voice_types(),
                            value='standard',
                            interactive=True,
                            allow_custom_value=True
                        )
                    gossip_voice_change = gr.Button("Change Gossip Voice")
            with gr.Group() as main_panel_gossip:
                with gr.Row():
                    npc_info_display = gr.Markdown("### 🧐 NPC: [Choose Name]")

                with gr.Row(visible=False) as narrator_indicator_gossip:
                    npc_info_display_two_speakers = gr.Markdown("### 🧑‍‍🧑 <span style='color: red'>Narrator text detected</span>")

                with gr.Row(visible=False) as two_genders_indicator_gossip:
                    npc_info_display_two_genders = gr.Markdown("### ️♂️♀️ <span style='color: red'>Two genders text detected</span>")

            with gr.Group() as gossip_group:
                manager = GossipManager()
                manager.create_ui(df_state)

                all_outputs = [seed_gossip, npc_info_display, current_npc_name, use_modified_gossip_text, use_emotions_gossip_text, apply_effects_override_gossip]  # Start with container
                for comp in manager.components:
                    # Add each component from the dict in the order Gradio expects
                    all_outputs.extend([
                        comp['group'],
                        comp['name'],
                        comp['text'],
                        comp['original_text'],
                        comp['original_cleaned_text'],
                        comp['audio_og'],
                        comp['audio_vanilla'],
                        comp['audio_tbc'],
                        comp['audio_wotlk'],
                        comp['audio_output'],
                        comp['btn']
                    ])

                all_outputs.append(manager.v_sample_og_gossip)
                all_outputs.append(manager.v_sample_vanilla_gossip)
                all_outputs.append(manager.v_sample_tbc_gossip)
                all_outputs.append(manager.v_sample_wotlk_gossip)
                all_outputs.append(manager.v_sample_game_gossip)

                all_outputs.append(manager.narrator_voices_gossip)
                all_outputs.append(manager.n_sample_og_gossip)
                all_outputs.append(manager.n_sample_vanilla_gossip)
                all_outputs.append(manager.n_sample_tbc_gossip)
                all_outputs.append(manager.n_sample_wotlk_gossip)
                all_outputs.append(manager.monitors_global_gossip['narrator'])
                all_outputs.append(narrator_indicator_gossip)
                all_outputs.append(two_genders_indicator_gossip)
                all_outputs.append(manager.monitors_gossip['npc'])
                all_outputs.append(gossip_voice)
                all_outputs.append(gossip_voice_type)

                btn_load_gossip.click(
                    manager.update_from_df,
                    inputs=[df_state, npc_name, current_npc_name, manager.gossip_texts_state, use_modified_gossip_text],
                    outputs=all_outputs
                )

                gossip_voice.change(
                    fn=manager.update_voice_sample_gossip,
                    inputs=[df_state, module_name_single, npc_name, gossip_voice, gossip_voice_type],
                    outputs=[seed_gossip, gossip_voice_type, manager.v_sample_og_gossip, manager.v_sample_vanilla_gossip, manager.v_sample_tbc_gossip, manager.v_sample_wotlk_gossip, manager.v_sample_game_gossip, manager.monitors_gossip['npc']]
                )

                gossip_voice_change.click(
                    fn=manager.update_voice_sample_gossip,
                    inputs=[df_state, module_name_single, npc_name, gossip_voice, gossip_voice_type],
                    outputs=[seed_gossip, gossip_voice_type, manager.v_sample_og_gossip, manager.v_sample_vanilla_gossip, manager.v_sample_tbc_gossip, manager.v_sample_wotlk_gossip, manager.v_sample_game_gossip, manager.monitors_gossip['npc']]
                )

            with gr.Group():
                with gr.Row():
                    but3 = gr.Button("Generate lookup tables", variant="primary")
                    info3 = gr.Textbox(label="Output", value="")

                    but3.click(
                        tts_processor.generate_lookup_tables,
                        inputs=[
                            df_state,
                            expansions_gossip,
                            module_name_gossip,
                        ],
                        outputs=info3,
                    )
        with gr.TabItem("Batch TTS"):
            with gr.Group():
                with gr.Accordion("Select voices to produce audio on"):
                    load_voices_button = gr.Button("Load available voices", variant="primary")
                    selected_voice_names = gr.CheckboxGroup(
                            label="Choose all the voices that you want to generate audio for. By default, all voices are selected." ,
                            choices=[],
                            value = [],
                            show_label = True,
                            show_select_all = True,
                            interactive = True
                    )
                    load_voices_button.click(
                        load_voices,
                        inputs=[df_state],
                        outputs=selected_voice_names
                    )
            with gr.Group():
                with gr.Row():
                    with gr.Column():
                        xtts_param_label = gr.Markdown("XTTS Parameters")
                        temperature_batch = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label="Temperature",
                            value=5.0,
                            interactive=True,
                        )
                        length_penalty_batch = gr.Slider(
                            minimum=-10,
                            maximum=10,
                            label="Length penalty",
                            value=1,
                            interactive=True,
                        )
                        repetition_penalty_batch = gr.Slider(
                            minimum=0,
                            maximum=15,
                            label="Repetition penalty",
                            value=10,
                            step =0.5,
                            interactive=True,
                        )
                        top_k_batch = gr.Slider(
                            minimum=1,
                            maximum=100,
                            label="Top k",
                            value=1,
                            step = 1,
                            interactive=True,
                        )
                        top_p_batch = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label="Top p",
                            value=1.0,
                            interactive=True,
                        )
                        speed_batch = gr.Slider(
                            minimum=0.01,
                            maximum=2,
                            label="Speed",
                            value=1.05,
                            interactive=True,
                        )
                    with gr.Column():
                        rvc_param_label = gr.Markdown("RVC Parameters")
                        with gr.Row():
                            f0_method_batch = gr.Radio(
                                    label=" Select the pitch extraction algorithm",
                                    choices=(
                                        ["pm", "harvest", "crepe", "rmvpe"]
                                    ),
                                    value = "rmvpe"
                                )
                        f0_up_key_batch = gr.Textbox(
                            value = 0,
                            visible = False
                        )
                        index_rate_batch = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label="Index rate to control accent strength",
                            value=0.75,
                            interactive=True,
                        )
                        filter_radius_batch = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label="Apply median filter to harvested pitch results",
                            value=3,
                            step=1,
                            interactive=True,
                        )
                        resample_sr_batch = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label="Target resample rate after processing, set to 0 for no resampling",
                            value=0,
                            step=1,
                            interactive=True,
                        )
                        rms_mix_rate_batch = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label="Adjust the volume envelope scaling.",
                            value=0.25,
                            interactive=True,
                        )
                        protect_batch = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label="Protect rate",
                            value=0.25,
                            step=0.01,
                            interactive=True,
                        )
            with gr.Group():
                with gr.Row():
                    with gr.Column():
                        expansion_label = gr.Markdown("Choose Expansion(s) to produce audio for")
                        expansions = gr.Dropdown(
                            label="",
                            choices=(
                                ['Vanilla', 'TBC', 'WoTLK']
                            ),
                            value = ['Vanilla', 'TBC', 'WoTLK'],
                            type = 'index',
                            multiselect = True,
                            interactive = True
                            )
                    with gr.Column():
                        module_label = gr.Markdown("Module to export audio to")
                        module_name = gr.Dropdown(
                            label="",
                            choices=(
                                ['AI_VoiceOverData_VanillaExtra', 'AI_VoiceOverData_TBC', 'AI_VoiceOverData_WoTLK','Testing']
                            ),
                            value = 'Testing',
                            allow_custom_value = True,
                            interactive = True
                            )
            with gr.Group():
                with gr.Row():
                    but0 = gr.Button("Generate audio", variant="primary")
                    info0 = gr.Textbox(label="Output", value="")

                    but0.click(
                        tts_processor.tts_dataframe,
                        inputs=[
                            df_state,
                            selected_voice_names,
                            expansions,
                            module_name,
                            temperature_batch,
                            length_penalty_batch,
                            repetition_penalty_batch,
                            top_k_batch,
                            top_p_batch,
                            speed_batch,
                            f0_up_key_batch,
                            f0_method_batch,
                            index_rate_batch,
                            filter_radius_batch,
                            resample_sr_batch,
                            rms_mix_rate_batch,
                            protect_batch

                        ],
                        outputs=info0,
                    )
                    with gr.Row():
                        but1 = gr.Button("Generate lookup tables", variant="primary")
                        info1 = gr.Textbox(label="Output", value="")

                        but1.click(
                            tts_processor.generate_lookup_tables,
                            inputs=[
                                df_state,
                                expansions,
                                module_name,
                            ],
                            outputs=info1,
                        )

        with gr.TabItem("TTS by quest ZONE"):
            with gr.Group():
                with gr.Accordion("Select Zone to track quest Voice Over progress")  as ttszone_quest_list:
                    load_available_zones_ttszone = gr.Button("Load available zones", variant="primary")
                    selected_zones_ttszone = gr.Radio(
                            label="Choose the zone to load quests from that zone" ,
                            choices=[],
                            show_label = True,
                            interactive = True
                    )
                    load_available_zones_ttszone.click(
                        load_zones,
                        inputs=[df_state],
                        outputs=[selected_zones_ttszone, load_available_zones_ttszone]
                    )
            
                    selected_zones_ttszone.change(
                        hide_stuff,
                        inputs=None,
                        outputs=[ttszone_quest_list]
                    )

            with gr.Group():
                btn_load_quests_ttszone = gr.Button("📥 Load Quests from that zone", variant="primary")
            with gr.Group():
                zone_name_ttszone = gr.Markdown("### Quests in (Zone)", visible=False)
            with gr.Group():
                with gr.Row():
                    ttszone_fix_accept = gr.Button("📥 Fix all Accept texts", variant="primary")
                    ttszone_fix_complete = gr.Button("📥 Fix all Complete texts", variant="primary")
                    ttszone_fix_progress = gr.Button("📥 Fix all Progress texts", variant="primary")
            with gr.Group():
                with gr.Row():
                        dataframe_display_ttszone = gr.Dataframe(
                            show_search = 'search',
                            wrap = True,
                            interactive=False,
                            datatype=['str'] * 10 + ['html'] * 7,
                            column_widths = ["90px","70px","90px","90px","60px","150px", "150px", "300px","80px","100px", "50px", "50px", "50px", "50px", "50px", "50px", "50px", "50px", "50px"]
                            #header_tooltips=["Type of quest text: accept, complete, progress", "Quest ID", "Quest Title", "Quest Giver/Taker Name", "Quest Giver/Taker ID", "Voice name", "Begining of quest text", "Expansion", "Zone", "Has VO in Original Addon", "Has VO in VanillaExtra module", "Has VO in TBC module", "Has VO in WoTLK module", "Has Narrator Voice part in <>", "Has G: male:female part"],
                        )
            # with gr.Group() as quests_zones_group:
            #     btn_load_quests_ttszone.click(
            #         select_columns_ttszone,
            #         inputs=[df_state, selected_zones_ttszone],
            #         outputs=[dataframe_display_ttszone, zone_name_ttszone, zone_name_ttszone, btn_zip_quests_ttszone, btn_zip_quests_ttszone]
            #     )

            with gr.Group():
                with gr.Row():
                    with gr.Column():
                        btn_zip_quests_ttszone = gr.Button("📦 ZIP Quests for that zone", variant="primary", visible=False)
                    with gr.Column():
                        zip_quests_ttszone_prefix = gr.Dropdown(
                            label="ZIP prefix",
                            choices=["-silence", "-narrator", "-combined"],
                            value="-combined"
                        )
                with gr.Row():
                    zip_quests_output_ttszone = gr.File(label="Download Zone Zip", visible=False)

                btn_zip_quests_ttszone.click(
                    zip_generated_ttszone,
                    inputs=[df_state, selected_zones_ttszone, zip_quests_ttszone_prefix],
                    outputs=[zip_quests_output_ttszone, zip_quests_output_ttszone]
                )
                btn_load_quests_ttszone.click(
                    select_columns_ttszone,
                    inputs=[df_state, selected_zones_ttszone],
                    outputs=[dataframe_display_ttszone, zone_name_ttszone, zone_name_ttszone, btn_zip_quests_ttszone, btn_zip_quests_ttszone]
                )

        with gr.TabItem("TTS by gossip ZONE"):
            with gr.Group():
                with gr.Accordion("Select Zone to track gossip Voice Over progress"):
                    load_available_zones_ttszone_gossip = gr.Button("Load available zones", variant="primary")
                    selected_zones_ttszone_gossip = gr.Radio(
                            label="Choose the zone to load gossips from that zone" ,
                            choices=[],
                            show_label = True,
                            interactive = True
                    )
                    load_available_zones_ttszone_gossip.click(
                        load_zones,
                        inputs=[df_state],
                        outputs=[selected_zones_ttszone_gossip, load_available_zones_ttszone_gossip]
                    )

            with gr.Group():
                btn_load_gossip_ttszone = gr.Button("📥 Load Gossips from that zone", variant="primary")
            with gr.Group():
                zone_name_ttszone_gossip = gr.Markdown("### Gossips in (Zone)", visible=False)
            with gr.Group():
                with gr.Row():
                    ttszone_fix_gossip = gr.Button("📥 Fix all Gossips", variant="primary")
            with gr.Group():
                with gr.Row():
                        dataframe_display_ttszone_gossip = gr.Dataframe(
                            show_search = 'search',
                            wrap = True,
                            interactive=False,
                            datatype=['str'] * 8 + ['html'] * 7,
                            column_widths = ["90px","60px","150px", "150px", "300px","80px","100px", "100px", "50px", "50px", "50px", "50px", "50px", "50px", "50px", "50px"]
                            #header_tooltips=["Type of quest text: accept, complete, progress", "Quest ID", "Quest Title", "Quest Giver/Taker Name", "Quest Giver/Taker ID", "Voice name", "Begining of quest text", "Expansion", "Zone", "Has VO in Original Addon", "Has VO in VanillaExtra module", "Has VO in TBC module", "Has VO in WoTLK module", "Has Narrator Voice part in <>", "Has G: male:female part"],
                        )
            # with gr.Group() as quests_zones_group:
            #     btn_load_quests_ttszone.click(
            #         select_columns_ttszone,
            #         inputs=[df_state, selected_zones_ttszone],
            #         outputs=[dataframe_display_ttszone, zone_name_ttszone, zone_name_ttszone, btn_zip_quests_ttszone, btn_zip_quests_ttszone]
            #     )

            with gr.Group():
                with gr.Row():
                    with gr.Column():
                        btn_zip_gossips_ttszone = gr.Button("📦 ZIP Gossips for that zone", variant="primary", visible=False)
                    with gr.Column():
                        zip_gossip_ttszone_prefix = gr.Dropdown(
                            label="ZIP prefix",
                            choices=["-silence", "-narrator", "-combined"],
                            value="-combined"
                        )
                with gr.Row():
                    zip_gossips_output_ttszone = gr.File(label="Download Zone Zip", visible=False)

                btn_zip_gossips_ttszone.click(
                    zip_generated_ttszone_gossip,
                    inputs=[df_state, selected_zones_ttszone_gossip, zip_gossip_ttszone_prefix],
                    outputs=[zip_gossips_output_ttszone, zip_gossips_output_ttszone]
                )
                btn_load_gossip_ttszone.click(
                    select_columns_ttszone_gossip,
                    inputs=[df_state, selected_zones_ttszone_gossip],
                    outputs=[dataframe_display_ttszone_gossip, zone_name_ttszone_gossip, zone_name_ttszone_gossip, btn_zip_gossips_ttszone, btn_zip_gossips_ttszone]
                )

        with gr.TabItem("TTS by Zone"):
            with gr.Group():
                with gr.Accordion("Select Zone to track Voice Over progress")  as ttszone_quest_list_full:
                    load_available_zones_ttszone_full = gr.Button("Load available zones", variant="primary")
                    selected_zones_ttszone_full = gr.Radio(
                            label="Choose the zone to load voice over from that zone" ,
                            choices=[],
                            show_label = True,
                            interactive = True
                    )
                    load_available_zones_ttszone_full.click(
                        load_zones,
                        inputs=[df_state],
                        outputs=[selected_zones_ttszone_full, load_available_zones_ttszone_full]
                    )
            
                    selected_zones_ttszone_full.change(
                        hide_stuff,
                        inputs=None,
                        outputs=[ttszone_quest_list_full]
                    )

            with gr.Group():
                btn_load_quests_ttszone_full = gr.Button("📥 Load VO stats from that zone", variant="primary")
            with gr.Group():
                zone_name_ttszone_full = gr.Markdown("### VoiceOver in (Zone)", visible=True)
            with gr.Group():
                with gr.Row():
                    ttszone_fix_all_full = gr.Button("📥 Fix all VoiceOver Auto", variant="primary")
            with gr.Group():
                with gr.Row():
                    dataframe_display_ttszone_full = gr.Dataframe(
                        show_search = 'search',
                        wrap = True,
                        interactive=False,
                        datatype=['str'] * 10 + ['html'] * 7,
                        column_widths = ["90px","70px","90px","90px","60px","150px", "150px", "300px","80px","100px", "50px", "50px", "50px", "50px", "50px", "50px", "50px", "50px", "50px"]
                        #header_tooltips=["Type of quest text: accept, complete, progress", "Quest ID", "Quest Title", "Quest Giver/Taker Name", "Quest Giver/Taker ID", "Voice name", "Begining of quest text", "Expansion", "Zone", "Has VO in Original Addon", "Has VO in VanillaExtra module", "Has VO in TBC module", "Has VO in WoTLK module", "Has Narrator Voice part in <>", "Has G: male:female part"],
                    )
            with gr.Group():
                btn_remove_bad_ttszone_full = gr.Button("📥 Check and remove bad VO", variant="primary")

            with gr.Group():
                btn_load_quests_ttszone_full.click(
                    select_columns_ttszone_full,
                    inputs=[df_state, selected_zones_ttszone_full],
                    outputs=[dataframe_display_ttszone_full, zone_name_ttszone_full]
                )
                btn_remove_bad_ttszone_full.click(
                    select_columns_ttszone_full,
                    inputs=[df_state, selected_zones_ttszone_full, gr.State('remove_bad')],
                    outputs=[dataframe_display_ttszone_full, zone_name_ttszone_full]
                )

        with gr.TabItem("Sample Manager"):
            with gr.Group():
                with gr.Accordion("Select voices to select best sample for"):
                    load_voices_button_sample = gr.Button("Load available voices", variant="primary")
                    selected_voice_names_sample = gr.Radio(
                            label="Choose the voice that you want to select sample for." ,
                            choices=[],
                            value = None,
                            show_label = True,
                            interactive = True,
                    )
                    load_voices_button_sample.click(
                        load_voices_radio,
                        inputs=[df_state],
                        outputs=[selected_voice_names_sample, load_voices_button_sample]
                    )

            with gr.Group():
                btn_load_samples = gr.Button("📥 Load 20 Samples", variant="primary")
            with gr.Group():
                voice_name_samples = gr.Markdown("### Voice: ", visible=False)
            with gr.Group() as samples_group:
                samples_manager = VoiceSampleManager()
                samples_manager.create_ui_samples(df_state)
                all_outputs = [voice_name_samples, voice_name_samples, samples_manager.sample_voice_current]
                for comp in samples_manager.components:
                    all_outputs.extend([
                        comp['sample']
                    ])

                btn_load_samples.click(
                    samples_manager.update_samples,
                    inputs=[df_state, selected_voice_names_sample],
                    outputs=all_outputs
                )

        with gr.TabItem("Seed Manager"):

            def get_seeds_table_data(voice_name, voice_type):
                """
                Возвращает данные для таблицы в формате list of lists
                """
                voice_key = tts_processor.seed_manager._get_key(voice_name, voice_type)
                if not voice_key or voice_key not in tts_processor.seed_manager.data:
                    return [["Нет данных", "", ""]]

                voice_data = tts_processor.seed_manager.data[voice_key]
                seeds_dict = voice_data.get("seeds", {})

                if not seeds_dict:
                    return [["Нет сохранённых сидов", "", ""]]

                default_seed = voice_data.get("default_seed")

                # Формируем данные для таблицы
                rows = []
                for file_name, seed_info in seeds_dict.items():
                    seed = seed_info["seed"]
                    is_default = "⭐" if seed == default_seed else ""
                    rows.append([file_name, str(seed), is_default])

                # Сортируем: сначала дефолтные, потом по имени
                rows.sort(key=lambda x: (x[2] != "⭐", x[0]))

                return rows

            def refresh_table(voice_name, voice_type):
                """Обновляет таблицу"""
                if not voice_name:
                    return [["Выберите голос", "", ""]]

                data = get_seeds_table_data(voice_name, voice_type)
                return data

            def set_as_default(voice_name, voice_type, evt: gr.SelectData):
                """
                Устанавливает выбранный seed как дефолтный
                evt: событие выбора строки в таблице
                """
                if not voice_name:
                    gr.Warning("Сначала выберите голос")
                    return gr.update(), "❌ Голос не выбран"

                # Получаем данные таблицы
                table_data = get_seeds_table_data(voice_name, voice_type)

                if evt.index[0] >= len(table_data):
                    gr.Warning("Ошибка выбора строки")
                    return gr.update(), "❌ Ошибка"

                # Берём имя файла из выбранной строки
                file_name = table_data[evt.index[0]][0]

                # Получаем seed
                voice_key = tts_processor.seed_manager._get_key(voice_name, voice_type)
                seed_info = tts_processor.seed_manager.data[voice_key]["seeds"].get(file_name)

                if seed_info:
                    seed = seed_info["seed"]
                    if tts_processor.seed_manager.get_default_seed(voice_name, voice_type) == seed:
                        tts_processor.seed_manager.mark_as_default(voice_name, voice_type, None)
                        gr.Info(f"⭐ Дефолтный seed для {voice_key} удален")
                        return refresh_table(voice_name, voice_type), f"⭐ Дефолтный seed: Нет"
                    else:
                        tts_processor.seed_manager.mark_as_default(voice_name, voice_type, seed)
                        gr.Info(f"⭐ Дефолтный seed для {voice_key} = {seed} ({file_name})")
                        return refresh_table(voice_name, voice_type), f"⭐ Дефолтный seed: {seed}"

                return refresh_table(voice_name, voice_type), "❌ Ошибка"

            with gr.Group():
                with gr.Accordion("Select voices to select best seed for"):
                    load_voices_button_seed = gr.Button("Load available voices", variant="primary")
                    selected_voice_names_seed = gr.Radio(
                            label="Choose the voice that you want to select seed for." ,
                            choices=[],
                            value = None,
                            show_label = True,
                            interactive = True,
                    )
                    load_voices_button_seed.click(
                        load_voices_radio,
                        inputs=[df_state],
                        outputs=[selected_voice_names_seed, load_voices_button_seed]
                    )
                    seed_voice_type = gr.Dropdown(
                            show_label=False,
                            choices=get_possible_voice_types(),
                            value='standard',
                            interactive=True
                        )
                with gr.Row():
                    with gr.Column(scale=1):
                        refresh_btn = gr.Button("🔄 Обновить таблицу", variant="secondary")
                        status_text = gr.Textbox(label="📌 Статус", interactive=False)
                    
                with gr.Column(scale=2):
                    seeds_table = gr.Dataframe(
                        headers=["File Name", "Seed", "Default"],
                        datatype=["str", "str", "str"],
                        interactive=False,
                        wrap=True,
                        show_search = 'search',
                        label="📋 Saved seeds"
                    )

                # Обработчики событий
                refresh_btn.click(
                    fn=refresh_table,
                    inputs=[selected_voice_names_seed, seed_voice_type],
                    outputs=[seeds_table]
                )
                
                # selected_voice_names_seed.change(
                #     fn=refresh_table,
                #     inputs=[selected_voice_names_seed],
                #     outputs=seeds_table
                # )

                seed_voice_type.change(
                    fn=refresh_table,
                    inputs=[selected_voice_names_seed, seed_voice_type],
                    outputs=[seeds_table]
                )

                selected_voice_names_seed.change(
                    fn=get_possible_voice_types_dropdown,
                    inputs=[selected_voice_names_seed],
                    outputs=[seed_voice_type]
                )
                
                # При клике на строку таблицы
                seeds_table.select(
                    fn=set_as_default,
                    inputs=[selected_voice_names_seed, seed_voice_type],
                    outputs=[seeds_table, status_text]
                )

        with gr.TabItem("Export Module & Lookup tables"):
            with gr.Group():
                with gr.Row():
                    module_name_export = gr.Dropdown(
                            label="Module to export",
                            choices=(
                                ['AI_VoiceOverData_VanillaExtra', 'AI_VoiceOverData_TBC', 'AI_VoiceOverData_WoTLK', 'AI_VoiceOverData_Turtle']
                            ),
                            value='AI_VoiceOverData_VanillaExtra',
                            allow_custom_value=False,
                            interactive=True
                        )
                    with gr.Column():
                        module_volume_fix_export = gr.Checkbox(
                            label="Increase volume",
                            value=False
                        )
                        module_volume_fix_export_value = gr.Slider(
                            label="Target Volume", 
                            minimum=-24.0, maximum=-12.0, value=-16.0, step=0.5
                        )
                with gr.Row():
                    module_export_start = gr.Button("📥 Start Export", variant="primary")
                with gr.Row():
                    info_export = gr.Textbox(label="Output", value="")
                    module_export_start.click(
                        tts_processor.export_module_full,
                        inputs=[
                            df_state,
                            module_name_export,
                            module_volume_fix_export,
                            module_volume_fix_export_value
                        ],
                        outputs=info_export,
                    )

    dataframe_display_ttszone.select(fn=on_cell_select, inputs=[dataframe_display_ttszone],
                             outputs=[copy_trigger, quest_id, npc_name, tabs_container])
    
    dataframe_display_ttszone.select(fn=on_cell_select_ttszone, inputs=[df_state, dataframe_display_ttszone],
                             outputs=[dataframe_display_ttszone, module_name_single, module_name_gossip])

    dataframe_display_ttszone_gossip.select(fn=on_cell_select, inputs=[dataframe_display_ttszone_gossip],
                             outputs=[copy_trigger, quest_id, npc_name, tabs_container])
    
    dataframe_display_ttszone_gossip.select(fn=on_cell_select_ttszone_gossip, inputs=[df_state, dataframe_display_ttszone_gossip],
                             outputs=[dataframe_display_ttszone_gossip, module_name_single, module_name_gossip])

    dataframe_display.select(fn=on_cell_select, inputs=[dataframe_display],
                             outputs=[copy_trigger, quest_id, npc_name, tabs_container])
    
    ttszone_fix_accept.click(fn=auto_fix_ttszone, inputs=[df_state, dataframe_display_ttszone, selected_zones_ttszone, gr.State('accept')], outputs=[dataframe_display_ttszone, ttszone_fix_accept])
    ttszone_fix_complete.click(fn=auto_fix_ttszone, inputs=[df_state, dataframe_display_ttszone, selected_zones_ttszone, gr.State('complete')], outputs=[dataframe_display_ttszone, ttszone_fix_complete])
    ttszone_fix_progress.click(fn=auto_fix_ttszone, inputs=[df_state, dataframe_display_ttszone, selected_zones_ttszone, gr.State('progress')], outputs=[dataframe_display_ttszone, ttszone_fix_progress])
    
    ttszone_fix_gossip.click(fn=auto_fix_ttszone_gossip, inputs=[df_state, dataframe_display_ttszone_gossip, selected_zones_ttszone_gossip], outputs=[dataframe_display_ttszone_gossip, ttszone_fix_gossip])
    
    ttszone_fix_all_full.click(fn=auto_fix_ttszone_full, inputs=[df_state, selected_zones_ttszone_full], outputs=[dataframe_display_ttszone_full])
    # dataframe_display_turtle.select(fn=on_cell_select, inputs=[dataframe_display_turtle],
    #                          outputs=[copy_trigger, quest_id, npc_name, tabs_container])

    copy_trigger.change(None, inputs=[copy_trigger], outputs=None, js=copy_to_clipboard_js)

    but_turtle_load.click(
                        lambda: clean_quest_data_turtle(tts_processor),
                        inputs=[],
                        outputs=[info_turtle_load,df_state, module_name_single, module_name_gossip],
                    )
    

app.launch(
    theme = gr.themes.Ocean(),
    css=css,
    server_name="0.0.0.0",
    server_port=7280
)
