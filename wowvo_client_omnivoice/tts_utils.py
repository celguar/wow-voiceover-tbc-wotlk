import requests
import os
import pandas as pd
from tqdm import tqdm
import hashlib
from concurrent.futures import ThreadPoolExecutor
import re
from wowvo_client.consts import RACE_DICT, GENDER_DICT, VOICE_MODEL_MAP, NPC_EFFECTS
from wowvo_client.length_table import write_sound_length_table_lua
from wowvo_client.utils import get_first_n_words, get_last_n_words, replace_dollar_bs_with_space
from slpp import slpp as lua
from wowvo_client.audio_effects import *
from pydub import AudioSegment
import io
from wowvo_client.tts_engine import TTSEngine, get_available_models
import subprocess
import glob
from pathlib import Path
import shutil
