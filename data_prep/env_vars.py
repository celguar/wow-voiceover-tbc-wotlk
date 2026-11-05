from dotenv import load_dotenv
load_dotenv()
import os

MYSQL_HOST = "127.0.0.1"#os.getenv("MYSQL_HOST")
MYSQL_PORT = "3312"#int(os.getenv("MYSQL_PORT"))
MYSQL_USER = "root"#os.getenv("MYSQL_USER")
MYSQL_PASSWORD = "cmangos"#os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = "voiceover_data"#os.getenv("MYSQL_DATABASE")
MYSQL_PATH = r"I:\voiceover\mysql-server\bin\mysql.exe"
SOX_PATH = r"I:\voiceover\sox\sox.exe"
MODELS_DIR = r"I:\voiceover\model"
ADDON_PATH = r"I:\voiceover\script\AddOn"
ADDON_PATH_WORKS = r"I:\voiceover\script\AddOn_Works"
ADDON_PATH_EXPORT = r"I:\voiceover\script\AddOn_Export"
VOICE_SAMPLE_PATH = r"I:\voiceover\script\voices"
VOICE_PROMPTS_PATH = r"I:\voiceover\script\voices\prompts"
MODULE_NAMES = ['AI_VoiceOverData_Vanilla', 'AI_VoiceOverData_VanillaExtra', 'AI_VoiceOverData_TBC',
                      'AI_VoiceOverData_WoTLK']
MODULE_NAMES_SHORT = ['Vanilla', 'VanillaExtra', 'TBC',
                      'WoTLK']
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
