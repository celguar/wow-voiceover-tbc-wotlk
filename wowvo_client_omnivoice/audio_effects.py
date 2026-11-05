from pydub import AudioSegment
import subprocess, os
import librosa
import soundfile as sf
from data_prep.env_vars import SOX_PATH
import tempfile
import torch
import numpy as np

def is_sox_command(effect_string):
    import re
    """Check if string looks like a SOX command"""
    # SOX commands typically have words like: pitch, reverb, lowpass, echo, gain, etc.
    sox_keywords = ['pitch', 'reverb', 'lowpass', 'highpass', 'echo', 'gain', 
                    'tremolo', 'tempo', 'bass', 'treble', 'overdrive', 'flanger',
                    'chorus', 'phaser', 'compand', 'norm', 'vol']
    
    effect_lower = effect_string.lower().strip()
    
    # Check if it contains any SOX keywords
    for keyword in sox_keywords:
        if keyword in effect_lower:
            return True
    
    # Check if it has spaces and numbers (typical SOX command pattern)
    if re.search(r'\b\d+\s*\w+\b', effect_lower):  # e.g., "3000", "0.6", "15"
        return True
    
    return False

def dk_effects(mp3_path):
    # Load the mp3
    sound = AudioSegment.from_file(mp3_path)

    # Work in-place using temporary intermediate files with .wav extension

    base_path = os.path.join(mp3_path[:-4])  # strip .mp3
    working_path = base_path + "_proc.wav"
    sound.export(working_path, format="wav")

    reverbed = base_path + "_reverb.wav"
    subprocess.run([
        SOX_PATH, working_path, reverbed,
        'reverb', '50', '50', '75', '100', '25', '-5',
    ], check=True)
    working_path = reverbed

    bassed = base_path + "_bassed.wav"

    subprocess.run([
        SOX_PATH, working_path, bassed,
        'bass', '10'], check=True)

    working_path = bassed


    tremmed = base_path + "_tremolo.wav"
    subprocess.run([SOX_PATH, working_path, tremmed,
                    'tremolo', '200',
                    'gain', '-n'], check=True)
    working_path = tremmed


    y, sr = librosa.load(working_path, sr=None)
    processed = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=-0.50) #raspy voice

    # Save processed audio to temp wav file
    temp_final_wav = base_path + "_final.wav"
    sf.write(temp_final_wav, processed, sr)

    # Final: overwrite original .mp3 with processed audio
    final = AudioSegment.from_file(temp_final_wav)
    final.export(mp3_path, format="mp3", bitrate="64k")

    # Clean up temp wavs
    for f in os.listdir(os.path.dirname(mp3_path)):
        if f.startswith(os.path.basename(base_path)) and f.endswith(".wav"):
            os.remove(os.path.join(os.path.dirname(mp3_path), f))

def robot_effects(mp3_path):
    # Load the mp3
    sound = AudioSegment.from_file(mp3_path)

    # Work in-place using temporary intermediate files with .wav extension

    base_path = os.path.join(mp3_path[:-4])  # strip .mp3
    working_path = base_path + "_proc.wav"
    sound.export(working_path, format="wav")

    changed = base_path + "_changed.wav"
    subprocess.run([SOX_PATH, working_path, changed,
                    'tremolo', '100000',
                    'pitch', '-250',
                    'lowpass', '4000',      # Remove high frequencies
                    'gain', '-n',], check=True)
    working_path = changed


    # Final: overwrite original .mp3 with processed audio
    final = AudioSegment.from_file(working_path)
    final.export(mp3_path, format="mp3", bitrate="64k")

    # Clean up temp wavs
    for f in os.listdir(os.path.dirname(mp3_path)):
        if f.startswith(os.path.basename(base_path)) and f.endswith(".wav"):
            os.remove(os.path.join(os.path.dirname(mp3_path), f))


def ghost_effects(mp3_path):
    # Load the mp3
    sound = AudioSegment.from_file(mp3_path)

    # Work in-place using temporary intermediate files with .wav extension

    base_path = os.path.join(mp3_path[:-4])  # strip .mp3
    working_path = base_path + "_proc.wav"
    sound.export(working_path, format="wav")

    changed = base_path + "_changed.wav"
    subprocess.run([
        SOX_PATH, working_path, changed,
        'reverb', '100', '70', '50', '80', '50', '0',
        'gain','-n',
    ], check=True)
    working_path = changed

    # Final: overwrite original .mp3 with processed audio
    final = AudioSegment.from_file(working_path)
    final.export(mp3_path, format="mp3", bitrate="64k")

    # Clean up temp wavs
    for f in os.listdir(os.path.dirname(mp3_path)):
        if f.startswith(os.path.basename(base_path)) and f.endswith(".wav"):
            os.remove(os.path.join(os.path.dirname(mp3_path), f))

def undead_effects(mp3_path):

    # Load the mp3
    sound = AudioSegment.from_file(mp3_path)

    # Work in-place using temporary intermediate files with .wav extension

    base_path = os.path.join(mp3_path[:-4])  # strip .mp3
    working_path = base_path + "_proc.wav"
    sound.export(working_path, format="wav")

    changed = base_path + "_changed.wav"
    subprocess.run([
        SOX_PATH, working_path, changed,
        'reverb', '40', '40', '40', '50', '25', '-5'
    ], check=True)
    working_path = changed

    # Final: overwrite original .mp3 with processed audio
    final = AudioSegment.from_file(working_path)
    final.export(mp3_path, format="mp3", bitrate="64k")

    # Clean up temp wavs
    for f in os.listdir(os.path.dirname(mp3_path)):
        if f.startswith(os.path.basename(base_path)) and f.endswith(".wav"):
            os.remove(os.path.join(os.path.dirname(mp3_path), f))

def demon_effects(mp3_path):

    # Load the mp3
    sound = AudioSegment.from_file(mp3_path)

    # Work in-place using temporary intermediate files with .wav extension

    base_path = os.path.join(mp3_path[:-4])  # strip .mp3
    working_path = base_path + "_proc.wav"
    sound.export(working_path, format="wav")

    changed = base_path + "_changed.wav"
    subprocess.run([
        SOX_PATH, working_path, changed,
        'pitch', '-100',
        'reverb', '32', '40', '40', '50', '25', '-5',
        'gain','-n',
    ], check=True)
    working_path = changed

    # Final: overwrite original .mp3 with processed audio
    final = AudioSegment.from_file(working_path)
    final.export(mp3_path, format="mp3", bitrate="64k")

    # Clean up temp wavs
    for f in os.listdir(os.path.dirname(mp3_path)):
        if f.startswith(os.path.basename(base_path)) and f.endswith(".wav"):
            os.remove(os.path.join(os.path.dirname(mp3_path), f))

def ethereal_effects(mp3_path):

    # Load the mp3
    sound = AudioSegment.from_file(mp3_path)

    # Work in-place using temporary intermediate files with .wav extension

    base_path = os.path.join(mp3_path[:-4])  # strip .mp3
    working_path = base_path + "_proc.wav"
    sound.export(working_path, format="wav")

    changed = base_path + "_changed.wav"
    subprocess.run([
        SOX_PATH, working_path, changed,
        'reverb', '50', '60', '100', '100', '100', '-6'
    ], check=True)
    working_path = changed

    # Final: overwrite original .mp3 with processed audio
    final = AudioSegment.from_file(working_path)
    final.export(mp3_path, format="mp3", bitrate="64k")

    # Clean up temp wavs
    for f in os.listdir(os.path.dirname(mp3_path)):
        if f.startswith(os.path.basename(base_path)) and f.endswith(".wav"):
            os.remove(os.path.join(os.path.dirname(mp3_path), f))

def giant_effects(mp3_path, voice):

    # Load the mp3
    sound = AudioSegment.from_file(mp3_path)

    # Work in-place using temporary intermediate files with .wav extension

    base_path = os.path.join(mp3_path[:-4])  # strip .mp3
    working_path = base_path + "_proc.wav"
    sound.export(working_path, format="wav")

    changed = base_path + "_changed.wav"
    if voice in (["ogrila_ogre"]):
        subprocess.run([
            SOX_PATH, working_path, changed,
            'pitch', '-350',
            'tempo', '0.85',
            'gain','-n',
        ], check=True)
    elif voice in (['ogre_male']):
        subprocess.run([
            SOX_PATH, working_path, changed,
            'pitch', '-200',
            'tempo', '0.90',
            'gain','-n',
        ], check=True)
    elif voice in (['ancient']):
        print("Ancient effects...", flush = True)
        subprocess.run([
            SOX_PATH, working_path, changed,
            'pitch', '-250',
            'reverb', '43', '40', '40', '50', '25', '-5',
            'gain','-n',

        ], check=True)
    else:
        subprocess.run([
            SOX_PATH, working_path, changed,
            'pitch', '-100',
            'tempo', '1.2',
        ], check=True)
    working_path = changed

    # Final: overwrite original .mp3 with processed audio
    final = AudioSegment.from_file(working_path)
    final.export(mp3_path, format="mp3", bitrate="64k")

    # Clean up temp wavs
    for f in os.listdir(os.path.dirname(mp3_path)):
        if f.startswith(os.path.basename(base_path)) and f.endswith(".wav"):
            os.remove(os.path.join(os.path.dirname(mp3_path), f))

def small_effects(mp3_path):

    # Load the mp3
    sound = AudioSegment.from_file(mp3_path)

    # Work in-place using temporary intermediate files with .wav extension

    base_path = os.path.join(mp3_path[:-4])  # strip .mp3
    working_path = base_path + "_proc.wav"
    sound.export(working_path, format="wav")

    changed = base_path + "_changed.wav"
    subprocess.run([
        SOX_PATH, working_path, changed,
        'pitch', '400'
    ], check=True)
    working_path = changed

    # Final: overwrite original .mp3 with processed audio
    final = AudioSegment.from_file(working_path)
    final.export(mp3_path, format="mp3", bitrate="64k")

    # Clean up temp wavs
    for f in os.listdir(os.path.dirname(mp3_path)):
        if f.startswith(os.path.basename(base_path)) and f.endswith(".wav"):
            os.remove(os.path.join(os.path.dirname(mp3_path), f))

def earthen_effects(mp3_path):

    # Load the mp3
    sound = AudioSegment.from_file(mp3_path)

    # Work in-place using temporary intermediate files with .wav extension

    base_path = os.path.join(mp3_path[:-4])  # strip .mp3
    working_path = base_path + "_proc.wav"
    sound.export(working_path, format="wav")

    changed = base_path + "_changed.wav"
    subprocess.run([
       SOX_PATH, working_path, changed,
       'pitch', '-25',
       'gain', '2',
    ], check=True)
    working_path = changed

    y, sr = librosa.load(working_path, sr=None)
    processed = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=-1.50) #for raspy voice

    # Save processed audio to temp wav file
    temp_final_wav = base_path + "_final.wav"
    sf.write(temp_final_wav, processed, sr)

    # Final: overwrite original .mp3 with processed audio
    final = AudioSegment.from_file(temp_final_wav)
    final.export(mp3_path, format="mp3", bitrate="64k")

    # Clean up temp wavs
    for f in os.listdir(os.path.dirname(mp3_path)):
        if f.startswith(os.path.basename(base_path)) and f.endswith(".wav"):
            os.remove(os.path.join(os.path.dirname(mp3_path), f))

def naaru_effects(mp3_path):
    # Load the mp3
    sound = AudioSegment.from_file(mp3_path)

    # Work in-place using temporary intermediate files with .wav extension

    base_path = os.path.join(mp3_path[:-4])  # strip .mp3
    print(base_path)
    working_path = base_path + "_proc.wav"
    sound.export(working_path, format="wav")

    changed = base_path + "_changed.wav"
    subprocess.run([
       SOX_PATH, working_path, changed,
       'pitch', '-150',
       'tremolo','100',
       'reverb', '60', '50', '60', '100', '50', '-5',
       'gain', '-n',

    ], check=True)
    working_path = changed

    # Final: overwrite original .mp3 with processed audio
    final = AudioSegment.from_file(working_path)
    final.export(mp3_path, format="mp3", bitrate = "64k")

    # Clean up temp wavs
    for f in os.listdir(os.path.dirname(mp3_path)):
        if f.startswith(os.path.basename(base_path)) and f.endswith(".wav"):
            os.remove(os.path.join(os.path.dirname(mp3_path), f))

def comms_effects(mp3_path):
    # Load the mp3
    sound = AudioSegment.from_file(mp3_path)

    # Work in-place using temporary intermediate files with .wav extension

    base_path = os.path.join(mp3_path[:-4])  # strip .mp3
    print(base_path)
    working_path = base_path + "_proc.wav"
    sound.export(working_path, format="wav")

    changed = base_path + "_changed.wav"
    subprocess.run([
       SOX_PATH, working_path, changed,
       'highpass', '300',
       'lowpass','3400',

    ], check=True)
    working_path = changed

    # Final: overwrite original .mp3 with processed audio
    final = AudioSegment.from_file(working_path)
    final.export(mp3_path, format="mp3", bitrate = "64k")

    # Clean up temp wavs
    for f in os.listdir(os.path.dirname(mp3_path)):
        if f.startswith(os.path.basename(base_path)) and f.endswith(".wav"):
            os.remove(os.path.join(os.path.dirname(mp3_path), f))


def _apply_sox_effects_to_waveform(audio_dict, sox_args, also_pitch_shift=None):
    """
    Internal helper to apply sox effects to waveform dict

    Args:
        audio_dict: Dict with 'waveform' (tensor) and 'sample_rate'
        sox_args: List of sox arguments
        also_pitch_shift: Optional (n_steps, sr) for librosa pitch shift after sox

    Returns:
        Processed audio_dict
    """
    # 1. Если пришел список, достаем первый элемент
    audio_data = audio_dict
    if isinstance(audio_dict, list):
        audio_data = audio_dict[0]
    
    # 2. Определяем, где сама вейвформа
    if isinstance(audio_data, dict):
        # Если это словаaudio_dictрь (как ожидал старый код)
        waveform = audio_data["waveform"]
    else:
        # Если это уже массив NumPy (как показывает твой дебаг)
        waveform = audio_data

    # print(f"Input shape: {audio_dict['waveform'].shape}")
    sample_rate = 24000

    # Convert to numpy
    if torch.is_tensor(waveform):
        waveform_np = waveform.cpu().numpy()
    else:
        waveform_np = waveform

    # CRITICAL: Reshape from (1, 1, samples) to (samples,) for soundfile
    # Remove batch and channel dimensions
    if waveform_np.ndim == 3:
        # (1, 1, samples) → (samples,)
        waveform_np = waveform_np.squeeze()  # Removes dimensions of size 1
        # print(f"Reshaped for soundfile: {waveform_np.shape}")

    # Ensure float32
    if waveform_np.dtype != np.float32:
        waveform_np = waveform_np.astype(np.float32)

    # Create temp files
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f_in:
        input_path = f_in.name
        sf.write(input_path, waveform_np, sample_rate)

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f_out:
        output_path = f_out.name

    try:
        # Apply sox effects
        cmd = [SOX_PATH, input_path, output_path] + sox_args
        subprocess.run(cmd, check=True)

        # Read result
        processed_np, _ = sf.read(output_path)
        # processed_np = np.squeeze(processed_np)  # Remove extra dims
        # print(f"mid tensor shape: {processed_np.shape}")

        # Optional librosa pitch shift
        if also_pitch_shift:
            n_steps, sr = also_pitch_shift
            processed_np = librosa.effects.pitch_shift(
                y=processed_np, sr=sr, n_steps=n_steps
            )

        # CRITICAL: Reshape back to original format (1, 1, samples)
        # Add batch and channel dimensions back
        if processed_np.ndim == 1:
            # (samples,) → (1, 1, samples)
            processed_np = processed_np.reshape(1, 1, -1)
        elif processed_np.ndim == 2:
            # (samples, channels) → (1, channels, samples)
            processed_np = processed_np.transpose(1, 0)  # (channels, samples)
            processed_np = processed_np.reshape(1, processed_np.shape[0], -1)

        # print(f"Reshaped back: {processed_np.shape}")

        # Convert back to tensor
        processed_tensor = torch.from_numpy(processed_np).float()

        # One more squeeze for safety
        # processed_tensor = processed_tensor.squeeze()
        # print(f"Final tensor shape: {processed_tensor.shape}")

        return processed_tensor

        return {
            "waveform": processed_tensor,
            "sample_rate": sample_rate
        }

    finally:
        # Cleanup
        os.unlink(input_path)
        os.unlink(output_path)

def match_peak_volume_on_waveform(audio_dict, volume_match_sample=None, target_dbfs_override=None):
    """
    Apply gain to match target dBFS level.
    """
    from pydub import AudioSegment
    
    # 1. Если пришел список, достаем первый элемент
    audio_data = audio_dict
    if isinstance(audio_dict, list):
        audio_data = audio_dict[0]
    
    # 2. Определяем, где сама вейвформа
    if isinstance(audio_data, dict):
        # Если это словаaudio_dictрь (как ожидал старый код)
        waveform = audio_data["waveform"]
    else:
        # Если это уже массив NumPy (как показывает твой дебаг)
        waveform = audio_data

    # print(f"Input shape: {audio_dict['waveform'].shape}")
    sample_rate = 24000

    # Convert to numpy
    if torch.is_tensor(waveform):
        waveform_np = waveform.cpu().numpy()
    else:
        waveform_np = waveform

    # CRITICAL: Reshape from (1, 1, samples) to (samples,) for soundfile
    # Remove batch and channel dimensions
    if waveform_np.ndim == 3:
        # (1, 1, samples) → (samples,)
        waveform_np = waveform_np.squeeze()  # Removes dimensions of size 1
        # print(f"Reshaped for soundfile: {waveform_np.shape}")

    # Ensure float32
    if waveform_np.dtype != np.float32:
        waveform_np = waveform_np.astype(np.float32)
    
    # Determine target dBFS
    target_dbfs = -16.0
    
    if target_dbfs_override:
        target_dbfs = target_dbfs_override
    elif volume_match_sample:
        sample = AudioSegment.from_file(volume_match_sample)
        target_dbfs = sample.dBFS
    
    print(f"target_dbfs: {target_dbfs}")
    
    # Measure current dBFS
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = f.name
        
        sf.write(temp_path, waveform_np, sample_rate)
        current = AudioSegment.from_file(temp_path)
        current_dbfs = current.dBFS
    
    os.unlink(temp_path)
    
    print(f"current_dbfs: {current_dbfs:.2f}")
    
    # Apply gain
    gain_db = target_dbfs - current_dbfs
    print(f"gain: {gain_db:.2f} dB")
    
    processed = _apply_sox_effects_to_waveform(audio_dict, ['vol', f'{gain_db:.2f}dB'])
    
    gain_linear = 10 ** (gain_db / 20)
    print(f"Volume diff {gain_linear:.4f} normalized.")
    
    return processed

# Now each effect function
def dk_effects_on_waveform(audio_dict):
    """DK effects on waveform"""
    # Apply effects in sequence
    audio = _apply_sox_effects_to_waveform(
        audio_dict,
        ['reverb', '50', '50', '75', '100', '25', '-5']
    )
    audio = _apply_sox_effects_to_waveform(
        audio,
        ['bass', '10']
    )
    audio = _apply_sox_effects_to_waveform(
        audio,
        ['tremolo', '200', 'gain', '-n']
    )
    # Final pitch shift with librosa
    return _apply_sox_effects_to_waveform(
        audio,
        [],  # No sox args
        also_pitch_shift=(-0.50, audio["sample_rate"])
    )

def robot_effects_on_waveform(audio_dict):
    """Robot effects on waveform"""
    return _apply_sox_effects_to_waveform(
        audio_dict,
        ['tremolo', '100000', 'pitch', '-250',
         'lowpass', '4000', 'gain', '-n']
    )

def ghost_effects_on_waveform(audio_dict):
    """Ghost effects on waveform"""
    return _apply_sox_effects_to_waveform(
        audio_dict,
        ['reverb', '100', '70', '50', '80', '50', '0', 'gain', '-n']
    )

def undead_effects_on_waveform(audio_dict):
    """Undead effects on waveform"""
    return _apply_sox_effects_to_waveform(
        audio_dict,
        ['reverb', '40', '40', '40', '50', '25', '-5']
    )

def demon_effects_on_waveform(audio_dict):
    """Demon effects on waveform"""
    return _apply_sox_effects_to_waveform(
        audio_dict,
        ['pitch', '-100', 'reverb', '32', '40', '40', '50', '25', '-5', 'gain', '-n']
    )

def ethereal_effects_on_waveform(audio_dict):
    """Ethereal effects on waveform"""
    return _apply_sox_effects_to_waveform(
        audio_dict,
        ['reverb', '50', '60', '100', '100', '100', '-6']
    )

def giant_effects_on_waveform(audio_dict, voice):
    """Giant effects on waveform with voice-specific variants"""
    if voice in ["ogrila_ogre"]:
        return _apply_sox_effects_to_waveform(
            audio_dict,
            ['pitch', '-350', 'tempo', '0.85', 'gain', '-n']
        )
    elif voice in ['ogre_male']:
        return _apply_sox_effects_to_waveform(
            audio_dict,
            ['pitch', '-200', 'tempo', '0.90', 'gain', '-n']
        )
    elif voice in ['ancient']:
        print("Ancient effects...", flush=True)
        return _apply_sox_effects_to_waveform(
            audio_dict,
            ['pitch', '-250', 'reverb', '43', '40', '40', '50', '25', '-5', 'gain', '-n']
        )
    else:
        return _apply_sox_effects_to_waveform(
            audio_dict,
            ['pitch', '-100', 'tempo', '1.2']
        )

def small_effects_on_waveform(audio_dict):
    """Small effects on waveform"""
    return _apply_sox_effects_to_waveform(
        audio_dict,
        ['pitch', '400']
    )

def earthen_effects_on_waveform(audio_dict):
    """Earthen effects on waveform"""
    audio = _apply_sox_effects_to_waveform(
        audio_dict,
        ['pitch', '-25', 'gain', '2']
    )
    return _apply_sox_effects_to_waveform(
        audio,
        [],
        also_pitch_shift=(-1.50, audio["sample_rate"])
    )

def naaru_effects_on_waveform(audio_dict):
    """Naaru effects on waveform"""
    return _apply_sox_effects_to_waveform(
        audio_dict,
        ['pitch', '-150', 'tremolo', '100',
         'reverb', '60', '50', '60', '100', '50', '-5', 'gain', '-n']
    )

def comms_effects_on_waveform(audio_dict):
    """Comms (radio) effects on waveform"""
    return _apply_sox_effects_to_waveform(
        audio_dict,
        ['highpass', '300', 'lowpass', '3400']
    )

def letter_effect_on_waveform(audio_dict):
    """
    Apply "reading a letter" effect to audio
    - Slight muffling (low-pass filter)
    - Small room reverb
    - Slight volume normalization
    """
    return _apply_sox_effects_to_waveform(
        audio_dict,
        [
            'lowpass', '4000',        # Muffle high frequencies (paper sound)
            'reverb', '30', '40', '30', '40', '20', '-5',  # Small room reverb
            'gain', '-n',              # Normalize volume
            'compand', '0.01,0.1', '-60,-60,-40,-10,-20,-5',  # Gentle compression
        ]
    )

def wretched_effect_on_waveform(audio_dict):
    return _apply_sox_effects_to_waveform(
        audio_dict,
        [
            'pitch', '-250',
            'tremolo', '15',
            'echo', '0.8', '0.8', '60', '0.5',
        ]
    )