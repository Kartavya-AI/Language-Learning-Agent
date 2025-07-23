import os
import tempfile
import numpy as np
import traceback
import librosa

def convert_audio_to_pcm16(audio_bytes, sample_rate=16000):
    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        data, sr = librosa.load(temp_path, sr=None, mono=True)
        if sr != sample_rate:
            data = librosa.resample(data, orig_sr=sr, target_sr=sample_rate)

        data = np.clip(data, -1.0, 1.0)
        pcm = (data * 32767).astype(np.int16).tobytes()
        os.unlink(temp_path)
        return pcm

    except Exception as e:
        print(f"[Audio Conversion Error] {e}")
        print(traceback.format_exc())
        return None
