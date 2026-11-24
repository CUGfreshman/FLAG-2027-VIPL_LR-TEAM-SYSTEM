"""Original mono 16 kHz audio decoding for the Whisper feature window."""
from pathlib import Path
import subprocess

import numpy as np
import soundfile as sf
import torch


def decode(row):
    path = row['io_path']
    if Path(path).suffix.lower() in ('.wav', '.flac'):
        with sf.SoundFile(path) as stream:
            rate = stream.samplerate
            waveform = stream.read(frames=20 * rate, dtype='float32', always_2d=True).mean(axis=1)
        if rate != 16000:
            import torchaudio.functional as AF
            waveform = AF.resample(torch.from_numpy(waveform), rate, 16000).numpy()
        waveform = np.asarray(waveform[:320000], dtype=np.float32)
    else:
        result = subprocess.run(['ffmpeg', '-v', 'error', '-threads', '1', '-i', path,
                                 '-map', '0:a:0', '-vn', '-t', '20', '-ac', '1', '-ar', '16000',
                                 '-f', 'f32le', 'pipe:1'], capture_output=True, check=True)
        waveform = np.frombuffer(result.stdout, dtype='<f4').copy()[:320000]
    assert len(waveform) >= 400 and np.isfinite(waveform).all(), (path, len(waveform))
    return waveform
