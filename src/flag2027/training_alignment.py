"""Check the row keys supplied by training feature archives."""
import numpy as np


def validate_training_alignment(features, rows):
    """Reject mismatched embedded keys; feature-only caches remain positional."""
    keys = {
        'face_path': ('face_path',),
        'audio_path': ('audio_path', 'voice_path'),
        'voice_path': ('voice_path', 'audio_path'),
        'trial_id': ('trial_id',),
        'subset': ('subset',),
    }
    for key, columns in keys.items():
        if key not in features:
            continue
        expected = [next((row[c] for c in columns if row.get(c)), None) for row in rows]
        if key in ('trial_id', 'subset') and all(value is None for value in expected):
            # Older training lists can supply media paths without trial keys.
            continue
        actual = np.asarray(features[key]).astype(str)
        if actual.shape != (len(rows),):
            raise ValueError(f'training feature key {key} must have one value per CSV row')
        mismatch = np.flatnonzero(actual != np.asarray(expected, dtype=str))
        if len(mismatch):
            raise ValueError(f'training feature key {key} does not match CSV data row {mismatch[0] + 1}')
