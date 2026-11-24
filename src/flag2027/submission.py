"""Accepted four-file submission layout, with no score calibration."""
from __future__ import annotations
import csv
import io
import json
import math
from pathlib import Path
import zipfile
import numpy as np

OUTPUTS = {
    'english_no_gender': 'no_gender/sub_score_v4_English_heard.txt',
    'bangla_no_gender': 'no_gender/sub_score_v4_Bangla_unheard.txt',
    'english_gender': 'gender/sub_score_v4_English_heard.txt',
    'bangla_gender': 'gender/sub_score_v4_Bangla_unheard.txt',
}
HEAD_NAMES = ('triplet', 'facenet_wespeaker', 'imagebind_wespeaker',
              'arcface_wespeaker', 'dinov2_imagebind_wespeaker', 'imagebind_whisper13')


def read_trials(path):
    """CSV order is the order supplied by the organizer; IDs are not re-sorted."""
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    required = {'trial_id', 'subset', 'face_path', 'audio_path'}
    if not rows or not required <= rows[0].keys():
        raise ValueError('trial manifest requires trial_id,subset,face_path,audio_path')
    seen = set()
    for row in rows:
        if row['subset'] not in OUTPUTS:
            raise ValueError('unknown subset: ' + row['subset'])
        if not row['trial_id'] or any(c.isspace() for c in row['trial_id']):
            raise ValueError('trial IDs must be nonempty and contain no whitespace')
        key = (row['subset'], row['trial_id'])
        if key in seen:
            raise ValueError('duplicate subset/trial ID')
        seen.add(key)
    return rows


def linear_fusion(components):
    """The deployed rule: arithmetic mean of six distances, not their squares."""
    values = np.asarray(components, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError('components must have shape [trials, 6]')
    if not np.isfinite(values).all() or (values < 0).any() or (values > 2).any():
        raise ValueError('branch distances must be finite and in [0,2]')
    return np.sum(values, axis=1) / 6


def write_submission(rows, distances, output, require_all_subsets=True):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if output.suffix.lower() != '.zip' or len(output.name) > 64:
        raise ValueError('output must be a ZIP filename of at most 64 characters')
    values = np.asarray(distances, dtype=np.float64)
    if values.shape != (len(rows),) or not np.isfinite(values).all():
        raise ValueError('distance vector does not match manifest')
    if (values < 0).any() or (values > 2).any():
        raise ValueError('distances must lie in [0,2]')
    grouped = {key: [] for key in OUTPUTS}
    seen = set()
    for row, value in zip(rows, values):
        key = (row['subset'], row['trial_id'])
        if key in seen:
            raise ValueError('duplicate subset/trial ID')
        seen.add(key)
        grouped[row['subset']].append(row['trial_id'] + ' ' + format(float(value), '.17g') + '\n')
    if require_all_subsets and any(not group for group in grouped.values()):
        raise ValueError('official submission requires all four subsets')
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED) as archive:
        for key, member in OUTPUTS.items():
            archive.writestr(member, ''.join(grouped[key]).encode('utf-8'))
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None or archive.namelist() != list(OUTPUTS.values()):
            raise RuntimeError('submission archive failed verification')
    return {'files': {OUTPUTS[key]: len(group) for key, group in grouped.items()},
            'rows': len(rows), 'crc': 'PASS', 'formula': 'sum(six_branch_distances)/6'}
