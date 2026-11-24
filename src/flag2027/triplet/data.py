"""Training data IO; the sampling implementation is the original production code."""
from pathlib import Path
import csv
import json
import numpy as np
import torch
from .original.original_objective import UniquePairSampler


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def load_training_data(directory):
    directory = Path(directory)
    rows = read_rows(directory / 'rows.csv')
    ready = json.loads((directory / 'DATA_READY.json').read_text(encoding='utf-8'))
    required = ('identity', 'source', 'language', 'gender', 'face_path', 'voice_path')
    if not rows or not all(all(key in row for key in required) for row in rows):
        raise ValueError('Missing training metadata columns')
    if not all(row['language'] == 'English' for row in rows):
        raise ValueError('The deployed triplet is trained only on exact English rows')
    identities = np.asarray([row['identity'] for row in rows])
    people = sorted(set(identities))
    if len(rows) != 22976 or len(people) != 198:
        raise ValueError('Exact reproduction requires 22,976 English rows / 198 identities')
    lookup = {identity: i for i, identity in enumerate(people)}
    labels = torch.tensor([lookup[identity] for identity in identities], dtype=torch.long)
    sources = torch.tensor([{'v1': 0, 'v2': 1, 'v4': 2}[row['source']] for row in rows], dtype=torch.long)
    sampler = UniquePairSampler(identities, [r['face_path'] for r in rows], [r['voice_path'] for r in rows], p=32, k=4)
    raw = {m: np.ascontiguousarray(np.load(directory / (m + '.npy'), allow_pickle=False), dtype=np.float32)
           for m in ('face', 'voice')}
    for modality, width in [('face', 1536), ('voice', 1280)]:
        if raw[modality].shape != (22976, width) or not np.isfinite(raw[modality]).all():
            raise ValueError('Unexpected prepared features')
    info = dict(fold=0, training_rows=len(rows), training_identity_count=len(people),
                training_ids=people, held_ids=[], steps_per_epoch=179, total_updates=53700,
                unique_candidate_pairs=sum(len(v) for v in sampler.rows.values()),
                unique_faces=len(set(r['face_path'] for r in rows)),
                unique_audio=len(set(r['voice_path'] for r in rows)),
                input_widths={'face': 1536, 'voice': 1280}, feature_ready=str(directory / 'DATA_READY.json'),
                language_filter='Exact English metadata; V1+V2+V4',
                source_rows={s: sum(r['source'] == s for r in rows) for s in ('v1', 'v2', 'v4')})
    if (info['unique_candidate_pairs'], info['unique_faces'], info['unique_audio']) != (22965, 22187, 21043):
        raise ValueError('Media identifiers/order do not match the released training pool')
    return rows, np.arange(len(rows)), labels, sources, sampler, raw, info
