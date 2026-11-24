#!/usr/bin/env python3
"""Project unique strict-English training media before fitting the T16 INLP3."""
from pathlib import Path
import argparse
import csv
import json
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True, help='prepared English rows.csv/face.npy/voice.npy')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with (args.data / 'rows.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 22976 or len({row['identity'] for row in rows}) != 198:
        raise ValueError('only the fixed strict-English198 training pool may fit released INLP')
    if any(row.get('language') != 'English' or row.get('split', 'train') != 'train' for row in rows):
        raise ValueError('INLP training rows must be English training media')
    genders = {}
    for row in rows:
        gender = row['gender'].lower()
        gender = {'male': 'm', 'female': 'f'}.get(gender, gender)
        if gender not in ('m', 'f') or (row['identity'] in genders and genders[row['identity']] != gender):
            raise ValueError('invalid or conflicting training gender')
        genders[row['identity']] = gender
    from flag2027.triplet.model import load_projection, project_inputs
    model = load_projection(args.checkpoint, device=args.device)
    face = np.load(args.data / 'face.npy', mmap_mode='r', allow_pickle=False)
    voice = np.load(args.data / 'voice.npy', mmap_mode='r', allow_pickle=False)
    if len(face) != len(rows) or len(voice) != len(rows):
        raise ValueError('training media and embedding rows differ')
    projected = project_inputs(model, face, voice)
    arrays = {}
    for modality in ('face', 'voice'):
        first = {}
        for index, row in enumerate(rows):
            key = row.get(modality + '_path') or row.get('audio_path')
            if not key:
                raise ValueError('missing explicit unique-media key')
            first.setdefault(key, index)
        take = np.asarray(list(first.values()), dtype=np.int64)
        arrays[modality] = projected[modality][take]
        arrays[modality + '_identity'] = np.asarray([rows[int(i)]['identity'] for i in take])
    people = sorted(genders)
    arrays['identities'] = np.asarray(people)
    arrays['genders'] = np.asarray([genders[p] for p in people])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    print(json.dumps({'training_identities': 198, 'face_files': len(arrays['face']),
                      'voice_files': len(arrays['voice']), 'head_space': 256}))


if __name__ == '__main__':
    main()
