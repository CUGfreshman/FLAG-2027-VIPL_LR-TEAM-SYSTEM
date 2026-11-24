#!/usr/bin/env python3
"""Score trial-aligned encoder features with the fixed six-branch model."""
from pathlib import Path
import argparse
import csv
import json
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from flag2027.submission import read_trials, write_submission, HEAD_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, help='trial_id,subset,face_path,audio_path CSV')
    parser.add_argument('--features', required=True, help='NPZ emitted by extract_features.py')
    parser.add_argument('--config', default=str(Path(__file__).resolve().parents[1] / 'configs/system.json'))
    parser.add_argument('--output', required=True, help='new official-format ZIP; never overwrite')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--components', help='optional new per-branch distance CSV')
    args = parser.parse_args()
    rows = read_trials(args.manifest)
    if Path(args.output).exists() or (args.components and Path(args.components).exists()):
        raise FileExistsError('requested output already exists')
    from flag2027.inference import score_features
    with np.load(args.features, allow_pickle=False) as archive:
        if 'trial_id' not in archive or 'subset' not in archive:
            raise ValueError('feature archive requires explicit trial_id and subset arrays')
        expected = [(row['subset'], row['trial_id']) for row in rows]
        actual = list(zip(archive['subset'].astype(str), archive['trial_id'].astype(str)))
        if actual != expected:
            raise ValueError('feature rows do not match the trial manifest; positional guessing is forbidden')
        for key in ('face_path', 'audio_path'):
            if key in archive and archive[key].astype(str).tolist() != [row[key] for row in rows]:
                raise ValueError('feature media keys do not match the trial manifest: ' + key)
        distances, components = score_features(archive, rows, args.config, device=args.device)
    receipt = write_submission(rows, distances, args.output)
    if args.components:
        with Path(args.components).open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['subset', 'trial_id', *HEAD_NAMES, 'distance'])
            writer.writeheader()
            for row, part, value in zip(rows, components, distances):
                writer.writerow({'subset': row['subset'], 'trial_id': row['trial_id'],
                    **{name: format(float(d), '.17g') for name, d in zip(HEAD_NAMES, part)},
                    'distance': format(float(value), '.17g')})
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
