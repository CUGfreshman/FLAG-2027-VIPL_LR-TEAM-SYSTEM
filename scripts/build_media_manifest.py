#!/usr/bin/env python3
"""Resolve released training media keys beneath caller-supplied dataset roots."""
from pathlib import Path
import argparse
import csv
import gzip
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=Path, required=True)
    parser.add_argument('--roots', type=Path, required=True,
        help='JSON source-to-directory map, for example {"v1":"/data/V1","v2":"/data/V2","v4":"/data/V4","vox":"/data/vox2/mp4"}')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--subset', default='training')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    roots = json.loads(args.roots.read_text(encoding='utf-8'))
    open_rows = gzip.open if args.rows.suffix == '.gz' else open
    with open_rows(args.rows, 'rt', encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('empty training row manifest')
    result = []
    for index, row in enumerate(rows):
        source = row.get('source', 'vox')
        if source not in roots:
            raise ValueError('dataset root not supplied for source ' + source)
        base = Path(roots[source]).expanduser()
        face = row.get('face_path') or row.get('canonical_key') or row.get('utterance_key')
        voice = row.get('voice_path') or row.get('audio_path') or face
        if not face or not voice:
            raise ValueError('row has no explicit media path/key')
        paths = []
        for key in (face, voice):
            p = Path(key)
            if '..' in p.parts:
                raise ValueError('media key contains parent traversal')
            p = p if p.is_absolute() else base / p
            if not p.is_file():
                raise FileNotFoundError(p)
            paths.append(p.as_posix())
        result.append({**row, 'trial_id': str(row.get('row_index', index)), 'subset': args.subset,
                       'face_path': paths[0], 'voice_path': paths[1], 'audio_path': paths[1]})
    fields = list(dict.fromkeys(['trial_id', 'subset', *rows[0].keys(), 'face_path', 'voice_path', 'audio_path']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(result)
    print(json.dumps({'rows': len(result), 'output': str(args.output), 'media_exist': True}))


if __name__ == '__main__':
    main()
