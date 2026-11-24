"""Prepare exact English-row features without resampling or reordering them."""
from pathlib import Path
import argparse
import csv
import json
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from flag2027.training_alignment import validate_training_alignment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', required=True, help='Original-row CSV, aligned with all feature arrays')
    parser.add_argument('--features', required=True, help='NPZ with the four raw encoder blocks')
    parser.add_argument('--output', required=True, help='New empty directory for the prepared training inputs')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    with Path(args.rows).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        rows = list(reader)
    required = {'identity', 'source', 'language', 'gender', 'face_path', 'voice_path'}
    if not fieldnames or not required.issubset(fieldnames):
        raise ValueError('Training rows are missing required metadata columns')
    indices = np.array([i for i, row in enumerate(rows) if row['language'] == 'English'], dtype=np.int64)
    selected = [rows[int(i)] for i in indices]
    if len(selected) != 22976 or len(set(r['identity'] for r in selected)) != 198:
        raise ValueError('Expected the original 22,976 English rows / 198 identities')
    with np.load(args.features, allow_pickle=False) as features:
        validate_training_alignment(features, rows)
        keys = ['face_peav_final', 'face_facenet', 'voice_peav_final', 'voice_wespeaker']
        if any(len(features[key]) != len(rows) for key in keys):
            raise ValueError('Features and metadata CSV must have the same row count')
        from flag2027.triplet.model import compose_inputs
        face, voice = compose_inputs(*(features[key][indices] for key in keys))
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'rows.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    np.save(output / 'face.npy', face)
    np.save(output / 'voice.npy', voice)
    (output / 'DATA_READY.json').write_text(json.dumps(dict(rows=len(selected), identities=198,
        input_widths={'face': 1536, 'voice': 1280}, each_block_L2=True, concatenation_L2=False,
        metadata_language_filter='English exact', original_row_order_preserved=True), indent=2), encoding='utf-8')
    from flag2027.triplet.data import load_training_data
    _, _, _, _, _, _, info = load_training_data(output)
    print(json.dumps(info, ensure_ascii=False))


if __name__ == '__main__':
    main()
