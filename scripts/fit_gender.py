#!/usr/bin/env python3
"""Fit original three-round INLP in an already trained 192-D or 256-D head."""
from pathlib import Path
import argparse
import json
import numpy as np
from flag2027.cca.fitting import fit_inlp3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-projections', required=True, type=Path,
        help='NPZ: face, voice (unique training media); face_identity, voice_identity; identities, genders')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.training_projections, allow_pickle=False) as data:
        train = {modality: data[modality] for modality in ('face', 'voice')}
        ids = {modality: data[modality + '_identity'].astype(str) for modality in ('face', 'voice')}
        genders = dict(zip(data['identities'].astype(str), data['genders'].astype(str)))
    if len(genders) != 198:
        raise ValueError('production INLP uses 198 strict-English training identities')
    own, rounds, diagnostics = fit_inlp3(train, ids, genders)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        np.savez_compressed(stream, INLP3_directions=own['directions'], INLP3_basis=own['basis'])
    args.output.with_suffix('.fit.json').write_text(json.dumps({'rounds': rounds,
        'training_only_diagnostics': diagnostics}, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'COMPLETE', 'output': str(args.output)}))


if __name__ == '__main__':
    main()
