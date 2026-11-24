#!/usr/bin/env python3
"""Fit one fixed CCA192/ridge10/sqrt head using training-only identity means."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
from flag2027.cca.data import english_deployment, identity_prototypes, separate_domain_prototypes
from flag2027.cca.fitting import fit_complete
from flag2027.cca.runtime import BRANCH_VIEWS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch', required=True, choices=tuple(BRANCH_VIEWS))
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--prototypes', type=Path,
        help='NPZ: face_prototypes, voice_prototypes, identities, identity_weights; no prototype L2')
    selection.add_argument('--fit-features', type=Path, help='Selected Vox+English training feature NPZ')
    selection.add_argument('--vox-features', type=Path,
        help='Vox feature NPZ; its prototypes are formed separately from English to avoid a mixed media array')
    parser.add_argument('--fit-rows', type=Path,
        help='Required with --fit-features; aligned CSV also has domain=vox/competition, split=train')
    parser.add_argument('--vox-rows', type=Path, help='Branch-specific selected Vox training rows aligned to --vox-features')
    parser.add_argument('--training-features', required=True, type=Path, help='Strict-English198 feature NPZ')
    parser.add_argument('--training-rows', required=True, type=Path,
        help='CSV aligned to features: identity, gender, language, face_path, voice_path; split=train')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.prototypes:
        with np.load(args.prototypes, allow_pickle=False) as data:
            prototypes = {key: data[key] for key in data.files}
    elif args.vox_features:
        if not args.vox_rows:
            parser.error('--vox-features requires --vox-rows')
        prototypes = separate_domain_prototypes(args.vox_features, args.vox_rows,
            args.training_features, args.training_rows, args.branch)
    else:
        if not args.fit_rows:
            parser.error('--fit-features requires --fit-rows')
        prototypes = identity_prototypes(args.fit_features, args.fit_rows, args.branch)
    weights = np.asarray(prototypes['identity_weights'], np.float64)
    if len(weights) != 6058 or np.sum(weights == 1) != 5860 or np.sum(weights == 10) != 198:
        raise ValueError('production pool requires 5860 Vox identities x1 and 198 English identities x10')
    train, identities, genders = english_deployment(args.training_features, args.training_rows, args.branch)
    model, rounds, diagnostics = fit_complete(prototypes['face_prototypes'], prototypes['voice_prototypes'], weights,
        train['face'], train['voice'], identities['face'], identities['voice'], genders,
        identities=prototypes['identities'], rank=192, ridge=10.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        np.savez_compressed(stream, **model)
    audit = {'branch': args.branch, 'rank': 192, 'ridge': 10, 'spectrum': 'sqrt',
             'prototype_identities': 6058, 'deployment_identities': len(genders),
             'fit_inputs': {'prototypes': str(args.prototypes) if args.prototypes else None,
                            'fit_features': str(args.fit_features) if args.fit_features else None,
                            'fit_rows': str(args.fit_rows) if args.fit_rows else None,
                            'vox_features': str(args.vox_features) if args.vox_features else None,
                            'vox_rows': str(args.vox_rows) if args.vox_rows else None,
                            'training_features': str(args.training_features),
                            'training_rows': str(args.training_rows)},
             'rounds': rounds, 'diagnostics': diagnostics,
             'parameters_estimated_from': 'training data only; no evaluation media or scores'}
    args.output.with_suffix('.fit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'COMPLETE', 'output': str(args.output), 'branch': args.branch}))


if __name__ == '__main__':
    main()
