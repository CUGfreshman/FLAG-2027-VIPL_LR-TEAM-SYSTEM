"""Explicit row-to-media joins for training-only feature manifests."""
import csv
from pathlib import Path
import numpy as np
from ..training_alignment import validate_training_alignment
from .runtime import BRANCH_VIEWS, compose_view


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if not row.get('voice_path') and row.get('audio_path'):
            row['voice_path'] = row['audio_path']
    return rows


def unique_media(values, rows, modality):
    """Repeated original-row media are counted once, joined by file key."""
    first = {}
    owners = {}
    for index, row in enumerate(rows):
        key = row[modality + '_path']
        identity = row['identity']
        if key in owners and owners[key] != identity:
            raise ValueError('one media key has multiple identities')
        owners[key] = identity
        first.setdefault(key, index)
    indices = np.asarray(list(first.values()), dtype=np.int64)
    return values[indices], np.asarray([rows[index]['identity'] for index in indices]).astype(str)


def feature_views(path, branch, rows=None):
    face_keys, voice_keys = BRANCH_VIEWS[branch]
    with np.load(path, allow_pickle=False) as data:
        if rows is not None:
            validate_training_alignment(data, rows)
        face = compose_view(data, face_keys)
        voice = compose_view(data, voice_keys)
    return {'face': face, 'voice': voice}


def english_deployment(features_path, rows_path, branch, expected_identities=198):
    rows = read_rows(rows_path)
    if not rows or any(row.get('language', '').strip().lower() not in ('english', 'en') for row in rows):
        raise ValueError('deployment/INLP input must contain only strict English training rows')
    if any(row.get('split', 'train').strip().lower() not in ('train', 'training') for row in rows):
        raise ValueError('only training rows may estimate deployment parameters')
    views = feature_views(features_path, branch, rows)
    if any(len(values) != len(rows) for values in views.values()):
        raise ValueError('feature and CSV row counts differ')
    genders = {}
    for row in rows:
        gender = row['gender'].lower()
        gender = {'male': 'm', 'female': 'f'}.get(gender, gender)
        if gender not in ('m', 'f') or row['identity'] in genders and genders[row['identity']] != gender:
            raise ValueError('missing or conflicting true training gender')
        genders[row['identity']] = gender
    if len(genders) != expected_identities:
        raise ValueError(f'expected {expected_identities} English training identities, got {len(genders)}')
    data, identities = {}, {}
    for modality, values in views.items():
        data[modality], identities[modality] = unique_media(values, rows, modality)
    return data, identities, genders


def identity_prototypes(features_path, rows_path, branch, domain=None):
    """Prepare the paired means from an already selected, training-only pool.

    Selection/LID/overlap removal is performed before this function. A `domain`
    column distinguishes Vox and competition identities even if strings match.
    """
    rows = read_rows(rows_path)
    views = feature_views(features_path, branch, rows)
    if any(len(values) != len(rows) for values in views.values()):
        raise ValueError('feature and CSV row counts differ')
    qualified = []
    for row in rows:
        row_domain = domain or row['domain'].strip().lower()
        if row_domain == 'target':
            row_domain = 'competition'
        if row_domain not in ('vox', 'competition'):
            raise ValueError('domain must be vox or competition')
        if row.get('split', 'train').strip().lower() not in ('train', 'training'):
            raise ValueError('held/evaluation data cannot enter CCA')
        if row_domain == 'competition' and row.get('language', '').strip().lower() not in ('en', 'english'):
            raise ValueError('only competition English rows enter the released CCA pool')
        if row_domain == 'vox' and row.get('top1_language', '').strip().lower() in ('bn', 'as', 'or'):
            raise ValueError('excluded Vox language remains in the selected manifest')
        q = dict(row)
        q['identity'] = ('target:' if row_domain == 'competition' else 'vox:') + row['identity']
        qualified.append(q)
    people = sorted({row['identity'] for row in qualified})
    prototypes = {}
    for modality, values in views.items():
        unique_values, ids = unique_media(values, qualified, modality)
        prototypes[modality + '_prototypes'] = np.stack([
            unique_values[ids == identity].mean(0) for identity in people])
    prototypes['identities'] = np.asarray(people)
    prototypes['identity_weights'] = np.asarray([10.0 if p.startswith('target:') else 1.0 for p in people])
    return prototypes


def separate_domain_prototypes(vox_features, vox_rows, english_features, english_rows, branch):
    """Form each domain's means separately; never concatenate media arrays."""
    rows = read_rows(vox_rows)
    expected_rows = 117194 if branch == 'facenet_wespeaker' else 117193
    if len(rows) != expected_rows:
        raise ValueError(f'{branch} production Vox manifest needs {expected_rows} selected rows; got {len(rows)}. '
                         'Use the archived branch-specific selection; do not silently drop/reinsert a row.')
    # The unusable row is identified by original metadata, never current CSV position.
    bad = [r for r in rows if str(r.get('union_index', r.get('original_union_index', ''))) == '104443']
    if branch != 'facenet_wespeaker' and bad:
        raise ValueError('the branch-specific 117193-row manifest must exclude original union_index=104443')
    english = identity_prototypes(english_features, english_rows, branch, domain='competition')
    vox = identity_prototypes(vox_features, vox_rows, branch, domain='vox')
    if len(english['identities']) != 198 or len(vox['identities']) != 5860:
        raise ValueError('production prototype domains must contain 198 English and 5860 Vox identities')
    return {key: np.concatenate([english[key], vox[key]], axis=0) for key in english}
