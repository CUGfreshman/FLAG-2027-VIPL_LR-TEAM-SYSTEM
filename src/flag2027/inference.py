"""Feature-to-score inference using the exact released downstream models."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .submission import HEAD_NAMES, linear_fusion

FEATURE_DIMS = {
    'face_peav_final': 1024, 'face_facenet': 512, 'face_imagebind': 1024,
    'face_arcface': 512, 'face_dino_cls': 1024, 'face_dino_patch': 1024,
    'face_dino_layer12': 1024, 'voice_peav_final': 1024,
    'voice_wespeaker': 256, 'voice_whisper13': 1280,
}


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(x).all() or not (norm > 1e-12).all():
        raise ValueError('zero or nonfinite feature vector')
    return x / norm


def normalized_distance(face, voice):
    return np.sqrt(np.maximum(2 - 2 * np.clip(np.sum(face * voice, axis=1), -1, 1), 0))


def validate_features(features, count):
    for name, width in FEATURE_DIMS.items():
        if name not in features:
            raise ValueError('missing encoder feature: ' + name)
        if np.shape(features[name]) != (count, width):
            raise ValueError('incorrect shape for ' + name + ': ' + str(np.shape(features[name])))
        if not np.isfinite(features[name]).all():
            raise ValueError('nonfinite encoder feature: ' + name)


def score_features(features, rows, config_path, device='cpu'):
    """All centers, projections and gender directions are frozen training artifacts."""
    from .cca import project_pair, apply_inlp
    from .triplet.model import load_projection, compose_inputs, project_inputs
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config['head_order'] != list(HEAD_NAMES) or config['fusion'] != 'linear_mean':
        raise ValueError('configuration is not the fixed six-branch system')
    if config['weights'] != [1 / 6] * 6:
        raise ValueError('six-branch release uses fixed equal weights')
    root = config_path.parent.parent
    validate_features(features, len(rows))
    gender = np.asarray([row['subset'].endswith('_gender') and not row['subset'].endswith('_no_gender') for row in rows])
    components = np.empty((len(rows), 6), dtype=np.float64)
    model = load_projection(root / config['triplet_checkpoint'], device=device)
    face_input, voice_input = compose_inputs(features['face_peav_final'], features['face_facenet'],
                                             features['voice_peav_final'], features['voice_wespeaker'])
    projected = project_inputs(model, face_input, voice_input, batch_size=config['triplet_batch_size'])
    with np.load(root / config['triplet_inlp'], allow_pickle=False) as parameter:
        directions = parameter['directions' if 'directions' in parameter else 'INLP3_directions'].copy()
    for flag in (False, True):
        take = np.flatnonzero(gender == flag)
        if not len(take):
            continue
        face = apply_inlp(projected['face'][take], directions, 3 if flag else 0)
        voice = apply_inlp(projected['voice'][take], directions, 3 if flag else 0)
        components[take, 0] = normalized_distance(face, voice)
    # Equal contribution of the four 1024-d visual blocks, then whole-view L2 in CCA.
    dino_imagebind = np.concatenate([unit(features[key]) for key in (
        'face_dino_cls', 'face_dino_patch', 'face_dino_layer12', 'face_imagebind')], axis=1)
    views = {
        'facenet_wespeaker': (features['face_facenet'], features['voice_wespeaker']),
        'imagebind_wespeaker': (features['face_imagebind'], features['voice_wespeaker']),
        'arcface_wespeaker': (features['face_arcface'], features['voice_wespeaker']),
        'dinov2_imagebind_wespeaker': (dino_imagebind, features['voice_wespeaker']),
        'imagebind_whisper13': (features['face_imagebind'], features['voice_whisper13']),
    }
    for column, name in enumerate(HEAD_NAMES[1:], 1):
        face, voice = views[name]
        for flag in (False, True):
            take = np.flatnonzero(gender == flag)
            if len(take):
                projected_pair = project_pair(root / config['cca_models'][name], face[take], voice[take], gender=flag)
                components[take, column] = normalized_distance(projected_pair['face'], projected_pair['voice'])
    return linear_fusion(components), components
