"""Portable path adapter around the accepted projection/distance functions.

All numerical operations below retain the original float64 inference order.
The optional whole-view L2 adapts raw extractor outputs to the frozen view
contract; pass normalize_inputs=False only for already unit-normalized views.
"""
from pathlib import Path
from collections.abc import Mapping
import numpy as np
from .numerics.inference_primitives import unit, embedded, dist

BRANCH_VIEWS = {
    'facenet_wespeaker': (('face_facenet',), ('voice_wespeaker',)),
    'imagebind_wespeaker': (('face_imagebind',), ('voice_wespeaker',)),
    'arcface_wespeaker': (('face_arcface',), ('voice_wespeaker',)),
    'dinov2_imagebind_wespeaker': (('face_dino_cls', 'face_dino_patch', 'face_dino_layer12', 'face_imagebind'), ('voice_wespeaker',)),
    'imagebind_whisper13': (('face_imagebind',), ('voice_whisper13',)),
}


def load_model(path):
    with np.load(Path(path), allow_pickle=False) as archive:
        model = {key: archive[key] for key in archive.files}
    for modality in ('face', 'voice'):
        projection = model['fit_' + modality + '_projection']
        center = model['deployment_' + modality + '_mean']
        if projection.ndim != 2 or center.shape != (projection.shape[0],):
            raise ValueError('invalid CCA projection/center shape')
    rank = model['fit_face_projection'].shape[1]
    if model['fit_voice_projection'].shape[1] != rank:
        raise ValueError('face/voice CCA ranks differ')
    if model['INLP3_directions'].shape != (3, rank):
        raise ValueError('expected three directions in the head own space')
    return model


def compose_view(features: Mapping, names):
    """L2 every atomic feature, concatenate in production order, then view L2.

    DINO's three 1024-D blocks and ImageBind's 1024-D block are four atomic
    blocks. Do not first normalize DINO3072 as one group against ImageBind.
    """
    arrays = [unit(features[name]) for name in names]
    if len({len(array) for array in arrays}) != 1:
        raise ValueError('atomic feature row counts differ')
    return unit(np.concatenate(arrays, axis=1))


def apply_inlp(x, directions, k=3):
    """Exact original inlp.py::apply body, without importing fitting packages."""
    x = unit(x)
    for direction in directions[:k]:
        x = unit(x - np.outer(x @ direction, direction))
    return x


def project_pair(model_path, face, voice, gender=False, normalize_inputs=True):
    """Return face/voice embeddings; no cross-trial statistics are calculated."""
    model = load_model(model_path) if not isinstance(model_path, Mapping) else model_path
    data = {'face': np.asarray(face, np.float64), 'voice': np.asarray(voice, np.float64)}
    if len(data['face']) != len(data['voice']):
        raise ValueError('face/voice pair counts differ')
    if normalize_inputs:
        data = {modality: unit(values) for modality, values in data.items()}
    fit = {modality + '_projection': model['fit_' + modality + '_projection'] for modality in data}
    centers = {modality: model['deployment_' + modality + '_mean'] for modality in data}
    projected = embedded(data, fit, centers)
    if gender:
        projected = {modality: apply_inlp(values, model['INLP3_directions'], 3)
                     for modality, values in projected.items()}
    return projected


def score_pair(model_path, face, voice, gender=False, normalize_inputs=True):
    """Return the original sqrt(2-2*clipped cosine) branch distance."""
    return dist(project_pair(model_path, face, voice, gender, normalize_inputs))
