"""Portable IO adapter around the unmodified production projection model."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .original.original_model import TripletModel

FACE_WIDTH = 1536
VOICE_WIDTH = 1280


def make_model():
    """Retain production constructor/replacement order and RNG consumption."""
    model = TripletModel({'use_fop': False})
    model.face.block[0] = nn.Linear(FACE_WIDTH, 256)
    model.voice.block[0] = nn.Linear(VOICE_WIDTH, 256)
    return model


def load_projection(checkpoint, device='cpu'):
    """Load the released or automatically exported tensor-only epoch0300.pth.

    The original recovery *.pth.tar includes optimizer/NumPy RNG objects and
    must not be supplied to this weights_only inference loader.
    """
    payload = torch.load(Path(checkpoint), map_location='cpu', weights_only=True)
    if payload.get('epoch') != 300 or payload.get('global_step') != 53700:
        raise ValueError('This release expects the fixed epoch300/global_step53700 endpoint')
    model = make_model()
    model.load_state_dict(payload['model_state'], strict=True)
    for name in ('face', 'voice'):
        if int(getattr(model, name).block[1].num_batches_tracked) != 53700:
            raise ValueError('Unexpected native English-training BN counters')
    return model.to(device).eval().requires_grad_(False)


def compose_inputs(face_peav_final, face_facenet, voice_peav_final, voice_wespeaker):
    """Four aligned raw encoder arrays -> separately unit-normalized concatenations.

    The four named arguments are aligned [N,D] encoder arrays. PE-AV is the same-pipeline final
    1024 block, never the distinct historical legacy_final representation.
    """
    features = dict(face_peav_final=face_peav_final, face_facenet=face_facenet,
                    voice_peav_final=voice_peav_final, voice_wespeaker=voice_wespeaker)
    specifications = {
        'face': [('face_peav_final', 1024), ('face_facenet', 512)],
        'voice': [('voice_peav_final', 1024), ('voice_wespeaker', 256)],
    }
    result = {}
    count = None
    for modality, blocks in specifications.items():
        pieces = []
        for key, width in blocks:
            array = np.ascontiguousarray(features[key], dtype=np.float32)
            if array.ndim != 2 or array.shape[1] != width or not np.isfinite(array).all():
                raise ValueError(f'{key}: expected finite [N,{width}]')
            if count is None:
                count = len(array)
            if len(array) != count:
                raise ValueError('Feature rows are not aligned')
            if np.any(np.linalg.norm(array, axis=1) <= 1e-12):
                raise ValueError(f'{key}: zero embedding')
            pieces.append(F.normalize(torch.from_numpy(array), p=2, dim=1, eps=1e-12).numpy())
        result[modality] = np.ascontiguousarray(np.concatenate(pieces, axis=1), dtype=np.float32)
    return result['face'], result['voice']


def _unit64(array):
    array = np.asarray(array, dtype=np.float64)
    norm = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.isfinite(array).all() or np.any(norm <= 1e-12):
        raise ValueError('Zero or non-finite projected embedding')
    return array / norm


def project_inputs(model, face, voice, batch_size=1024):
    """Apply native eval BN, then reproduce production float64 output L2.

    ``face`` and ``voice`` are the already composed float32 inputs. Do not
    normalize the concatenation before calling this function. Outputs are a
    dict of float64 [N,256] face/voice arrays, before optional own-space INLP3.
    """
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    device = next(model.parameters()).device
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.eval()
    result = {}
    with torch.inference_mode():
        for modality, value, width in [('face', face, FACE_WIDTH), ('voice', voice, VOICE_WIDTH)]:
            value = np.ascontiguousarray(value, dtype=np.float32)
            if value.ndim != 2 or value.shape[1] != width or not len(value) or not np.isfinite(value).all():
                raise ValueError(f'{modality}: expected nonempty finite [N,{width}]')
            output = [getattr(model, modality)(torch.from_numpy(value[i:i+batch_size]).to(device)).cpu().numpy()
                      for i in range(0, len(value), batch_size)]
            result[modality] = _unit64(np.concatenate(output, axis=0))
    if any(not torch.equal(value, model.state_dict()[key].detach().cpu()) for key, value in before.items()):
        raise RuntimeError('Frozen projection/BN state changed during inference')
    return result
