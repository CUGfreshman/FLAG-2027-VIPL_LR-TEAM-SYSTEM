"""Frozen DINOv2-L feature blocks from the original extractor."""
import numpy as np
import torch
import torch.nn.functional as F


def check(array, count):
    assert array.dtype == np.float32 and array.shape == (count, 3072)
    assert np.isfinite(array).all()
    norms = np.linalg.norm(array.reshape(count, 3, 1024).astype(np.float64), axis=2)
    assert np.max(np.abs(norms - 1)) < 1e-5


def encode(model, pixels):
    with torch.inference_mode():
        out = model(pixel_values=pixels, output_hidden_states=True, return_dict=True)
        assert len(out.hidden_states) == 25
        assert out.last_hidden_state.dtype == torch.float32 and out.last_hidden_state.shape[1:] == (257, 1024)
        blocks = (out.last_hidden_state[:, 0], out.last_hidden_state[:, 1:].mean(dim=1),
                  out.hidden_states[12][:, 0])
        assert all(x.dtype == torch.float32 and x.shape[1] == 1024 for x in blocks)
        assert all(torch.isfinite(x).all() and (torch.linalg.vector_norm(x, dim=1) > 1e-8).all() for x in blocks)
        result = torch.cat([F.normalize(x, dim=1) for x in blocks], dim=1).cpu().numpy()
    check(result, len(result))
    return result
