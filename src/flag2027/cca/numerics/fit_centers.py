"""Whitened CCA with the release ridge and spectrum convention."""
import numpy as np


def fit(stat, ridge, rank=192):
    roots = {}
    for name, x in stat['modalities'].items():
        roots[name] = (x['eigenvectors'] / np.sqrt(x['eigenvalues'] + ridge*x['scale'])) @ x['eigenvectors'].T
    f = stat['modalities']['face']['centered_means']
    v = stat['modalities']['voice']['centered_means']
    cross = f.T @ v / stat['n']
    u, singular, vt = np.linalg.svd(roots['face'] @ cross @ roots['voice'], full_matrices=False)
    assert np.isfinite(singular).all() and singular[rank-1] > 1e-12 and singular.max() <= 1+1e-8
    weight = np.sqrt(singular[:rank])
    return dict(face_mean=stat['modalities']['face']['mean'],
        voice_mean=stat['modalities']['voice']['mean'],
        face_projection=(roots['face'] @ u[:, :rank])*weight,
        voice_projection=(roots['voice'] @ vt.T[:, :rank])*weight,
        singular_values=singular, rank=np.asarray(rank), ridge=np.asarray(ridge))
