"""Original weighted identity-mean CCA and three-round INLP, with portable I/O."""
import numpy as np
from .numerics import weighted_identity_centers as weighted
from .numerics import fit_centers as original_fit
from .runtime import unit


def fit_projection(face_prototypes, voice_prototypes, identity_weights, rank=192, ridge=10.0):
    """Do not L2-renormalize identity means. Uses the original functions."""
    stat = weighted.statistics(face_prototypes, voice_prototypes, identity_weights)
    return original_fit.fit(stat, float(ridge), int(rank))


def deployment_center(values, identities):
    """Equal identities; equal unique media within identity, as original center()."""
    identities = np.asarray(identities).astype(str)
    values = np.asarray(values, np.float64)
    return np.stack([values[identities == identity].mean(0)
                     for identity in sorted(set(identities))]).mean(0)


def fit_inlp3(data, identities, genders):
    """Fit using the production numerical routine."""
    from .numerics.fit3 import fit3
    return fit3(data, identities, genders)


def fit_complete(face_prototypes, voice_prototypes, identity_weights,
                 training_face, training_voice, face_identity, voice_identity, genders,
                 identities=None, rank=192, ridge=10.0):
    """Fit mapping, English-only centers, and own-space INLP3 in three stages."""
    fit = fit_projection(face_prototypes, voice_prototypes, identity_weights, rank, ridge)
    training = {'face': unit(training_face), 'voice': unit(training_voice)}
    ids = {'face': np.asarray(face_identity).astype(str), 'voice': np.asarray(voice_identity).astype(str)}
    centers = {modality: deployment_center(values, ids[modality]) for modality, values in training.items()}
    from .numerics.inference_primitives import embedded
    own, rounds, diagnostics = fit_inlp3(embedded(training, fit, centers), ids, genders)
    model = {'fit_' + key: value for key, value in fit.items()}
    model.update({'deployment_' + modality + '_mean': value for modality, value in centers.items()})
    model.update(INLP3_directions=own['directions'], INLP3_basis=own['basis'],
                 training_identity_weights=np.asarray(identity_weights, np.float64))
    if identities is not None:
        model['training_identities'] = np.asarray(identities).astype(str)
    return model, rounds, diagnostics
