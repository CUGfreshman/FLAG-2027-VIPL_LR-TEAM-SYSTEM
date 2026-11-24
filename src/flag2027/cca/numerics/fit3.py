"""Fit three independent gender directions in one branch."""
import numpy as np
from .inlp import assess, classifier, unit


def fit3(train,identities,genders):
    """Fit three rounds in the branch feature space."""
    data={m:unit(x) for m,x in train.items()}
    dimension=data['face'].shape[1]
    assert dimension in (192,256) and data['voice'].shape[1]==dimension
    basis=np.empty((dimension,0),np.float64);directions=[];rounds=[];diagnostics=[]
    for state in range(4):
        probe=classifier(data,identities,genders)
        diagnostics+=assess(probe,data,identities,genders,'train_resubstitution',state)
        if state==3:break
        w=probe.coef_[0].copy()
        for _ in range(2):w-=(w@basis)@basis.T
        norm=float(np.linalg.norm(w));rawnorm=float(np.linalg.norm(probe.coef_))
        tolerance=max(dimension,state+1)*np.finfo(np.float64).eps*max(1.,rawnorm)
        active=norm>tolerance
        direction=w/norm if active else np.zeros(dimension,np.float64)
        directions.append(direction)
        if active:basis=np.column_stack((basis,direction))
        for modality,z in data.items():data[modality]=unit(z-np.outer(z@direction,direction)) if active else unit(z)
        residual=max(float(np.max(np.abs(z@basis))) for z in data.values()) if basis.shape[1] else 0.
        orth=float(np.max(np.abs(basis.T@basis-np.eye(basis.shape[1])))) if basis.shape[1] else 0.
        assert residual<1e-10 and orth<1e-12
        rounds.append(dict(round=state+1,raw_classifier_norm=rawnorm,novel_direction_norm=norm,
            novelty_tolerance=tolerance,zero_direction=not active,effective_rank=basis.shape[1],
            max_basis_orthogonality_error=orth,max_training_removed_axis_residual=residual))
    return dict(directions=np.stack(directions),basis=basis),rounds,diagnostics
