"""Training-only shared-space INLP and freshly fitted residual probes for T029."""
from __future__ import annotations
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.exceptions import ConvergenceWarning

LR = dict(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, tol=1e-8,
          fit_intercept=True, class_weight=None, random_state=20261103)

def unit(x):
    x=np.asarray(x,np.float64)
    norm=np.linalg.norm(x,axis=1,keepdims=True)
    assert np.isfinite(x).all() and (norm>1e-12).all(), 'zero/nonfinite projected vector'
    return x/norm

def weights(ids):
    """Equal identities, equal files within identity; normalize sum to file count."""
    ids=np.asarray(ids).astype(str)
    _,inverse,counts=np.unique(ids,return_inverse=True,return_counts=True)
    w=1.0/counts[inverse]
    return w*(len(w)/w.sum())

def merged(data,ids,genders):
    x=np.concatenate([data[m] for m in ('face','voice')])
    labels={m:np.asarray([int(genders[str(p)]=='m') for p in ids[m]]) for m in data}
    # Each modality contributes one half of the total weight; each identity has equal mass.
    w=np.concatenate([weights(ids[m])/len(ids[m]) for m in ('face','voice')])
    w*=len(w)/w.sum()
    y=np.concatenate([labels[m] for m in ('face','voice')])
    assert np.isclose(w.sum(),len(w))
    return x,y,w

def classifier(data,ids,genders):
    x,y,w=merged(data,ids,genders)
    probe=LogisticRegression(**LR)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        probe.fit(x,y,sample_weight=w)
    assert probe.n_iter_[0]<LR['max_iter']
    return probe

def assess(probe,data,ids,genders,scope,state):
    records=[]
    for modality in ('combined','face','voice'):
        if modality=='combined':x,y,w=merged(data,ids,genders)
        else:
            x=data[modality]
            y=np.asarray([int(genders[str(p)]=='m') for p in ids[modality]])
            w=weights(ids[modality])
        score=probe.decision_function(x);pred=(score>=0)
        records.append(dict(state=state,scope=scope,modality=modality,files=len(x),
            identities=len(genders),male_identities=sum(g=='m' for g in genders.values()),
            female_identities=sum(g=='f' for g in genders.values()),
            weighted_accuracy=float(accuracy_score(y,pred,sample_weight=w)),
            balanced_accuracy=float(balanced_accuracy_score(y,pred,sample_weight=w)),
            gender_auc=float(roc_auc_score(y,score,sample_weight=w)),
            majority_baseline=max(sum(g=='m' for g in genders.values()),sum(g=='f' for g in genders.values()))/len(genders),
            probe_intercept=float(probe.intercept_[0]),probe_coef_norm=float(np.linalg.norm(probe.coef_)),
            probe_iterations=int(probe.n_iter_[0]),threshold=0.0,
            fit_scope='training identities only; fresh linear probe in current projected space',
            weighting='identity equal; combined modality half; sum weights = sample count'))
    return records

def trajectory(train,held,ids,hids,genders,hgenders):
    """One 8-removal trajectory, with probes at states 0..8 (ninth is diagnostic only)."""
    train={m:unit(x) for m,x in train.items()}
    held=None if held is None else {m:unit(x) for m,x in held.items()}
    dimension=train['face'].shape[1]
    q=np.empty((dimension,0),np.float64)
    directions=[];coefs=[];intercepts=[];audits=[];records=[]
    for state in range(9):
        probe=classifier(train,ids,genders)
        records+=assess(probe,train,ids,genders,'train_resubstitution',state)
        if held is not None:records+=assess(probe,held,hids,hgenders,'held_unseen',state)
        coefs.append(probe.coef_[0].copy());intercepts.append(probe.intercept_[0])
        if state==8:break
        w=probe.coef_[0].copy()
        # Two modified Gram-Schmidt passes retain only the learned novel component.
        for _ in range(2):w-=(w@q)@q.T
        wn=float(np.linalg.norm(w));rawnorm=float(np.linalg.norm(probe.coef_))
        tolerance=max(dimension,state+1)*np.finfo(np.float64).eps*max(1.0,rawnorm)
        active=wn>tolerance
        direction=w/wn if active else np.zeros(dimension,np.float64)
        directions.append(direction)
        if active:q=np.column_stack((q,direction))
        for data in (train,held):
            if data is None:continue
            for m in data:
                z=data[m]
                data[m]=unit(z-np.outer(z@direction,direction)) if active else unit(z)
        residual=max(float(np.max(np.abs(x@q))) for x in train.values()) if q.shape[1] else 0.0
        orth=float(np.max(np.abs(q.T@q-np.eye(q.shape[1])))) if q.shape[1] else 0.0
        assert residual<1e-10 and orth<1e-12
        audits.append(dict(round=state+1,raw_classifier_norm=rawnorm,novel_direction_norm=wn,
            novelty_tolerance=tolerance,zero_direction=not active,effective_rank=q.shape[1],
            max_basis_orthogonality_error=orth,max_training_removed_axis_residual=residual))
    fitted=dict(directions=np.stack(directions),probe_coefficients=np.stack(coefs),
                probe_intercepts=np.asarray(intercepts),basis=q)
    return fitted,audits,records

def apply(x,directions,k):
    x=unit(x)
    for direction in directions[:k]:x=unit(x-np.outer(x@direction,direction))
    return x

def residual_probe(train,held,ids,hids,genders,hgenders,state):
    probe=classifier(train,ids,genders)
    result=assess(probe,train,ids,genders,'train_resubstitution',state)
    if held is not None:result+=assess(probe,held,hids,hgenders,'held_unseen',state)
    return result
