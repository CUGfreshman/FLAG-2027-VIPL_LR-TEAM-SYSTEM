"""Only identity mass changes; modal covariance remains between identity means.

The returned n-scaled centered matrix lets the unchanged fit_centers.fit form
weighted cross-covariance using its original centered.T @ centered / n path.
"""
import numpy as np

def statistics(face,voice,identity_weights):
    n=len(face);assert n==len(voice) and n>192
    w=np.asarray(identity_weights,np.float64);assert w.shape==(n,) and np.isfinite(w).all() and (w>0).all()
    normalized=w/w.mean()  # n*alpha; all1 exactly1, and invariant to overall scale.
    result=dict(n=n,identity_weights=w,identity_mass=normalized/n,modalities={})
    for name,raw in [('face',face),('voice',voice)]:
        raw=np.asarray(raw,np.float64);assert np.isfinite(raw).all()
        mean=(raw*normalized[:,None]).mean(0)
        centered=(raw-mean)*np.sqrt(normalized)[:,None]
        covariance=centered.T@centered/n;covariance=(covariance+covariance.T)*.5
        ev,q=np.linalg.eigh(covariance);assert ev.min()>-1e-10
        scale=np.trace(covariance)/covariance.shape[0];assert scale>0
        result['modalities'][name]=dict(mean=mean,centered_means=centered,covariance=covariance,
            eigenvalues=np.maximum(ev,0),eigenvectors=q,scale=float(scale))
    return result

def cross(stat):
    return stat['modalities']['face']['centered_means'].T@stat['modalities']['voice']['centered_means']/stat['n']

def verify_math(face,voice,weights):
    weights=np.asarray(weights,np.float64);alpha=weights/weights.sum()
    s=statistics(face,voice,weights);t=statistics(face,voice,weights*7)
    means={};errors={}
    for name,raw in [('face',face),('voice',voice)]:
        raw=np.asarray(raw,np.float64);mu=np.average(raw,axis=0,weights=weights);means[name]=mu
        # Independent weighted second-moment expansion.
        covariance=raw.T@(alpha[:,None]*raw)-np.outer(mu,mu)
        errors[name+'_mean']=float(np.max(np.abs(mu-s['modalities'][name]['mean'])))
        errors[name+'_identity_covariance']=float(np.max(np.abs(covariance-s['modalities'][name]['covariance'])))
        errors[name+'_overall_weight_scale_mean']=float(np.max(np.abs(s['modalities'][name]['mean']-t['modalities'][name]['mean'])))
        errors[name+'_overall_weight_scale_covariance']=float(np.max(np.abs(s['modalities'][name]['covariance']-t['modalities'][name]['covariance'])))
    reference=(face-means['face']).T@(alpha[:,None]*(voice-means['voice']))
    errors['identity_cross_covariance']=float(np.max(np.abs(reference-cross(s))))
    errors['overall_weight_scale_cross']=float(np.max(np.abs(cross(s)-cross(t))))
    assert max(errors.values())<=1e-12,errors
    return s,errors
