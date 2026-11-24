"""Unchanged inference functions extracted from the original fit_remote.py."""
import numpy as np

def unit(a):
    a=np.asarray(a,np.float64)
    norm=np.linalg.norm(a,axis=1,keepdims=True)
    assert np.isfinite(a).all() and np.all(norm>1e-12)
    return a/norm

def embedded(x,fit,center):
    return {m:unit((a-center[m])@fit[m+'_projection']) for m,a in x.items()}

def dist(z):
    return np.sqrt(np.maximum(2-2*np.clip(np.sum(z['face']*z['voice'],axis=1),-1,1),0))
