"""Frozen CCA projection and original training-only fitting functions."""
from .runtime import load_model, project_pair, score_pair, unit, compose_view, apply_inlp

__all__ = ['load_model', 'project_pair', 'score_pair', 'unit', 'compose_view', 'apply_inlp']
