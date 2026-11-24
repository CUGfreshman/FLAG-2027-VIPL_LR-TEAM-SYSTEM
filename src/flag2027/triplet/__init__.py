"""Competition-English final-layer triplet projection used by the six-head system."""
from .model import make_model, load_projection, compose_inputs, project_inputs

__all__ = ['make_model', 'load_projection', 'compose_inputs', 'project_inputs']
