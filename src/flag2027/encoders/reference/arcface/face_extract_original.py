"""Load SCRFD, ArcFace and alignment from the caller's pinned InsightFace checkout."""
from pathlib import Path
import importlib
import sys
import types


def official_modules(source):
    """Load the original small inference modules without the full package init."""
    source = Path(source).resolve()
    root = '_encoder_eval_insightface'
    for name, path in [(root, source), (root + '.model_zoo', source / 'model_zoo'),
                       (root + '.utils', source / 'utils')]:
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    scrfd = importlib.import_module(root + '.model_zoo.scrfd')
    arcface = importlib.import_module(root + '.model_zoo.arcface_onnx')
    face_align = importlib.import_module(root + '.utils.face_align')
    return scrfd.SCRFD, arcface.ArcFaceONNX, face_align
