"""Frozen FaceNet and WeSpeaker adapters with their original preprocessing.

``load_encoder(name, {"resources": resource_root}, device)`` returns an object
with ``encode_batch(modality, records) -> float32[N,D]``. Records are consumed in
their given order and contain unique_index, canonical_path, and io_path. Each
file is forwarded separately, matching the original official extraction.
``last_records`` contains the latest batch's preprocessing diagnostics;
``audit_records`` accumulates them for the worker to save as JSON.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
import sys
import types

import numpy as np


RESOURCE_PATHS = {
    "facenet_package": "vendor/facenet_pytorch",
    "facenet_checkpoint": "weights/facenet/20180402-114759-vggface2.pt",
    "wespeaker_checkpoint": "weights/wespeaker/avg_model",
    "wespeaker_resnet": "vendor/wespeaker/release__wespeaker__models__resnet.py",
    "wespeaker_pooling": "vendor/wespeaker/release__wespeaker__models__pooling_layers.py",
}


def _source_module(name, path, *, package=False):
    path = Path(path).resolve()
    if name in sys.modules:
        previous = sys.modules[name]
        if Path(previous.__file__).resolve() != path:
            raise RuntimeError(f"Conflicting source for {name}: {previous.__file__} versus {path}")
        return previous
    kwargs = {"submodule_search_locations": [str(path.parent)]} if package else {}
    spec = importlib.util.spec_from_file_location(name, path, **kwargs)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load original source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _facenet_code(package_path):
    return _source_module("_t068_original_facenet", package_path / "__init__.py", package=True)


def _install_reference_nms(package):
    """Preserve the recorded tie ordering in the bundled FaceNet NMS."""
    from argsort_compat import argsort_reference
    source = sys.modules[package.__name__ + ".models.utils.detect_face"]
    if getattr(source.nms_numpy, "_t068_reference_nms", False):
        return source._t068_nms_audit
    text = inspect.getsource(source.nms_numpy)
    original_call = "I = np.argsort(s)"
    if text.count(original_call) != 1:
        raise ValueError("Unexpected original FaceNet NMS sorting implementation")
    namespace = dict(vars(source), argsort_reference=argsort_reference)
    exec(compile(text.replace(original_call, "I = argsort_reference(s)"),
                 str(source.__file__) + "::reference_avx512_argsort", "exec"), namespace)
    portable_nms = namespace["nms_numpy"]
    audit = []

    def reference_nms(boxes, scores, threshold, method):
        audit.append(dict(candidates=len(scores),
                          exact_score_ties=len(scores) - len(np.unique(scores))))
        return portable_nms(boxes, scores, threshold, method)

    reference_nms._t068_reference_nms = True
    source.nms_numpy = reference_nms
    source._t068_nms_audit = audit
    return audit


def _wespeaker_resnet(resnet_path, pooling_path):
    """Load the bundled WeSpeaker source, changing only its pooling import."""
    namespace = "_t068_original_wespeaker"
    if namespace not in sys.modules:
        package = types.ModuleType(namespace)
        package.__path__ = [str(resnet_path.parent)]
        sys.modules[namespace] = package
    pooling = _source_module(namespace + ".pooling_layers", pooling_path)
    setattr(sys.modules[namespace], "pooling_layers", pooling)
    name = namespace + ".resnet"
    if name in sys.modules:
        if Path(sys.modules[name].__file__).resolve() != resnet_path.resolve():
            raise RuntimeError("Conflicting original WeSpeaker release source")
        return sys.modules[name].ResNet34
    text = resnet_path.read_text(encoding="utf-8-sig")
    original_import = "import wespeaker.models.pooling_layers as pooling_layers"
    if text.count(original_import) != 1:
        raise ValueError(f"Unexpected release pooling import: {resnet_path}")
    text = text.replace(original_import, f"from {namespace} import pooling_layers")
    module = types.ModuleType(name)
    module.__file__ = str(resnet_path)
    module.__package__ = namespace
    sys.modules[name] = module
    exec(compile(text, str(resnet_path), "exec"), module.__dict__)
    return module.ResNet34


class _StandardEncoder:
    def __init__(self, name, config, device):
        import torch
        self.name = name
        self.device = torch.device(device)
        self.resources = Path(config["resources"]).expanduser().resolve()
        self.paths = {key: self.resources / relative for key, relative in RESOURCE_PATHS.items()}
        self.dimension = {"facenet": 512, "wespeaker": 256}[name]
        self.modality = "face" if name == "facenet" else "voice"
        specification = json.loads(Path(__file__).with_name("standard_contract.json").read_text(encoding="utf-8"))
        dependency_keys = {
            "facenet": ("facenet_package", "facenet_checkpoint"),
            "wespeaker": ("wespeaker_checkpoint", "wespeaker_resnet", "wespeaker_pooling"),
        }[name]
        self.contract = {
            "encoder": name,
            **specification[name],
            "resources": str(self.resources),
            "resource_paths": {key: str(self.paths[key]) for key in dependency_keys},
            "frozen_eval": True,
            "requires_grad": False,
            "inference_mode": True,
            "dtype": "float32",
            "allow_tf32": False,
            "forward_batch_size": 1,
            "fitting_or_adaptation": False,
        }
        if name == "facenet":
            nms_sort = config.get("encoders", {}).get("facenet", {}).get(
                "nms_sort", "reference_numpy1264_avx512")
            if nms_sort != "reference_numpy1264_avx512":
                raise ValueError(f"Unsupported frozen FaceNet NMS sort contract: {nms_sort}")
            self.contract["nms_sort"] = nms_sort
        self.last_records = []
        self.audit_records = []
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self._load()

    def _load(self):
        import torch
        if self.name == "facenet":
            package = _facenet_code(self.paths["facenet_package"])
            self._nms_audit = _install_reference_nms(package)
            self.model = package.InceptionResnetV1(pretrained=None, classify=True, num_classes=8631)
            state = torch.load(self.paths["facenet_checkpoint"], map_location="cpu", weights_only=True)
            self.model.load_state_dict(state, strict=True)
            self.model.classify = False
            self.detector = package.MTCNN(device=self.device)
            self.detector.to(self.device).eval().requires_grad_(False)
            mtcnn = sys.modules[package.__name__ + ".models.mtcnn"]
            utils = sys.modules[package.__name__ + ".models.utils.detect_face"]
            self.standardize = mtcnn.fixed_image_standardization
            self.extract_face = utils.extract_face
        else:
            state = torch.load(self.paths["wespeaker_checkpoint"], map_location="cpu", weights_only=True)
            if not isinstance(state, dict) or not all(isinstance(value, torch.Tensor) for value in state.values()):
                raise ValueError("Expected original WeSpeaker tensor state dictionary")
            if tuple(state["projection.weight"].shape) != (17982, 256):
                raise ValueError("Unexpected original WeSpeaker classifier dimensions")
            backbone = {key: value for key, value in state.items() if not key.startswith("projection.")}
            if set(state) - set(backbone) != {"projection.weight"}:
                raise ValueError("Unexpected original WeSpeaker projection keys")
            factory = _wespeaker_resnet(self.paths["wespeaker_resnet"], self.paths["wespeaker_pooling"])
            self.model = factory(feat_dim=80, embed_dim=256, pooling_func="TSTP", two_emb_layer=False)
            self.model.load_state_dict(backbone, strict=True)
        self.model.to(self.device).float().eval().requires_grad_(False)

    def _face(self, path):
        import torch
        from PIL import Image
        with Image.open(path) as opened:
            picture = opened.convert("RGB")
        nms_before = len(self._nms_audit)
        crop, probability = self.detector(picture, return_prob=True)
        info = dict(detection_probability=None if probability is None else float(probability),
                    original_status="ok", processing="original_standard_encoder",
                    final_nms_sort="NumPy1.26.4 reference AVX512 argsort, portable scalar network/partition",
                    final_nms_calls=self._nms_audit[nms_before:])
        if crop is None:
            # Frozen official rule: only MTCNN no_face triggers full-source crop.
            box = [0, 0, picture.width, picture.height]
            crop = self.standardize(self.extract_face(picture, box, image_size=160, margin=0))
            info.update(original_status="no_face", processing="original_no_face__official_full_source_crop160",
                        box=box, original_image_size=list(picture.size))
        if tuple(crop.shape) != (3, 160, 160) or not torch.isfinite(crop).all():
            raise ValueError(f"Invalid original FaceNet crop: {tuple(crop.shape)}")
        value = self.model(crop.unsqueeze(0).to(self.device))[0].cpu().numpy()
        return value, info

    def _wespeaker(self, path):
        import torchaudio
        import torchaudio.compliance.kaldi as kaldi
        pcm, rate = torchaudio.load(path, normalize=False)
        original_dtype = str(pcm.dtype)
        if pcm.ndim != 2 or pcm.shape[0] != 1:
            raise ValueError(f"Original WeSpeaker expects mono, got {tuple(pcm.shape)}")
        pcm = pcm.float()
        if rate != 16000:
            pcm = torchaudio.transforms.Resample(rate, 16000)(pcm)
        feats = kaldi.fbank(pcm, num_mel_bins=80, frame_length=25, frame_shift=10, sample_frequency=16000, dither=0.)
        feats = feats - feats.mean(0, keepdim=True)
        if len(feats) < 20:
            raise ValueError(f"Insufficient WeSpeaker fbank frames: {len(feats)}")
        value = self.model(feats.unsqueeze(0).to(self.device))[-1][0].cpu().numpy()
        return value, dict(input_dtype=original_dtype, input_sample_rate=rate, duration_seconds=pcm.shape[1] / 16000,
                           fbank_frames=len(feats), processing="original_integer_PCM_fbank80_CMN")

    def encode_batch(self, modality, records):
        import torch
        if modality != self.modality:
            raise ValueError(f"{self.name} supports {self.modality}, received {modality}")
        records = list(records)
        self.last_records = []
        outputs = []
        encode_one = {"facenet": self._face, "wespeaker": self._wespeaker}[self.name]
        for record in records:
            label = {key: record.get(key) for key in ("unique_index", "canonical_path", "io_path")}
            try:
                path = str(record["io_path"])
                with torch.inference_mode():
                    value, info = encode_one(path)
                value = np.asarray(value)
                if value.shape != (self.dimension,) or value.dtype != np.float32 or not np.isfinite(value).all() or np.linalg.norm(value) <= 0:
                    raise ValueError(f"Invalid {self.name} output shape/dtype/values: {value.shape}/{value.dtype}")
                # FaceNet's original forward already emits unit512.
                # WeSpeaker retains raw scale; consumers normalize their blocks.
                outputs.append(np.ascontiguousarray(value))
                audit = dict(**label, encoder=self.name, status="ok", **info,
                             upstream_norm=float(np.linalg.norm(value)))
                self.last_records.append(audit)
                self.audit_records.append(audit)
            except Exception as error:
                audit = dict(**label, encoder=self.name, status="encoding_error",
                             error_type=type(error).__name__, error=str(error))
                self.last_records.append(audit)
                self.audit_records.append(audit)
                raise RuntimeError(f"{self.name} encoding failed for {label}: {type(error).__name__}: {error}") from error
        return np.stack(outputs).astype(np.float32, copy=False) if outputs else np.empty((0, self.dimension), dtype=np.float32)


def load_encoder(name, config, device):
    if name not in ("facenet", "wespeaker"):
        raise ValueError(f"Unsupported standard encoder: {name}")
    return _StandardEncoder(name, config, device)
