"""Path-configurable wrappers around the archived production encoder bodies.

No fitted statistic is computed here. Each item is encoded independently in
eval mode. Feature dimensions and per-model scaling match the deployed views.
"""
from __future__ import annotations

import ast
import csv
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / 'reference'
REPOSITORY = HERE.parents[2]
FEATURE_DIMENSIONS = {
    'face_peav_final': 1024, 'voice_peav_final': 1024,
    'face_facenet': 512, 'voice_wespeaker': 256,
    'face_imagebind': 1024, 'face_arcface': 512,
    'face_dino_cls': 1024, 'face_dino_patch': 1024, 'face_dino_layer12': 1024,
    'voice_whisper13': 1280,
}
ENCODER_KEYS = {
    'peav': ('face_peav_final', 'voice_peav_final'),
    'facenet': ('face_facenet',), 'wespeaker': ('voice_wespeaker',),
    'imagebind': ('face_imagebind',), 'arcface': ('face_arcface',),
    'dinov2': ('face_dino_cls', 'face_dino_patch', 'face_dino_layer12'),
    'whisper13': ('voice_whisper13',),
}


def read_manifest(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'trial_id', 'subset', 'face_path', 'audio_path'}
        if not required <= set(reader.fieldnames or []):
            raise ValueError('Manifest needs trial_id,subset,face_path,audio_path')
        rows = list(reader)
    keys = [(row['subset'], row['trial_id']) for row in rows]
    if len(keys) != len(set(keys)) or not rows:
        raise ValueError('Manifest must be nonempty and (subset,trial_id) unique')
    return rows


def config_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPOSITORY / path


def encoder_source(cfg, name, default, entry):
    root = config_path(cfg.get('source', default)).resolve()
    if not (root / entry).is_file():
        raise FileNotFoundError(f'{name}: expected {root / entry}. See section 2.2 "Encoder sources and weights" in README.md for the pinned source checkout.')
    return root


def module(name, path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def original_functions(path, names, namespace):
    """Load named definitions from the bundled encoder source.

    Callers pass fixed paths below `REFERENCE`, not caller-supplied Python.
    Compile only the named definitions, with their dependencies supplied by
    the adapter, so archived top-level setup is not executed.
    """
    path = Path(path)
    text = path.read_text(encoding='utf-8-sig')
    tree = ast.parse(text)
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise RuntimeError('Archived source does not contain requested production bodies')
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def rgb(path):
    from PIL import Image
    path = Path(path)
    if path.suffix.lower() not in ('.mp4', '.mkv', '.avi', '.mov'):
        with Image.open(path) as image:
            return image.convert('RGB').copy()
    import cv2
    capture = cv2.VideoCapture(str(path))
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or count <= 0:
            raise ValueError('Video cannot be decoded: ' + str(path))
        capture.set(cv2.CAP_PROP_POS_FRAMES, count // 2)
        ok, bgr = capture.read()
        if not ok:
            raise ValueError('Middle frame cannot be decoded: ' + str(path))
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()


def native_audio(path):
    """Whole native-rate float waveform, arithmetic channel mean; PE-AV path."""
    import soundfile as sf
    path = Path(path)
    if path.suffix.lower() in ('.wav', '.flac', '.ogg'):
        audio, rate = sf.read(path, dtype='float32', always_2d=True)
    else:
        info = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0',
            '-show_entries', 'stream=sample_rate,channels', '-of', 'json', str(path)],
            capture_output=True, check=True)
        stream = json.loads(info.stdout)['streams'][0]
        rate, channels = int(stream['sample_rate']), int(stream['channels'])
        data = subprocess.run(['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(path),
            '-map', '0:a:0', '-vn', '-f', 'f32le', 'pipe:1'], capture_output=True, check=True).stdout
        audio = np.frombuffer(data, dtype='<f4').reshape(-1, channels)
    mono = np.ascontiguousarray(audio.mean(axis=1, dtype=np.float32))
    if not mono.size or not np.isfinite(mono).all():
        raise ValueError('Invalid native audio: ' + str(path))
    return mono, rate


class PEAV:
    def __init__(self, cfg, device):
        # Reuse the real Encoders.visual/audio methods, whose output was stored
        # as hook.final. No intermediate pooling or legacy_final replacement.
        ref = REFERENCE / 'peav'
        native = module('_flag_peav_original', ref / 'extract_peav.py')
        self.encoder = native.Encoders(SimpleNamespace(weights=config_path(cfg['weights']),
            threads=4, device=device, image_batch=1))
        route = cfg.get('visual_resize_route', 'source60_float32_round_uint8_bilinear_aa')
        if route == 'source60_float32_round_uint8_bilinear_aa':
            processor = self.encoder.processor.video_processor
            resize = processor.resize
            torch = self.encoder.torch
            if processor.to_dict()['resample'] != 2:
                raise ValueError('Unexpected original PE-AV resize method')
            def source60_resize(image, *args, **kwargs):
                if image.device.type == 'cpu' and image.dtype == torch.uint8:
                    return resize(image.float(), *args, **kwargs).round().to(torch.uint8)
                return resize(image, *args, **kwargs)
            processor.resize = source60_resize
        elif route != 'native':
            raise ValueError('Unknown PE-AV resize route')

    def encode(self, row):
        face = self.encoder.visual([np.asarray(rgb(row['face_path']))])[0]
        voice = self.encoder.audio([native_audio(row['audio_path'])])[0][0]
        return {'face_peav_final': face, 'voice_peav_final': voice}


class Standard:
    def __init__(self, name, cfg, device):
        ref = REFERENCE / 'standard'
        sys.modules['argsort_compat'] = module('_flag_argsort_reference', ref / 'argsort_compat.py')
        native = module('_flag_standard_original_' + name, ref / 'standard.py')
        native.RESOURCE_PATHS.update({
            'facenet_package': str(REPOSITORY / 'vendor/facenet_pytorch'),
            'facenet_checkpoint': str(config_path(cfg.get('weights', 'weights/facenet/20180402-114759-vggface2.pt'))),
            'wespeaker_checkpoint': str(config_path(cfg.get('weights', 'weights/wespeaker/avg_model'))),
            'wespeaker_resnet': str(REPOSITORY / 'vendor/wespeaker/release__wespeaker__models__resnet.py'),
            'wespeaker_pooling': str(REPOSITORY / 'vendor/wespeaker/release__wespeaker__models__pooling_layers.py'),
        })
        self.encoder = native.load_encoder(name, {'resources': str(REPOSITORY)}, device)
        self.name = name

    def encode(self, row):
        modality = 'face' if self.name == 'facenet' else 'voice'
        path = row['face_path'] if modality == 'face' else row['audio_path']
        # The original FaceNet callable takes a PIL image; retain its exact
        # MTCNN and historical fallback for both image and Vox video inputs.
        if self.name == 'facenet' and Path(path).suffix.lower() in ('.mp4', '.mkv', '.avi', '.mov'):
            import torch
            picture = rgb(path)
            crop, probability = self.encoder.detector(picture, return_prob=True)
            if crop is None:
                crop = self.encoder.standardize(self.encoder.extract_face(picture,
                    [0, 0, picture.width, picture.height], image_size=160, margin=0))
            with torch.inference_mode():
                vector = self.encoder.model(crop.unsqueeze(0).to(self.encoder.device))[0].cpu().numpy()
        else:
            vector = self.encoder.encode_batch(modality, [{'unique_index': 0, 'canonical_path': path, 'io_path': path}])[0]
        return {('face_facenet' if self.name == 'facenet' else 'voice_wespeaker'): vector}


class ArcFace:
    def __init__(self, cfg, device):
        source = encoder_source(cfg, 'InsightFace', 'external/insightface/python-package/insightface',
                                'model_zoo/scrfd.py')
        import onnxruntime as ort
        native = module('_flag_arcface_entry', REFERENCE / 'arcface/face_extract_original.py')
        SCRFD, ArcFaceONNX, self.align = native.official_modules(source)
        gpu = device.startswith('cuda')
        gpu_id = int(device.partition(':')[2] or '0') if gpu else -1
        providers = [('CUDAExecutionProvider', {'device_id': gpu_id}), 'CPUExecutionProvider'] if gpu else ['CPUExecutionProvider']
        if gpu and 'CUDAExecutionProvider' not in ort.get_available_providers():
            raise RuntimeError('Install the declared GPU ONNX Runtime build')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4; options.inter_op_num_threads = 1
        options.use_deterministic_compute = True; options.log_severity_level = 3
        weights = config_path(cfg['weights'])
        detector = weights / 'det_10g.onnx'; recognizer = weights / 'w600k_r50.onnx'
        ds = ort.InferenceSession(str(detector), sess_options=options, providers=providers)
        rs = ort.InferenceSession(str(recognizer), sess_options=options, providers=providers)
        if gpu and (ds.get_providers()[0] != 'CUDAExecutionProvider' or rs.get_providers()[0] != 'CUDAExecutionProvider'):
            raise RuntimeError('ArcFace sessions did not activate CUDA')
        self.sizes = [(128, 128), (640, 640)]
        self.detector = SCRFD(model_file=str(detector), session=ds, static_shape_sessions=False)
        self.detector.prepare(ctx_id=0 if gpu else -1, input_size=self.sizes, det_thresh=.5)
        self.encoder = ArcFaceONNX(model_file=str(recognizer), session=rs)
        self.encoder.prepare(ctx_id=0 if gpu else -1)

    def encode(self, row):
        import cv2
        from PIL import Image
        path = Path(row['face_path'])
        if path.suffix.lower() in ('.mp4', '.mkv', '.avi', '.mov'):
            bgr = np.asarray(rgb(path))[:, :, ::-1].copy()
        else:
            # Preserve the original OpenCV JPEG/image decoder for stills.
            bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError('Unreadable ArcFace image: ' + str(path))
        boxes, landmarks = self.detector.detect(bgr, input_size=self.sizes, max_num=0)
        if len(boxes):
            aligned = self.align.norm_crop(bgr, landmark=landmarks[0], image_size=112)
        else:
            image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            aligned = np.asarray(image.resize((112, 112), Image.Resampling.BICUBIC))[:, :, ::-1].copy()
        return {'face_arcface': np.asarray(self.encoder.get_feat(aligned), dtype=np.float32).reshape(-1)}


class ImageBind:
    def __init__(self, cfg, device):
        source = encoder_source(cfg, 'ImageBind', 'external/imagebind', 'imagebind/models/imagebind_model.py')
        import torch
        import torchvision.transforms._functional_tensor as ft
        sys.modules.setdefault('torchvision.transforms.functional_tensor', ft)
        sys.path.insert(0, str(source))
        from imagebind.models.imagebind_model import imagebind_huge
        from torchvision import transforms
        self.model = imagebind_huge(pretrained=False)
        self.model.load_state_dict(torch.load(config_path(cfg['weights']), map_location='cpu', weights_only=True), strict=True)
        self.model.float().to(device).eval().requires_grad_(False)
        self.device = device
        self.transform = transforms.Compose([transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])

    def encode(self, row):
        import torch
        with torch.inference_mode():
            value = self.model({'vision': self.transform(rgb(row['face_path'])).unsqueeze(0).to(self.device)})['vision']
        return {'face_imagebind': value.float().cpu().numpy()[0]}


class DINO:
    def __init__(self, cfg, device):
        import torch
        import torch.nn.functional as F
        from transformers import AutoImageProcessor, Dinov2Model
        weights = config_path(cfg['weights'])
        self.processor = AutoImageProcessor.from_pretrained(weights, local_files_only=True, use_fast=False)
        self.model = Dinov2Model.from_pretrained(weights, local_files_only=True).float().to(device).eval().requires_grad_(False)
        self.device = device
        self.native = original_functions(REFERENCE / 'dinov2/extract_dino.py', ['check', 'encode'],
            {'torch': torch, 'np': np, 'F': F})['encode']

    def encode(self, row):
        pixels = self.processor(images=rgb(row['face_path']), return_tensors='pt')['pixel_values'].to(self.device)
        value = self.native(self.model, pixels)[0].reshape(3, 1024)
        return dict(zip(ENCODER_KEYS['dinov2'], value))


class Whisper13:
    def __init__(self, cfg, device):
        import torch
        from transformers import WhisperConfig, WhisperFeatureExtractor
        from transformers.models.whisper.modeling_whisper import WhisperEncoder
        from safetensors.torch import load_file
        weights = config_path(cfg['weights'])
        config = WhisperConfig.from_pretrained(weights, local_files_only=True)
        config._attn_implementation = 'sdpa'
        self.model = WhisperEncoder(config)
        self.model.load_state_dict(load_file(str(weights / 'encoder.safetensors')), strict=True)
        if not device.startswith('cuda'):
            raise ValueError('Production Whisper uses fp16 CUDA; CPU is not the validated production route')
        self.model.to(device).half().eval().requires_grad_(False)
        self.processor = WhisperFeatureExtractor.from_pretrained(weights, local_files_only=True)
        self.device = device
        self.decode = original_functions(REFERENCE / 'whisper/decode_audio_original.py', ['decode'],
            {'Path': Path, 'np': np, 'subprocess': subprocess, 'sf': __import__('soundfile'), 'torch': torch})['decode']

    def encode(self, row):
        import torch
        import torch.nn.functional as F
        wave = self.decode({'io_path': row['audio_path']})
        inputs = self.processor([wave], sampling_rate=16000, return_tensors='pt', return_attention_mask=True,
            padding='max_length', max_length=480000, truncation=True, device=self.device)
        length = math.ceil(len(wave) / 320)
        with torch.inference_mode():
            output = self.model(inputs.input_features.to(self.device).half(), output_hidden_states=True, return_dict=True)
            z = output.hidden_states[13].float()
            pooled = z[:, :length].sum(1) / length
            vector = F.normalize(pooled, dim=-1).cpu().numpy()[0]
        return {'voice_whisper13': vector}


def extract_manifest(rows, config, output, device='cuda:0', names=None):
    import torch
    torch.set_num_threads(4); torch.manual_seed(0); np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    names = names or list(ENCODER_KEYS)
    factories = {'peav': PEAV, 'facenet': lambda cfg, dev: Standard('facenet', cfg, dev),
        'wespeaker': lambda cfg, dev: Standard('wespeaker', cfg, dev), 'imagebind': ImageBind,
        'arcface': ArcFace, 'dinov2': DINO, 'whisper13': Whisper13}
    arrays = {}
    for name in names:
        if name not in factories:
            raise ValueError('Unknown encoder: ' + name)
        encoder = factories[name](config[name], device)
        for key in ENCODER_KEYS[name]:
            arrays[key] = np.empty((len(rows), FEATURE_DIMENSIONS[key]), np.float32)
        for index, row in enumerate(rows):
            values = encoder.encode(row)
            for key, vector in values.items():
                vector = np.asarray(vector, dtype=np.float32)
                if vector.shape != (FEATURE_DIMENSIONS[key],) or not np.isfinite(vector).all() or np.linalg.norm(vector) <= 0:
                    raise ValueError('Invalid encoder vector: ' + key + ', row ' + str(index))
                arrays[key][index] = vector
            if (index + 1) % 25 == 0 or index + 1 == len(rows):
                print(json.dumps({'encoder': name, 'done': index + 1, 'total': len(rows)}), flush=True)
        del encoder
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    arrays['trial_id'] = np.asarray([r['trial_id'] for r in rows])
    arrays['subset'] = np.asarray([r['subset'] for r in rows])
    arrays['face_path'] = np.asarray([r['face_path'] for r in rows])
    arrays['audio_path'] = np.asarray([r['audio_path'] for r in rows])
    for key in ['identity', 'gender', 'source', 'language']:
        if all(key in row for row in rows):arrays[key] = np.asarray([r[key] for r in rows])
    output = Path(output)
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:np.savez_compressed(stream, **arrays)
    return {'rows': len(rows), 'features': {key: list(a.shape) for key, a in arrays.items() if key in FEATURE_DIMENSIONS},
        'encoders': names, 'device': device, 'estimated_statistics': False}
