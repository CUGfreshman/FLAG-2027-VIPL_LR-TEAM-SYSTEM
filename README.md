# FLAG 2027 VIPL_LR TEAM SYSTEM

This repository reproduces the six-branch face–voice matching system submitted by the VIPL_LR team to the FLAG 2027 challenge (official average EER of 16.25% in the evaluation phase).

**Contents**

- [Repository structure](#repository-structure)
- [1. Environment](#environment)
- [2. Mode 1: run with the released checkpoints](#mode-checkpoints)
  - [2.1 Checkpoints](#checkpoints)
  - [2.2 Encoder sources and weights](#encoder-sources-and-weights)
  - [2.3 Trial manifest](#trial-manifest)
  - [2.4 Run](#run-inference)
- [3. Mode 2: retrain and refit](#mode-retrain)
  - [3.1 Data](#data)
  - [3.2 Training and fitting](#retrain)
  - [3.3 Scoring with the new parameters](#score-new)
- [4. Tests](#tests)
- [Third-party notices and license](#license)

<a name="repository-structure"></a>

## Repository structure

```text
flag2027-sixbranch/
├── README.md
├── LICENSE                          # MIT license for the code written for this system
├── pyproject.toml                   # installable Python package `flag2027`
├── requirements.txt                 # numerical dependencies for scoring and fitting
├── requirements-encoders.txt        # frozen-encoder stack (PyTorch, Transformers, ONNX Runtime, ...)
├── configs/
│   ├── system.json                  # fixed six-branch system: branch order, 1/6 weights, checkpoint paths
│   ├── triplet.json                 # recorded recipe of the triplet branch (not a tuning configuration)
│   ├── cca.json                     # recorded recipe of the CCA branches (not a tuning configuration)
│   ├── encoders.example.json        # local paths of encoder sources and weights (edit before use)
│   └── data_roots.example.json      # dataset roots for MAV-Celeb V1/V2/V4 (edit before use)
├── scripts/
│   ├── run_inference.sh             # one-command inference: extract features, then score
│   ├── extract_features.py          # raw images/audio -> frozen-encoder features (NPZ)
│   ├── score_trials.py              # features -> six branch distances -> official-format ZIP
│   ├── build_media_manifest.py      # training rows + dataset roots -> resolved media CSV
│   ├── prepare_triplet_data.py      # features + CSV -> triplet training inputs
│   ├── train_triplet.py             # train the two embedding heads (encoders stay frozen)
│   ├── project_training.py          # project training media with the trained heads (input to INLP)
│   ├── fit_gender.py                # fit the three-round INLP of the triplet branch
│   ├── fit_cca.py                   # fit one CCA branch
│   ├── prepare_vox.py               # restore the fixed VoxCeleb2 selection as a media manifest
│   └── download_whisper_encoder.py  # extract the encoder-only Whisper weights
├── src/flag2027/
│   ├── inference.py                 # features -> branch distances -> equal-weight mean
│   ├── submission.py                # trial-manifest reader and official-format ZIP writer
│   ├── encoders/                    # frozen-encoder wrappers and reference implementations
│   ├── triplet/                     # embedding heads, loss, sampler and training loop
│   └── cca/                         # CCA fitting, centering, INLP and runtime projection
├── checkpoints/
│   ├── triplet/                     # triplet-branch parameters and their INLP directions
│   ├── cca/                         # parameters of the five CCA branches
│   └── SHA256SUMS                   # checksums of the seven parameter files
├── data_lists/
│   ├── triplet_training_rows.csv    # fixed MAV-Celeb English training pairs
│   └── vox_selected.csv.gz          # fixed VoxCeleb2 selection
├── tests/                           # unit tests
└── vendor/                          # included third-party source (FaceNet, WeSpeaker) with their licenses
```

`weights/`, `external/`, `data/`, `features/` and `outputs/` are directories that you create at run time; they are excluded in `.gitignore`.

<a name="environment"></a>

## 1. Environment

- Operating system: Linux or WSL.
- Python 3.12.
- FFmpeg and FFprobe (to decode video and audio).
- CUDA is required for feature extraction and training; scoring already-extracted features can run on CPU only.

From the root of the checked-out repository (full environment, PyTorch built for CUDA 12.8):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-encoders.txt -e .
```

To score **already-extracted** features on CPU only, install the base dependencies and the CPU build of PyTorch instead:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt -e .
```

Use the editable install (`-e .`) and keep the whole repository; `scripts/`, `configs/`, `checkpoints/` and `vendor/` are not included in a wheel.

<a name="mode-checkpoints"></a>

## 2. Mode 1: run with the released checkpoints

Load the checkpoints provided in the repository (triplet-branch parameters, the five CCA-branch matrices, and so on) and score a trial manifest. Complete 2.1–2.4 in order.

<a name="checkpoints"></a>

### 2.1 Checkpoints

The seven learned parameter files are included in `checkpoints/` (tracked with plain Git). They can be verified with `checkpoints/SHA256SUMS`; no separate download or training is needed.

| File | Content |
|---|---|
| `triplet/epoch0300.pth` | Triplet branch, epoch 300 (53,700 updates in total), including the BatchNorm statistics obtained in training |
| `triplet/inlp198.npz` | The three gender directions (INLP) of the triplet branch |
| `cca/facenet_wespeaker.npz` | CCA branch: FaceNet–WeSpeaker |
| `cca/imagebind_wespeaker.npz` | CCA branch: ImageBind–WeSpeaker |
| `cca/arcface_wespeaker.npz` | CCA branch: ArcFace–WeSpeaker |
| `cca/dinov2_imagebind_wespeaker.npz` | CCA branch: three DINOv2 feature views + ImageBind face features, with WeSpeaker |
| `cca/imagebind_whisper13.npz` | CCA branch: ImageBind–Whisper block 13 |

Each CCA file already contains the projection of its branch, the centering mean vectors and the three-round INLP directions; keep them together. The triplet checkpoint holds inference weights only, without the optimizer and random-number state needed to resume training.

<a name="encoder-sources-and-weights"></a>

### 2.2 Encoder sources and weights

The one-command script starts from raw images and audio, so the frozen encoders must be prepared first. If you already have extracted features, you can skip this section (see 2.4).

**Step 1: get the ImageBind and InsightFace sources.** From the repository root, clone them and check out the pinned commits:

```bash
mkdir -p external
git clone https://github.com/facebookresearch/ImageBind.git external/imagebind
git -C external/imagebind checkout --detach 53680b02d7e37b19b124fa37bae4b6c98c38f5be
git clone https://github.com/deepinsight/insightface.git external/insightface
git -C external/insightface checkout --detach c6c3af7bd2d0a0bf4b78c78437357f3521aa5d52
```

The `source` fields in `configs/encoders.example.json` already point to `external/imagebind` and `external/insightface/python-package/insightface`. The adapters load the source directly from these two paths; do not install them as Python packages.

**Step 2: get the pretrained weights.** Put the weights at the locations listed below, or edit the `weights` fields in `configs/encoders.example.json`. Keep each model's original configuration and processor files together with its weights.

| Encoder | Weight source and version | Local location |
|---|---|---|
| PE-AV Base | [facebook/pe-av-base](https://huggingface.co/facebook/pe-av-base/tree/1c4c329b78d80a18d860ebfb05daebff4bc44518), revision `1c4c329b78d80a18d860ebfb05daebff4bc44518` | `weights/pe-av-base/{model.safetensors,config.json,processor_config.json,preprocessor_config.json,...tokenizer files}` |
| FaceNet | [VGGFace2 weights released by facenet-pytorch](https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt) | `weights/facenet/20180402-114759-vggface2.pt` |
| WeSpeaker ResNet34-LM | [Wespeaker/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/tree/f0c48c298fd835726c27956a5d617bad7115627e), revision `f0c48c298fd835726c27956a5d617bad7115627e` | `weights/wespeaker/avg_model` |
| ImageBind Huge | [official ImageBind weights](https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth) | `weights/imagebind/imagebind_huge.pth` |
| ArcFace buffalo_l | [InsightFace buffalo_l](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip) | `weights/arcface/buffalo_l/{w600k_r50.onnx,det_10g.onnx}` |
| DINOv2 Large | [facebook/dinov2-large](https://huggingface.co/facebook/dinov2-large/tree/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c), revision `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c` | `weights/dinov2-large/{model.safetensors,config.json,preprocessor_config.json}` |
| Whisper Large v3 | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3/tree/06f233fe06e710322aca913c1bc4249a0d71fce1), revision `06f233fe06e710322aca913c1bc4249a0d71fce1` | `weights/whisper-large-v3-encoder/{encoder.safetensors,config.json,preprocessor_config.json}` (see below) |

For PE-AV, DINOv2 and WeSpeaker, use the Hugging Face download tools with the revision listed in the table. Whisper is needed only as an encoder; extract it from the pinned weights with the download script (it keeps only `model.encoder.*` and strips the key prefix; the tensor data are unchanged):

```bash
python scripts/download_whisper_encoder.py --output weights/whisper-large-v3-encoder
```

<a name="trial-manifest"></a>

### 2.3 Trial manifest

Build a CSV with the four columns `trial_id,subset,face_path,audio_path`. The `subset` values are `english_no_gender`, `bangla_no_gender`, `english_gender` and `bangla_gender`. Keep the original trial IDs and the row order within each subset, and use absolute media paths.

<a name="run-inference"></a>

### 2.4 Run

**One command.** The script extracts the features and then scores them:

```bash
bash scripts/run_inference.sh --manifest data/trials.csv --output outputs/scores.zip --device cuda:0
```

By default it uses `configs/encoders.example.json` (encoder paths) and `configs/system.json` (the six branches and checkpoint paths); use `--encoders-config` and `--system-config` to supply other configurations. If you already have an extracted NPZ feature file, add `--features <file>` to skip extraction (the file must keep the `trial_id`, `subset` and media paths written during extraction). The output ZIP contains four files in the challenge format, one per subset, with one `trial_id distance` pair per line; a smaller distance means the pair is more likely to be the same person.

**Step by step.** You can also run the two stages separately:

```bash
python scripts/extract_features.py \
  --manifest data/trials.csv --config configs/encoders.example.json \
  --output features/trials.npz --device cuda:0

python scripts/score_trials.py \
  --manifest data/trials.csv --features features/trials.npz \
  --config configs/system.json --device cpu \
  --output outputs/scores.zip
```

<a name="mode-retrain"></a>

## 3. Mode 2: retrain and refit

Retrain the triplet branch, then refit the five CCA branches and the INLP of every branch. In addition to the encoder sources and weights of 2.2, you need the data below.

<a name="data"></a>

### 3.1 Data

**MAV-Celeb English training set.** It combines the 56 training identities of the official training set (MAV-Celeb V4, English) with the English subsets of the external V1 and V2 data: 5,004 rows from these 56 V4 identities and 5,786 and 12,186 rows from V1 and V2, for 198 identities and 22,976 original pairs in total. The official training set has 70 identities, split by identity 80/20 at random (seed 1); the other 14 identities are held out and are not used for any training or fitting. The identities and pairs in the V4 training subset are listed in [`triplet_training_rows.csv`](data_lists/triplet_training_rows.csv) in the rows with `source=v4`. The datasets can be downloaded from: V1 and V2 at [mavceleb/mavceleb_baseline](https://github.com/mavceleb/mavceleb_baseline); V4 at the [FLAG_2027 repository](https://github.com/SwapnilKhandoker101/FLAG_2027#dataset). Set the dataset roots in `configs/data_roots.example.json`: set the V1 and V2 roots to the directories containing `v1/` and `v2/`, respectively, and the V4 root to the directory containing `faces/` and `voices/`.

**VoxCeleb2 selection.** The fixed selection list is included in the repository as `data_lists/vox_selected.csv.gz` (117,194 training rows and 2,000 held-out rows, 119,194 in total); the source media (the VoxCeleb2 development-set MP4 files) must be downloaded separately, as described below. The list was pre-filtered with the SpeechBrain VoxLingua107 language-identification model ([speechbrain/lang-id-voxlingua107-ecapa](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/tree/0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9), revision `0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9`), which removed utterances whose top-1 label is Bengali (`bn`) or Assamese (`as`).

**VoxCeleb2 media download.** You can use the Hugging Face mirror [Reverb/voxceleb2](https://huggingface.co/datasets/Reverb/voxceleb2); download only the nine MP4 archive parts `vox2_dev_mp4_partaa` to `vox2_dev_mp4_partai` (about 267 GB). The samples in this mirror are cropped face MP4 videos (224 × 224, H.264, 25 fps) that keep the AAC audio track (16 kHz, mono), laid out as `dev/mp4/<identity ID>/<video ID>/<clip number>.mp4`, which is the path structure the repository scripts expect. When running `prepare_vox.py`, point `--mp4-root` at the `dev/mp4` directory.

<a name="retrain"></a>

### 3.2 Training and fitting

**Step 1: rebuild the training inputs.** Extraction and training must use the same CSV with resolved media paths.

```bash
python scripts/build_media_manifest.py \
  --rows data_lists/triplet_training_rows.csv \
  --roots configs/data_roots.example.json --output data/english198.csv

python scripts/extract_features.py \
  --manifest data/english198.csv --config configs/encoders.example.json \
  --output features/english198.npz --device cuda:0

python scripts/prepare_triplet_data.py \
  --rows data/english198.csv --features features/english198.npz \
  --output features/triplet_english198
```

**Step 2: restore the VoxCeleb2 selection and extract features.** The FaceNet–WeSpeaker branch uses 117,194 rows; the other CCA branches use `--drop-zero-audio` to drop one recorded all-zero-audio row (117,193 rows).

```bash
python scripts/prepare_vox.py --mp4-root /data/VoxCeleb2/video/dev/mp4 \
  --split train --output data/vox_fw.csv --check-exists
python scripts/prepare_vox.py --mp4-root /data/VoxCeleb2/video/dev/mp4 \
  --split train --drop-zero-audio --output data/vox_other.csv --check-exists

python scripts/extract_features.py --manifest data/vox_fw.csv \
  --config configs/encoders.example.json --encoders facenet wespeaker \
  --output features/vox_fw.npz --device cuda:0
python scripts/extract_features.py --manifest data/vox_other.csv \
  --config configs/encoders.example.json \
  --encoders imagebind arcface dinov2 whisper13 wespeaker \
  --output features/vox_other.npz --device cuda:0
```

**Step 3: train the triplet branch and fit its INLP.**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_triplet.py \
  --data features/triplet_english198 --output outputs/triplet
python scripts/project_training.py \
  --checkpoint outputs/triplet/epoch0300.pth \
  --data features/triplet_english198 --output features/triplet_training_embeddings.npz
python scripts/fit_gender.py \
  --training-projections features/triplet_training_embeddings.npz \
  --output outputs/triplet_INLP3.npz
```

Training uses a single CUDA device, and the inference weights are written to `outputs/triplet/epoch0300.pth` when it finishes. To resume, point `--output` at the directory that contains `checkpoint_latest.pth.tar`; to train a new model, use a new output directory.

**Step 4: fit the five CCA branches.**

```bash
python scripts/fit_cca.py --branch facenet_wespeaker \
  --vox-features features/vox_fw.npz --vox-rows data/vox_fw.csv \
  --training-features features/english198.npz --training-rows data/english198.csv \
  --output outputs/facenet_wespeaker.npz

for branch in imagebind_wespeaker arcface_wespeaker dinov2_imagebind_wespeaker imagebind_whisper13; do
  python scripts/fit_cca.py --branch "$branch" \
    --vox-features features/vox_other.npz --vox-rows data/vox_other.csv \
    --training-features features/english198.npz --training-rows data/english198.csv \
    --output "outputs/${branch}.npz"
done
```

<a name="score-new"></a>

### 3.3 Scoring with the new parameters

Copy `configs/system.json` (keep the copy in `configs/`) and change its parameter paths to the newly generated files. Relative paths in the configuration are resolved against the parent of the directory that contains the configuration file; absolute paths also work. Then run:

```bash
bash scripts/run_inference.sh --manifest data/trials.csv --output outputs/scores_new.zip \
  --system-config <your-config> --device cuda:0
```

`configs/triplet.json` and `configs/cca.json` only record the fixed recipes; the training and fitting commands do not read them to set hyperparameters.

<a name="tests"></a>

## 4. Tests

```bash
python -m unittest discover -s tests -v
```

<a name="license"></a>

## Third-party notices and license

- The code written for this system in `src/`, `scripts/`, `tests/` and `configs/` is released under the MIT License (see `LICENSE`).
- Third-party source in `vendor/` keeps its own license: FaceNet (`vendor/facenet_pytorch/LICENSE.upstream`) is MIT and WeSpeaker (`vendor/wespeaker/LICENSE.upstream`) is Apache License 2.0. Files that carry their own notice, such as the argsort compatibility code (`src/flag2027/encoders/reference/standard/argsort_compat.LICENSE.md`), are governed by that notice.
- The ImageBind source is released by Meta under CC BY-NC-SA 4.0 (non-commercial use, share-alike), and the terms for its weights are in the [official model card](https://github.com/facebookresearch/ImageBind/blob/53680b02d7e37b19b124fa37bae4b6c98c38f5be/model_card.md); the pretrained InsightFace weights may be used for non-commercial research only. This repository distributes neither; obtain them as described in [2.2](#encoder-sources-and-weights) and comply with their terms.
- The terms of the other public weights are set by their upstream sources; please check them yourself.
- This repository does not distribute dataset media; their use is subject to the terms of the dataset providers.
