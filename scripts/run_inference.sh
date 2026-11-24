#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: run_inference.sh --manifest PATH --output PATH [options]

  --manifest PATH          Trial CSV: trial_id,subset,face_path,audio_path
  --output PATH            New challenge-format ZIP (must not exist)
  --device DEV             Feature extraction device (default: cuda:0)
  --encoders-config PATH   Encoder config (default: configs/encoders.example.json)
  --system-config PATH     Scoring config (default: configs/system.json)
  --features PATH          Reuse an existing NPZ, or extract to this new path
                           Default: features/<output stem>.npz; refuse reuse
                           of this default unless given explicitly
  --no-verify              Skip the shipped checkpoint checksums
  --dry-run                Check inputs and print commands without running inference
  -h, --help               Show this help

Scoring always runs on CPU. Set PYTHON to select the Python executable.
Explicit paths are relative to the caller; default paths are relative to the
repository. Extraction also writes a .contract.json sidecar next to the NPZ.
USAGE
}

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

caller_dir="$PWD"
manifest=''
output=''
device='cuda:0'
encoders_config=''
system_config=''
features=''
verify=1
dry_run=0

while (( $# )); do
    case "$1" in
        --manifest|--output|--device|--encoders-config|--system-config|--features)
            if (( $# < 2 )) || [[ -z "$2" || "$2" == --* || "$2" == -h ]]; then
                fail "$1 requires a value"
            fi
            case "$1" in
                --manifest) manifest="$2" ;;
                --output) output="$2" ;;
                --device) device="$2" ;;
                --encoders-config) encoders_config="$2" ;;
                --system-config) system_config="$2" ;;
                --features) features="$2" ;;
            esac
            shift 2
            ;;
        --no-verify) verify=0; shift ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown argument: $1 (see --help)" ;;
    esac
done
[[ -n "$manifest" ]] || fail "--manifest is required (see --help)"
[[ -n "$output" ]] || fail "--output is required (see --help)"

absolute_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *)  printf '%s/%s\n' "$caller_dir" "$1" ;;
    esac
}

path_exists() { [[ -e "$1" || -L "$1" ]]; }

repo_root="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
manifest="$(absolute_path "$manifest")"
output="$(absolute_path "$output")"
if [[ -n "$encoders_config" ]]; then
    encoders_config="$(absolute_path "$encoders_config")"
else
    encoders_config="$repo_root/configs/encoders.example.json"
fi
if [[ -n "$system_config" ]]; then
    system_config="$(absolute_path "$system_config")"
else
    system_config="$repo_root/configs/system.json"
fi

[[ -f "$manifest" ]] || fail "Manifest file not found: $manifest"
[[ -f "$encoders_config" ]] || fail "Encoder config file not found: $encoders_config"
[[ -f "$system_config" ]] || fail "System config file not found: $system_config"
if path_exists "$output"; then
    fail "Output already exists: $output"
fi

extract=1
if [[ -n "$features" ]]; then
    features="$(absolute_path "$features")"
    if path_exists "$features"; then
        [[ -f "$features" ]] || fail "Feature path is not a file: $features"
        extract=0
    fi
else
    output_name="${output##*/}"
    features="$repo_root/features/${output_name%.*}.npz"
    if path_exists "$features"; then
        fail "Default feature path already exists: $features (use --features to reuse it or select a new path)"
    fi
fi

python_bin="${PYTHON:-python}"
if [[ "$python_bin" == */* ]]; then
    python_bin="$(absolute_path "$python_bin")"
fi
cd -- "$repo_root"
command -v "$python_bin" >/dev/null || fail "Python executable not found: $python_bin (set PYTHON)"

"$python_bin" -B - "$system_config" "$output" "$features" "$extract" <<'PY'
import json
import sys
from pathlib import Path

config_path, output, features = map(Path, sys.argv[1:4])
if output.suffix.lower() != ".zip" or len(output.name) > 64:
    sys.exit("Error: Output must be a ZIP filename of at most 64 characters")

if sys.argv[4] == "1":
    contract = features.with_suffix(".contract.json")
    destinations = {path.resolve() for path in (output, features, contract)}
    if len(destinations) != 3:
        sys.exit("Error: ZIP, feature NPZ, and extraction sidecar paths must be distinct")
    if contract.exists() or contract.is_symlink():
        sys.exit(f"Error: Extraction sidecar already exists: {contract}")

# Match score_features: relative checkpoint paths start two levels above the config file.
try:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cca = config["cca_models"]
    if not isinstance(cca, dict) or len(cca) != 5:
        raise ValueError("expected five CCA parameter paths")
    params = [config["triplet_checkpoint"], config["triplet_inlp"], *cca.values()]
    if not all(isinstance(path, str) and path for path in params):
        raise ValueError("expected nonempty string parameter paths")
except (OSError, ValueError, KeyError, TypeError) as exc:
    sys.exit(f"Error: Invalid system config {config_path}: {exc}")

for param in params:
    path = config_path.parent.parent / param
    if not path.is_file():
        sys.exit(f"Error: Checkpoint file not found: {path}")
PY

if (( verify )); then
    command -v sha256sum >/dev/null || fail "sha256sum not found (use --no-verify to skip checksums)"
    [[ -f "$repo_root/checkpoints/SHA256SUMS" ]] || fail "Checkpoint checksum list not found: $repo_root/checkpoints/SHA256SUMS"
    if ! (cd -- "$repo_root/checkpoints" && sha256sum --quiet -c SHA256SUMS); then
        fail "Shipped checkpoint checksum verification failed"
    fi
fi

extract_cmd=("$python_bin" -B "$repo_root/scripts/extract_features.py"
    --manifest "$manifest" --config "$encoders_config" --output "$features" --device "$device")
score_cmd=("$python_bin" -B "$repo_root/scripts/score_trials.py"
    --manifest "$manifest" --features "$features" --config "$system_config" --device cpu --output "$output")

print_command() {
    printf '%q' "$1"
    shift
    printf ' %q' "$@"
    printf '\n'
}

if (( dry_run )); then
    if (( extract )); then
        print_command "${extract_cmd[@]}"
    fi
    print_command "${score_cmd[@]}"
    exit 0
fi

if (( extract )); then
    "${extract_cmd[@]}"
fi
"${score_cmd[@]}"
printf 'Output ZIP: %s\n' "$output"
