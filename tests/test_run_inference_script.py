import csv
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np

from flag2027.inference import FEATURE_DIMS
from flag2027.submission import OUTPUTS


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / 'scripts/run_inference.sh'


class RunInferenceScriptTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='flag-inference-test-')
        self.addCleanup(temp.cleanup)
        self.work = Path(temp.name).resolve() / 'caller directory'
        self.work.mkdir()
        self.manifest = self.work / 'trial list.csv'
        self.output = self.work / 'out folder' / 'trial scores.zip'
        self.rows = [
            {'trial_id': f'{i}-{j}', 'subset': subset,
             'face_path': str(self.work / 'unused.jpg'),
             'audio_path': str(self.work / 'unused.wav')}
            for i, subset in enumerate(OUTPUTS)
            for j in range(i + 1)
        ]
        with self.manifest.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def run_script(self, *args, script=SCRIPT, python=sys.executable):
        return subprocess.run(
            ['bash', str(script), *(str(arg) for arg in args)],
            cwd=self.work,
            env={**os.environ, 'PYTHON': str(python), 'PYTHONDONTWRITEBYTECODE': '1',
                 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'},
            capture_output=True, text=True, timeout=30,
        )

    def inputs(self):
        return ('--manifest', self.manifest.name,
                '--output', self.output.relative_to(self.work))

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def assert_failure(self, result, text):
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(text, result.stderr)

    def repo_fixture(self):
        repo = self.work.parent / 'repo fixture'
        (repo / 'scripts').mkdir(parents=True)
        shutil.copy2(SCRIPT, repo / 'scripts' / SCRIPT.name)
        for name in ('configs', 'checkpoints', 'src'):
            (repo / name).symlink_to(REPO / name, target_is_directory=True)
        for name in ('extract_features.py', 'score_trials.py'):
            (repo / 'scripts' / name).symlink_to(REPO / 'scripts' / name)
        return repo

    def score_config(self, *, missing_checkpoint=False):
        config = json.loads((REPO / 'configs/system.json').read_text(encoding='utf-8'))
        for key in ('triplet_checkpoint', 'triplet_inlp'):
            config[key] = str(REPO / config[key])
        config['cca_models'] = {key: str(REPO / value)
                                for key, value in config['cca_models'].items()}
        if missing_checkpoint:
            config['triplet_checkpoint'] = str(self.work / 'missing checkpoint.pth')
        path = self.work / 'custom configs' / 'scoring config.json'
        path.parent.mkdir()
        path.write_text(json.dumps(config), encoding='utf-8')
        return path

    def assert_plan(self, result, *, features, encoders_config, system_config, device):
        self.assert_success(result)
        commands = [shlex.split(line) for line in result.stdout.splitlines() if line.strip()]
        self.assertEqual(len(commands), 2, result.stdout)
        for command, script, options in (
            (commands[0], 'extract_features.py', {
                '--manifest': self.manifest, '--config': encoders_config,
                '--output': features, '--device': device,
            }),
            (commands[1], 'score_trials.py', {
                '--manifest': self.manifest, '--features': features, '--config': system_config,
                '--output': self.output, '--device': 'cpu',
            }),
        ):
            script_positions = [i for i, arg in enumerate(command) if Path(arg).name == script]
            self.assertEqual(len(script_positions), 1, command)
            args = command[script_positions[0] + 1:]
            self.assertEqual(len(args), 2 * len(options), command)
            self.assertEqual(dict(zip(args[::2], args[1::2])),
                             {name: str(value) for name, value in options.items()})

    def test_help_lists_options(self):
        result = self.run_script('--help')
        self.assert_success(result)
        for flag in ('-h', '--help', '--manifest', '--output', '--device', '--encoders-config',
                     '--system-config', '--features', '--no-verify', '--dry-run'):
            self.assertIn(flag, result.stdout)

    def test_required_arguments(self):
        for args, missing in (
            (('--output', self.output), '--manifest'),
            (('--manifest', self.manifest), '--output'),
        ):
            with self.subTest(missing=missing):
                self.assert_failure(self.run_script(*args), missing)

    def test_dry_run_prints_both_commands_without_writing(self):
        repo = self.repo_fixture()
        features = repo / 'features' / 'trial scores.npz'
        result = self.run_script(*self.inputs(), '--dry-run',
                                 script=repo / 'scripts' / SCRIPT.name)
        self.assert_plan(result, features=features, encoders_config=repo / 'configs/encoders.example.json',
                         system_config=repo / 'configs/system.json', device='cuda:0')
        self.assertFalse(self.output.parent.exists())
        self.assertFalse(features.parent.exists())

    def test_dry_run_resolves_explicit_paths_and_device(self):
        features = self.work / 'feature folder' / 'cached rows.npz'
        encoders = self.work / 'encoder setup.json'
        shutil.copy2(REPO / 'configs/encoders.example.json', encoders)
        config = self.score_config()
        result = self.run_script(
            *self.inputs(), '--features', features.relative_to(self.work),
            '--encoders-config', encoders.name, '--system-config', config.relative_to(self.work),
            '--device', 'cuda:3', '--dry-run', '--no-verify',
            python=os.path.relpath(sys.executable, self.work),
        )
        self.assert_plan(result, features=features, encoders_config=encoders,
                         system_config=config, device='cuda:3')
        self.assertFalse(self.output.parent.exists())
        self.assertFalse(features.parent.exists())

    def test_output_is_not_overwritten(self):
        self.output.parent.mkdir()
        original = b'existing output'
        self.output.write_bytes(original)
        self.assert_failure(self.run_script(*self.inputs(), '--no-verify'), str(self.output))
        self.assertEqual(self.output.read_bytes(), original)

    def test_existing_default_cache_is_not_overwritten(self):
        repo = self.repo_fixture()
        features = repo / 'features' / 'trial scores.npz'
        features.parent.mkdir()
        original = b'existing feature file'
        features.write_bytes(original)
        result = self.run_script(*self.inputs(), '--no-verify', script=repo / 'scripts' / SCRIPT.name)
        self.assert_failure(result, '--features')
        self.assertEqual(features.read_bytes(), original)
        self.assertFalse(self.output.exists())

    def test_extraction_sidecar_is_not_overwritten(self):
        features = self.work / 'custom features.npz'
        sidecar = features.with_suffix('.contract.json')
        original = b'existing feature contract'
        sidecar.write_bytes(original)
        self.assert_failure(
            self.run_script(*self.inputs(), '--features', features.name, '--no-verify'), str(sidecar),
        )
        self.assertEqual(sidecar.read_bytes(), original)
        self.assertFalse(features.exists())
        self.assertFalse(self.output.exists())

    def test_missing_custom_parameter_is_rejected_even_without_checksum(self):
        config = self.score_config(missing_checkpoint=True)
        features = self.work / 'new cache.npz'
        result = self.run_script(
            *self.inputs(), '--features', features.name, '--system-config',
            config.relative_to(self.work), '--no-verify', '--dry-run',
        )
        self.assert_failure(result, 'missing checkpoint.pth')
        self.assertFalse(features.exists())
        self.assertFalse(self.output.exists())

    def test_reuse_synthetic_features_with_released_checkpoints(self):
        features = self.work / 'synthetic cache.npz'
        rng = np.random.default_rng(2027)
        arrays = {key: rng.standard_normal((len(self.rows), width), dtype=np.float32)
                  for key, width in FEATURE_DIMS.items()}
        arrays.update({key: np.asarray([row[key] for row in self.rows]) for key in self.rows[0]})
        np.savez(features, **arrays)

        config = self.score_config()
        result = self.run_script(
            *self.inputs(), '--features', features.name, '--system-config',
            config.relative_to(self.work), '--no-verify',
        )
        self.assert_success(result)
        self.assertIn(str(self.output), result.stdout)
        self.assertFalse(features.with_suffix('.contract.json').exists())
        with ZipFile(self.output) as archive:
            self.assertEqual(archive.namelist(), list(OUTPUTS.values()))
            for subset, member in OUTPUTS.items():
                rows = archive.read(member).decode('utf-8').splitlines()
                parts = [row.split() for row in rows]
                self.assertTrue(all(len(part) == 2 for part in parts))
                self.assertEqual([part[0] for part in parts],
                                 [row['trial_id'] for row in self.rows if row['subset'] == subset])
                self.assertTrue(all(0 <= float(part[1]) <= 2 for part in parts))
