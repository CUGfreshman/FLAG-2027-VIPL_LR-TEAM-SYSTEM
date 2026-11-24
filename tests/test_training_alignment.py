import csv
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from flag2027.cca.data import english_deployment, identity_prototypes
from flag2027.training_alignment import validate_training_alignment


def write_rows(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class TrainingAlignmentTests(unittest.TestCase):
    def test_checks_media_and_available_trial_keys(self):
        rows = [{'face_path': 'a.jpg', 'voice_path': 'a.wav', 'trial_id': '8', 'subset': 'training'}]
        archive = {'face_path': ['a.jpg'], 'audio_path': ['a.wav'], 'trial_id': [8], 'subset': ['training']}
        validate_training_alignment(archive, rows)
        for key in ('audio_path', 'trial_id', 'subset'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key + '.*row 1'):
                validate_training_alignment({**archive, key: ['wrong']}, rows)
        validate_training_alignment({'face_path': ['a.jpg'], 'trial_id': [8]},
                                    [{'face_path': 'a.jpg', 'voice_path': 'a.wav'}])

    def test_cca_checks_mismatched_rows_and_accepts_aligned_or_legacy_features(self):
        rows = [dict(identity=f'v4:{name}', language='English', gender='m', split='train',
                     face_path=f'{name}.jpg', voice_path=f'{name}.wav') for name in ('a', 'b')]
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            manifest = path / 'rows.csv'
            write_rows(manifest, rows)
            def write_features(name, **media):
                feature = path / (name + '.npz')
                np.savez(feature, face_facenet=np.eye(2), voice_wespeaker=np.eye(2), **media)
                return feature

            aligned = write_features('aligned', face_path=['a.jpg', 'b.jpg'], audio_path=['a.wav', 'b.wav'])
            swapped = write_features('swapped', face_path=['b.jpg', 'a.jpg'], audio_path=['b.wav', 'a.wav'])
            legacy = write_features('legacy')
            def deployment(features):
                return english_deployment(features, manifest, 'facenet_wespeaker', expected_identities=2)[0]['face']

            def prototypes(features):
                return identity_prototypes(features, manifest, 'facenet_wespeaker',
                                           domain='competition')['face_prototypes']

            for reader in (deployment, prototypes):
                np.testing.assert_array_equal(reader(aligned), reader(legacy))
                with self.assertRaisesRegex(ValueError, 'face_path.*row 1'):
                    reader(swapped)

    def test_triplet_preparation_rejects_misorder_before_creating_output(self):
        script = Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_triplet_data.py'
        n = 22976
        rows = [dict(identity=f'v4:{i % 198}', source='v4', language='English', gender='m',
                     face_path='a.jpg', voice_path='a.wav') for i in range(n)]
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            manifest, features, output = path / 'rows.csv', path / 'features.npz', path / 'prepared'
            write_rows(manifest, rows)
            np.savez(features, face_path=['wrong.jpg'] + ['a.jpg'] * (n - 1))
            with patch.object(sys, 'argv', [str(script), '--rows', str(manifest), '--features', str(features),
                                             '--output', str(output)]):
                with self.assertRaisesRegex(ValueError, 'face_path.*row 1'):
                    runpy.run_path(str(script), run_name='__main__')
            self.assertFalse(output.exists())
