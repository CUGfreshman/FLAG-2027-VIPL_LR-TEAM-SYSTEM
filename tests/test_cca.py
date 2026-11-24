"""Numerical regression checks; synthetic inputs only, no challenge trials."""
from pathlib import Path
import unittest
import tempfile
import csv
import numpy as np
from flag2027.cca import load_model, project_pair, score_pair, unit, apply_inlp
from flag2027.cca.numerics import inference_primitives as original
from flag2027.cca.fitting import fit_projection
from flag2027.cca.numerics import weighted_identity_centers, fit_centers
from flag2027.cca.data import identity_prototypes, read_rows

ROOT = Path(__file__).resolve().parents[1]


class CCARegression(unittest.TestCase):
    def test_all_released_heads_use_original_projection(self):
        rng = np.random.default_rng(2)
        for path in sorted((ROOT / 'checkpoints/cca').glob('*.npz')):
            with self.subTest(model=path.name):
                model = load_model(path)
                face = unit(rng.normal(size=(7, model['fit_face_projection'].shape[0])))
                voice = unit(rng.normal(size=(7, model['fit_voice_projection'].shape[0])))
                fitting = {m + '_projection': model['fit_' + m + '_projection'] for m in ('face', 'voice')}
                centers = {m: model['deployment_' + m + '_mean'] for m in ('face', 'voice')}
                reference = original.embedded({'face': face, 'voice': voice}, fitting, centers)
                actual = project_pair(model, face, voice, normalize_inputs=False)
                for modality in actual:
                    np.testing.assert_array_equal(actual[modality], reference[modality])
                np.testing.assert_array_equal(score_pair(model, face, voice, normalize_inputs=False), original.dist(reference))
                gender = {m: apply_inlp(values, model['INLP3_directions'], 3) for m, values in reference.items()}
                np.testing.assert_array_equal(score_pair(model, face, voice, True, False), original.dist(gender))

    def test_original_weighted_fit_path_is_unchanged(self):
        rng = np.random.default_rng(2)
        latent = rng.normal(size=(240, 6))
        face = latent @ rng.normal(size=(6, 16)) + rng.normal(size=(240, 16)) * .2
        voice = latent @ rng.normal(size=(6, 12)) + rng.normal(size=(240, 12)) * .2
        weights = np.r_[np.full(40, 10.), np.ones(200)]
        reference = fit_centers.fit(weighted_identity_centers.statistics(face, voice, weights), 10., 8)
        actual = fit_projection(face, voice, weights, rank=8, ridge=10.)
        for key in reference:
            np.testing.assert_array_equal(actual[key], reference[key])

    def test_apply_keeps_original_normalization_order(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(11, 192))
        q, _ = np.linalg.qr(rng.normal(size=(192, 3)))
        directions = q.T
        reference = unit(x)
        for direction in directions:
            reference = unit(reference - np.outer(reference @ direction, direction))
        np.testing.assert_array_equal(apply_inlp(x, directions, 3), reference)

    def test_identity_means_deduplicate_media_and_keep_mean_norm(self):
        # Three original rows contain only two unique media; the duplicate must
        # not make the first file count twice. audio_path is the manifest alias.
        face = np.asarray([[1., 0.], [1., 0.], [0., 1.]])
        voice = np.asarray([[0., 1.], [0., 1.], [1., 0.]])
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            features = folder / 'features.npz'
            np.savez(features, face_facenet=face, voice_wespeaker=voice)
            rows = folder / 'rows.csv'
            with rows.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['identity', 'gender', 'language', 'face_path', 'audio_path', 'split'])
                writer.writeheader()
                for face_key, audio_key in [('f1', 'a1'), ('f1', 'a1'), ('f2', 'a2')]:
                    writer.writerow(dict(identity='person', gender='m', language='English',
                        face_path=face_key, audio_path=audio_key, split='train'))
            result = identity_prototypes(features, rows, 'facenet_wespeaker', domain='competition')
            np.testing.assert_array_equal(result['face_prototypes'], [[.5, .5]])
            np.testing.assert_array_equal(result['voice_prototypes'], [[.5, .5]])
            np.testing.assert_array_equal(result['identity_weights'], [10.])
            self.assertEqual(result['identities'].tolist(), ['target:person'])
            self.assertEqual(read_rows(rows)[0]['voice_path'], 'a1')


if __name__ == '__main__':
    unittest.main()
