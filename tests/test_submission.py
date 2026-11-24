from pathlib import Path
import csv
import tempfile
import unittest
import zipfile
import numpy as np
from flag2027.submission import OUTPUTS, linear_fusion, read_trials, write_submission


class SubmissionTests(unittest.TestCase):
    def test_mean_and_accepted_layout(self):
        components = np.arange(24, dtype=np.float64).reshape(4, 6) / 12
        expected = np.array([sum(row) / 6 for row in components])
        np.testing.assert_array_equal(linear_fusion(components), expected)
        rows = [{'subset': subset, 'trial_id': 'case_' + str(i), 'face_path': 'f', 'audio_path': 'a'}
                for i, subset in enumerate(OUTPUTS)]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'try_synthetic.zip'
            receipt = write_submission(rows, expected, path)
            self.assertEqual(receipt['rows'], 4)
            with zipfile.ZipFile(path) as archive:
                self.assertEqual(archive.namelist(), list(OUTPUTS.values()))
                for i, member in enumerate(OUTPUTS.values()):
                    self.assertEqual(archive.read(member).decode(),
                        'case_' + str(i) + ' ' + format(float(expected[i]), '.17g') + '\n')
            with self.assertRaises(FileExistsError):
                write_submission(rows, expected, path)

    def test_invalid_distances_are_rejected(self):
        for value in (-0.01, 2.01, np.nan, np.inf):
            part = np.ones((2, 6)); part[0, 0] = value
            with self.assertRaises(ValueError):
                linear_fusion(part)

    def test_duplicate_trial_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'trials.csv'
            with path.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream)
                writer.writerow(['trial_id', 'subset', 'face_path', 'audio_path'])
                writer.writerow(['a', 'english_gender', 'f', 'a'])
                writer.writerow(['a', 'english_gender', 'f2', 'a2'])
            with self.assertRaises(ValueError):
                read_trials(path)


if __name__ == '__main__':
    unittest.main()
