from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from flag2027.triplet.model import make_model
from flag2027.triplet.original.original_objective import UniquePairSampler
from flag2027.triplet import training
from flag2027.triplet.training import same_recipe, train_one_epoch


class TripletTrainingTests(unittest.TestCase):
    def test_fixed_seed_epoch_uses_both_towers_and_replays_sampling(self):
        device = torch.device('cpu')
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        sources = torch.tensor([0, 0, 0, 0, 2, 2, 2, 2])
        index = np.arange(len(labels))
        sampler = UniquePairSampler(labels.numpy(), index, index, p=4, k=2)
        features = torch.Generator().manual_seed(32)
        tensors = {'face': torch.randn(len(labels), 1536, generator=features),
                   'voice': torch.randn(len(labels), 1280, generator=features)}

        def run():
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(2)
                model = make_model()
                before = {k: v.clone() for k, v in model.state_dict().items()}
                opt = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=.01,
                                       betas=(.9, .999), eps=1e-8)
                negative_rng = torch.Generator(device='cpu').manual_seed(2)
                trace = []
                exposure = np.zeros(len(labels), np.int64)
                stats = train_one_epoch(model, opt, sampler, 1, 2, labels, sources,
                                        labels, sources, negative_rng, tensors, device,
                                        index, exposure, trace)
                for modality in ('face', 'voice'):
                    state = model.state_dict()
                    self.assertEqual(int(state[f'{modality}.block.1.num_batches_tracked']), 2)
                    linear_key = f'{modality}.block.0.weight'
                    self.assertFalse(torch.equal(state[linear_key], before[linear_key]))
                return model.state_dict(), trace, exposure, stats

        a, a_trace, a_exposure, a_stats = run()
        b, b_trace, b_exposure, b_stats = run()
        self.assertEqual(a_trace, b_trace)
        self.assertEqual(len(a_trace), 2)
        np.testing.assert_array_equal(a_exposure, np.full(len(labels), 2))
        np.testing.assert_array_equal(a_exposure, b_exposure)
        self.assertEqual(a_stats, b_stats)
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))

    def test_recipe_ignores_old_source_paths_but_not_training_settings(self):
        old = {'model_source': '/previous/tree/model.py', 'arm': 'historical', 'fold': 0,
               'task': 'historical', 'data': {'fold': 0, 'steps_per_epoch': 179}, 'lr': 1e-5, 'seed': 2}
        new = {'model_source': 'triplet/model.py', 'data': {'steps_per_epoch': 179}, 'lr': 1e-5, 'seed': 2}
        self.assertTrue(same_recipe(old, new))
        self.assertFalse(same_recipe(old, {**new, 'lr': 1e-3}))
        self.assertFalse(same_recipe(old, {**new, 'data': {'steps_per_epoch': 180}}))

    def test_export_uses_output_directory_directly(self):
        def finish(data, dest):
            dest.mkdir(parents=True)
            torch.save({'epoch': 300, 'global_step': 53700, 'model_state': {'weight': torch.ones(1)}},
                       dest / 'epoch0300.pth.tar')

        with TemporaryDirectory() as tmp:
            data = Path(tmp) / 'features'
            output = Path(tmp) / 'outputs' / 'triplet'
            with patch.object(training, 'run_training', side_effect=finish) as run:
                training.train(data, output)
                run.assert_called_once_with(data.resolve(), output.resolve())
            self.assertTrue((output / 'epoch0300.pth.tar').is_file())
            released = torch.load(output / 'epoch0300.pth', weights_only=True)
            self.assertEqual(released['global_step'], 53700)
            self.assertFalse((output / 'outputs').exists())

    def test_nested_run_needs_explicit_output(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            leaf = root / 'outputs' / 'historical' / 'fold0'
            leaf.mkdir(parents=True)
            (leaf / 'config.json').touch()
            torch.save({'epoch': 300, 'global_step': 53700, 'model_state': {}}, leaf / 'epoch0300.pth.tar')
            with patch.object(training, 'run_training') as run:
                with self.assertRaisesRegex(FileExistsError, 'exact run directory'):
                    training.train(root / 'features', root)
                run.assert_not_called()
                training.train(root / 'features', leaf)
                run.assert_called_once_with((root / 'features').resolve(), leaf.resolve())
            self.assertTrue((leaf / 'epoch0300.pth').is_file())
            self.assertFalse((root / 'epoch0300.pth').exists())
