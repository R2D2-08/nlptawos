import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ps1.prepare import prepare_file
from ps1.splits import make_splits
from ps1.models import TokenCNN
from ps1.train_eval import metrics, fit_cnn, inference


class ProjectSmokeTests(unittest.TestCase):
    def test_split_has_no_leakage_and_each_dev_example_validated_once(self):
        s = make_splits(100, seed=42)
        self.assertEqual(len(s['folds']), 5)
        self.assertEqual(len(s['test']), 20)
        self.assertEqual(len(s['dev']), 80)
        test = set(s['test'])
        self.assertEqual(set(v for f in s['folds'] for v in f['val']), set(s['dev']))
        for f in s['folds']:
            self.assertFalse(set(f['train']) & set(f['val']))
            self.assertFalse(test & (set(f['train']) | set(f['val'])))

    def test_metrics(self):
        got = metrics([1, 2, 3], [2, 2, 1])
        self.assertAlmostEqual(got['MAE'], 1)
        self.assertAlmostEqual(got['MdAE'], 1)

    def test_preprocessing_with_duplicates_flags_and_project_filter(self):
        with tempfile.TemporaryDirectory() as temp:
            src, dst = Path(temp) / 'raw.csv', Path(temp) / 'clean.csv'
            data = []
            for i in range(50):
                data.append({'issue_key': f'A-{i}', 'project_key': 'A', 'title': f'Fix screen bug {i}',
                             'description_text': f'Test with case {i}', 'story_point': (i % 5) + 1,
                             'story_point_changed_after_estimation': 0})
            data.append(dict(data[0]))
            data.append({'issue_key':'B-1', 'project_key':'B', 'title':'OTHER PROJECT',
                         'description_text':'Should disappear', 'story_point':99})
            data.append({'issue_key':'A-EDIT', 'project_key':'A', 'title':'Edited after estimation',
                         'description_text':'Should disappear', 'story_point':10,
                         'story_point_changed_after_estimation':1})
            pd.DataFrame(data).to_csv(src, index=False)
            report = prepare_file(src, dst, 'A')
            self.assertEqual(report['final_unique_issues'], 50)
            self.assertTrue((pd.read_csv(dst)['project_key'] == 'A').all())

    def test_three_conv_forward_and_short_training(self):
        rng = np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as t:
            features = rng.standard_normal((12, 8, 16)).astype('float16')
            masks = np.ones((12, 8), dtype='uint8')
            y = np.arange(12, dtype='float32') % 5 + 1
            model = TokenCNN(16, (12,8,4), 3, .2)
            self.assertEqual(len(model.convs), 3)
            got = model(torch.tensor(features[:2], dtype=torch.float32), torch.tensor(masks[:2], dtype=torch.float32))
            self.assertEqual(got.shape, (2,))
            cfg = {'conv_channels': [12,8,4], 'kernel_size':3, 'learning_rate':.001,
                   'weight_decay':.01, 'train_batch_size':4, 'num_workers':0, 'epochs':2}
            model, history = fit_cnn(features, masks, y, list(range(8)), cfg, .2, 'cpu', seed=42)
            pred = inference(model, features, masks, y, [8,9,10,11], 2, 'cpu')
            self.assertEqual(len(pred), 4)
            self.assertEqual(len(history), 2)


if __name__ == '__main__':
    unittest.main()
