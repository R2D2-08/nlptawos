"""Full OFFLINE infrastructure smoke test, NOT a performance study of actual TAWOS or BERT."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from ps1.common import save_json
from ps1.prepare import prepare_file
from ps1.splits import make_splits
from ps1.train_eval import run_experiment
from ps1.make_report import build


class SyntheticPipelineTest(unittest.TestCase):
    def test_pipeline_and_pdf_with_mock_token_features(self):
        with tempfile.TemporaryDirectory() as location:
            root = Path(location)
            n = 40
            df = pd.DataFrame([
                {'issue_key':f'TEST-{i}', 'project_key':'SYNTHETIC',
                 'title':f'Fake sample title {i}', 'description_text':f'Artificial wording {i}',
                 'story_point':float(i % 5 + 1)} for i in range(n)
            ])
            df.to_csv(root/'raw.csv',index=False)
            (root/'data').mkdir()
            (root/'results').mkdir()
            audit = prepare_file(root/'raw.csv',root/'data/issues.csv','SYNTHETIC',root/'data/audit.json')
            self.assertEqual(audit['final_unique_issues'],n)
            rng = np.random.default_rng(12)
            features = rng.standard_normal((n,8,16)).astype('float16')
            masks = np.ones((n,8),dtype='uint8')
            np.save(root/'features.npy',features)
            np.save(root/'masks.npy',masks)
            conf = {
                'seed':42,'model_name':'google-bert/bert-base-uncased','max_length':8,
                'conv_channels':[12,8,4],'kernel_size':3,'cnn_dropouts':[.2,.4],
                'epochs':1,'train_batch_size':8,'num_workers':0,'learning_rate':.001,
                'weight_decay':.01,'ridge_alpha':10.0,
            }
            save_json(root/'results/reproduction.json', {'config':conf, 'audit':audit})
            summary = run_experiment(root/'data/issues.csv',make_splits(n),root/'features.npy',
                                     root/'masks.npy',conf,root/'results','cpu')
            self.assertEqual(len(summary['test_metrics']),3)
            self.assertEqual(len(pd.read_csv(root/'results/five_fold_results.csv')),20)
            build(root,root/'test_report.pdf','Fake Group','Synthetic Only','https://example.invalid')
            self.assertGreater((root/'test_report.pdf').stat().st_size,1000)


if __name__ == '__main__':
    unittest.main()
