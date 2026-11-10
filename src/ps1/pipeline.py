"""One entry point: prepare -> split -> frozen BERT -> 5 folds -> train -> final test."""
import argparse
from pathlib import Path
import pandas as pd
import torch
from .common import load_config, save_json, file_sha256
from .prepare import prepare_file
from .splits import save_splits
from .embeddings import extract_bert
from .train_eval import run_experiment


def main():
    ap = argparse.ArgumentParser(description='Complete PS-1 TAWOS within-project BERT pipeline')
    ap.add_argument('--csv', required=True, help='TAWOS exported labeled issue CSV')
    ap.add_argument('--project', required=True, help='ONE project key, e.g. MESOS')
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--outdir', default='runs', help='Output root, separated per project')
    ap.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    args = ap.parse_args()
    conf = load_config(args.config)
    root = Path(args.outdir) / args.project.upper()
    prepared = root / 'data' / 'issues.csv'
    split_path = root / 'data' / 'splits.json'
    cache = root / 'data' / 'bert_cache'
    result_dir = root / 'results'
    stats = prepare_file(args.csv, prepared, project=args.project, stats_json=root / 'data' / 'audit.json')
    split = save_splits(stats['final_unique_issues'], int(conf['seed']), split_path,
                        int(conf['cv_folds']), float(conf['test_size']))
    vectors, masks = extract_bert(prepared, cache, conf['model_name'],
                                  int(conf['max_length']), int(conf['embedding_batch_size']), args.device)
    result_dir.mkdir(parents=True, exist_ok=True)
    save_json(result_dir / 'reproduction.json', {
        'config': conf, 'source_csv_path': str(Path(args.csv).resolve()),
        'source_csv_sha256': file_sha256(args.csv), 'clean_csv_sha256': file_sha256(prepared),
        'torch_version': torch.__version__, 'pandas_version': pd.__version__,
        'audit': stats,
    })
    summary = run_experiment(prepared, split, vectors, masks, conf, result_dir, args.device)
    print(f'\nALL OUTPUTS: {result_dir.resolve()}')


if __name__ == '__main__':
    main()
