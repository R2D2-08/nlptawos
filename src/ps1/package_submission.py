"""Copy SMALL, measured, non-text outputs for GitHub and project submission."""
import argparse
import shutil
import json
from pathlib import Path

RESULTS = [
    'cv_summary.csv', 'five_fold_results.csv', 'heldout_test_results.csv',
    'summary.json', 'reproduction.json',
    'plots/cross_validation_mae.png',
    'plots/cross_validation_mdae.png',
    'plots/test_actual_vs_predicted.png',
    'plots/test_absolute_errors.png',
    'plots/training_loss.png',
]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True,help='runs/PROJECT path')
    p.add_argument('--out',default='submission_results')
    a=p.parse_args()
    root=Path(a.run)/'results'
    target=Path(a.out)/Path(a.run).name
    for rel in RESULTS:
        src=root/rel
        if not src.exists():
            raise FileNotFoundError('Missing measured artifact: '+str(src))
        dest=target/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        if rel == "reproduction.json":
            content = json.loads(src.read_text(encoding="utf-8"))
            if "source_csv_path" in content:
                content["source_csv_path"] = "[local path omitted from public submission]"
            dest.write_text(json.dumps(content, indent=2), encoding="utf-8")
        else:
            shutil.copyfile(src,dest)
    print(f'Copied measured tables/plots to {target}. Review files before publishing.')
    print('Held-out individual issue predictions, raw data, and large model weights intentionally excluded.')

if __name__ == '__main__':
    main()
