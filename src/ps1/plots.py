"""Matplotlib figures for actual (never fabricated) experiment outputs."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def make_plots(output_dir, fold_rows, actual, predicted, final_history):
    root = Path(output_dir) / 'plots'
    root.mkdir(exist_ok=True, parents=True)
    all_models = sorted(set(x['model'] for x in fold_rows))
    for metric in ('MAE', 'MdAE'):
        series = [[r[metric] for r in fold_rows if r['model'] == model] for model in all_models]
        fig, ax = plt.subplots(figsize=(10, 5))
        for j, vals in enumerate(series):
            ax.scatter([j + 1] * len(vals), vals, alpha=.65)
            ax.hlines(np.mean(vals), j + .75, j + 1.25, linewidth=3)
        ax.set_xticks(np.arange(1, len(all_models) + 1), labels=all_models, rotation=20, ha='right')
        ax.set_ylabel(metric + ' (lower is better)')
        ax.set_title(f'Five-fold validation {metric}: dots=individual folds, bars=means')
        fig.tight_layout()
        fig.savefig(root / f'cross_validation_{metric.lower()}.png', dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, predicted, alpha=.45)
    bound = max(float(np.max(actual)), float(np.max(predicted)), 1.)
    ax.plot([0, bound], [0, bound], linestyle='--')
    ax.set(xlabel='Actual story points', ylabel='Predicted story points', title='Held-out test predictions')
    fig.tight_layout()
    fig.savefig(root / 'test_actual_vs_predicted.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.abs(np.asarray(actual) - np.asarray(predicted)), bins=20)
    ax.set(xlabel='Absolute error in story points', ylabel='Count', title='Held-out test absolute error')
    fig.tight_layout()
    fig.savefig(root / 'test_absolute_errors.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([h['epoch'] for h in final_history], [h['train_smooth_l1'] for h in final_history])
    ax.set(xlabel='Training epoch', ylabel='SmoothL1 loss', title='Final CNN training loss (development set only)')
    fig.tight_layout()
    fig.savefig(root / 'training_loss.png', dpi=160)
    plt.close(fig)
