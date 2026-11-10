"""5-fold model selection on dev ONLY; exactly one untouched holdout evaluation."""
import csv
import json
from pathlib import Path
from statistics import mean, stdev
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, median_absolute_error
from .common import save_json, set_seed
from .models import TokenCNN


class EmbeddingDataset(Dataset):
    def __init__(self, features, masks, targets, indices):
        self.features, self.masks, self.targets = features, masks, targets
        self.indices = np.asarray(indices, dtype=int)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        # Explicit copy makes read-only memory maps safe in PyTorch.
        return (torch.from_numpy(np.array(self.features[idx], dtype=np.float32, copy=True)),
                torch.from_numpy(np.array(self.masks[idx], dtype=np.float32, copy=True)),
                torch.tensor(self.targets[idx], dtype=torch.float32))


def metrics(y_true, y_pred):
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return {'MAE': float(mean_absolute_error(y_true, y_pred)),
            'MdAE': float(median_absolute_error(y_true, y_pred))}


def inference(model, features, masks, targets, indices, batch_size, device):
    loader = DataLoader(EmbeddingDataset(features, masks, targets, indices), batch_size=batch_size)
    model.eval()
    preds = []
    with torch.inference_mode():
        for x, mask, _ in loader:
            predictions = model(x.to(device), mask.to(device))
            preds.extend(np.maximum(predictions.cpu().numpy(), 0).tolist())
    return np.array(preds, dtype=float)


def fit_cnn(features, masks, targets, train_indices, c, dropout, device, seed):
    set_seed(seed)
    model = TokenCNN(embedding_dim=features.shape[2], channels=tuple(c['conv_channels']),
                     kernel_size=int(c['kernel_size']), dropout=float(dropout)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(c['learning_rate']),
                                  weight_decay=float(c['weight_decay']))
    loss_fn = torch.nn.SmoothL1Loss(beta=1.0)
    loader = DataLoader(EmbeddingDataset(features, masks, targets, train_indices),
                        batch_size=int(c['train_batch_size']), shuffle=True,
                        num_workers=int(c['num_workers']))
    history = []
    for epoch in range(int(c['epochs'])):
        model.train()
        total, total_examples = 0., 0
        for x, mask, y in loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, mask)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * len(y)
            total_examples += len(y)
        history.append({'epoch': epoch + 1, 'train_smooth_l1': total / total_examples})
    return model, history


def mean_pooled(features, masks, chunk_size=256):
    """BERT average token vectors for cheap Ridge baseline (also word-based)."""
    out = np.zeros((len(features), features.shape[2]), dtype=np.float32)
    for start in range(0, len(features), chunk_size):
        stop = min(start + chunk_size, len(features))
        x = np.asarray(features[start:stop], dtype=np.float32)
        mask = np.asarray(masks[start:stop], dtype=np.float32)
        out[start:stop] = (x * mask[:, :, None]).sum(axis=1) / np.maximum(mask.sum(axis=1, keepdims=True), 1)
    return out


def save_rows_csv(path, rows, fields):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run_experiment(clean_csv, splits, features_path, masks_path, config, results_dir, device='auto'):
    from .plots import make_plots
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(clean_csv)
    y = df['story_point'].to_numpy(dtype=np.float32)
    features = np.load(features_path, mmap_mode='r')
    masks = np.load(masks_path, mmap_mode='r')
    assert len(y) == len(features) == len(masks) == splits['n']
    assert len(splits['folds']) == 5
    target_device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    print(f'Training on {target_device}; {len(y)} labeled issues, {len(splits["dev"])} development, {len(splits["test"])} held-out test')
    pooled = mean_pooled(features, masks)
    dev, test = np.asarray(splits['dev']), np.asarray(splits['test'])
    cv_rows = []
    train_histories = {}
    candidate_models = [('cnn_dropout_' + str(drop), float(drop)) for drop in config['cnn_dropouts']]

    for model_name, dropout in candidate_models:
        for fold_idx, fold in enumerate(splits['folds'], start=1):
            tr, va = np.asarray(fold['train']), np.asarray(fold['val'])
            assert not (set(tr) & set(va) or set(va) & set(test) or set(tr) & set(test))
            model, hist = fit_cnn(features, masks, y, tr, config, dropout, target_device,
                                  seed=int(config['seed']) + fold_idx)
            val_pred = inference(model, features, masks, y, va, config['train_batch_size'], target_device)
            result = metrics(y[va], val_pred)
            cv_rows.append({'model': model_name, 'fold': fold_idx, 'train_n': len(tr),
                            'val_n': len(va), **result})
            train_histories[f'{model_name}_fold_{fold_idx}'] = hist
            print(f'{model_name} | fold {fold_idx} | MAE={result["MAE"]:.3f} MdAE={result["MdAE"]:.3f}', flush=True)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for fold_idx, fold in enumerate(splits['folds'], start=1):
        tr, va = np.asarray(fold['train']), np.asarray(fold['val'])
        for name, estimator, use_features in [
            ('ridge_bert_mean', Ridge(alpha=float(config['ridge_alpha'])), True),
            ('dummy_median', DummyRegressor(strategy='median'), False),
        ]:
            train_x = pooled[tr] if use_features else np.zeros((len(tr), 1))
            val_x = pooled[va] if use_features else np.zeros((len(va), 1))
            estimator.fit(train_x, y[tr])
            result = metrics(y[va], estimator.predict(val_x))
            cv_rows.append({'model': name, 'fold': fold_idx, 'train_n': len(tr),
                            'val_n': len(va), **result})

    save_rows_csv(out / 'five_fold_results.csv', cv_rows, ['model', 'fold', 'train_n', 'val_n', 'MAE', 'MdAE'])
    aggregates = []
    for name in sorted(set(r['model'] for r in cv_rows)):
        x = [r for r in cv_rows if r['model'] == name]
        aggregates.append({'model': name,
                           'MAE_mean': mean(r['MAE'] for r in x),
                           'MAE_std': stdev(r['MAE'] for r in x),
                           'MdAE_mean': mean(r['MdAE'] for r in x),
                           'MdAE_std': stdev(r['MdAE'] for r in x)})
    save_rows_csv(out / 'cv_summary.csv', aggregates,
                  ['model', 'MAE_mean', 'MAE_std', 'MdAE_mean', 'MdAE_std'])
    # The CNN hyperparameter is selected using CV on the development portion only.
    # The median/ridge baselines are reported but do not determine the CNN config.
    cnn_summary = [row for row in aggregates if row['model'].startswith('cnn_dropout_')]
    winner = min(cnn_summary, key=lambda r: (r['MAE_mean'], r['MdAE_mean']))
    selected_dropout = float(winner['model'].removeprefix('cnn_dropout_'))
    print(f'Chosen CNN by CV MAE: {winner["model"]} (test set NEVER used for selection)')

    final_model, final_history = fit_cnn(features, masks, y, dev, config, selected_dropout,
                                         target_device, seed=int(config['seed']))
    checkpoint = {
        'model_state_dict': {key: tensor.detach().cpu() for key, tensor in final_model.state_dict().items()},
        'model_name': config['model_name'], 'max_length': int(config['max_length']),
        'embedding_dim': int(features.shape[2]),
        'conv_channels': list(config['conv_channels']), 'kernel_size': int(config['kernel_size']),
        'dropout': selected_dropout, 'project': df['project_key'].iloc[0],
    }
    torch.save(checkpoint, out / 'final_cnn.pt')
    pred_cnn = inference(final_model, features, masks, y, test, config['train_batch_size'], target_device)
    test_metrics = [{'model': winner['model'], **metrics(y[test], pred_cnn)}]
    prediction_table = pd.DataFrame({
        'issue_key': df.iloc[test]['issue_key'].astype(str).to_list(),
        'actual': y[test], 'predicted_cnn': pred_cnn,
        'absolute_error': np.abs(y[test] - pred_cnn)})
    for name, estimator, use_features in [
        ('ridge_bert_mean', Ridge(alpha=float(config['ridge_alpha'])), True),
        ('dummy_median', DummyRegressor(strategy='median'), False),
    ]:
        tr_x = pooled[dev] if use_features else np.zeros((len(dev), 1))
        te_x = pooled[test] if use_features else np.zeros((len(test), 1))
        estimator.fit(tr_x, y[dev])
        pred = np.maximum(estimator.predict(te_x), 0)
        prediction_table['predicted_' + name] = pred
        test_metrics.append({'model': name, **metrics(y[test], pred)})
    prediction_table.to_csv(out / 'heldout_predictions.csv', index=False)
    save_rows_csv(out / 'heldout_test_results.csv', test_metrics, ['model','MAE','MdAE'])
    save_json(out / 'training_histories.json', {**train_histories, 'final': final_history})
    summary = {
        'project': df['project_key'].iloc[0],
        'n': len(y), 'development_n': len(dev), 'heldout_test_n': len(test),
        'seed': config['seed'], 'folds': len(splits['folds']),
        'chosen_cnn': winner['model'], 'chosen_using': 'development five-fold mean MAE; MdAE tie-breaker',
        'five_fold_cnn': winner, 'test_metrics': test_metrics,
        'note': 'Baselines are separate comparisons. The test set was not used in model selection.',
    }
    save_json(out / 'summary.json', summary)
    make_plots(out, cv_rows, y[test], pred_cnn, final_history)
    print('\nFinished. Evaluation outputs for supplied dataset (confirm TAWOS provenance before citing):')
    for t in test_metrics:
        print(f'  {t["model"]}: test MAE={t["MAE"]:.4f}, test MdAE={t["MdAE"]:.4f}')
    return summary
