from pathlib import Path
import json
import random
import hashlib
import numpy as np
import torch
import yaml


def load_config(path):
    with open(path, encoding="utf-8") as f:
        c = yaml.safe_load(f)
    if int(c["cv_folds"]) != 5:
        raise ValueError("Guidelines require exactly five validation folds")
    if float(c["test_size"]) != 0.20:
        raise ValueError("Guidelines require 20% held-out test split")
    if len(c["conv_channels"]) != 3:
        raise ValueError("Use exactly three CNN convolution layers")
    if int(c["kernel_size"]) % 2 != 1:
        raise ValueError("kernel_size must be odd to preserve sequence length")
    return c


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Stronger reproducibility, at a possible performance cost.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
