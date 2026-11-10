"""Frozen BERT contextual *token* embeddings, cached on disk as float16."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .common import file_sha256, save_json


def extract_bert(csv_path, output_dir, model_name, max_length, batch_size=16, device='auto'):
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError as e:
        raise RuntimeError('Please install transformers: pip install -e .') from e
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    embeddings_path = out / 'token_embeddings.npy'
    masks_path = out / 'attention_masks.npy'
    metadata_path = out / 'embeddings_manifest.json'
    source_hash = file_sha256(csv_path)
    if all(p.exists() for p in (embeddings_path, masks_path, metadata_path)):
        meta = json.loads(metadata_path.read_text(encoding='utf-8'))
        if meta['csv_sha256'] == source_hash and meta['model_name'] == model_name and meta['max_length'] == max_length:
            print('Found matching, cached BERT token embeddings.')
            return embeddings_path, masks_path
        raise RuntimeError('Stale embeddings found. Use a different --outdir or delete data/processed/cache first.')
    texts = pd.read_csv(csv_path)['text'].fillna('').astype(str).tolist()
    target_device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert = AutoModel.from_pretrained(model_name).to(target_device)
    if bert.config.model_type != 'bert':
        raise ValueError(f'This PS-1 specifically requires BERT, got: {bert.config.model_type}')
    bert.eval()  # frozen embeddings; no gradient, no fine-tuning
    d = int(bert.config.hidden_size)
    token_vectors = np.lib.format.open_memmap(embeddings_path, 'w+', dtype='float16', shape=(len(texts), max_length, d))
    masks = np.lib.format.open_memmap(masks_path, 'w+', dtype='uint8', shape=(len(texts), max_length))
    try:
        for start in range(0, len(texts), batch_size):
            stop = min(start + batch_size, len(texts))
            tokens = tokenizer(texts[start:stop], padding='max_length', truncation=True,
                               max_length=max_length, return_tensors='pt')
            input_ids = tokens['input_ids'].to(target_device)
            attention_mask = tokens['attention_mask'].to(target_device)
            token_type_ids = tokens.get('token_type_ids')
            kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask}
            if token_type_ids is not None:
                kwargs['token_type_ids'] = token_type_ids.to(target_device)
            with torch.inference_mode():
                representation = bert(**kwargs).last_hidden_state
                representation = representation * attention_mask.unsqueeze(-1)
            token_vectors[start:stop] = representation.cpu().float().numpy().astype('float16')
            masks[start:stop] = attention_mask.cpu().numpy().astype('uint8')
            if (start // batch_size) % 25 == 0:
                print(f'Extracted {stop}/{len(texts)} BERT examples', flush=True)
        token_vectors.flush()
        masks.flush()
        save_json(metadata_path, {
            'csv_sha256': source_hash, 'model_name': model_name,
            'max_length': max_length, 'count': len(texts), 'hidden_size': d,
            'dtype': 'float16', 'frozen_bert': True,
            'note': 'Train, validation and test examples use the same PRETRAINED frozen encoder; labels are never used during extraction.'
        })
    finally:
        del token_vectors, masks, bert
    return embeddings_path, masks_path
