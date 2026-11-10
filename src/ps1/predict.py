"""Predict on a new issue using the *trained* CNN and the matching frozen BERT."""
import argparse
import numpy as np
import torch
from .models import TokenCNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True, help='runs/PROJECT/results/final_cnn.pt')
    ap.add_argument('--text', required=True, help='New software task title and description')
    ap.add_argument('--device', choices=['cpu','cuda','auto'], default='auto')
    args = ap.parse_args()
    from transformers import AutoModel, AutoTokenizer
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    # Only load checkpoints you trained or explicitly trust.
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    tokenizer = AutoTokenizer.from_pretrained(saved['model_name'])
    bert = AutoModel.from_pretrained(saved['model_name']).to(device).eval()
    model = TokenCNN(embedding_dim=saved['embedding_dim'], channels=saved['conv_channels'],
                     kernel_size=saved['kernel_size'], dropout=saved['dropout']).to(device)
    model.load_state_dict(saved['model_state_dict'])
    model.eval()
    tokens = tokenizer(args.text, padding='max_length', truncation=True,
                       max_length=saved['max_length'], return_tensors='pt')
    kwargs = {k: v.to(device) for k, v in tokens.items()}
    with torch.inference_mode():
        vector = bert(**kwargs).last_hidden_state
        mask = kwargs['attention_mask'].float()
        prediction = model(vector * mask.unsqueeze(-1), mask)
    print(f'Trained on: {saved["project"]}')
    print(f'Predicted story points: {max(0.0, float(prediction.item())):.2f}')
    print('A numerical estimate, NOT an objectively correct complexity rating.')


if __name__ == '__main__':
    main()
