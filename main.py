import argparse
import sys
import yaml
import torch
import numpy as np

# Add src to path if running from root
sys.path.append('src')

from src.dataset import get_dataloaders
from src.models.base_gnn import BaselineGNN
from src.models.geometry_gnn import GeometricGNN
from src.train import Trainer

def main():
    parser = argparse.ArgumentParser(description='Molecular Property Prediction GNN')
    parser.add_argument('--model_type', type=str, default='geometric', choices=['base', 'geometric'], help='Model type: base (GCN) or geometric (SchNet-like)')
    parser.add_argument('--target_idx', type=int, default=7, help='Target index from QM9 (Default 7: U0)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    
    args = parser.parse_args()

    # Configuration
    config = {
        'lr': args.lr,
        'weight_decay': 1e-5,
        'log_dir': f'experiments/logs_{args.model_type}_target{args.target_idx}'
    }

    print(f"--- Molecular Property Prediction Pipeline ---")
    print(f"Model: {args.model_type}")
    print(f"Target Index: {args.target_idx}")
    
    # 1. Load Data
    root = 'data/qm9'
    train_loader, val_loader, test_loader, stats = get_dataloaders(
        root=root, 
        target_idx=args.target_idx, 
        batch_size=args.batch_size
    )

    # Add normalization stats to config for Trainer
    config['target_mean'] = stats['mean']
    config['target_std'] = stats['std']

    # 2. Initialize Model
    if args.model_type == 'base':
        model = BaselineGNN(hidden_channels=args.hidden_dim)
    else:
        model = GeometricGNN(hidden_channels=args.hidden_dim)

    # 3. Train
    trainer = Trainer(model, train_loader, val_loader, test_loader, config)
    trainer.run(num_epochs=args.epochs)

if __name__ == '__main__':
    main()
