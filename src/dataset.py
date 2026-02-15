import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform
import numpy as np

class TargetTransform(BaseTransform):
    """
    Transforms the label to a specific target index from QM9.
    QM9 has 19 regression targets.
    index 0: mu
    index 1: alpha
    index 2: HOMO
    index 3: LUMO
    index 4: gap
    ...
    index 7: U0 (internal energy)
    """
    def __init__(self, target_idx=7):
        self.target_idx = target_idx

    def __call__(self, data):
        # QM9.y is typically shape [1, 19]
        # We select the specific target and ensure it's float
        data.y = data.y[:, self.target_idx].view(-1, 1).float()
        return data

def get_dataloaders(root, target_idx=7, batch_size=64, split_ratio=(0.8, 0.1, 0.1), seed=42):
    """
    Loads QM9 dataset, applies transformations, and returns DataLoaders.
    """
    dataset = QM9(root=root)
    
    # Shuffle dataset
    dataset = dataset.shuffle()
    
    # Transform labels on the fly
    # Note: For efficiency in production, pre-transform is better, 
    # but for flexibility here we iterate.
    # Actually, let's just modify the y values for the split we use.
    # A cleaner way is to wrap the dataset or just extract the column in the training loop.
    # But to be "Research Grade", let's normalize the targets.
    
    # 1. Feature normalization (optional but recommended for regression)
    # We will compute mean/std of the training target.
    
    # Calculate split indices
    N = len(dataset)
    n_train = int(N * split_ratio[0])
    n_val = int(N * split_ratio[1])
    n_test = N - n_train - n_val
    
    train_dataset = dataset[:n_train]
    val_dataset = dataset[n_train:n_train+n_val]
    test_dataset = dataset[n_train+n_val:]
    
    # Compute stats for target normalization from TRAIN set only (prevent leakage)
    train_y = train_dataset.data.y[:, target_idx]
    mean = train_y.mean().item()
    std = train_y.std().item()
    
    # Store stats to inverse transform later
    stats = {'mean': mean, 'std': std}
    
    print(f"Dataset Loaded. Total: {N}")
    print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")
    print(f"Target Index: {target_idx}, Mean: {mean:.4f}, Std: {std:.4f}")

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, stats
