import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import os
import yaml

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, config, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config['lr'], weight_decay=config.get('weight_decay', 0.0))
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        
        self.start_epoch = 0
        self.best_val_mae = float('inf')
        self.target_mean = config.get('target_mean', 0.0)
        self.target_std = config.get('target_std', 1.0)
        
        # Logging Setup
        self.log_dir = config.get('log_dir', 'experiments/logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_mae = 0
        
        for data in tqdm(self.train_loader, desc="Training", leave=False):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            out = self.model(data)
            
            # Loss Calculation (against normalized targets!)
            # But the dataset loader might not normalize y itself if we are doing it on the fly.
            # In `dataset.py`, we didn't normalize y in `get_dataloaders`. 
            # We calculated stats but returned raw y data.
            # So let's normalize y here for loss calculation.
            target_norm = (data.y - self.target_mean) / self.target_std
            
            loss = F.mse_loss(out, target_norm)
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item() * data.num_graphs
            
            # Calculate MAE in original scale for monitoring
            with torch.no_grad():
                pred_orig = out * self.target_std + self.target_mean
                mae = F.l1_loss(pred_orig, data.y)
                total_mae += mae.item() * data.num_graphs

        return total_loss / len(self.train_loader.dataset), total_mae / len(self.train_loader.dataset)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_mae = 0
        total_loss = 0 # MSE
        
        for data in loader:
            data = data.to(self.device)
            out = self.model(data)
            
            target_norm = (data.y - self.target_mean) / self.target_std
            loss = F.mse_loss(out, target_norm) # MSE on normalized scale
            
            pred_orig = out * self.target_std + self.target_mean
            mae = F.l1_loss(pred_orig, data.y) # MAE on original scale
            
            total_loss += loss.item() * data.num_graphs
            total_mae += mae.item() * data.num_graphs
            
        return total_loss / len(loader.dataset), total_mae / len(loader.dataset)

    def run(self, num_epochs=100):
        print(f"Starting training on {self.device}...")
        
        for epoch in range(1, num_epochs + 1):
            train_loss, train_mae = self.train_epoch()
            val_loss, val_mae = self.evaluate(self.val_loader)
            
            self.scheduler.step(val_mae)
            
            print(f"Epoch {epoch:03d}: Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f}")
            
            # Save Checkpoint based on MAE
            if val_mae < self.best_val_mae:
                self.best_val_mae = val_mae
                torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'best_model.pth'))
                print(f"  -> New Best Model Saved (MAE: {val_mae:.4f})")
                
        # Final Test
        self.model.load_state_dict(torch.load(os.path.join(self.log_dir, 'best_model.pth')))
        test_loss, test_mae = self.evaluate(self.test_loader)
        print(f"\nFinal Test MAE: {test_mae:.4f}")
