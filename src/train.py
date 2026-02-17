import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(os.path.dirname(__file__))
from dataset import get_dataloaders
from model import WaveformUNet, count_parameters


# ── Loss Function ──────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    MAE + SSIM-inspired loss.
    MAE is robust for velocity regression.
    """
    def __init__(self, mae_weight=1.0, mse_weight=0.5):
        super().__init__()
        self.mae = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.mae_weight = mae_weight
        self.mse_weight = mse_weight

    def forward(self, pred, target):
        return self.mae_weight * self.mae(pred, target) + \
               self.mse_weight * self.mse(pred, target)


# ── Metric ─────────────────────────────────────────────────────────────────────

def mae_metric(pred, target):
    """MAE in original velocity units (m/s)."""
    pred_vel   = pred   * 3000.0 + 1500.0
    target_vel = target * 3000.0 + 1500.0
    return torch.mean(torch.abs(pred_vel - target_vel)).item()


# ── Train one epoch ────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_mae  = 0.0

    for batch_idx, (seis, vel) in enumerate(loader):
        seis = seis.to(device)
        vel  = vel.to(device)

        optimizer.zero_grad()
        pred = model(seis)
        loss = criterion(pred, vel)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mae  += mae_metric(pred, vel)

        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)} "
                  f"| Loss: {loss.item():.4f} "
                  f"| MAE: {mae_metric(pred, vel):.1f} m/s")

    return total_loss / len(loader), total_mae / len(loader)


# ── Validate one epoch ─────────────────────────────────────────────────────────

def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_mae  = 0.0

    with torch.no_grad():
        for seis, vel in loader:
            seis = seis.to(device)
            vel  = vel.to(device)
            pred = model(seis)
            loss = criterion(pred, vel)
            total_loss += loss.item()
            total_mae  += mae_metric(pred, vel)

    return total_loss / len(loader), total_mae / len(loader)


# ── Plot training curves ───────────────────────────────────────────────────────

def plot_curves(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'],   label='Val Loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(history['train_mae'], label='Train MAE (m/s)')
    axes[1].plot(history['val_mae'],   label='Val MAE (m/s)')
    axes[1].set_title('MAE (m/s)')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Curves saved to {save_path}")


# ── Main training function ─────────────────────────────────────────────────────

def train(
    data_root       = "../data/train_samples",
    checkpoint_dir  = "../outputs/checkpoints",
    epochs          = 20,
    batch_size      = 8,
    lr              = 1e-3,
    base_ch         = 64,
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")       # Apple Silicon GPU
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader = get_dataloaders(
        data_root, batch_size=batch_size, num_workers=0
    )

    # Model
    model = WaveformUNet(base_ch=base_ch).to(device)
    count_parameters(model)

    # Optimizer, scheduler, loss
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    criterion = CombinedLoss()

    # History
    history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
    best_val_loss = float('inf')

    print(f"\n🚀 Starting training for {epochs} epochs...\n")

    for epoch in range(1, epochs + 1):
        print(f"── Epoch {epoch}/{epochs} ──────────────────────────")

        train_loss, train_mae = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_mae   = val_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)

        print(f"  Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.1f} m/s")
        print(f"  Val   Loss: {val_loss:.4f} | Val   MAE: {val_mae:.1f} m/s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
            }, ckpt_path)
            print(f"  ✅ Best model saved (val_loss={val_loss:.4f})")

        # Save latest checkpoint every 5 epochs
        if epoch % 5 == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch}.pth")
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, ckpt_path)

    # Plot and save curves
    plot_curves(history, os.path.join(checkpoint_dir, "training_curves.png"))
    print("\n🎉 Training complete!")
    return model, history


if __name__ == "__main__":
    train()