import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SeismicDataset(Dataset):
    """
    PyTorch Dataset for Full Waveform Inversion.
    
    Input  (X): Seismic waveforms  — shape (5, 1000, 70)
    Target (y): Velocity maps      — shape (1, 70, 70)
    """

    def __init__(self, data_root, normalize=True):
        """
        Args:
            data_root  : path to train_samples folder
            normalize  : whether to normalize inputs and targets
        """
        self.normalize = normalize
        self.samples = []  # list of (seis_path, vel_path, sample_idx) tuples

        self._load_file_pairs(data_root)

        print(f"✅ Dataset ready: {len(self.samples)} total samples")

    def _load_file_pairs(self, data_root):
        """Walk through all subfolders and pair seismic with velocity files."""

        for folder in sorted(os.listdir(data_root)):
            folder_path = os.path.join(data_root, folder)
            if not os.path.isdir(folder_path):
                continue

            # Style 1: model/ and data/ subfolders (e.g. CurveVel_A)
            model_dir = os.path.join(folder_path, "model")
            data_dir  = os.path.join(folder_path, "data")

            if os.path.isdir(model_dir) and os.path.isdir(data_dir):
                model_files = sorted(os.listdir(model_dir))
                data_files  = sorted(os.listdir(data_dir))
                for mf, df in zip(model_files, data_files):
                    vel_path  = os.path.join(model_dir, mf)
                    seis_path = os.path.join(data_dir, df)
                    arr = np.load(vel_path, allow_pickle=True)
                    for i in range(arr.shape[0]):
                        self.samples.append((seis_path, vel_path, i))

            # Style 2: vel/seis files directly in folder (e.g. CurveFault_A)
            else:
                all_files = os.listdir(folder_path)
                vel_files  = sorted([f for f in all_files if f.startswith("vel")])
                seis_files = sorted([f for f in all_files if f.startswith("seis")])
                for vf, sf in zip(vel_files, seis_files):
                    vel_path  = os.path.join(folder_path, vf)
                    seis_path = os.path.join(folder_path, sf)
                    arr = np.load(vel_path, allow_pickle=True)
                    for i in range(arr.shape[0]):
                        self.samples.append((seis_path, vel_path, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seis_path, vel_path, sample_idx = self.samples[idx]

        seis = np.load(seis_path, allow_pickle=True)[sample_idx]  # (5, 1000, 70)
        vel  = np.load(vel_path,  allow_pickle=True)[sample_idx]  # (1, 70, 70)

        if self.normalize:
            # Normalize seismic: zero mean, unit std per sample
            mean = seis.mean()
            std  = seis.std() + 1e-8
            seis = (seis - mean) / std

            # Normalize velocity to [0, 1] using known physical range
            vel = (vel - 1500.0) / (4500.0 - 1500.0)

        seis = torch.tensor(seis, dtype=torch.float32)
        vel  = torch.tensor(vel,  dtype=torch.float32)

        return seis, vel


def get_dataloaders(data_root, batch_size=8, val_split=0.1, num_workers=2):
    """Create train and validation DataLoaders."""

    dataset = SeismicDataset(data_root, normalize=True)

    total    = len(dataset)
    val_size = int(total * val_split)
    trn_size = total - val_size

    train_ds, val_ds = torch.utils.data.random_split(dataset, [trn_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Train samples: {trn_size} | Val samples: {val_size}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    return train_loader, val_loader