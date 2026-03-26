import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils import pad_or_truncate

class GuitarChordDataset(Dataset):
    def __init__(self, df, le):
        self.spectrograms = np.array([pad_or_truncate(s) for s in df["spectrogram"]])
        # Normalize
        self.spectrograms = (self.spectrograms - self.spectrograms.min()) / \
                            (self.spectrograms.max() - self.spectrograms.min())
        self.labels = torch.tensor(le.transform(df["label"]), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Add channel dimension → (1, 128, 128)
        spec = torch.tensor(self.spectrograms[idx], dtype=torch.float32).unsqueeze(0)
        return spec, self.labels[idx]