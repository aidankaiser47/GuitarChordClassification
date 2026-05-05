import torch
from torch.utils.data import Dataset
import numpy as np
from utils import pad_or_truncate

# parts of this file were AI assisted with Claude

def frequency_mask(spec, max_mask_size=20):
    """Zero out a random horizontal band (frequency axis)."""
    spec = spec.copy()
    h = spec.shape[0]
    mask_size = np.random.randint(1, max(2, max_mask_size))
    start = np.random.randint(0, max(1, h - mask_size))
    spec[start:start + mask_size, :] = 0.0
    return spec


def time_mask(spec, max_mask_size=20):
    """Zero out a random vertical band (time axis)."""
    spec = spec.copy()
    w = spec.shape[1]
    mask_size = np.random.randint(1, max(2, max_mask_size))
    start = np.random.randint(0, max(1, w - mask_size))
    spec[:, start:start + mask_size] = 0.0
    return spec


def add_gaussian_noise(spec, std=0.01):
    """Add small Gaussian noise to simulate recording variation."""
    return spec + np.random.normal(0, std, spec.shape).astype(np.float32)


def random_time_shift(spec, max_shift=10):
    """Shift the spectrogram along the time axis by a random amount."""
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(spec, shift, axis=1)


def pitch_shift_approx(spec, max_shift=4):
    """Approximate pitch shift by rolling along the frequency axis."""
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(spec, shift, axis=0)


def random_amplitude_scale(spec, low=0.8, high=1.2):
    """Scale overall amplitude — simulates volume variation."""
    scale = np.random.uniform(low, high)
    return np.clip(spec * scale, 0.0, 1.0)


def apply_train_augmentations(spec):
    """
    Chain of augmentations for training. Each is applied
    probabilistically so not all fire on every sample.
    """
    if np.random.rand() < 0.5:
        spec = frequency_mask(spec, max_mask_size=20)
    if np.random.rand() < 0.5:
        spec = time_mask(spec, max_mask_size=20)
    if np.random.rand() < 0.5:
        spec = add_gaussian_noise(spec, std=0.01)
    if np.random.rand() < 0.5:
        spec = random_time_shift(spec, max_shift=10)
    if np.random.rand() < 0.4:
        spec = pitch_shift_approx(spec, max_shift=4)
    if np.random.rand() < 0.5:
        spec = random_amplitude_scale(spec, low=0.8, high=1.2)
    return spec

class GuitarChordDataset(Dataset):
    def __init__(self, df, le, transform=None, augment=False):
        """
        Args:
            df        : DataFrame with 'spectrogram' and 'label' columns
            le        : fitted LabelEncoder
            transform : torchvision transform (for ResNet-18 pipeline)
            augment   : if True, apply spectrogram augmentations (training only)
        """
        self.spectrograms = np.array(
            [pad_or_truncate(s) for s in df["spectrogram"]], dtype=np.float32
        )

        # Per-sample normalization — more robust than global min/max on small datasets
        self.spectrograms = self._normalize(self.spectrograms)

        self.labels    = torch.tensor(le.transform(df["label"]), dtype=torch.long)
        self.transform = transform
        self.augment   = augment

    @staticmethod
    def _normalize(specs):
        """Normalize each spectrogram independently to [0, 1]."""
        out = np.zeros_like(specs, dtype=np.float32)
        for i, s in enumerate(specs):
            s_min, s_max = s.min(), s.max()
            out[i] = (s - s_min) / (s_max - s_min + 1e-8)  # epsilon avoids div-by-zero
        return out

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        spec = self.spectrograms[idx].copy()  # (128, 128) float32

        # Apply spectrogram-domain augmentations (training only)
        if self.augment:
            spec = apply_train_augmentations(spec)

        if self.transform:
            # ResNet path: numpy → PIL (grayscale) → transform → 3-channel tensor
            from PIL import Image
            spec_uint8 = (spec * 255).astype(np.uint8)
            img  = Image.fromarray(spec_uint8, mode='L')
            spec = self.transform(img)
        else:
            # Custom CNN path: add channel dim → (1, 128, 128)
            spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)

        return spec, self.labels[idx]