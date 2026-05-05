import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import pandas as pd
import librosa
import numpy as np

def get_dataloaders(batch_size=128, augmentation=False, pretrained=False):
    """
    Returns CIFAR-10 train, validation, and test DataLoaders.
    Args:
        batch_size: mini-batch size
        augmentation: if True, apply data augmentation to training set only
        pretrained: if True, resize and normalize for pretrained VGG16
    Returns:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        test_loader: DataLoader for test set
    """
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)
        transform_plain = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_train = transform_plain
    else:
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2470, 0.2435, 0.2616)
        transform_plain = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        # TODO (for Step 2 Data Augmentation):
        transforms.RandomHorizontalFlip(p=0.5) # randomly will flip an image 
        transform_aug = transforms.Compose([
            # add augmentation 
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_train = transform_aug if augmentation else transform_plain

    full_dataset = datasets.CIFAR10(root="./data", train=True, download=True)
    train_size   = int(0.8 * len(full_dataset))
    generator    = torch.Generator().manual_seed(42)
    perm         = torch.randperm(len(full_dataset), generator=generator)
    train_indices = perm[:train_size].tolist()
    val_indices   = perm[train_size:].tolist()

    train_dataset = Subset(
        datasets.CIFAR10(root="./data", train=True, download=False, transform=transform_train),
        train_indices
    )
    val_dataset = Subset(
        datasets.CIFAR10(root="./data", train=True, download=False, transform=transform_plain),
        val_indices
    )
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_plain)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def plot_training(history, title="Training Curve"):
    """
    Plot training/validation loss and accuracy.
    Args:
        history: dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        title: plot title
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (loss_plt, acc_plt) = plt.subplots(1, 2, figsize=(14, 5)) # makes 2 plots; one for loss, one for accuracy
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # loss plot
    loss_plt.plot(epochs, history["train_loss"], label="Train Loss")
    loss_plt.plot(epochs, history["val_loss"],   label="Val Loss")
    loss_plt.set_title("Loss")
    loss_plt.set_xlabel("Epoch")
    loss_plt.set_ylabel("Loss")
    loss_plt.legend()
    loss_plt.grid(True)

    # accuracy plot
    acc_plt.plot(epochs, history["train_acc"], label="Train Acc")
    acc_plt.plot(epochs, history["val_acc"],   label="Val Acc")
    acc_plt.set_title("Accuracy")
    acc_plt.set_xlabel("Epoch")
    acc_plt.set_ylabel("Accuracy")
    acc_plt.set_ylim(0, 1)
    acc_plt.legend()
    acc_plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(true_classes, pred_classes, class_names):
    """
    Plots a confusion matrix heatmap.
    Args:
        true_classes: list or array of true class indices (0-9 for CIFAR-10)
        pred_classes: list or array of predicted class indices
        class_names: list of class names corresponding to indices
    """
    cm = confusion_matrix(true_classes, pred_classes)

    fig, axis = plt.subplots(figsize=(10, 8))
    
    #seaborn heat map
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axis)
    axis.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def get_feature_map_sizes(model, input_size=(1, 3, 32, 32)):  # Note: PyTorch uses (batch, channels, height, width)
    """
    Prints input/output shapes and parameter count for each layer.
    Useful for understanding how spatial dimensions change through your CNN and how many parameters are trainable.
    
    Note: use input_size= (1, 3, 224, 224) for pretrained VGG16
    """
    hooks  = []
    shapes = []

    # get device from model automatically
    device = next(model.parameters()).device

    # this hook will record input/output shapes and parameter counts for eahc layer
    def hook_fn(module, input, output):
        params = sum(p.numel() for p in module.parameters(recurse=False))
        shapes.append((
            module.__class__.__name__,
            list(input[0].shape),
            list(output.shape),
            params
        ))

    # register hooks on all layers
    for layer in model.modules():
        hooks.append(layer.register_forward_hook(hook_fn))

    # run a dummy forward pass
    with torch.no_grad():
        model(torch.zeros(input_size).to(device))

    # clean up hooks
    for h in hooks:
        h.remove()

    # print nicely formatted table
    print(f"{'Layer':<15} {'Input Shape':<25} {'Output Shape':<25} {'Params':>10}")
    print("-" * 80)
    for name, in_shape, out_shape, params in shapes:
        print(f"{name:<15} {str(in_shape):<25} {str(out_shape):<25} {params:>10,}")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("-" * 80)
    print(f"{'Total params':<65} {total:>10,}")
    print(f"{'Trainable params':<65} {trainable:>10,}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_audio_dataset(folder_path):
# used to go into each chord subfolder, and read it
    data = []
    for chord_folder in os.listdir(folder_path):
        chord_path = os.path.join(folder_path, chord_folder)
        if os.path.isdir(chord_path):
            for file in os.listdir(chord_path):
                if file.endswith(('.wav', '.mp3', '.flac')):
                    data.append({"file_path": os.path.join(chord_path, file), "label": chord_folder})
    return pd.DataFrame(data)

def wav_to_melspectrogram(file_path, sample_rate=22050, n_mels=128, n_fft=2048, hop_length=512):
    # Load audio
    audio, sr = librosa.load(file_path, sr=sample_rate)
    
    # Convert to mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, 
                                               n_fft=n_fft, hop_length=hop_length)
    
    # Convert to decibels (log scale) — important for CNNs
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db

MAX_LEN = 128  # adjust based on your audio length

def pad_or_truncate(spec, max_len=MAX_LEN):
    if spec.shape[1] < max_len:
        # Pad with zeros
        pad_width = max_len - spec.shape[1]
        spec = np.pad(spec, ((0, 0), (0, pad_width)), mode="constant")
    else:
        spec = spec[:, :max_len]
    return spec