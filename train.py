import torch
import torch.nn as nn
import torch.optim as optim

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from guitar_chord_dataset import GuitarChordDataset


# parts of this file were AI assisted with Claude

def build_resnet18(num_classes: int, dropout_rate: float = 0.5) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    # Only fine-tune layer4 — less capacity = less overfitting on tiny datasets
    for param in model.layer4.parameters():
        param.requires_grad = True

    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float('inf')
        self.counter    = 0
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Run one full pass over the training data
    Args:
        model: nn.Module, the neural network being trained
        loader: DataLoader, provides batches of training data
        criterion: loss function
        optimizer: optimizer for updating model parameters
        device: the variable that tells PyTorch whether to run on CPU or GPU
    Returns:
        avg_loss: average loss over all batches
        avg_acc:  average accuracy over all batches
    """
    model.train()

    total_loss = 0.0
    correct = 0
    total_ins = 0

    for inputs, labels in loader: # for each batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad() # reset stored gradients

        outputs = model(inputs)
        loss = criterion(outputs, labels) # gets loss

        loss.backward() # back props
        optimizer.step()

        total_loss += loss.item() * inputs.size(0) # gathers loss for batch, asdds to total
        _, predicted = outputs.max(1) # gets predicted class from the index with the max value
        correct += predicted.eq(labels).sum().item() # gets num correct
        total_ins += labels.size(0) # gets total inputs

    avg_loss = total_loss / total_ins
    avg_acc = correct / total_ins
    return avg_loss, avg_acc


def validate(model, loader, criterion, device):
    """
    Evaluate model on validation set
    Args:
        model: nn.Module, the neural network being evaluated
        loader: DataLoader, provides batches of validation data
        criterion: loss function
        device: the variable that tells PyTorch whether to run on CPU or GPU
    Returns:
        avg_loss: average loss over all batches
        avg_acc:  average accuracy over all batches
    """
    model.eval()

    # super similar to train_one_epoch
    total_loss = 0.0
    correct = 0
    total_ins = 0

    with torch.no_grad():
        for inputs, labels in loader: # for each batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1) # gets predicted class from the index with max value
            correct += predicted.eq(labels).sum().item()
            total_ins += labels.size(0)

    avg_loss = total_loss / total_ins
    avg_acc = correct / total_ins
    return avg_loss, avg_acc


def test(model, loader, device):
    """
    Evaluate final model performance on test set
    Args:
        model: nn.Module, the neural network being evaluated
        loader: DataLoader, provides batches of test data
        device: the variable that tells PyTorch whether to run on CPU or GPU
    Returns:
        avg_acc: average test accuracy
    """
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader: # for each batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = outputs.max(1) # gets the predicted class via max 
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_acc = correct / total
    return avg_acc


def get_predictions(model, loader, device):
    """
    Collect all predictions and true labels from a data loader
    This will be helpful for confusion matrix heatmap 
    Args:
        model: nn.Module, the neural network being evaluated
        loader: DataLoader, provides batches of data
        device: the variable that tells PyTorch whether to run on CPU or GPU
    Returns:
        all_labels: list of true class indices, i.e., the class label (0-9 for CIFAR-10)
        all_preds:  list of predicted class indices
    """
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_labels.extend(labels.cpu().tolist()) # adds labels to complete set
            all_preds.extend(predicted.cpu().tolist()) # adds predictions to our complete set

    return all_labels, all_preds


def run_experiment(model, train_loader, val_loader, device,
                   num_epochs=40, lr=0.001, weight_decay=1e-3):
    """
    Regularization applied here:
      - weight_decay (L2) via AdamW
      - label smoothing in CrossEntropyLoss
      - cosine LR annealing
      - early stopping
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay   # L2 regularization
    )
    scheduler     = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    early_stopper = EarlyStopping(patience=7)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        # ── Train ──
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item() * imgs.size(0)
            t_correct += (out.argmax(1) == labels).sum().item()
            t_total   += imgs.size(0)

        # validate
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out  = model(imgs)
                loss = criterion(out, labels)
                v_loss    += loss.item() * imgs.size(0)
                v_correct += (out.argmax(1) == labels).sum().item()
                v_total   += imgs.size(0)

        train_loss = t_loss / t_total
        train_acc  = t_correct / t_total
        val_loss   = v_loss / v_total
        val_acc    = v_correct / v_total

        scheduler.step()
        early_stopper.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  ✓ Saved best model (val_acc={val_acc:.3f})")

        if early_stopper.should_stop:
            print(f"  Early stopping triggered at epoch {epoch}")
            break

    return history