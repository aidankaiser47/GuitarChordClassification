import torch
import torch.nn as nn
import torch.optim as optim


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


def run_experiment(model, train_loader, val_loader, device, num_epochs=20, lr=0.001):
    """
    Full training loop for one experiment.
    Args:
        model: nn.Module, the neural network being trained
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: the variable that tells PyTorch whether to run on CPU or GPU
        num_epochs (int): number of training epochs
        lr (float): learning rate
    Returns:
        history: dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    for epoch in range(1, num_epochs + 1): # for however many epochs
        # we train and then validate
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # store all the losses and accuracies for later
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        #epoch progress
        print(f"Epoch [{epoch:>3}/{num_epochs}]  "
              f"Train Loss: {train_loss:.2f}  Train Acc: {train_acc:.2f}  |  "
              f"Val Loss: {val_loss:.2f}  Val Acc: {val_acc:.2f}")

    return history