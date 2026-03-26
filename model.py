import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CustomCNN(nn.Module):
    """
    Custom convolutional neural network for image classification.
    
    Args:
        num_layers (int): number of convolutional layers
        filters (list of int): number of filters for each conv layer
        kernel_sizes (list of int): kernel size for each conv layer
        stride (int): stride for all convolutional layers
        padding (int): padding for all convolutional layers
        use_pooling (bool): whether to apply MaxPool after conv layers
        dropout (float): dropout probability before fully connected layer
        num_classes (int): number of output classes
    """
    def __init__(self,
                 num_layers=3,
                 filters=None,
                 kernel_sizes=None,
                 stride=1,
                 padding=1,
                 use_pooling=True,
                 dropout=0.3,
                 num_classes=10):
        super().__init__()

        if filters is None:
            filters = [32, 64, 128]
        if kernel_sizes is None:
            kernel_sizes = [3] * num_layers
        # Note: PyTorch uses (batch, channels, height, width)

        layers = []
        in_channels = 1 # grayscale layers
        for i in range(num_layers): # loops for each layer specified
            layers.append(nn.Conv2d(in_channels, filters[i], kernel_size=kernel_sizes[i], stride=stride, padding=padding))
            layers.append(nn.ReLU())
            
            if use_pooling:
                layers.append(nn.MaxPool2d(kernel_size=2))
                
            in_channels = filters[i]

        self.features = nn.Sequential(*layers)


        # use a dummy forward pass to compute the flattened feature size, then define self.fc
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 128, 128) # size of the spectrogram
            flat_size = self.features(dummy).view(1, -1).shape[1]

        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(flat_size, num_classes))


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

