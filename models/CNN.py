import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, in_channels, conv_channels, num_classes):
        super().__init__()
        self.convs = nn.ModuleList()

        prev_channels = in_channels
        for out_channels in conv_channels:
            self.convs.append(nn.Conv2d(prev_channels, out_channels, kernel_size=3, padding=1))
            prev_channels = out_channels

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.classifier = nn.Linear(prev_channels, num_classes)

    def forward(self, x):
        for conv in self.convs:
            x = F.relu(conv(x))
        x = self.pool(x)  # shape: (batch, channels, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
