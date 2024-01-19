import torch
import torch.nn as nn


class MyNeuralNet(nn.Module):
    """Basic neural network class.

    Args:
        in_features: number of input features
        out_features: number of output features

    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialize the neural network.

        Args:
            in_features: number of input features
            out_features: number of output features

        """
        super(MyNeuralNet, self).__init__()

        self.conv1 = nn.Conv2d(in_features, 32, 3)  # [B, 1, 70, 395] -> [B, 32, 68, 393]
        self.relu1 = nn.LeakyReLU()
        self.maxpool1 = nn.MaxPool2d(2)  # [B, 32, 68, 393] -> [B, 32, 34, 196]
        self.conv2 = nn.Conv2d(32, 64, 3)  # [B, 32, 34, 196] -> [B, 64, 32, 194]
        self.relu2 = nn.LeakyReLU()
        self.maxpool2 = nn.MaxPool2d(2)  # [B, 64, 32, 194] -> [B, 64, 16, 97]
        self.flatten = nn.Flatten()  # [B, 64, 16, 97] -> [B, 64 * 16 * 97] = [B, 100352]
        self.fc1 = nn.Linear(64 * 16 * 97, 512)
        self.relu3 = nn.LeakyReLU()
        self.fc2 = nn.Linear(512, 128)
        self.out = nn.Linear(128, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: input tensor expected to be of shape [N, in_features]

        Returns:
            Output tensor with shape [N, out_features]

        """
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)

        x = self.relu3(self.fc1(x))
        x = self.relu3(self.fc2(x))
        x = self.out(x)

        return x
