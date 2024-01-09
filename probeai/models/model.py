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

        self.conv1 = nn.Conv2d(in_features, 32, 3)  # [B, 1, 28, 28] -> [B, 32, 26, 26]
        self.relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(32, 64, 3)  # [B, 32, 26, 26] -> [B, 64, 24, 24]
        self.relu2 = nn.LeakyReLU()
        self.maxpool = nn.MaxPool2d(2)  # [B, 64, 24, 24] -> [B, 64, 12, 12]
        self.flatten = nn.Flatten()  # [B, 64, 12, 12] -> [B, 64 * 12 * 12]
        self.fc1 = nn.Linear(64 * 12 * 12, 500)
        self.relu3 = nn.LeakyReLU()
        self.fc2 = nn.Linear(500, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: input tensor expected to be of shape [N, in_features]

        Returns:
            Output tensor with shape [N, out_features]

        """
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool(x)
        x = self.flatten(x)

        # Extract the intermediate representation from fc1
        intermediate_representation = self.fc1(x)

        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        return x, intermediate_representation
