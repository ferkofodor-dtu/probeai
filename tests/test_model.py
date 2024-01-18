import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from probeai.models.model import MyNeuralNet
from torchvision.datasets import MNIST
import pytest

@pytest.fixture
def probeai_model():
    return MyNeuralNet(1, 2)

def test_model_output_shape(probeai_model):
    model = probeai_model
    dummy_input = torch.randn(1, 1, 395, 70)  # assuming MNIST image size
    output, _ = model(dummy_input)
    assert output.shape == torch.Size([1, 2])
