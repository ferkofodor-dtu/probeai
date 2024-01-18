import torch
from probeai.models.model import MyNeuralNet
import pytest


@pytest.fixture
def probeai_model():
    return MyNeuralNet(1, 2)


def test_model_output_shape(probeai_model):
    model = probeai_model
    dummy_input = torch.randn(1, 1, 395, 70)  # assuming MNIST image size
    output, _ = model(dummy_input)
    assert output.shape == torch.Size([1, 2])
