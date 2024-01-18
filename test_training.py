import os
from tests.utils import run_training_script
import torch
from mlops.models.model import MyNeuralNet
import pytest
from tests import _PATH_DATA


@pytest.fixture
def mnist_model():
    return MyNeuralNet(1, 10)

def test_training_script(mnist_model):
    # Create a temporary directory for the model checkpoint
    checkpoint_dir = "temp_checkpoint"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Run the training script with a small number of epochs for testing
    config = {
        "model_conf": {
            "in_features": 1,
            "out_features": 10
        },
        "train_conf": {
            "n_epochs": 1,
            "batch_size": 64,
            "learning_rate": 0.001,
            "seed": 42,
            "dataset_path": _PATH_DATA
        }
    }

    # You should replace "path/to/mnist_dataset" with the actual path to your MNIST dataset
    run_training_script(config, mnist_model, checkpoint_dir)

    # Check if the model checkpoint is created
    assert os.path.exists(os.path.join(checkpoint_dir, "model.pth"))

    # Clean up
    os.rmdir(checkpoint_dir)
