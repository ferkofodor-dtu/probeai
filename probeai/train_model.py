from pathlib import Path

import click
import matplotlib.pyplot as plt
import torch
import hydra
import logging
from data.make_dataset import load_probeai
from omegaconf import OmegaConf
from models.model import MyNeuralNet
from torch.profiler import profile, ProfilerActivity
from torch.profiler import profile, tensorboard_trace_handler

log = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# @click.command()
# @click.option("--epochs", default=5, help="number of epochs to train for")
# @click.option("--batch", default=8, help="batch size to use for training")
# @click.option("--lr", default=1e-3, help="learning rate to use for training")
@hydra.main(version_base="1.1", config_path="config", config_name="defaults.yaml")
def train(config):
    log.info(OmegaConf.to_yaml(config))
    log.info("Training day and night")

    # Get the config
    modelc_ = config.model_conf
    hyperpms = config.train_conf
    torch.manual_seed(hyperpms['seed'])

    # Initialize the model
    model = MyNeuralNet(modelc_['in_features'], modelc_['out_features']).to(device)

    log.info(f"Dataset path: {hyperpms['dataset_path']}")
    # Get the data
    train_data, test_data = load_probeai(hyperpms['dataset_path'])
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=hyperpms['batch_size'], shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=hyperpms['batch_size'], shuffle=True)

    # Train the model
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperpms['lr'])


    # Profiling
    prof = torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],

    # In this example with wait=1, warmup=1, active=2, repeat=1,
    # profiler will skip the first step/iteration,
    # start warming up on the second, record
    # the third and the forth iterations,
    # after which the trace will become available
    # and on_trace_ready (when set) is called;
    # the cycle repeats starting with the next step

    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=2,
        repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profile')
    # used when outputting for tensorboard
    )

    train_losses, test_losses = [], []
    # Training loop
    for e in range(hyperpms['n_epochs']):
        running_loss = 0
        with prof:
            for images, labels in trainloader:
                # Flatten MNIST images into a 784 long vector
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                output, _ = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                prof.step()

                running_loss += loss.cpu().item()
            train_losses.append(running_loss / len(trainloader))

            # Turn off gradients for validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                running_loss = 0
                for data in testloader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    log_ps, _ = model(inputs)
                    _, predicted = torch.max(log_ps.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    loss = criterion(log_ps, labels)
                    running_loss += loss.cpu().item()
                test_losses.append(running_loss / len(testloader))

            model.train()
            # if e % 1 == 0:
            log.info(f"Epoch: {e}, Training loss: {train_losses[-1]:.4f}, Validation loss: {test_losses[-1]:.4f}")
    
    
    # Save the model
    src = Path.cwd() / "models" / "model.pth"
    src.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, src)

    # Save the losses
    src_fig = Path.cwd() / "models" / "losses.png"
    plt.plot(train_losses, label="Training loss")
    plt.plot(test_losses, label="Validation loss")
    plt.legend(frameon=False)
    plt.savefig(src_fig)
    plt.close()


if "__main__" == __name__:
    train()
