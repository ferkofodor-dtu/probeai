from pathlib import Path

import matplotlib.pyplot as plt
import torch
import hydra
import logging
from data.make_dataset import load_probeai
from omegaconf import OmegaConf
from models.model import MyNeuralNet
import wandb

log = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )
    return loader


def make(config):
    # Make the data
    train, test = load_probeai()
    train_loader = make_loader(train, batch_size=config.train_conf["batch_size"])
    test_loader = make_loader(test, batch_size=config.train_conf["batch_size"])

    # Make the model
    model = MyNeuralNet(
        config.model_conf["in_features"], config.model_conf["out_features"]
    ).to(device)

    # Make the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.train_conf["learning_rate"]
    )

    return model, train_loader, test_loader, criterion, optimizer



def train_batch(images, labels, model, optimizer, criterion):
    images, labels = images.to(device), labels.to(device)

    # Forward pass ➡
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss


def test(model, test_loader, criterion, epoch, save_model=False):
    model.eval()
    # Run the model on some test examples
    with torch.no_grad():
        running_loss = 0
        correct, total = 0, 0
        for images, labels in test_loader:
            inputs, labels = images.to(device), labels.to(device)
            log_ps = model(inputs)
            _, predicted = torch.max(log_ps.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(log_ps, labels)
            running_loss += loss.cpu().item()

        print(
            f"Accuracy of the model on the {total} "
            + f"test images: {correct / total:%}"
        )
    wandb.log({"test_accuracy": correct / total})

    return running_loss / len(test_loader)



# @click.command()
# @click.option("--epochs", default=5, help="number of epochs to train for")
# @click.option("--batch", default=8, help="batch size to use for training")
# @click.option("--lr", default=1e-3, help="learning rate to use for training")
@hydra.main(version_base=None, config_path="config", config_name="defaults.yaml")
def train(config):
    wandb.login()

    wandb.init(project="probeai", entity="s220356")

    log.info(OmegaConf.to_yaml(config))
    log.info("Training day and night")

    torch.manual_seed(config.train_conf["seed"])


    model, train_loader, test_loader, criterion, optimizer = make(config)

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
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/profile")
        # used when outputting for tensorboard
    )

    train_losses, test_losses = [], []
    # Training loop
    for e in range(config.train_conf["n_epochs"]):
        running_loss = 0
        with prof:
            for images, labels in train_loader:
                loss = train_batch(images, labels, model, optimizer, criterion)
                running_loss += loss.cpu().item()
            train_losses.append(running_loss / len(train_loader))

            wandb.log({"epoch": e, "loss": loss})
            test_loss = test(model, test_loader, criterion, e)
            test_losses.append(test_loss)

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
