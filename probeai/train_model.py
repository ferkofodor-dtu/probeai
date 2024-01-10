from pathlib import Path

import click
import matplotlib.pyplot as plt
import torch

from data.make_dataset import load_probeai
from models.model import MyNeuralNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@click.command()
@click.option("--epochs", default=5, help="number of epochs to train for")
@click.option("--batch", default=8, help="batch size to use for training")
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(epochs, batch, lr):
    """Train a model on MNIST."""
    print("Training day and night")
    print(f"E: {epochs}, B: {batch}, LR: {lr}")

    # Initialize the model
    model = MyNeuralNet(1, 2).to(device)

    # Get the data
    train_data, test_data = load_probeai()
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch, shuffle=True)

    # Train the model
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, test_losses = [], []
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output, _ = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
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

        # Make sure the model is back in training mode
        model.train()

        if e % 5 == 0:
            print(f"Epoch: {e}, Training loss: {train_losses[-1]:.4f}, Validation loss: {test_losses[-1]:.4f}")

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
