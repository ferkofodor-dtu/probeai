from pathlib import Path

import click
import torch


def mnist(src=None):
    """Return train and test dataloaders for MNIST."""
    if src is None:
        src = Path.cwd() / "data" / "raw" / "mnist"
    else:
        src = Path(src)

    train_data, train_labels = [], []
    for i in range(5):
        train_data.append(torch.load(src / f"train_images_{i}.pt"))
        train_labels.append(torch.load(src / f"train_target_{i}.pt"))

    train_data = torch.cat(train_data, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    test_data = torch.load(src / "test_images.pt")
    test_labels = torch.load(src / "test_target.pt")

    train_data = train_data.unsqueeze(1)
    test_data = test_data.unsqueeze(1)

    # print(train_data.shape)
    # print(train_labels.shape)
    # print(test_data.shape)
    # print(test_labels.shape)

    return (
        torch.utils.data.TensorDataset(train_data, train_labels),
        torch.utils.data.TensorDataset(test_data, test_labels),
    )


@click.command()
@click.argument("src", type=click.Path(exists=True))
@click.argument("dst", type=click.Path())
@click.argument("dataset", type=str)
def main(src, dst, dataset):
    """Process the data and save it."""
    src = Path.cwd() / src / dataset
    dst = Path.cwd() / dst / dataset
    dst.mkdir(parents=True, exist_ok=True)

    train_data, test_data = mnist(src)
    torch.save(train_data, dst / "train_data.pt")
    torch.save(test_data, dst / "test_data.pt")


if __name__ == "__main__":
    # Get the data and process it
    main()
