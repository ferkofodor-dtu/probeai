from pathlib import Path
import click
import torch
from visualizations.visualize import visualize_tsne
from data.make_dataset import load_probeai


def predict(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> None:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """

    interms = []
    labels = []

    model = model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, label = data
            inputs, label = inputs, label
            log_ps, interm = model(inputs)
            _, predicted = torch.max(log_ps.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            interms.append(interm)
            labels.append(label)

    accuracy = correct / total
    print(f"Accuracy: {accuracy*100:.2f}%")

    src_fig = Path.cwd() / "models" / "tsne.png"
    src_fig.parent.mkdir(parents=True, exist_ok=True)

    # Concatenate the intermediate representations
    features = torch.cat(interms, dim=0)
    labels = torch.cat(labels, dim=0)

    # Visualize the features and save the figure
    visualize_tsne(features, labels, src_fig)

    # return torch.cat([model(batch)[1] for batch, _ in dataloader], 0).cpu()


@click.command()
@click.argument("src", type=click.Path(exists=True))
def main(src):
    """Run prediction for a given model and dataloader."""
    src = Path.cwd() / src
    model = torch.load(src).cpu()
    _, test_data = load_probeai()
    testloader = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=True)
    predict(model, testloader)


if __name__ == "__main__":
    # Get the data and process it
    main()
