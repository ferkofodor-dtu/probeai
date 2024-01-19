import torch
from data.make_dataset import load_probeai
from models.model import MyLightningModel
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import hydra


def predict(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> None:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """

    model.eval()

    preds = []
    targets = []
    with torch.no_grad():
        for data, label in dataloader:
            output = model(data)
            preds.append(output)
            targets.append(label)

    preds = torch.cat(preds, dim=0).argmax(dim=1).cpu()
    targets = torch.cat(targets, dim=0).cpu()

    report = classification_report(targets, preds)
    with open("./reports/classification_report.txt", "w") as outfile:
        outfile.write(report)
    confmat = confusion_matrix(targets, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=confmat)
    disp.plot()
    plt.savefig("./reports/figures/confusion_matrix.png")
    plt.close()


# @click.command()
# @click.argument("src", type=click.Path(exists=True))
@hydra.main(version_base=None, config_path="config", config_name="defaults.yaml")
def main(config):
    """Run prediction for a given model and dataloader."""
    model = MyLightningModel.load_from_checkpoint("./models/best_model.ckpt", config=config)
    _, test_data = load_probeai()
    testloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    predict(model, testloader)


if __name__ == "__main__":
    # Get the data and process it
    main()
