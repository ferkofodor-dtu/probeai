from pathlib import Path

import click
import torch
from torchvision import transforms
import cv2
from torch.utils.data import Dataset


class ProbeAIDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform:
            x = cv2.resize(x, (70,395), interpolation = cv2.INTER_AREA)
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)
    

def load_probeai(src=None):
    """Return train and test dataloaders for ProbeAI."""
    if src is None:
        src = Path.cwd() / "data" / "raw"
    else:
        src = Path(src).expanduser()

    train_data = torch.load(str(src / "train_data.pt"))
    test_data = torch.load(src / "test_data.pt")

    train_labels = torch.load(src / "train_labels.pt")
    test_labels = torch.load(src / "test_labels.pt")

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.0), (1.0)),])

    train_dataset = ProbeAIDataset(train_data, train_labels, transform=transform)
    test_dataset = ProbeAIDataset(test_data, test_labels, transform=transform)

    # train_dataset = train_dataset.unsqueeze(1)
    # test_dataset = test_dataset.unsqueeze(1)

    return train_dataset, test_dataset


@click.command()
@click.argument("src", type=click.Path(exists=True))
@click.argument("dst", type=click.Path())
def main(src, dst):
    """Process the data and save it."""
    src = Path.cwd() / src
    dst = Path.cwd() / dst
    dst.mkdir(parents=True, exist_ok=True)
    
    src.expanduser()
    dst.expanduser()

    train_data, test_data = load_probeai(src)
    torch.save(train_data, dst / "train_data.pt")
    torch.save(test_data, dst / "test_data.pt")


if __name__ == "__main__":
    # Get the data and process it
    main()
