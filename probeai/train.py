import torch
from torch.utils.data import DataLoader
import lightning as L
import wandb
from data.make_dataset import load_probeai
from models.model import MyNeuralNet
import hydra

from torchmetrics.classification import Accuracy


class MyLightningModel(L.LightningModule):
    def __init__(self, config):
        super(MyLightningModel, self).__init__()
        self.config = config
        self.model = MyNeuralNet(
            config.model_conf.in_features, config.model_conf.out_features
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="binary")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs, _ = self(images)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs, _ = self(images)
        loss = self.criterion(outputs, labels)
        acc = self.accuracy(outputs, labels)
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.train_conf["learning_rate"]
        )
        return optimizer


def make_loader(dataset, batch_size):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )


def train(config):
    wandb.login()
    # Make the data
    train, test = load_probeai(config.train_conf.dataset_path)
    train_loader = make_loader(train, batch_size=config.train_conf.batch_size)
    test_loader = make_loader(test, batch_size=config.train_conf.batch_size)

    # Initialize PyTorch Lightning model
    model = MyLightningModel(config)

    # Initialize wandb logger
    wandb.init(project="probeai", entity="ferkofodor-dtu")

    # Initialize PyTorch Lightning trainer
    trainer = L.Trainer(
        max_epochs=config.train_conf.n_epochs,
        profiler="simple",
        gpus=1 if torch.cuda.is_available() else 0,
        log_every_n_steps=1,
        logger=wandb.loggers.WandbLogger(),
        progress_bar_refresh_rate=1,
    )

    # Train the model
    trainer.fit(model, train_loader, test_loader)


@hydra.main(version_base="1.1", config_path="config", config_name="defaults.yaml")
def hydra_train(config):
    train(config)


if __name__ == "__main__":
    hydra_train()
