import torch
from torch.utils.data import DataLoader
import lightning as L
from data.make_dataset import load_probeai
from models.model import MyNeuralNet
import hydra
from torchmetrics.classification import Accuracy
from lightning.pytorch.loggers import WandbLogger


class MyLightningModel(L.LightningModule):
    def __init__(self, config):
        super(MyLightningModel, self).__init__()
        self.config = config
        self.model = MyNeuralNet(config.model_conf["in_features"], config.model_conf["out_features"])
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="binary")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.criterion(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        preds = self(images)
        loss = self.criterion(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.train_conf["learning_rate"])
        return optimizer


def make_loader(dataset, batch_size, shuffle=True):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=2,
    )


def train(config):
    # Make the data
    train, test = load_probeai()
    train_loader = make_loader(train, batch_size=config.train_conf["batch_size"])
    test_loader = make_loader(test, batch_size=config.train_conf["batch_size"], shuffle=False)

    # Initialize PyTorch Lightning model
    model = MyLightningModel(config)

    # # Initialize wandb logger
    wandb_logger = WandbLogger(project="probeai-lightning", entity="s220356")

    # Initialize PyTorch Lightning trainer
    trainer = L.Trainer(
        max_epochs=config.train_conf["n_epochs"],
        profiler="simple",
        accelerator="auto",
        logger=wandb_logger,
        log_every_n_steps=27,
        enable_checkpointing=True,
    )

    # Train the model
    trainer.fit(model, train_loader, test_loader)
    model = MyLightningModel.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        config=config,
    )

    # save model
    torch.save(model.state_dict(), "model.pth")

    return model


@hydra.main(version_base=None, config_path="config", config_name="defaults.yaml")
def hydra_train(config):
    L.seed_everything(config.train_conf["seed"])
    train(config)


if __name__ == "__main__":
    hydra_train()
