from torch.utils.data import DataLoader
import lightning as L
from data.make_dataset import load_probeai
from models.model import MyLightningModel
import hydra
from lightning.pytorch.loggers import WandbLogger
from predict_model import predict


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

    # save model
    trainer.save_checkpoint("./models/best_model.ckpt", weights_only=True)

    model = MyLightningModel.load_from_checkpoint(
        "./models/best_model.ckpt",
        config=config,
    )

    # Test the model
    predict(model, test_loader)


@hydra.main(version_base=None, config_path="config", config_name="defaults.yaml")
def hydra_train(config):
    L.seed_everything(config.train_conf["seed"])
    train(config)


if __name__ == "__main__":
    hydra_train()
