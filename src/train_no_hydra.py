# TODO: Implement it using PyTorch Lightning but no Hydra at first.

# Essentially, I replicate `src/train.py::train()` but without Hydra
# and thereby hard-coded settings like data and model.

import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
import torch
from torch.optim import Adam
from lightning.pytorch.trainer import Trainer

from src.utils import (
    RankedLogger,
)
from src.data.online_handwriting_datasets import XournalPagewiseDatasetPyTorch
from src.data.online_handwriting_datasets import get_alphabet_from_dataset
from src.data.online_handwriting_datasets import get_number_of_channels_from_dataset
from src.data.online_handwriting_datamodule import SimpleOnlineHandwritingDataModule
from src.models.carbune_module import CarbuneLitModule
from src.models.components.carbune2020_net import Carbune2020NetAttempt1
from src.data.transforms import TwoChannels

# ==============================================
# ================== Settings ==================
# ==============================================

SEED = 42

# ==============================================
# ================== Main Code =================
# ==============================================

log = RankedLogger(__name__, rank_zero_only=True)

L.seed_everything(SEED, workers=True)

path = '/storage/datastore-personal/s1691089/data/code/carbune2020_implementation/data/datasets/2024-01-20-xournal_dataset.xoj'
path = '/storage/datastore-personal/s1691089/data/code/carbune2020_implementation/data/datasets/2024-02-16-xournal_dataset.xoj'
output_dir = '/storage/datastore-personal/s1691089/data/code/carbune2020_implementation/data/train_no_hydra'

log.info(f"Wee test: XournalPagewiseDatasetPyTorch can be initialised")
ds = XournalPagewiseDatasetPyTorch(
    path,
    transform=TwoChannels(),
)
# Seems to work - good!
# -> This can become a test later.
# del ds
print( ds[0] )

alphabet = get_alphabet_from_dataset( ds )

number_of_channels = get_number_of_channels_from_dataset( ds )

log.info(f"Instantiating datamodule")
datamodule: LightningDataModule = SimpleOnlineHandwritingDataModule(
    path, (1, 0, 0), 64, 0, False,
)

log.info(f"Instantiating model")


# model: LightningModule = # TODO
net = Carbune2020NetAttempt1(
    number_of_channels,
    nodes_per_layer=64,
    number_of_layers=3,
    dropout=0.0,
    alphabet=alphabet,
)
# optimiser = Adam(
#     # TODO: missing params,
#     lr=0.001,
#     weight_decay=0.0,
# )
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     mode='min',
#     factor=0.1,
#     patience=10, 
# )
model: LightningModule = CarbuneLitModule(
    net,
    optimizer=None,
    scheduler=None,
    compile=False,
)

# log.info("Instantiating callbacks...")
# callbacks: List[Callback] = # TODO

# log.info("Instantiating loggers...")
# logger: List[Logger] = # TODO

log.info(f"Instantiating trainer")
trainer: Trainer = Trainer(
    default_root_dir=output_dir,
    min_epochs=1, # prevents early stopping
    max_epochs=10,
    accelerator='gpu',
    devices=1,
    check_val_every_n_epoch=1,
    deterministic=False,
    # Added back in later:
    # callbacks=callbacks,
    # logger=logger,
)

datamodule.setup()
train_dataloader = datamodule.train_dataloader()

print('>>>>>>>>>>>>>>>>>>')
print(train_dataloader)
print(len(train_dataloader))
print(next( iter( train_dataloader )))
print(train_dataloader[0])
print('>>>>>>>>>>>>>>>>>>')

# TODO: Add collate function to dataloaders.

exit()

log.info("Starting training!")
trainer.fit(
    model=model,
    datamodule=datamodule,
    # ckpt_path=cfg.get("ckpt_path"),
)