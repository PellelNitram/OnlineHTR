from typing import Any, Dict, Tuple
from pathlib import Path

import torch
from torch import nn
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.functional.text import word_error_rate
from torchmetrics.functional.text import char_error_rate
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

from src.utils.decoders import GreedyCTCDecoder
from src.models.components.carbune2020_net import Carbune2020NetAttempt1
from src.utils.io import store_alphabet
from src.data.tokenisers import AlphabetMapper


class LitModule1(LightningModule):
    """Attempt 1 of neural network described in [Carbune2020] paper.

    The description in [Carbune2020] is not clear w.r.t. the last layer,
    hence this is my first interpretation of the network that they use.
    """

    def __init__(
        self,
        nodes_per_layer: int,
        number_of_layers: int,
        dropout: float,
        decoder,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        alphabet: list[str],
        number_of_channels:int,
    ) -> None:
        """Initialize a `LitModule1` module.

        :param nodes_per_layer: The dimension of the hidden state in the stack of LSTM cells.
        :param number_of_layers: The number of LSTM layers.
        :param dropout: Dropout value to use in stack of LSTM cells.
        :param decoder: CTC decoder for decoding log probabilities to characters.
        :param optimizer: The optimizer for training.
        :param scheduler: The learning rate scheduler.
        :param alphabet: The alphabet of tokens to use.
        :param number_of_channels: The number of input channels per time step in the time series.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.alphabet = list(alphabet)
        self.alphabet_mapper = AlphabetMapper(self.alphabet)

        # loss function
        self.criterion = torch.nn.CTCLoss(blank=0, reduction='mean')

        # ==============
        # Network layers
        # ==============

        # I would have loved to set up this module in `setup` but I was not able to
        # load checkpoints then, see e.g. https://github.com/Lightning-AI/pytorch-lightning/issues/5410

        # Output layer to be fed into CTC loss; the output must be log probabilities
        # according to https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
        # with shape (T, N, C) where C is the number of classes (= here alphabet letters)
        # N is the batch size and T is the sequence length
        self.log_softmax = nn.LogSoftmax(dim=2) # See this documentation:
                                                # https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html#torch.nn.LogSoftmax

        # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.lstm_stack = torch.nn.LSTM(
            input_size=number_of_channels,
            hidden_size=nodes_per_layer,
            num_layers=number_of_layers,
            bias=True,
            batch_first=False,
            dropout=dropout,
            bidirectional=True,
            proj_size=0,
        )

        # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.linear = torch.nn.Linear(
            in_features=2 * nodes_per_layer, # 2 b/c bidirectional=True
            out_features=len(self.alphabet) + 1, # +1 for blank
            bias=True,
        )

        self.hp_metric = MinMetric() # Set min(val_loss) as hp_metric

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model.

        :param x: The input tensor.
        :return: A tensor of predictions that is compatible with CTC loss.
                 Shape: `[sequence length, batch size, len(self.alphabet) + 1]`.
        """
        result, (h_n, c_n) = self.lstm_stack(x)
        result = self.linear(result)
        result = self.log_softmax(result)
        return result

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        log_softmax = self.forward(batch['ink'])
        loss = self.criterion(
            log_softmax,
            batch['label'],
            batch['ink_lengths'],
            batch['label_lengths'],
        )

        metrics = {
        }

        return loss, metrics

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, metrics = self.model_step(batch)

        self.batch_size = batch['ink'].shape[1]

        # update and log metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

        # # TODO: Add text to log like tensorboard - what to save exactly? saving full text is too wasteful every step - every few steps??
        # for logger in self.loggers:
        #     if isinstance(logger, TensorBoardLogger):
        #         tensorboard = logger.experiment
        #         tensorboard.add_text('test_text', f'this is a test - {self.global_step}', self.global_step)

        # return loss or backpropagation will fail
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, metrics = self.model_step(batch)

        self.batch_size = batch['ink'].shape[1]

        # update and log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

        # Log hyperparameter metric as explained here:
        # https://lightning.ai/docs/pytorch/stable/extensions/logging.html#logging-hyperparameters
        self.hp_metric.update(loss.item())
        self.log("hp_metric", self.hp_metric.compute(), batch_size=self.batch_size)
        # TODO: CHECK IF THIS WORKS!!! I need it for larger parameter sweep.

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, metrics = self.model_step(batch)

        self.batch_size = batch['ink'].shape[1]

        # update and log metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": self.trainer.check_val_every_n_epoch, # I.e. every available val epoch
                },
            }
        return {"optimizer": optimizer}

    def on_fit_start(self):

        # Store data for subsequent inference: alphabet
        dm = self.trainer.datamodule
        store_alphabet(
            outfile=Path(self.trainer.default_root_dir) / 'alphabet.json',
            alphabet=dm.alphabet,
        )

    # def on_fit_end(self):

    #     # Store hparams
    #     print('finiiiished test')

    #     for logger in self.trainer.loggers:
    #         logger.log_hyperparams(
    #             {
    #                 'p1': 3.0,
    #                 'p2': 'foo'
    #                 # TODO: Add all relevant parameters here
    #             },
    #             {
    #                 "hp/metric_1": 0,
    #                 "hp/metric_2": 0,
    #                 # TODO: Add best val_loss and CER/WER w/ all specified decoders
    #             }
    #         )