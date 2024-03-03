from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.functional.text import word_error_rate
from torchmetrics.functional.text import char_error_rate
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

from src.utils.decoders import GreedyCTCDecoder


class CarbuneLitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        alphabet_mapper,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        self.alphabet_mapper = alphabet_mapper
        self.decoder = GreedyCTCDecoder()

        # loss function
        self.criterion = torch.nn.CTCLoss(blank=0, reduction='mean')

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for averaging wer across batches
        self.train_wer = MeanMetric()
        self.val_wer = MeanMetric()
        self.test_wer = MeanMetric()

        # for averaging cer across batches
        self.train_cer = MeanMetric()
        self.val_cer = MeanMetric()
        self.test_cer = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

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
        decoded_texts = self.decoder(log_softmax, self.alphabet_mapper)

        # TODO: Could be pre-computed (using list0 in batch to avoid endless recomputation
        labels = []
        for i_batch in range(log_softmax.shape[1]):
            label_length = batch['label_lengths'][i_batch]
            label = batch['label'][i_batch, :label_length]
            label = [ self.alphabet_mapper.index_to_character(c) for c in label ]
            label = "".join(label)
            labels.append(label)

        cer = char_error_rate(preds=decoded_texts, target=labels)
        wer = word_error_rate(preds=decoded_texts, target=labels)

        metrics = {
            'cer': cer,
            'wer': wer,
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

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.train_wer(metrics['wer'])
        self.log("train/wer", self.train_wer, on_step=False, on_epoch=True, prog_bar=True)

        self.train_cer(metrics['cer'])
        self.log("train/cer", self.train_cer, on_step=False, on_epoch=True, prog_bar=True)

        # # TODO: Add text to log like tensorboard - what to save exactly? saving full text is too wasteful every step - every few steps??
        # for logger in self.loggers:
        #     if isinstance(logger, TensorBoardLogger):
        #         tensorboard = logger.experiment
        #         tensorboard.add_text('test_text', f'this is a test - {self.global_step}', self.global_step)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, metrics = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, metrics = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    # def configure_optimizers(self) -> Dict[str, Any]:
    #     """Choose what optimizers and learning-rate schedulers to use in your optimization.
    #     Normally you'd need one. But in the case of GANs or similar you might have multiple.

    #     Examples:
    #         https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

    #     :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
    #     """
    #     optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
    #     if self.hparams.scheduler is not None:
    #         scheduler = self.hparams.scheduler(optimizer=optimizer)
    #         return {
    #             "optimizer": optimizer,
    #             "lr_scheduler": {
    #                 "scheduler": scheduler,
    #                 "monitor": "val/loss",
    #                 "interval": "epoch",
    #                 "frequency": 1,
    #             },
    #         }
    #     return {"optimizer": optimizer}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=0.001,
            weight_decay=0.0,)
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = MNISTLitModule(None, None, None, None)
