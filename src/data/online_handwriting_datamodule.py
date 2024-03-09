from typing import Any, Dict, Optional, Tuple
from pathlib import Path

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from src.data.online_handwriting_datasets import XournalPagewiseDatasetPyTorch
from src.data.online_handwriting_datasets import IAM_OnDB_Dataset
from src.data.transforms import TwoChannels
from src.data.transforms import CharactersToIndices
from src.data.collate_functions import my_collator
from src.data.transforms import TwoChannels
from src.data.tokenisers import AlphabetMapper
from src.data.online_handwriting_datasets import get_alphabet_from_dataset
from src.data.online_handwriting_datasets import get_number_of_channels_from_dataset


class SimpleOnlineHandwritingDataModule(LightningDataModule):
    """`LightningDataModule` for online handwriting datasets using `OnlineHandwritingDataset`."""

    def __init__(
        self,
        alphabet: list,
        data_dir: str = "data/", # TODO: Should I supply the path to build
                                 #       OnlineHandwritingDataset in here or
                                 #       should I supply the OnlineHandwritingDataset.
                                 #       I prefer the latter although it requires
                                 #       to adapt the configs accordingly to load
                                 #       correct OnlineHandwritingDataset.
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # TODO: Fix format of online_handwriting_datasets here by switching
        #       channels. Alternatively, do so in o_h_d (?). I do so b/c model
        #       consumes it as such, hence o_h_d doesn't need to know about.
        #       hence rather here. maybe w/ switch or parameter setting?

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

        self.transform = transforms.Compose([
            TwoChannels(),
            CharactersToIndices(alphabet),
        ])

    def prepare_data(self) -> None:
        """Not implemented because no data needs to be downloaded."""
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data.
        
        Set variables `self.data_train`, `self.data_val` and `self.data_test`.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """

        # 0. Transform OnlineHandwritingDataset according to settings

        # 1. Create dataset from 0 data.
        # Build PyTorch Dataset from my OnlineHandwritingDataset by performing transform on my dataset; both can be configured later on

        # 2. perform train/val/test splits

        # 3. build vocabulary

        # load and split datasets only if not loaded already
        # TODO
        # if not self.data_train and not self.data_val and not self.data_test:
        #     trainset = MNIST(self.hparams.data_dir, train=True, transform=self.transforms)
        #     testset = MNIST(self.hparams.data_dir, train=False, transform=self.transforms)
        #     dataset = ConcatDataset(datasets=[trainset, testset])
        #     self.data_train, self.data_val, self.data_test = random_split(
        #         dataset=dataset,
        #         lengths=self.hparams.train_val_test_split,
        #         generator=torch.Generator().manual_seed(42),
        #     )

        if not self.data_train and not self.data_val and not self.data_test:

            dataset = XournalPagewiseDatasetPyTorch(self.hparams.data_dir, transform=self.transform)
            dataset = IAM_OnDB_Dataset(
                    Path('data/datasets/IAM-OnDB'),
                    transform=self.transform,
                    limit=-1000,
            )

            if sum(self.hparams.train_val_test_split) > len(dataset):
                raise RuntimeError(
                    f"Dataset (len={len(dataset)}) too short for requested splits ({self.hparams.train_val_test_split})."
                )
            
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )
            
    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=my_collator,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=my_collator,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=my_collator,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

class IAMOnDBDataModule(LightningDataModule):
    """TODO."""

    # TODO: Write test!

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        limit: int = -1,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `IAMOnDBDataModule`.

        TODO.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data.
        
        Set variables `self.data_train`, `self.data_val` and `self.data_test`.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """

        if not self.data_train and not self.data_val and not self.data_test:

            if sum(self.hparams.train_val_test_split) > len(dataset):
                raise RuntimeError(
                    f"Dataset (len={len(dataset)}) too short for requested splits ({self.hparams.train_val_test_split})."
                )

            dataset = IAM_OnDB_Dataset(
                    Path(self.hparams.data_dir),
                    transform=None,
                    limit=self.hparams.limit,
            )

            self.alphabet = get_alphabet_from_dataset( dataset )
            self.alphabet_mapper = AlphabetMapper( self.alphabet )

            self.number_of_channels = get_number_of_channels_from_dataset( dataset )

            # TODO: Add transforms as parameter that are then used in setup. So that they
            #       can be parameterised w/ Hydra later on.
            transform = transforms.Compose([
                TwoChannels(),
                CharactersToIndices( self.alphabet ),
            ])

            dataset.transform = transform
            
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )
            
    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=my_collator,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=my_collator,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=my_collator,
        )