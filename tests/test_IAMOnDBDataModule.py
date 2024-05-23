from pathlib import Path

import pytest
import torch

from src.data.online_handwriting_datamodule import IAMOnDBDataModule


PATH = Path('data/datasets/IAM-OnDB') # Needs to be parameterised

@pytest.mark.parametrize(
    "batch_size,limit",
    [
        pytest.param(32, 100, marks=pytest.mark.data),
        pytest.param(64, 200, marks=pytest.mark.data),
        pytest.param(16, 75, marks=pytest.mark.data),
        pytest.param(512, -1, marks=[
            pytest.mark.data,
            pytest.mark.slow,
        ], id='all'),
    ])
def test_carbune2020_xytn_transform(batch_size: int, limit: int) -> None:
    """Tests the data loading and transformation functionality of `IAMOnDBDataModule` using
    the `carbune2020_xytn` transform.

    This function initializes the `IAMOnDBDataModule` with the specified batch size and data limit, 
    applies the `carbune2020_xytn` transform, and verifies the integrity and properties of the 
    resulting data loaders and batches. As such, it ensures that the necessary attributes were
    created (e.g., the dataloader objects) and that dtypes and batch sizes correctly match.

    :param batch_size: The size of the batches to be used during data loading.
    :type batch_size: int
    :param limit: The limit on the number of data points to be loaded. If non-negative, the total 
                  number of data points across train, validation, and test sets should match this limit.
    :type limit: int
    :raises AssertionError: If any of the assertions regarding the setup, data properties, and batch 
                            integrity fail.
    """

    dm = IAMOnDBDataModule(
        data_dir=PATH,
        train_val_test_split=(0.7, 0.2, 0.1),
        batch_size=batch_size,
        num_workers=0,
        limit=limit,
        pin_memory=False,
        transform="carbune2020_xytn",
    )

    assert not dm.data_train and not dm.data_val and not dm.data_test

    dm.setup()
    assert dm.dataset
    assert dm.alphabet
    assert dm.alphabet_mapper
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.number_of_channels
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    if limit >= 0:
        assert num_datapoints == limit

    batch = next(iter(dm.train_dataloader()))
    assert len(batch) == 5
    assert type(batch) == dict
    assert len(batch['ink_lengths']) == batch_size
    assert len(batch['label_lengths']) == batch_size
    assert len(batch['label_str']) == batch_size
    assert batch['ink'].shape[1] == batch_size
    assert batch['ink'].shape[2] == dm.number_of_channels
    assert batch['label'].shape[0] == batch_size
    assert batch['ink'].dtype == torch.float32 # Hard-coded for this dataset but
                                               # data collator adapts to samples'
                                               # ink data types
    assert batch['label'].dtype == torch.int64