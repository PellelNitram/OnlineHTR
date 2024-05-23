from pathlib import Path

import pytest
import numpy as np

from src.data.online_handwriting_datasets import IAM_OnDB_Dataset_Carbune2020
from src.data.online_handwriting_datasets import IAM_OnDB_Dataset
from src.data.transforms import Carbune2020


PATH = Path('data/datasets/IAM-OnDB') # Needs to be parameterised

@pytest.mark.parametrize(
    "limit",
    [
        pytest.param(10,
                     marks=[pytest.mark.data,]),
        pytest.param(-1,
                     marks=[pytest.mark.data, pytest.mark.slow,], id='all'),
    ])
def test_correctness(limit: int) -> None:
    """Tests the correctness of the `IAM_OnDB_Dataset_Carbune2020` dataset processing 
    by comparing it against the raw `IAM_OnDB_Dataset` with the `Carbune2020` transform.

    This function checks that the number of samples in both datasets are equal, and then
    verifies that each corresponding sample in both datasets has the same `sample_name`
    and `label`. It also ensures that the numerical data fields `x`, `y`, `t` and `n`
    are identical.

    :param limit: The number of samples to include in the datasets for comparison.
    :type limit: int
    :raises AssertionError: If the number of samples in either dataset are not equal.
    :raises AssertionError: If the number of samples in either dataset are not equal to limit if limit >= 0.
    :raises AssertionError: If any of the `sample_name` or `label` fields differ between corresponding samples.
    :raises AssertionError: If the numerical fields `x`, `y`, `t` and `n` differ between corresponding samples.
    """

    ds_carbune = IAM_OnDB_Dataset_Carbune2020(path=PATH, transform=None, limit=limit)
    ds_raw = IAM_OnDB_Dataset(path=PATH, transform=Carbune2020(), limit=limit, skip_carbune2020_fails=True)

    assert len(ds_carbune) == len(ds_raw)
    if limit >= 0:
        assert len(ds_carbune) == len(ds_raw) == limit

    for i_sample in range(limit):

        sample_carbune = ds_carbune[i_sample]
        sample_raw = ds_raw[i_sample]

        assert sample_carbune['sample_name'] == sample_raw['sample_name']
        assert sample_carbune['label'] == sample_raw['label']
        assert np.abs(sample_carbune['x']-sample_raw['x']).max() == 0.0
        assert np.abs(sample_carbune['y']-sample_raw['y']).max() == 0.0
        assert np.abs(sample_carbune['t']-sample_raw['t']).max() == 0.0
        assert np.abs(sample_carbune['n']-sample_raw['n']).max() == 0.0