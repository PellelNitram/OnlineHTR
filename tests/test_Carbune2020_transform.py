from pathlib import Path

import pytest
import numpy as np

from src.data.online_handwriting_datasets import IAM_OnDB_Dataset
from src.data.transforms import Carbune2020
from src.data import FAILED_SAMPLE


PATH = Path('data/datasets/IAM-OnDB') # Needs to be parameterised

@pytest.mark.martin
def test_construction_with_limit():

    limit = 5

    ds = IAM_OnDB_Dataset(path=PATH, transform=Carbune2020(), limit=limit)

    assert len(ds) == limit

    sample = ds[0]

@pytest.mark.martin
@pytest.mark.slow
def test_construction_no_limit():

    ds = IAM_OnDB_Dataset(path=PATH, transform=Carbune2020(), limit=-1)

    assert len(ds) == 12187

    samples_transformed = []
    for i_sample in range(len(ds)):
        samples_transformed.append( ds[i_sample] )

    # Test types: either dict or type of failed indicator variable
    type_failed = type(FAILED_SAMPLE)
    type_dict = dict
    ctr = 0
    for sample_transformed in samples_transformed:
        assert type(sample_transformed) in [ type_dict, type_failed ]
        if type(sample_transformed) == dict:
            ctr += 1
    
    assert ctr == 12120