from pathlib import Path

import pytest
import numpy as np

from src.data.online_handwriting_datasets import IAM_OnDB_Dataset
from src.data.transforms import Carbune2020
from src.data import FAILED_SAMPLE


PATH = Path('data/datasets/IAM-OnDB') # Needs to be parameterised

@pytest.mark.data
def test_construction_with_limit():

    limit = 5

    ds = IAM_OnDB_Dataset(path=PATH, transform=Carbune2020(), limit=limit)

    assert len(ds) == limit

    for i_sample in range(len(ds)):
        sample = ds[i_sample]

@pytest.mark.data
@pytest.mark.slow
def test_construction_no_limit():

    ds = IAM_OnDB_Dataset(path=PATH, transform=Carbune2020(), limit=-1)

    assert len(ds) == IAM_OnDB_Dataset.LENGTH

    samples_transformed = []
    for i_sample in range(len(ds)):
        samples_transformed.append( ds[i_sample] )

    # Test types: either dict or type of failed indicator variable
    type_failed = type(FAILED_SAMPLE)
    type_dict = dict
    succeeded_sample_names = []
    for sample_transformed in samples_transformed:
        assert type(sample_transformed) in [ type_dict, type_failed ]
        if type(sample_transformed) == type_dict:
            succeeded_sample_names.append( sample_transformed['sample_name'] )
    
    assert len(ds) == len(succeeded_sample_names) + len(IAM_OnDB_Dataset.SAMPLES_TO_SKIP_BC_CARBUNE2020_FAILS)

@pytest.mark.data
@pytest.mark.slow
def test_construction_no_limit_skip_carbune2020_fails():

    ds = IAM_OnDB_Dataset(path=PATH, transform=Carbune2020(), limit=-1, skip_carbune2020_fails=True)

    assert len(ds) == IAM_OnDB_Dataset.LENGTH - len(IAM_OnDB_Dataset.SAMPLES_TO_SKIP_BC_CARBUNE2020_FAILS)

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

    assert ctr == IAM_OnDB_Dataset.LENGTH - len(IAM_OnDB_Dataset.SAMPLES_TO_SKIP_BC_CARBUNE2020_FAILS)