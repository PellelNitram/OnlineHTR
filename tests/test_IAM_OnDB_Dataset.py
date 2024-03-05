from pathlib import Path

import pytest
import numpy as np

from src.data.online_handwriting_datasets import IAM_OnDB_Dataset


PATH = Path('data/datasets/IAM-OnDB') # Needs to be parameterised

@pytest.mark.martin
def test_construction_with_limit():

    limit = 5

    ds = IAM_OnDB_Dataset(path=PATH, transform=None, limit=limit)

    assert len(ds) == limit

@pytest.mark.martin
@pytest.mark.slow
def test_construction_no_limit():

    ds = IAM_OnDB_Dataset(path=PATH, transform=None, limit=-1)

    length = 12187 # Determined empirically

    assert len(ds) == length