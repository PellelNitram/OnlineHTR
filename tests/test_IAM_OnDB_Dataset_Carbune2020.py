from pathlib import Path

import pytest
import numpy as np

from src.data.online_handwriting_datasets import IAM_OnDB_Dataset_Carbune2020


PATH = Path('data/datasets/IAM-OnDB') # Needs to be parameterised

@pytest.mark.martin
def test_todo():

    assert 0 == 1, "Add tests! Check if samples are indeed transformed ones. Check for all of the samples!"