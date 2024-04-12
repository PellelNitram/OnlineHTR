from pathlib import Path

import pytest
import numpy as np

from src.data.online_handwriting_datasets import IAM_OnDB_Dataset
from src.data.online_handwriting_datasets import Own_Dataset


PATH_IAM = Path('data/datasets/IAM-OnDB') # Needs to be parameterised
PATH_OWN_DATASET = Path('data/datasets/own_test_dataset') # Needs to be parameterised

# TODO: See more tests to implement here in `test_IAM_OnDB_Dataset.py`

@pytest.mark.martin
def test_correctness_against_IAM_manually(tmp_path: Path):
    # This is to ensure that the data captured by `draw_and_store_sample.py` (which is
    # then cast into `Own_Dataset` class) matches that of the IAM_OnDB_Dataset. For example
    # the orientation of the y axis can be confirmed to be the same thereby.

    # *Result*: My own dataset and `draw_and_store_sample.py` script saves the data correctly!

    print()
    print(f'Samples saved at: "{tmp_path}"')
    print()

    ds_own = Own_Dataset(PATH_OWN_DATASET)
    ds_IAM = IAM_OnDB_Dataset(PATH_IAM, limit=len(ds_own))

    # Get NR_SAMPLES reproducible random draws
    index_list = [0, 1]

    for index in index_list:
        # Own
        sample_name = ds_own[index]['sample_name']
        ds_own.plot_sample_to_image_file(index, tmp_path / Path(f'OwnDataset_{sample_name}.png'))
        # IAM
        sample_name = ds_IAM[index]['sample_name']
        ds_IAM.plot_sample_to_image_file(index, tmp_path / Path(f'IAM_{sample_name}.png'))