from pathlib import Path
import logging

import pytest

from src.data.online_handwriting_datasets import XournalPagewiseDataset


PATH = Path('data/datasets/2024-02-16-xournal_dataset.xoj') # Needs to be parameterised

@pytest.mark.installation
def test_construction():

    ds = XournalPagewiseDataset(
        path=PATH,
    )

@pytest.mark.installation
def test_loaded_data():

    ds = XournalPagewiseDataset(
        path=PATH,
    )

    assert len( ds ) == len( ds.data )
    assert len( ds ) == 4

    sample = ds[0]
    assert type( sample ) == dict
    assert len( sample ) == 5
    assert min( sample['stroke_nr'] ) == 0
    assert max( sample['stroke_nr'] ) == 9
    assert len( sample['stroke_nr'] ) == len( sample['x'] ) == len( sample['y'] ) == 1026
    assert sample['label'] == 'Hello World!'
    assert sample['sample_name'] == 'hello_world'