from pathlib import Path
import logging

import pytest

from src.data.online_handwriting_datasets import XournalPagewiseDataset


@pytest.mark.martin
def test_construction():

    logger = logging.getLogger('test_construction')
    logger.setLevel(logging.INFO)

    ds = XournalPagewiseDataset(
        path=Path.home() / Path('data/code/carbune2020_implementation/data/datasets/2024-01-20-xournal_dataset.xoj'),
        logger=logger
    )

@pytest.mark.martin
def test_load_data():

    logger = logging.getLogger('test_load_data')
    logger.setLevel(logging.INFO)

    ds = XournalPagewiseDataset(
        path=Path.home() / Path('data/code/carbune2020_implementation/data/datasets/2024-01-20-xournal_dataset.xoj'),
        logger=logger
    )

    ds.load_data()

    assert len( ds.data ) == 1

    assert min( ds.data[0]['stroke_nr'] ) == 0
    assert max( ds.data[0]['stroke_nr'] ) == 9

    assert len( ds.data[0]['stroke_nr'] ) == len( ds.data[0]['x'] ) == len( ds.data[0]['y'] ) == 1026

    assert ds.data[0]['label'] == 'Hello World!'

    assert ds.data[0]['sample_name'] == 'hello_world'

@pytest.mark.martin
def test_to_images(tmp_path):

    logger = logging.getLogger('test_to_images')
    logger.setLevel(logging.INFO)

    ds = XournalPagewiseDataset(
        path=Path.home() / Path('data/code/carbune2020_implementation/data/datasets/2024-01-20-xournal_dataset.xoj'),
        logger=logger
    )
    ds.load_data()

    # TODO PATCH
    tmp_path = Path.home() / '00_TMP_2024-01-28_deleteMe'

    ds.to_images(path=tmp_path, format='jpg')

    print(tmp_path)

    raise NotImplementedError