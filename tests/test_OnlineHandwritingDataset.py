from pathlib import Path
import logging

import numpy as np

from src.data.online_handwriting_datasets import OnlineHandwritingDataset


def test_construction():

    logger = logging.getLogger('test_construction')
    logger.setLevel(logging.INFO)

    ds = OnlineHandwritingDataset(logger=logger)

def test_set_data():

    logger = logging.getLogger('test_set_data')
    logger.setLevel(logging.INFO)

    ds = OnlineHandwritingDataset(logger=logger)

    test_data = [ {'a': [ 0, 1 ], 'b': [ 2, 3 ]} ]

    ds.set_data(test_data)

    for d_ds, d_test in zip( ds.data, test_data ):
        assert d_ds.keys() == d_test.keys()
        for k in d_ds.keys():
            assert d_ds[k] == d_test[k]