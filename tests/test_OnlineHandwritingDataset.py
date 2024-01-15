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

def test_to_disc(tmpdir):

    logger = logging.getLogger('test_to_disc')
    logger.setLevel(logging.INFO)

    ds = OnlineHandwritingDataset(logger=logger)

    test_data = [ { 'a vector': [ 0, 1 ], 'one_number': 3.2, 'str': 'this is a test string' } ]

    ds.set_data(test_data)

    ds.to_disc( tmpdir / 'test_to_disc.h5' )

def test_from_disc(tmpdir):

    logger = logging.getLogger('test_from_disc')
    logger.setLevel(logging.INFO)

    # First create and store a dataset to then load and compare against
    ds = OnlineHandwritingDataset(logger=logger)
    test_data = [ { 'a vector': [ 0, 1 ], 'one_number': 3.2, 'str': 'this is a test string' } ]
    ds.set_data(test_data)
    ds.to_disc( tmpdir / 'test_to_disc.h5' )

    # Load dataset
    ds_loaded = OnlineHandwritingDataset(logger=logger)
    ds_loaded.from_disc( tmpdir / 'test_to_disc.h5' )

    # Compare datasets
    for s, s_loaded in zip( ds.data, ds_loaded.data ):
        assert s.keys() == s_loaded.keys()
        for key in s:
            value = s[key]
            value_loaded = s_loaded[key]
            assert np.alltrue( value == value_loaded )