from pathlib import Path
import logging

import numpy as np

from src.data.online_handwriting_datasets import OnlineHandwritingDataset


def test_construction():

    logger = logging.getLogger('test_construction')
    logger.setLevel(logging.INFO)

    ds = OnlineHandwritingDataset(logger=logger)