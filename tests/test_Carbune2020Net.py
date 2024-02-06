from pathlib import Path
import logging

import pytest

from src.models.components.carbune2020_net import Carbune2020NetAttempt1


# Note: I only test functions that I overwrote myself.

@pytest.mark.martin
def test_construction():

    logger = logging.getLogger('test_construction')
    logger.setLevel(logging.INFO)

    net = Carbune2020NetAttempt1(
        number_of_channels=4,
        nodes_per_layer=64,
        number_of_layers=3,
        dropout=0.25,
        alphabet=['a', 'b', 'c',],
    )