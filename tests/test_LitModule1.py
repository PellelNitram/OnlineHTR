from pathlib import Path
import logging

import pytest
import torch

from src.models.carbune_module import LitModule1


# Note: I only test functions that I overwrote myself.

@pytest.mark.martin
def test_construction():

    net = LitModule1(
        number_of_channels=4,
        nodes_per_layer=64,
        number_of_layers=3,
        dropout=0.25,
        alphabet=['a', 'b', 'c',],
        decoder=None,
        optimizer=None,
        scheduler=None,
    )