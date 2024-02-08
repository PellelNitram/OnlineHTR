from pathlib import Path
import logging

import pytest
import torch

from src.models.components.carbune2020_net import Carbune2020NetAttempt1


# Note: I only test functions that I overwrote myself.

@pytest.mark.martin
def test_construction():

    net = Carbune2020NetAttempt1(
        number_of_channels=4,
        nodes_per_layer=64,
        number_of_layers=3,
        dropout=0.25,
        alphabet=['a', 'b', 'c',],
    )

@pytest.mark.martin
def test_forward():

    alphabet = ['a', 'b', 'c', 'd']
    net = Carbune2020NetAttempt1(
        number_of_channels=4,
        nodes_per_layer=64,
        number_of_layers=3,
        dropout=0.25,
        alphabet=alphabet,
    )

    # Construct synthetic data
    time_series_length = 13
    batch_size = 32
    number_input_channels = 4
    batched_sample = torch.randn(time_series_length,
                                 batch_size,
                                 number_input_channels)

    # Call forward method
    result = net(batched_sample)