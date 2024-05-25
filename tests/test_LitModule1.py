from pathlib import Path
import logging

import pytest
import torch

from src.models.carbune_module import LitModule1


# Note: I only test functions that I overwrote myself.

@pytest.mark.installation
def test_construction():
    """Test the construction of a `LitModule1` instance.

    This test ensures that a `LitModule1` object is created correctly using
    a set of example parameters.
    """

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

@pytest.mark.installation
def test_forward():

    alphabet = ['a', 'b', 'c', 'd']
    net = LitModule1(
        number_of_channels=4,
        nodes_per_layer=64,
        number_of_layers=3,
        dropout=0.25,
        alphabet=alphabet,
        decoder=None,
        optimizer=None,
        scheduler=None,
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

    # Tests
    assert result.shape == ( time_series_length, batch_size, len(alphabet)+1 )

    assert_sum = torch.all(
        torch.abs(
            torch.sum( torch.exp( result ), axis=2)
             - 1
        ) < 3e-7
    )
    assert assert_sum.item()