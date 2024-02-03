import torch
from torch import nn


class Carbune2020NetAttempt1(nn.Module):
    """TODO. Check SimpleDenseNet for inspiration."""

    def __init__(
        self,
        number_of_channels: int,
        nodes_per_layer: int,
        number_of_layers: int,
        dropout: float,
        alphabet: list,
    ) -> None:
        """TODO. Check SimpleDenseNet for inspiration.

        :param number_of_channels: The number of channels per step in the time series.
        :param number_of_layers: The number of LSTM layers.
        """
        # TODO: Try to replicate also my previous TensorFlow attempt that worked (up to memory leakage).
        raise NotImplementedError # Check SimpleDenseNet for inspiration.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO. Check SimpleDenseNet for inspiration."""
        raise NotImplementedError # Check SimpleDenseNet for inspiration.
