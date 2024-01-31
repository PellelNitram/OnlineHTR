import torch
from torch import nn


class Carbune2020Net(nn.Module):
    """TODO. Check SimpleDenseNet for inspiration."""

    def __init__(
        number_of_channels: int,
        nodes_per_layer: int,
        number_of_layers: int,
        dropout: float,
        alphabet: list,
    ) -> None:
        """TODO. Check SimpleDenseNet for inspiration."""
        # TODO: Try to replicate also my previous TensorFlow attempt that worked (up to memory leakage).
        raise NotImplementedError # Check SimpleDenseNet for inspiration.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO. Check SimpleDenseNet for inspiration."""
        raise NotImplementedError # Check SimpleDenseNet for inspiration.