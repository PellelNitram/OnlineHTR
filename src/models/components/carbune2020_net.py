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
        super().__init__()

        # Output layer to be fed into CTC loss; the output must be log probabilities
        # according to https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
        # with shape (T, N, C) where C is the number of classes (= here alphabet letters)
        # N is the batch size and T is the sequence length
        self.log_softmax = nn.LogSoftmax(dim=2) # See this documentation:
                                                # https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html#torch.nn.LogSoftmax

        # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.lstm_stack = torch.nn.LSTM(
            input_size=number_of_channels,
            hidden_size=nodes_per_layer,
            num_layers=number_of_layers,
            bias=True,
            batch_first=False,
            dropout=dropout,
            bidirectional=True,
            proj_size=0,
        )

        # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.linear = torch.nn.Linear(
            in_features=2 * nodes_per_layer, # 2 b/c bidirectional=True
            out_features=len(alphabet) + 1, # +1 for blank
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO. Check SimpleDenseNet for inspiration."""
        raise NotImplementedError # Check SimpleDenseNet for inspiration.
