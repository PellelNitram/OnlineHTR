from typing import List

import torch


class GreedyCTCDecoder(torch.nn.Module):
    """Greedy CTC (Connectionist Temporal Classification) Decoder implementation.

    Loosely based on [this tutorial](https://pytorch.org/audio/main/tutorials/asr_inference_with_ctc_decoder_tutorial.html#greedy-decoder).
    """

    def __init__(self):
        super().__init__()

    def forward(self, log_probabilities: torch.Tensor, alphabet_mapper) -> List[str]:
        """Decode log probabilities into text using a greedy algorithm.

        :param log_probabilities: Log probability tensor of emissions.
            Shape `[seq_length, batch_size, num_labels]`.
        :type log_probabilities: torch.Tensor
        :param alphabet_mapper: An instance of AlphabetMapper class to map label indices to characters.
        :type alphabet_mapper: AlphabetMapper
        :return: List of decoded transcripts for each input sequence.
        :rtype: List[str]
        """

        probabilities = torch.exp(log_probabilities)

        batch_size = probabilities.shape[1]

        decoded_texts = []

        for i_batch in range(batch_size):
            p = probabilities[:, i_batch, :]
            indices = torch.argmax(p, dim=1)
            indices = torch.unique_consecutive(indices, dim=-1)
            indices = [i for i in indices if i != alphabet_mapper.BLANK_INDEX]
            joined = "".join([alphabet_mapper.index_to_character(i) for i in indices])
            decoded_texts.append(joined)

        return decoded_texts