from typing import List

import torch


class GreedyCTCDecoder(torch.nn.Module):
    """TODO.

    Loosely based on [this](https://pytorch.org/audio/main/tutorials/asr_inference_with_ctc_decoder_tutorial.html#greedy-decoder).
    """

    def __init__(self):
        super().__init__()

    def forward(self, log_probabilities: torch.Tensor, alphabet_mapper) -> List[str]:
        """TODO. Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
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