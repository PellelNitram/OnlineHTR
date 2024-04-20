# TODO: Add function here to allow usage of model with CTC loss.

import torch
import numpy as np


def ctc_loss_collator(batch):
    """TODO.

    TODO: Check https://github.com/pytorch/pytorch/blob/main/torch/utils/data/_utils/collate.py#L216 to check
    if this function here is efficient and good enough and how it compares to PyTorch authors' work.
    """

    batch_size = len(batch)

    labels = [ sample['label'] for sample in batch ]
    inks = [ sample['ink'] for sample in batch ]

    label_lengths = [ len(label) for label in labels ]
    max_label_length = max(label_lengths)

    ink_lengths = [ ink.shape[0] for ink in inks ]
    max_ink_length = max(ink_lengths)

    channel_dimensions = [ ink.shape[1] for ink in inks ]
    assert np.allclose( channel_dimensions[0], channel_dimensions )
    channel_dimension = channel_dimensions[0]

    ink_types = [ ink.dtype for ink in inks ]
    for ink_type in ink_types[1:]:
        assert ink_types[0] == ink_type
    ink_type = ink_types[0]

    # This is Log_probs parameter in CTC loss
    X_tensor_batched = torch.zeros((max_ink_length, batch_size, channel_dimension), dtype=ink_type)
    for i_batch in range(batch_size):
        X_tensor_batched[:ink_lengths[i_batch], i_batch, :] = inks[i_batch]

    # This is Targets parameter in CTC loss
    y_tensor_batched = torch.zeros((batch_size, max_label_length), dtype=torch.int64)
    for i_batch in range(batch_size):
        y_tensor_batched[i_batch, :label_lengths[i_batch]] = labels[i_batch]

    # Return the data as needed by CTC loss, see here https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html.
    return {
        'ink': X_tensor_batched,
        'label': y_tensor_batched,
        'ink_lengths': ink_lengths,
        'label_lengths': label_lengths,
        'label_str': [ sample['label_str'] for sample in batch ],
     }