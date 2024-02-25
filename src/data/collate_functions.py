# TODO: Add function here to allow usage of model with CTC loss.

import torch
import numpy as np


def my_collator(batch):

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

    print(ink_lengths)
    print(max_ink_length)
    print(channel_dimension)

    ink_types = [ ink.dtype for ink in inks ]
    for ink_type in ink_types[1:]:
        assert ink_types[0] == ink_type
    ink_type = ink_types[0]

    X_tensor_batched = torch.zeros((max_ink_length, batch_size, channel_dimension), dtype=ink_type)
    for i_batch in range(batch_size):
        X_tensor_batched[:ink_lengths[i_batch], i_batch, :] = inks[i_batch]

    # y_tensor_batched = ...

    # TODO: Also return the lengths as needed by CTC loss.

    # print(inks)
    # print(labels)

    # TODO: Check https://github.com/pytorch/pytorch/blob/main/torch/utils/data/_utils/collate.py#L216 to check
    #       if this function here is efficient and good enough and how it compares to PyTorch authors' work.

    # print(batch)
    exit()
    pass