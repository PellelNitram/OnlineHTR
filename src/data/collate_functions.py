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

    # TODO: Infer dtype.

    X_tensor_batched = torch.zeros((max_ink_length, batch_size, channel_dimension))
    print(X_tensor_batched)
    # TODO: Fill it.

    # y_tensor_batched = ...

    # print(inks)
    # print(labels)

    # print(batch)
    exit()
    pass