import torch
import numpy as np


def ctc_loss_collator(batch: list[dict]) -> dict:
    """Collator to prepare batches that are compatible with CTC loss.

    This collator is used in `torch.utils.data.DataLoader` instances.

    :param batch: List of samples that are converted into a batch.
    :returns: The batch formatted to be suitable for CTC loss.
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