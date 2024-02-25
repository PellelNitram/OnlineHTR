# Add transforms for datasets here.

# Datasets will return dicts with keys x, y, (optionally) t, stroke_nr, label, sample_name.

# Transforms that I need for sure:
# - As input a stack of (x, y, stroke_nr) and as output label.
# - Switch axis to fit axis of what model reads as input.
# - same as above but with t
# - label to alphabet
# - lower text in label
# - same as above but with transforms like differences, differences after equidistance transform, Bezier curves
# - left alone vs lowered label alphabet
# - TODO: What to code as transform vs what to code as transform?

import torch
import numpy as np

from src.data.tokenisers import AlphabetMapper


class TwoChannels(object):
    """TODO.

    Return { ink: (x, y), label: label } where label is list of letters.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        """TODO.

        :returns: { ink, label } with ink as PyTorch tensor of float type.
        """

        x = sample['x']
        y = sample['y']

        ink = np.vstack([x, y]).T

        label = sample['label']

        return {
            'ink': torch.from_numpy(ink).float(),
            'label': label,
        }


class CharactersToIndices(object):
    """TODO.

    Return { ink: (x, y), label: label } where label is list of integers. ink remains unchanged.
    """

    def __init__(self, alphabet: list):
        self.alphabet = alphabet
        self.alphabet_mapper = AlphabetMapper(alphabet)

    def __call__(self, sample):
        """TODO.

        :returns: { ink, label } with ink unmodified and label
                  as list integer indices instead of characters.
                  label is returned as torch.int64 tensor.
        """

        label = [ self.alphabet_mapper.character_to_index(c) for c in sample['label']]
        label = torch.as_tensor(label, dtype=torch.int64)

        return {
            'ink': sample['ink'],
            'label': label,
        }