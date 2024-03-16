from pathlib import Path

import pytest
import numpy as np

from src.data.transforms import DictToTensor


@pytest.mark.martin
def test_construction():

    DictToTensor(channel_names=['x', 'a', 'z'])

@pytest.mark.martin
def test_call():

    t = DictToTensor(channel_names=['x', 'z'])

    sample = {
        'a': [0.4, -1.1, 4.7],
        'label': 'foo bar',
        'x': [3.9, 2.4, 4.5],
        'y': [0.1, 2.4, 3.8],
        'z': [5.2, 4.1, 8.5],
    }

    sample_transformed = t(sample)

    assert sample_transformed['label'] == sample['label']
    assert sample_transformed['ink'].shape == (3, 2)
    assert np.allclose( sample_transformed['ink'][:, 0], sample['x'] )
    assert np.allclose( sample_transformed['ink'][:, 1], sample['z'] )