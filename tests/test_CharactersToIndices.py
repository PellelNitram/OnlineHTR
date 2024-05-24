import torch
import pytest

from src.data.transforms import CharactersToIndices


@pytest.mark.installation
def test_instantiate():

    alphabet = [
        'a',
        'b',
        'c',
    ]

    transform = CharactersToIndices(alphabet)

@pytest.mark.installation
def test_call():

    alphabet = [
        'h',
        'e',
        'l',
        'o',
    ]

    sample = {
        'label': 'hello',
        'channel1': 'abcdef',
        'channel2': [
            1,
            2,
            4,
            3,
            5,
        ],
    }

    transform = CharactersToIndices(alphabet)

    sample_transformed = transform(sample)

    # Check that other channels remain the same
    assert    sample['channel1']  ==    sample_transformed['channel1']
    assert id(sample['channel1']) == id(sample_transformed['channel1'])

    assert    sample['channel2']  ==    sample_transformed['channel2']
    assert id(sample['channel2']) == id(sample_transformed['channel2'])

    # Check that label is correct
    assert torch.equal( sample['label'], torch.as_tensor([1, 2, 3, 3, 4], dtype=torch.int64) )
    assert sample['label'].dtype == torch.int64