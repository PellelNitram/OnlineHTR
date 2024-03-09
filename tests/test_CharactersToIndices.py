import torch

from src.data.transforms import CharactersToIndices


def test_instantiate():

    alphabet = [
        'a',
        'b',
        'c',
    ]

    transform = CharactersToIndices(alphabet)