from pathlib import Path

import pytest
import numpy as np

from src.data.transforms import DictToTensor


@pytest.mark.martin
def test_construction():

    DictToTensor(channel_names=['x', 'a', 'z'])