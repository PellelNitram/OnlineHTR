from pathlib import Path

import pytest
import numpy as np

from src.data.online_handwriting_datamodule import IAMOnDBDataModule


PATH = Path('data/datasets/IAM-OnDB') # Needs to be parameterised

@pytest.mark.martin
def test_todo():

    assert 0 == 1, "Add tests! Check `test_datamodules.py` for nice examples."