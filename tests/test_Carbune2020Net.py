from pathlib import Path
import logging

import pytest

from src.models.components.carbune2020_net import Carbune2020Net


@pytest.mark.martin
def test_construction():

    logger = logging.getLogger('test_construction')
    logger.setLevel(logging.INFO)

    net = Carbune2020Net()