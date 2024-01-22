from pathlib import Path
import logging

import numpy as np
import pytest

from src.utils import documents


@pytest.mark.martin
def test_Stroke_class():

    # Create test data
    test_x = np.array([1.23, 2.34, 3.45])
    test_y = np.array([4, 5, 6])
    test_meta_data = {
        'foo': 'bar',
        42: 1337,
    }

    # Test construction of stroke
    stroke = documents.Stroke(
        test_x,
        test_y,
        test_meta_data
    )

    # Test storage
    np.testing.assert_array_equal(stroke.x, test_x)
    np.testing.assert_array_equal(stroke.y, test_y)
    np.testing.assert_array_equal(stroke.meta_data.keys(),
                                  test_meta_data.keys())
    for key in stroke.meta_data:
        assert stroke.meta_data[key] == test_meta_data[key]