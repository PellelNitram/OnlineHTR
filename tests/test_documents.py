from pathlib import Path
import logging

import numpy as np
import pytest

from src.utils import documents


# Note: I don't test the dataclasses in excess to what I have done so already for Stroke class above.

PATH = Path('data/datasets/2024-01-20-xournal_dataset.xoj')

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

@pytest.mark.martin
def test_XournalDocument():

    path = Path('data/datasets/2024-01-20-xournal_dataset.xoj')
    x_document = documents.XournalDocument( path )

    # Check document properties
    assert x_document.path == path
    assert x_document.DPI == 72
    assert len(x_document.pages) == 2

    # Check page properties
    assert x_document.pages[1].background == {'type': 'solid', 'color': 'white', 'style': 'lined'}
    assert x_document.pages[1].meta_data == {'width': '612.00', 'height': '792.00'}
    assert len( x_document.pages[1].layers ) == 1

    # Check layer properties
    assert len( x_document.pages[0].layers[0].texts ) == 3
    assert len( x_document.pages[1].layers[0].texts ) == 2
    assert len( x_document.pages[0].layers[0].strokes ) == 0
    assert len( x_document.pages[1].layers[0].strokes ) == 10

    # Check stroke properties
    assert x_document.pages[1].layers[0].strokes[0].meta_data == {
        'tool': 'pen', 'color': 'black', 'width': '1.41'
    }
    assert len( x_document.pages[1].layers[0].strokes[0].x ) == 29
    assert len( x_document.pages[1].layers[0].strokes[0].y ) == 29

    # Check text properties
    assert x_document.pages[1].layers[0].texts[0].text == 'sample_name: hello_world'
    assert x_document.pages[1].layers[0].texts[1].text == 'label: Hello World!'
    assert x_document.pages[0].layers[0].texts[0].meta_data == {
        'font': 'Sans', 'size': '12.00', 'x': '75.68', 'y': '88.34', 'color': 'black'
    }

@pytest.mark.installation
@pytest.mark.ci_cd
def test_XournalDocument_save_page_as_image(tmp_path: Path) -> None:
    """Tests `XournalDocument.save_page_as_image` method.

    This saves a page to a file so that one can inspect the correctness of the
    saved page manually. Enabling the `pytest` setting `-s` allows one to see where
    the files were saved temporarily.

    :param tmp_path: Temporary path the file is saved to; managed by `pytest`.
    :type tmp_path: Path
    """

    print()
    print(f'Samples saved at: "{tmp_path}"')
    print()

    x_document = documents.XournalDocument( PATH )

    output_path = tmp_path / 'page1.png'

    output_path_result = x_document.save_page_as_image(
        page_index=1,
        out_path=output_path,
        black_white=False,
        dpi=72.0,
    )

    assert output_path == output_path_result