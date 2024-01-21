"""
This module provides infrastructure to interact with documents like Xournal files.

This module was initially based on https://github.com/PellelNitram/xournalpp_htr/blob/dab111cab22891805d2eeaaececf2649eb70c58c/xournalpp_htr/documents.py.
"""

# TODO: Add tests for this module from the same source, namely xournalpp_htr.

from abc import ABC, abstractmethod
from dataclasses import dataclass
import gzip
from pathlib import Path

import numpy as np
from bs4 import BeautifulSoup as bs
import matplotlib.pyplot as plt


@dataclass
class Page:
    """Class for keeping track of document page."""
    meta_data: dict
    background: dict
    layers: list