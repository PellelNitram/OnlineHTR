from __future__ import annotations # See https://stackoverflow.com/a/33533514 why
                                   # I use it for OnlineHandwritingDataset.map()
                                   # type annotation.

from pathlib import Path
import logging

import h5py
import numpy as np
import matplotlib.pyplot as plt

from src.utils.documents import XournalDocument


class OnlineHandwritingDataset:

    FAILED_SAMPLE = -1

    def __init__(self, path=None, logger=None):
        """
        A class to represent an online handwriting dataset that is to be subclassed
        to unify multiple datasets in a modular way.

        This class serves as dataset provider to other machine learning library dataset classes
        like those from PyTorch, PyTorch Lightning or HuggingFace.

        This class keeps all data in RAM memory because the existing online handwriting datasets
        tend to be relatively small so that they easily fit in RAM memory.

        The data is stored in the `data` field and is organised in a list that stores
        a dict of all features. This format is well suitable for storing time series as
        features. This class therefore only stores datasets that can fit in memory. This
        is an example for the IAMonDB format:
            data = [
                    { 'x': [...], 'y': [...], 't': [...], ..., 'label': ..., 'sample_name': ..., ... }, 
                    ...
                   ]

        The input and output data for subsequent model trainings can easily be derived
        based on the features in each sample. Each sample should have the same features.
        This is not checked.

        :param path: Path to load raw data from.
        :param logger: Logger to use. A new one is created if set to None.
        """
        self.path = path
        if logger is None:
            self.logger = logging.getLogger('OnlineHandwritingDataset')
        else:
            self.logger = logger

        self.logger.info('Dataset created')

        self.data = []

    def load_data(self) -> None:
        """
        Needs to be implemented in subclasses.
        """
        raise NotImplementedError

    def set_data(self, data):
        """
        Set the data of this instance.

        :param data: Data to set as `self.data`.
        """
        self.data = data

    def to_disc(self, path: Path) -> None:
        """
        Store OnlineHandwritingDataset to disc.

        The OnlineHandwritingDataset is stored as HDF5 file of the structure:
        - one group per sample
        - one dataset per feature; those can be a time series as well as a single value

        :param path: Path to save dataset to.
        """
        with h5py.File(path, 'w') as f:
            for i, sample in enumerate( self.data ):
                group = f.create_group(f'sample_{i}')
                for key, value in sample.items():
                    group.create_dataset(key, data=value)

    def from_disc(self, path: Path) -> None:
        """
        Load OnlineHandwritingDataset from disc.

        The dataset must be in the format that is used in `to_disc()` to save the dataset.
        The data from disc is appended to the `data` attribute.

        :param path: Path to load dataset from.
        """
        with h5py.File(path, 'r') as f:
            for group_name in f:
                group = f[group_name]
                storage = {}
                for feature in group:
                    feature_dataset = group[feature]
                    value = feature_dataset[()]
                    if type(value) == bytes: # Convert bytes to string
                        value = value.decode('utf-8')
                    storage[feature] = value
                self.data.append(storage)

    def map(self, fct, logger=None) -> OnlineHandwritingDataset:
        """
        Applies a function to each sample and creates a new Dataset based on that.

        If the function indicates that the transformation of the sample has failed,
        then it is not added to the list of mapped samples.

        :param fct: The function that is applied. Its signature is `fct(sample)` with
                    `sample` being an element from `self.data`.
        :param logger: Logger that is used for resulting new dataset.
        :returns: New dataset.
        """
        new_dataset = OnlineHandwritingDataset(logger)
        data = []
        for sample in self.data:
            sample_mapped = fct( sample )
            if sample_mapped != self.FAILED_SAMPLE:
                data.append( sample_mapped )
        new_dataset.set_data( data )
        return new_dataset

    def fit_bezier_curve(self):
        """
        TODO: Implement it.

        Idea: Fit bezier curves recursively just as [Carbune2020] does.
        """
        raise NotImplementedError

    def to_images(self, path: Path, format: str = 'jpg') -> None:
        """
        Store dataset as images.

        :param path: Path to store the images at. Is created if it does not exist.
        :param format: The format to save the images with.

        Needs to be implemented in subclasses.
        """
        raise NotImplementedError

    def visualise(self):
        """
        TODO: Implement it.

        Idea: some visualisation methods, e.g. to plot image and also animated 2d and 3d video
        """
        raise NotImplementedError

class IAMonDB_Dataset(OnlineHandwritingDataset):

    # TODO: Should be compatible with the plain IAMonDB
    #       folder structure.

    pass

class XournalPagewiseDataset(OnlineHandwritingDataset):
    """
    Load an online text dataset from pages of a Xournal file.

    This class allows easy testing on real data.
    """

    # TODO: Given that I extend the Dataset, how do I extend the documentation?

    def load_data(self) -> None:
        """
        Loads a page-wise Xournal-based dataset.
        
        Loading is performed by constructing an `XournalDocument` instance and reading the
        data from there in line with the data format expected by `OnlineHandwritingDataset`
        class.

        Note: There is no time channel available.

        Data storage format is explained in the example dataset file and data generally
        starts on page 2.
        """

        self.logger.info('load_data: Start')

        xournal_document = XournalDocument(self.path)

        for i_page in range(1, len( xournal_document.pages )):

            page = xournal_document.pages[i_page]

            sample_name = page.layers[0].texts[0].text.replace('sample_name: ', '').strip()
            label = page.layers[0].texts[1].text.replace('label: ', '').strip()

            x_data = []
            y_data = []
            stroke_nr_data = []

            stroke_nr = 0
            for stroke in page.layers[0].strokes:
                assert len(stroke.x) == len(stroke.y)
                for i_point in range( len(stroke.x) ):
                    x_data.append( +stroke.x[i_point] )
                    y_data.append( -stroke.y[i_point] )
                    stroke_nr_data.append( stroke_nr )
                stroke_nr += 1

            self.data.append( {
                'x': np.array(x_data),
                'y': np.array(y_data),
                'stroke_nr': stroke_nr_data,
                'label': label,
                'sample_name': sample_name,
            } )

            self.logger.info(f'load_data: Stored {sample_name=}')

        self.logger.info(f'load_data: Finished')

    def to_images(self, path: Path, format: str = 'jpg') -> None:
        """
        Store dataset as images.

        :param path: Path to store the images at. Is created if it does not exist,
                     with parents created as well and no error raised if it already
                     exists.
        :param format: The format to save the images with.
        """

        path.mkdir(parents=True, exist_ok=True)

        for i_sample, sample in enumerate( self.data ):

            file_name = path / f'{i_sample}_{sample["sample_name"]}.{format}'

            plt.figure(dpi=300)
            plt.gca().set_aspect('equal')
            plt.scatter(sample['x'], sample['y'], c=sample['stroke_nr'],
                        cmap=plt.cm.get_cmap('Set1'),
                        s=1)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'{sample["sample_name"]=}')
            plt.savefig(file_name)
            plt.close()

            print(file_name)

            raise NotImplementedError