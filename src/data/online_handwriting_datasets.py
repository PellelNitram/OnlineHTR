from __future__ import annotations # See https://stackoverflow.com/a/33533514 why
                                   # I use it for OnlineHandwritingDataset.map()
                                   # type annotation.

from pathlib import Path
import logging

import h5py

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

    # TODO: load an online text from pages of a Xournal file

    # TODO: This class allows easy testing on real data.

    # TODO: Given that I extend the Dataset, how do I extend the documentation?

    def load_data(self):
        """
        Loads a page-wise Xournal-based dataset.
        
        Loading is performed by parsing the XML files and reading the text files.
        """

        raise NotImplementedError

        self.logger.info('load_data: Start')

        xournal_document = XournalDocument(self.path)

        sample_name = xournal_document.pages[1].layers[0].texts[0].text.replace('sample_name: ', '').strip()
        label = xournal_document.pages[1].layers[0].texts[1].text.replace('label: ', '').strip()

        print(sample_name, label)

        # ctr = 0 # Starts at 1

        # for root, dirs, files in os.walk(self.path / 'lineStrokes-all'):
        #     for f in files:
        #         if f.endswith('.xml'):

        #             ctr += 1

        #             sample_name = f.replace('.xml', '')

        #             self.logger.debug(f'Process {sample_name=} ({ctr})')

        #             if sample_name in self.SAMPLES_NOT_TO_STORE:
        #                 self.logger.warning(f'Skipped: {sample_name=}')
        #                 continue

        #             df, text_line = load_sample(sample_name, self.path)

        #             self.data.append( {
        #                 'x': df['x'].to_numpy(),
        #                 'y': df['y'].to_numpy(),
        #                 't': df['t'].to_numpy(),
        #                 'stroke_nr': list( df['stroke_nr'] ),
        #                 'label': text_line,
        #                 'sample_name': sample_name,
        #             } )

        # TODO: Check stroke_nr channel

        # TODO: check if x, y, t, stroke_nr are a single time series?

        self.logger.info(f'load_data: Finished')