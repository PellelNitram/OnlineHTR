from __future__ import annotations # See https://stackoverflow.com/a/33533514 why
                                   # I use it for OnlineHandwritingDataset.map()
                                   # type annotation.

from typing import List
import os

from pathlib import Path
import logging

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.documents import XournalDocument
from src.utils.io import load_IAM_OnDB_sample


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

        The strokes are colour-coded.

        :param path: Path to store the images at. Is created if it does not exist,
                     with parents created as well and no error raised if it already
                     exists.
        :param format: The format to save the images with.
        """

        path.mkdir(parents=True, exist_ok=True)

        for i_sample, sample in enumerate( self.data ):

            file_name = path / f'{i_sample}.{format}'

            plt.figure(dpi=300)
            plt.gca().set_aspect('equal')
            plt.scatter(sample['x'], sample['y'], c=sample['stroke_nr'],
                        cmap=plt.cm.get_cmap('Set1'),
                        s=1)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'{sample["sample_name"]=}\n{sample["label"]=}')
            plt.savefig(file_name)
            plt.close()

            self.logger.info(f'to_images: Saved sample {i_sample} to {file_name}')

class XournalPagewiseDatasetPyTorch(Dataset):

    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.data = self.load_data()

    def load_data(self):

        xournal_document = XournalDocument(self.path)

        result = []

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

            result.append( {
                'x': np.array(x_data),
                'y': np.array(y_data),
                'stroke_nr': stroke_nr_data,
                'label': label,
                'sample_name': sample_name,
            } )

        return result

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample

class IAM_OnDB_Dataset(Dataset):
    """IAM-OnDB dataset implementation in PyTorch.

    These are the links to the dataset:
    - https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database
    - https://doi.org/10.1109/ICDAR.2005.132

    This class encapsulates my own version of the IAM On-DB dataset in which I fixed a few small
    samples by fixing text formatting issues.

    This is the raw dataset which can be further processed using downstream transformations.
    """

    SAMPLES_NOT_TO_STORE = [
        'z01-000z-01', # There exists no text for that sample
        'z01-000z-02', # There exists no text for that sample
        'z01-000z-03', # There exists no text for that sample
        'z01-000z-04', # There exists no text for that sample
        'z01-000z-05', # There exists no text for that sample
        'z01-000z-06', # There exists no text for that sample
        'z01-000z-07', # There exists no text for that sample
        'z01-000z-08', # There exists no text for that sample
    ]

    def __init__(self, path: Path, transform=None, limit: int=-1) -> None:
        """
        TODO: Explain how the dataset needs to be stored on disk to allow access
        to it using this present class.

        :param path: Path to dataset.
        :param limit: Limit number of loaded samples to this value if positive.
        """
        self.path = path
        self.transform = transform
        self.limit = limit
        self.data = self.load_data()
        # TODO: I'd love to add logging here to understand the skipped images

    def load_data(self) -> List:
        """
        Returns IAM-OnDB data.
         
        In `__init__`, it is saved as `self.data`.
        
        Loading is performed by parsing the XML files and reading the text files.
        """

        result = []

        ctr = 0 # Starts at 1

        ended = False

        for root, dirs, files in os.walk(self.path / 'lineStrokes-all'):

            if ended:
                break

            for f in files:
                if f.endswith('.xml'):

                    sample_name = f.replace('.xml', '')

                    if self.limit >= 0 and ctr >= self.limit:
                        ended = True
                        break

                    if sample_name in self.SAMPLES_NOT_TO_STORE:
                        continue

                    df, text_line = load_IAM_OnDB_sample(sample_name, self.path)

                    result.append( {
                        'x': df['x'].to_numpy(),
                        'y': df['y'].to_numpy(),
                        't': df['t'].to_numpy(),
                        'stroke_nr': list( df['stroke_nr'] ),
                        'label': text_line,
                        'sample_name': sample_name,
                    } )

                    ctr += 1

        return result

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def plot_sample_to_image_file(self, sample_index: int, file_path: Path) -> None:
        """Plot sample data to image file.
        
        Helpful for debugging. It uses the `__getitem__` function and thereby applies transforms.
        
        :param sample_index: Index of sample to plot.
        :param file_path: Path to store image file as. Needs to come with suffix (this is not checked).
        """

        sample = self[sample_index]

        # TODO: each stroke gets a different colour
        # TODO: strokes plotted as dots
        # TODO: title is set to label
        
        plt.figure()
        plt.scatter(
            sample['x'],
            sample['y'],
            c=sample['stroke_nr'],
            s=1,
            cmap=matplotlib.colormaps.get_cmap('Set1'),
        )
        plt.title(f"{sample['sample_name']}: {sample['label']}")
        plt.gca().set_aspect('equal')
        plt.savefig(file_path)
        plt.close()
    
def get_alphabet_from_dataset(dataset: Dataset) -> List[str]:
    alphabet = []
    for sample in dataset:
        for letter in sample['label']:
            alphabet.append(letter)
    alphabet = list( set( alphabet ) )
    alphabet = sorted( alphabet )
    return alphabet

def get_number_of_channels_from_dataset(dataset: Dataset) -> List[str]:
    """TODO.
    
    Assumes data to be stored in "ink" field and number_of_channels to come last.
    This is in agreement with both LSTM and CTC loss from PyTorch.

    The ink needs to provide the field `.shape`, for example either as numpy
    array or as PyTorch tensor.
    """
    number_of_channels = []
    for sample in dataset:
        number_of_channels.append( sample['ink'].shape[-1] )
    number_of_channels = list( set( number_of_channels ) )
    if len(number_of_channels) > 1:
        raise ValueError('the dataset features multiple number of channels.')
    return number_of_channels[0]