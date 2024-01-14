from pathlib import Path

import h5py


class OnlineHandwritingDataset:

    # TODO: Should be compatible with PyTorch dataset and/or
    #       HuggingFace Dataset where I prefer the former.
    #
    #       The existing online handwriting datasets tend to be
    #       relatively small so that they easily fit in RAM memory.
    #
    #       Caching them is useful nevertheless.

    """
    TODO

    This class serves as dataset provider to other machine learning library dataset classes
    like those from PyTorch or PyTorch Lightning.
    """

    pass

    # Methods that might be useful:
    # - some visualisation methods - plot image and also animated 2d and 3d video
    # - fit to bezier curve

    def load_data(self) -> None:
        """
        Needs to be implemented in subclasses.
        """
        raise NotImplementedError

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

class IAMonDB_Dataset(OnlineHandwritingDataset):

    # TODO: Should be compatible with the plain IAMonDB
    #       folder structure.

    pass

class XournalPagewiseDataset(OnlineHandwritingDataset):

    # TODO: load an online text from pages of a Xournal file

    # TODO: This class allows easy testing on real data.

    pass
