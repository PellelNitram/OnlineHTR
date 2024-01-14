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
    # - (abstract) load()
    # - some visualisation methods - plot image and also animated 2d and 3d video
    # - fit to bezier curve
    # - (maybe) save to disk

class IAMonDB_Dataset(OnlineHandwritingDataset):

    # TODO: Should be compatible with the plain IAMonDB
    #       folder structure.

    pass

class XournalPagewiseDataset(OnlineHandwritingDataset):

    # TODO: load an online text from pages of a Xournal file

    # TODO: This class allows easy testing on real data.

    pass
