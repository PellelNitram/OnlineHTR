class AlphabetMapper:
    """Helper class to map between characters and indices that
    respects blank character used by CTC loss.
    
    The blank is inserted as 0th object in a hard-coded fashion.
    """

    BLANK = '<blank>'
    BLANK_INDEX = 0

    def __init__(self, alphabet: list):
        self.alphabet = alphabet
        self.alphabet_with_blank = [ AlphabetMapper.BLANK, ] + alphabet

    def character_to_index(self, character: str) -> int:
        return self.alphabet_with_blank.index(character)

    def index_to_character(self, index: int):
        return self.alphabet_with_blank[index]
