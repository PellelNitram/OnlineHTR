from tkinter import *
from pathlib import Path
import argparse

import torch

from src.models.carbune_module import LitModule1
from src.utils.io import load_alphabet
from src.data.tokenisers import AlphabetMapper
from src.data.acquisition import plot_strokes
from src.data.acquisition import reset_strokes
from src.data.acquisition import store_strokes
from src.data.acquisition import Sketchpad
from src.data.acquisition import predict


def parse_cli_args() -> dict:
    """Parse command-line arguments for this script."""

    parser = argparse.ArgumentParser()
    parser.add_argument('-dot-radius', '--dot-radius', type=int,
                        default=3)
    parser.add_argument('-base-path', '--base-path', type=Path,
                        default=Path('logs/train/multiruns/2024-04-10_20-09-25/8'))
    args = parser.parse_args()
    
    return vars(args)

def main(args: dict) -> None:
    """Main function of this script.

    :param args: Arguments to this main function.
    """

    # =====
    # Model
    # =====

    BASE_PATH = Path(args['base_path'])
    CHECKPOINT_PATH = BASE_PATH / 'checkpoints/epoch000749.ckpt'

    model = LitModule1.load_from_checkpoint(CHECKPOINT_PATH)

    model.eval()

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=lambda storage, loc: storage)

    alphabet = load_alphabet(BASE_PATH / 'alphabet.json')
    alphabet_mapper = AlphabetMapper( alphabet )
    decoder = checkpoint['hyper_parameters']['decoder']

    # ==
    # UI
    # ==

    global_strokes = []

    root = Tk()
    root.title("Draw and Predict Sample")
    root.geometry("1024x512")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    sketch = Sketchpad(root, global_strokes, args['dot_radius'])
    sketch.grid(column=0, row=0, sticky=(N, W, E, S))

    prediction_field = Text(root, height=5, width=100)
    prediction_field.place(x=300, y=400)
    prediction_field.tag_configure('big', font=('Arial', 20, 'bold', 'italic'))

    plot_button = Button(root, text="Plot strokes", command=lambda: plot_strokes(global_strokes))
    plot_button.place(x=50,y=50)

    store_button = Button(root, text="Store strokes", command=lambda: store_strokes(global_strokes))
    store_button.place(x=200,y=50)

    reset_button = Button(root, text="Reset strokes", command=lambda: reset_strokes(global_strokes, sketch, prediction_field))
    reset_button.place(x=800,y=50)

    predict_button = Button(root, text="Predict!", command=lambda: predict(global_strokes, prediction_field, alphabet, model, decoder, alphabet_mapper))
    predict_button.place(x=50,y=400)

    root.mainloop()

if __name__ == '__main__':

    args = parse_cli_args()

    main(args)