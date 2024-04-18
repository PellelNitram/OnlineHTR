from tkinter import Tk
from tkinter import Button
from tkinter import N, W, E, S
import argparse

from src.data.acquisition import Sketchpad
from src.data.acquisition import plot_strokes
from src.data.acquisition import store_strokes


def parse_cli_args() -> dict:
    """Parse command-line arguments for this script."""

    parser = argparse.ArgumentParser()
    parser.add_argument('-dot-radius', '--dot-radius', type=int,
                        default=3)
    args = parser.parse_args()
    
    return vars(args)

def main(args: dict) -> None:
    """Main function of this script.

    :param args: Arguments to this main function.
    """

    global_strokes = []

    root = Tk()
    root.geometry("1024x512")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    sketch = Sketchpad(root, global_strokes, args['dot_radius'])
    sketch.grid(column=0, row=0, sticky=(N, W, E, S))

    plot_button = Button(root, text="Plot strokes",
                         command=lambda: plot_strokes(global_strokes))
    plot_button.place(x=50,y=50)

    store_button = Button(root, text="Store strokes",
                          command=lambda: store_strokes(global_strokes))
    store_button.place(x=200,y=50)

    root.mainloop()

if __name__ == '__main__':

    args = parse_cli_args()

    main(args)