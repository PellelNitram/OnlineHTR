from tkinter import Canvas
from tkinter import END
from tkinter import filedialog
from tkinter import Event
from time import time
from pathlib import Path
from shutil import rmtree

import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

from src.data.online_handwriting_datasets import Own_Dataset
from src.data.online_handwriting_datasets import get_alphabet_from_dataset
from src.data.transforms import DictToTensor
from src.data.transforms import CharactersToIndices
from src.data.transforms import Carbune2020
from src.data.collate_functions import my_collator


def plot_strokes(strokes: list[list[(float, float, float)]]) -> None:
    plt.figure()
    for stroke in strokes:
        stroke = np.array(stroke)
        x = stroke[:, 0]
        y = stroke[:, 1]
        t = stroke[:, 2]
        plt.scatter(x, y)
    plt.gca().set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def reset_strokes(strokes, canvas, display):
    strokes.clear()
    canvas.delete("all")
    display.delete(1.0, END)

def store_strokes(strokes: list[list[(float, float, float)]], filename=None) -> None:
    
    if not filename:
        filename = filedialog.asksaveasfilename(
            title='Select a file to store strokes'
        )
    
    if not filename:
        return
    
    all_stroke_data = []
    for i_stroke, stroke in enumerate(strokes):
        for x, y, t in stroke:
            all_stroke_data.append((x, y, t, i_stroke))
    strokes = np.array(all_stroke_data)
       
    df = pd.DataFrame.from_dict({
        'x': strokes[:, 0],
        'y': strokes[:, 1],
        't': strokes[:, 2],
        'stroke_nr': strokes[:, 3],
    })
    
    df.to_csv(filename) 

def predict(strokes, display, alphabet, model, decoder, alphabet_mapper):

    if len(strokes) == 0:
        return

    TMP_FOLDER = Path('TMP')
    TMP_FOLDER.mkdir(parents=True, exist_ok=True)

    TMP_FILENAME = TMP_FOLDER / '0_TMP.csv'

    store_strokes(strokes, filename=TMP_FILENAME)

    # GET STANDALONE DATASET

    # This is the one that I really want to use
    dataset = Own_Dataset(
        Path(TMP_FOLDER),
        transform=None,
    )

    assert len(dataset) == 1

    alphabet_inference = get_alphabet_from_dataset( dataset )
    for letter in alphabet_inference: # Confirm that there are no OOV letters
        assert letter in alphabet

    transform = transforms.Compose([
        Carbune2020(),
        DictToTensor(['x', 'y', 't', 'n']),
        CharactersToIndices( alphabet ), # TODO: Why does it only work if CTI is last?
    ])

    dataset.transform = transform

    # # GET STANDALONE DATALOADER

    dl_inference = DataLoader(
        dataset=dataset,
        batch_size=64,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        collate_fn=my_collator,
    )

    assert len(dl_inference) == 1

    batch = next(iter(dl_inference))

    with torch.no_grad():
        log_softmax = model(batch['ink'].to('cuda'))

    decoded_texts = decoder(log_softmax, alphabet_mapper)
    decoded_text = decoded_texts[0]

    true_labels = batch['label_str']

    display.delete(1.0, END)
    display.insert(1.0, decoded_text, 'big')

    rmtree(TMP_FOLDER)

class Sketchpad(Canvas):
    def __init__(self, parent, strokes: list, dot_radius: int, **kwargs):
        super().__init__(parent, **kwargs)
        self.bind("<Button-1>", self.start_stroke)
        self.bind("<B1-Motion>", self.draw_and_store)
        self.bind("<ButtonRelease-1>", self.end_stroke)
        self.strokes = strokes
        self.dot_radius = dot_radius
        self.configure(bg='white')
        
    def start_stroke(self, event: Event) -> None:
        self.current_stroke = [
            (event.x, -event.y, time()),
        ]

    def draw_and_store(self, event: Event) -> None:
        x1, y1 = (event.x - self.dot_radius), (event.y - self.dot_radius)
        x2, y2 = (event.x + self.dot_radius), (event.y + self.dot_radius)
        self.create_oval(x1, y1, x2, y2, fill='#000000')
        self.current_stroke.append( (event.x, -event.y, time()) )
        
    def end_stroke(self, event: Event) -> None:
        self.draw_and_store(event)
        self.strokes.append(self.current_stroke)  