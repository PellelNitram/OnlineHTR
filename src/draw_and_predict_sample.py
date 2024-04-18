from tkinter import *
from tkinter import ttk
from time import time
from tkinter import filedialog
from pathlib import Path
from shutil import rmtree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

from src.models.carbune_module import LitModule1
from src.utils.io import load_alphabet
from src.data.tokenisers import AlphabetMapper
from src.data.online_handwriting_datasets import Own_Dataset
from src.data.online_handwriting_datasets import get_alphabet_from_dataset
from src.data.transforms import DictToTensor
from src.data.transforms import CharactersToIndices
from src.data.transforms import Carbune2020
from src.data.collate_functions import my_collator


# ========
# Settings
# ========

PATH = 'logs/train/multiruns/2024-04-10_20-09-25/8'
DOT_RADIUS = 1
DOT_RADIUS = 3

# =====
# Model
# =====

BASE_PATH = Path(PATH)
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

class Sketchpad(Canvas):
    def __init__(self, parent, strokes, dot_radius, **kwargs):
        super().__init__(parent, **kwargs)
        self.bind("<Button-1>", self.start_stroke)
        self.bind("<B1-Motion>", self.draw_and_store)
        self.bind("<ButtonRelease-1>", self.end_stroke)
        self.strokes = strokes
        self.dot_radius = dot_radius
        self.configure(bg='white')
        
    def start_stroke(self, event):
        self.current_stroke = [
            (event.x, -event.y, time()),
        ]

    def draw_and_store(self, event):
        x1, y1 = (event.x - self.dot_radius), (event.y - self.dot_radius)
        x2, y2 = (event.x + self.dot_radius), (event.y + self.dot_radius)
        self.create_oval(x1, y1, x2, y2, fill='#000000')
        self.current_stroke.append( (event.x, -event.y, time()) )
        
    def end_stroke(self, event):
        self.draw_and_store(event)
        self.strokes.append(self.current_stroke)

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

def reset_strokes(strokes, canvas, display):
    strokes.clear()
    canvas.delete("all")
    display.delete(1.0, END)
    
def predict(strokes, display, alphabet):

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

    print(f'Number of samples in dataset: {len(dataset)}')

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

    print(f'Number of samples in dataloader: {len(dl_inference)}')

    batch = next(iter(dl_inference))

    with torch.no_grad():
        log_softmax = model(batch['ink'].to('cuda'))

    decoded_texts = decoder(log_softmax, alphabet_mapper)
    decoded_text = decoded_texts[0]

    true_labels = batch['label_str']

    display.delete(1.0, END)
    display.insert(1.0, decoded_text, 'big')

    rmtree(TMP_FOLDER)

global_strokes = []

root = Tk()
root.title("Draw and Predict Sample")
root.geometry("1024x512")
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

sketch = Sketchpad(root, global_strokes, DOT_RADIUS)
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

predict_button = Button(root, text="Predict!", command=lambda: predict(global_strokes, prediction_field, alphabet))
predict_button.place(x=50,y=400)

root.mainloop()

# TODO: Add status bar