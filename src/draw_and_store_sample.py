from tkinter import *
from tkinter import ttk
from time import time
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.acquisition import Sketchpad
from src.data.acquisition import plot_strokes


def store_strokes(strokes: list[list[(float, float, float)]]) -> None:
    
    filename = filedialog.asksaveasfilename(
        title='Select a file to store strokes'
    )
    
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

global_strokes = []

root = Tk()
root.geometry("1024x512")
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

DOT_RADIUS = 3
sketch = Sketchpad(root, global_strokes, DOT_RADIUS)
sketch.grid(column=0, row=0, sticky=(N, W, E, S))

plot_button = Button(root, text="Plot strokes", command=lambda: plot_strokes(global_strokes))
plot_button.place(x=50,y=50)

store_button = Button(root, text="Store strokes", command=lambda: store_strokes(global_strokes))
store_button.place(x=200,y=50)

root.mainloop()

# TODO: Make sure that stored data format is same as IAMonDB!
# TODO: Later build an app that allows training data acquisition! It would use simple JSON. The UI will then include a button to clean the current sample and a text box to input the text.
