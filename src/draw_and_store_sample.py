from tkinter import *
from tkinter import ttk
from time import time
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.acquisition import Sketchpad
from src.data.acquisition import plot_strokes
from src.data.acquisition import store_strokes


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