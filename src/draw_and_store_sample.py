from tkinter import *
from tkinter import ttk
from time import time
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Sketchpad(Canvas):
    def __init__(self, parent, strokes, **kwargs):
        super().__init__(parent, **kwargs)
        self.bind("<Button-1>", self.start_stroke)
        self.bind("<B1-Motion>", self.draw_and_store)
        self.bind("<ButtonRelease-1>", self.end_stroke)
        self.strokes = strokes
        
    def start_stroke(self, event):
        self.current_stroke = [
            (event.x, -event.y, time()),
        ]

    def draw_and_store(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
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

sketch = Sketchpad(root, global_strokes)
sketch.grid(column=0, row=0, sticky=(N, W, E, S))

plot_button = Button(root, text="Plot strokes", command=lambda: plot_strokes(global_strokes))
plot_button.place(x=50,y=50)

store_button = Button(root, text="Store strokes", command=lambda: store_strokes(global_strokes))
store_button.place(x=200,y=50)

root.mainloop()

# TODO: Make sure that stored data format is same as IAMonDB!
# TODO: Later build an app that allows training data acquisition! It would use simple JSON. The UI will then include a button to clean the current sample and a text box to input the text.
