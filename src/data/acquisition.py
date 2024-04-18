from tkinter import Canvas
from tkinter import END
from tkinter import filedialog
from time import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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