from tkinter import END
from tkinter import filedialog

import matplotlib.pyplot as plt
import pandas as pd


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