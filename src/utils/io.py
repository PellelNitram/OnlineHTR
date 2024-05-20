import xml.etree.ElementTree as ET
from pathlib import Path
import json

import pandas as pd


def load_df_iam_ondb(path: Path) -> pd.DataFrame:
    """
    Load IAM OnDB strokes file as pd.DataFrame.

    Does not read any meta data.
    
    :param path: The path to the XML strokes file.
    :returns: pd.DataFrame with strokes stored in columns "x", "y", "t" and "stroke_nr".
    """
    tree = ET.parse(path)
    root = tree.getroot()

    for element in root:
        if element.tag == 'StrokeSet':
            stroke_set = element
            break

    data = { 'x': [], 'y': [], 't': [], 'stroke_nr': [] }

    stroke_nr = 0
    for stroke in stroke_set:
        for point in stroke:
            data['x'].append(float( point.attrib['x'] ))
            data['y'].append(float( point.attrib['y'] ))
            data['t'].append(float( point.attrib['time'] ))
            data['stroke_nr'].append(stroke_nr)
        stroke_nr += 1

    df = pd.DataFrame.from_dict(data)

    assert df['stroke_nr'].max() + 1 == len(stroke_set)

    return df

def load_IAM_OnDB_text_line(path: Path, line_nr: int) -> str:
    """
    Load text line of IAM OnDB sample.

    :param path: Path to lines file.
    :param line_nr: Number of line to extract. This is a 0-indexed value.
    :returns: The text line.
    """

    with open(path, 'r') as f:
        all_lines = [ xx.strip() for xx in f.readlines() ]

    for ii, line in enumerate( all_lines ):
        if line == 'CSR:':
            i_start = ii+1
            break

    all_lines = all_lines[i_start:]
    all_lines = [ xx for xx in all_lines if len(xx) > 0 ]

    return all_lines[line_nr]

def load_IAM_OnDB_sample(sample, base_path):
    """
    Load IAM On-DB data sample.

    With sample consisting of time series and text line as ground truth.

    :param sample: Sample code according to IAM On-DB encoding.
    :param base_path: Base path of IAM On-DB.
    :returns: (df, text_line) with df as time series.
    """

    SPLITTER = '-'

    code1, code2, code3 = sample.split(SPLITTER)
    code2_no_letters = ''.join( [ letter for letter in code2 if letter in '0123456789' ] )

    strokes_file = Path( base_path / f'lineStrokes-all/lineStrokes/{code1}/{code1}{SPLITTER}{code2_no_letters}/{code1}{SPLITTER}{code2}{SPLITTER}{code3}.xml' )
    text_line_file = Path( base_path / f'ascii-all/ascii/{code1}/{code1}{SPLITTER}{code2_no_letters}/{code1}{SPLITTER}{code2}.txt' )

    df = load_df_iam_ondb(strokes_file)
    df['y'] *= -1 # Correct text direction to natural direction facing upwards

    text_line = load_IAM_OnDB_text_line(text_line_file, int(code3)-1)

    return df, text_line

def store_alphabet(outfile: Path, alphabet: list[str]) -> None:
    """Stores the alphabet as JSON.

    :param outfile: The path to store the alphabet under.
    :param alphabet: The alphabet.
    """
    with open(outfile, 'w') as f:
        json.dump({'alphabet': alphabet}, f, indent=4)

def load_alphabet(infile: Path) -> list[str]:
    """Load alphabet from JSON.

    :param infile: The path to load the alphabet from.
    :returns: The alphabet as list of strings.
    """
    with open(infile, 'r') as f:
        json_data = json.load(f)
    return json_data['alphabet']

def get_best_checkpoint_path(checkpoints_path: Path) -> Path:
    """Get best checkpoint based on filename.

    The best checkpoint is determined by picking the checkpoint file with the lowest
    `val_loss` value in its filename.

    :param checkpoints_path: The path that contains the checkpoints.
    :returns: The path to the best checkpoint.
    """
    best_path = ''
    best_value = 100_000.0
    for f in checkpoints_path.glob('*'):
        if 'val_loss' in f.name:
            i_val_loss = f.name.index('val_loss')
            value = float(f.name[i_val_loss:].replace('.ckpt', '').split('=')[1])
            if value < best_value:
                best_value = value
                best_path = f
    return best_path