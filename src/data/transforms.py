# Add transforms for datasets here.

# Datasets will return dicts with keys x, y, (optionally) t, stroke_nr, label, sample_name.

# Transforms that I need for sure:
# - As input a stack of (x, y, stroke_nr) and as output label.
# - Switch axis to fit axis of what model reads as input.
# - same as above but with t
# - label to alphabet
# - lower text in label
# - same as above but with transforms like differences, differences after equidistance transform, Bezier curves
# - left alone vs lowered label alphabet
# - TODO: What to code as transform vs what to code as transform?

import torch
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from src.data.tokenisers import AlphabetMapper
from src.data import FAILED_SAMPLE


class TwoChannels(object):
    """TODO.

    Return { ink: (x, y), label: label } where label is list of letters.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        """TODO.

        :returns: { ink, label } with ink as PyTorch tensor of float type.
        """

        x = sample['x']
        y = sample['y']

        ink = np.vstack([x, y]).T

        label = sample['label']

        return {
            'ink': torch.from_numpy(ink).float(),
            'label': label,
        }

class DictToTensor(object):
    """Transform channels from a dict sample to a tensor."""

    def __init__(self, channel_names: list):
        self.channel_names = channel_names

    def __call__(self, sample: dict) -> dict:
        """Get transformed sample.

        :param sample: The sample to transform.
        :returns: { ink, label } with ink as PyTorch tensor of float type
                  and label remains unchanged from `sample`.
        """

        ink = np.vstack([ sample[channel_name] for channel_name in self.channel_names ]).T

        label = sample['label']

        return {
            'ink': torch.from_numpy(ink).float(),
            'label': label,
        }

class CharactersToIndices(object):
    """TODO.

    Returns { "label": label, < all others remain unchanged > } where label is list of integers.
    """

    def __init__(self, alphabet: list):
        self.alphabet = alphabet
        self.alphabet_mapper = AlphabetMapper(alphabet)

    def __call__(self, sample: dict) -> dict:
        """TODO.

        The sample is changed in-place.

        :returns: { "label": label, < all others remain unchanged > } with label
                  as list integer indices instead of characters.
                  label is returned as torch.int64 tensor.
        """

        label = [ self.alphabet_mapper.character_to_index(c) for c in sample['label']]
        label = torch.as_tensor(label, dtype=torch.int64)

        sample['label_str'] = sample['label']

        sample['label'] = label # Updated in-place

        return sample

class SimpleNormalise(object):
    """Simple normalisation based on [Carbune2020] without linear resampling and no time.

    From not using time follows no linear resampling.
    """

    def __init__(self):
        pass

    def __call__(self, sample: dict) -> dict:
        """Transform sample according to transform described in [Carbune2020]
        but without time and without resampling.

        The keys `sample_name` and `label` are copied over without modification.
        The keys `x`, `y` & `n` are transformed. The output sample is a new dict.

        :param sample: Input sample.
        :returns: Transformed sample as output with keys `sample_name`,
                  `x`, `y`, `n`, `label`.
        """

        sample_name = sample['sample_name']

        sample_sub = {
            'x': sample['x'],
            'y': sample['y'],
            'stroke_nr': sample['stroke_nr'],
        }
        df = pd.DataFrame.from_dict(sample_sub)

        df['x'] = df['x'] - df['x'].iloc[0]
        df['y'] = df['y'] - df['y'].min()
        scale_factor = df['y'].max()
        df['x'] = df['x'] / scale_factor
        df['y'] = df['y'] / scale_factor
        if df['x'].iloc[0] != 0:
            raise ValueError('x shift failed')
        if df['y'].min() != 0.0 or df['y'].max() != 1.0:
            raise ValueError('y shift and rescaling failed')

        # Set up X (a.k.a. ink)
        df_X = pd.DataFrame(index=np.arange(df.shape[0]))
        df_X['x'] = np.nan
        df_X['y'] = np.nan
        df_X['n'] = np.nan

        # Set up x
        df_X.loc[0, 'x'] = 0
        df_X.loc[1:, 'x'] = df['x'].diff(periods=1).iloc[1:]

        # Set up y
        df_X.loc[0, 'y'] = 0
        df_X.loc[1:, 'y'] = df['y'].diff(periods=1).iloc[1:]

        # Set up n
        n_values = df['stroke_nr'].diff(periods=1)
        n_values.iloc[0] = 1.0
        df_X.loc[:, 'n'] = n_values

        # Ensure that no NaN's are left
        if df_X.isnull().values.any():
            raise ValueError('NaN value found in df_X.')

        return {
            'sample_name': sample['sample_name'],
            'x': df_X['x'].to_numpy(),
            'y': df_X['y'].to_numpy(),
            'n': df_X['n'].to_numpy(),
            'label': sample['label'],
        }

class Carbune2020(object):
    """Transform described in [Carbune2020] paper.

    It first shifts and rescales the (x, y) coordinates. Then it
    resamples the strokes and computes differences as final result.
    """

    POINTS_PER_UNIT_LENGTH = 20.0

    def __init__(self):
        pass

    def __call__(self, sample: dict) -> dict:
        """Transform sample according to transform described in [Carbune2020].

        The keys `sample_name` and `label` are copied over without modification.
        The keys `x`, `y`, `t` & `n` are transformed according to transform
        described in [Carbune2020]. The output sample is a new dict.

        :param sample: Input sample.
        :returns: Transformed sample as output with keys `sample_name`,
                  `x`, `y`, `t`, `n`, `label`.
        """

        sample_name = sample['sample_name']

        sample_sub = {
            'x': sample['x'],
            'y': sample['y'],
            't': sample['t'],
            'stroke_nr': sample['stroke_nr'],
        }
        df = pd.DataFrame.from_dict(sample_sub)

        df['x'] = df['x'] - df['x'].iloc[0]
        df['y'] = df['y'] - df['y'].min()
        scale_factor = df['y'].max()
        df['x'] = df['x'] / scale_factor
        df['y'] = df['y'] / scale_factor
        if df['x'].iloc[0] != 0:
            raise ValueError('x shift failed')
        if df['y'].min() != 0.0 or df['y'].max() != 1.0:
            raise ValueError('y shift and rescaling failed')
        
        # Determine number of points for resampling based on length of strokes
        stroke_lengths = calculate_distance_to_prev_point(df).groupby('stroke_nr').sum()
        stroke_lengths['nr_points'] = stroke_lengths['distances'] * Carbune2020.POINTS_PER_UNIT_LENGTH
        stroke_lengths['nr_points_rounded_up'] = np.ceil( stroke_lengths['nr_points'] )
        stroke_lengths.loc[ stroke_lengths['nr_points_rounded_up'] == 1.0, 'nr_points_rounded_up' ] += 1 # Increase a single point to 2 points
        stroke_lengths['nr_points_rounded_up'] = stroke_lengths['nr_points_rounded_up'].astype(int)

        # Perform resampling
        data_resampled = {
            'x': [],
            'y': [],
            't': [],
            'stroke_nr': [],
        }
        discard_sample = False
        for stroke_nr, df_grouped in df.groupby('stroke_nr'):

            if df_grouped.shape[0] == 1:
                # print(f'Handle length-one stroke: {sample_name=} - {stroke_nr=} - use it without preprocessing')
                index_of_value = df_grouped.index[0]
                data_resampled['t'].append( df_grouped.loc[index_of_value, 't'] )
                data_resampled['x'].append( df_grouped.loc[index_of_value, 'x'] )
                data_resampled['y'].append( df_grouped.loc[index_of_value, 'y'] )
                data_resampled['stroke_nr'].append(stroke_nr)
                continue

            if np.allclose( df_grouped['t'].diff()[1:], 0 ):
                # print(f'{sample_name=} {stroke_nr=}: time channel is constant - discard sample')
                discard_sample = True # Remove full sample as one does not know which stroke the problem is
                break

            if not np.alltrue( df_grouped['t'].diff()[1:] >= 0.0 ):
                # print(f'{sample_name=} {stroke_nr=}: time channel is non-monotonous - discard sample')
                discard_sample = True # Remove full sample as one does not know which stroke the problem is
                break

            time_normalised = ( df_grouped['t'] - df_grouped['t'].min() ) / ( df_grouped['t'].max() - df_grouped['t'].min() )
            if time_normalised.min() != time_normalised.iloc[0] or time_normalised.max() != time_normalised.iloc[-1]:
                raise ValueError(f'min or max of time_normalised are not at the correct position: {sample_name=} {stroke_nr=}')

            time_normalised_resampled = np.linspace(time_normalised.min(),
                                                    time_normalised.max(),
                                                    stroke_lengths.loc[stroke_nr, 'nr_points_rounded_up'])
            
            x_resampled = interp1d(time_normalised, df_grouped['x'], kind='linear', bounds_error=True)(time_normalised_resampled)
            y_resampled = interp1d(time_normalised, df_grouped['y'], kind='linear', bounds_error=True)(time_normalised_resampled)
            t_resampled = interp1d(time_normalised, df_grouped['t'], kind='linear', bounds_error=True)(time_normalised_resampled)

            for (xx, yy, tt) in zip(x_resampled, y_resampled, t_resampled):
                data_resampled['t'].append(tt)
                data_resampled['x'].append(xx)
                data_resampled['y'].append(yy)
                data_resampled['stroke_nr'].append(stroke_nr)

        if discard_sample:
            # print(f'Skipped sample {sample_name}')
            return FAILED_SAMPLE
        
        df_resampled = pd.DataFrame.from_dict(data_resampled)

        # Set up X (a.k.a. ink)
        df_X = pd.DataFrame(index=np.arange(df_resampled.shape[0]))
        df_X['x'] = np.nan
        df_X['y'] = np.nan
        df_X['t'] = np.nan
        df_X['n'] = np.nan

        # Set up x
        df_X.loc[0, 'x'] = 0
        df_X.loc[1:, 'x'] = df_resampled['x'].diff(periods=1).iloc[1:]

        # Set up y
        df_X.loc[0, 'y'] = 0
        df_X.loc[1:, 'y'] = df_resampled['y'].diff(periods=1).iloc[1:]

        # Set up t
        df_X.loc[0, 't'] = 0
        df_X.loc[1:, 't'] = df_resampled['t'].diff(periods=1).iloc[1:]

        # Set up n
        n_values = df_resampled['stroke_nr'].diff(periods=1)
        n_values.iloc[0] = 1.0
        df_X.loc[:, 'n'] = n_values

        # Ensure that no NaN's are left
        if df_X.isnull().values.any():
            raise ValueError('NaN value found in df_X.')

        return {
            'sample_name': sample['sample_name'],
            'x': df_X['x'].to_numpy(),
            'y': df_X['y'].to_numpy(),
            't': df_X['t'].to_numpy(),
            'n': df_X['n'].to_numpy(),
            'label': sample['label'],
        }

def calculate_distance_to_prev_point(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every point in an ink with multiple strokes stored in a DataFrame, calculate difference to previous point of same stroke.

    The first point of a stroke does not have a previous point so that the distance is set to NaN.

    :param df: DataFrame that stores ink.
    :returns: DataFrame with distances.
    """
    distances = []
    stroke_nrs = []
    for stroke_nr, df_grouped in df.groupby('stroke_nr'):
        distance = np.sqrt( df_grouped['x'].diff()**2 + df_grouped['y'].diff()**2 )
        for d in distance:
            distances.append(d)
            stroke_nrs.append(stroke_nr)
    return pd.DataFrame.from_dict({
        'distances': distances,
        'stroke_nr': stroke_nrs,
    })