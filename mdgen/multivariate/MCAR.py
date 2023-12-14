import pandas as pd
import numpy as np


import warnings

# ==========================================================================
class MCAR:
    """
    Generate missing values in a dataset based on the MCAR (Missing Completely At Random) mechanism for multiple features simultaneously.

    Args:
        insertion_dataset (pd.DataFrame): The dataset to receive the missing values.
        method (str, optional): The method to use for generating missing values. Currently, only the "random" method is supported. Defaults to "random".

    Returns:
        pd.DataFrame: The inserted dataset with missing values based on the MCAR mechanism.
    """

    def __init__(self, X: pd.DataFrame, y: np.array, missing_rate: int = 10):
        if not isinstance(X, pd.DataFrame):
            raise TypeError('Dataset must be a Pandas Dataframe')
        if not isinstance(y, np.ndarray):
            raise TypeError('y must be a numpy array')

        if missing_rate >= 100:
            raise ValueError(
                'Missing Rate is too high, the whole dataset will be deleted!'
            )
        elif missing_rate < 0:
            raise ValueError('Missing rate must be between [0,100]')

        self.X = X
        self.y = y
        self.dataset = self.X.copy()
        self.dataset['target'] = y

    def random(self) -> pd.DataFrame:
        original_shape = self.dataset.shape
        mr = self.missing_rate / 100
        n = self.dataset.shape[0]
        p = self.dataset.shape[1]
        N = round(n * p * mr)

        array_values = self.dataset.values
        pos_miss = np.random.choice(
            self.dataset.shape[0] * self.dataset.shape[1], N, replace=False
        )
        array_values = array_values.flatten()
        array_values[pos_miss] = np.nan
        array_values = array_values.reshape(original_shape)

        return pd.DataFrame(array_values, columns=self.dataset.columns)

    def binomial(self, columns: list = None):
        warnings.warn(
            'Binomial sometimes does not generate the input missing rate in dataset'
        )

        if columns is None:
            raise TypeError(
                'You must inform columns from dataset to generate missing data'
            )

        for x_miss in columns:
            pos_xmiss = np.random.binomial(
                n=1, p=self.missing_rate / 100, size=self.dataset.shape[0]
            ).astype(bool)
            self.dataset.loc[pos_xmiss, x_miss] = np.nan
        return self.dataset
