# -*- coding: utf-8 -*

# =============================================================================
# Aeronautics Institute of Technologies (ITA) - Brazil
# University of Coimbra (UC) - Portugal
# Arthur Dantas Mangussi - mangussiarthur@gmail.com
# =============================================================================

__author__ = 'Arthur Dantas Mangussi'
__version__ = '1.0.0'

import warnings

import numpy as np
import pandas as pd


# ==========================================================================
class mMCAR:
    """
    A class to generate missing data in a dataset based on the Missing Completely At Random (MCAR) mechanism for multiple features simultaneously.

    Args:
        X (pd.DataFrame): The dataset to receive the missing data.
        y (np.array): The label values from dataset
        missing_rate (int, optional): The rate of missing data to be generated. Default is 10.

    Example Usage:
    ```python
    # Create an instance of the MCAR class
    generator = MCAR(X, y, missing_rate=20)

    # Generate missing values using the random strategy
    data_md = generator.random()
    ```
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
        """
        Function to randomly generate missing data in all dataset.

        Returns:
            dataset (DataFrame): The dataset with missing values generated under
            the MCAR mechanism.

        Reference:
        [1] Santos, M. S., R. C. Pereira, A. F. Costa, J. P. Soares, J. Santos, and
        P. H. Abreu. 2019. Generating Synthetic Missing Data: A Review by Missing Mechanism.
        IEEE Access 7: 11651–67.

        """
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
        """
        Function to generate missing data in columns by Bernoulli distribution
        for each attribute informed.

        Args:
            columns (list): A list of strings containing columns names.

        Returns:
            dataset (DataFrame): The dataset with missing values generated under
            the MCAR mechanism.

        Reference:
        [1] Santos, M. S., R. C. Pereira, A. F. Costa, J. P. Soares, J. Santos, and
        P. H. Abreu. 2019. Generating Synthetic Missing Data: A Review by Missing Mechanism.
        IEEE Access 7: 11651–67.

        """

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
