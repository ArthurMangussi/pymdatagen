# -*- coding: utf-8 -*-

# =============================================================================
# Aeronautics Institute of Technologies (ITA) - Brazil
# University of Coimbra (UC) - Portugal
# Arthur Dantas Mangussi - mangussiarthur@gmail.com
# =============================================================================

import numpy as np
import pandas as pd
from mdatagen.utils.math_calcs import MathCalcs


# ==========================================================================
class uMCAR:
    """
    A class to generate missing values in a dataset based on the Missing Completely At Random (MCAR) univariate mechanism.

    Args:
        X (pd.DataFrame): The dataset to receive the missing data.
        y (np.array): The label values from dataset
        missing_rate (int, optional): The rate of missing data to be generated. Default is 10.
        x_miss (string, optional): The name of feature to insert the missing data.
        method (str, optional): The method to choose x_miss. If x_miss not informed by user, x_miss will be choose randomly. The options to choose xmiss is ["random", "correlated", "min", "max"]. Default is "random"
        seed (int, optional): The seed for the random number generator.


    Example Usage:
    ```python
    # Create an instance of the MCAR class
    generator = MCAR(X, y, missing_rate=20, method="correlated")

    # Generate missing values using the random strategy
    data_md = generator.random()
    ```
    """

    # ------------------------------------------------------------------------
    def __init__(
        self,
        X: pd.DataFrame,
        y: np.array,
        missing_rate: int = 10,
        x_miss: str = None,
        method: str = 'random',
        seed: int = None
    ):
        if not isinstance(X, pd.DataFrame):
            raise TypeError('Dataset must be a Pandas Dataframe')
        if not isinstance(y, np.ndarray):
            raise TypeError('y must be a numpy array')

        self.missing_rate = missing_rate

        if self.missing_rate >= 100:
            raise ValueError(
                'Missing Rate is too high, the whole feature will be deleted!'
            )
        elif self.missing_rate < 0:
            raise ValueError('Missing rate must be between [0,100]')

        self.X = X
        self.y = y
        self.dataset = self.X.copy()
        self.dataset['target'] = y
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        self.N = round(self.missing_rate * self.dataset.shape[0] / 100)

        relevance = MathCalcs._nmi(self.X, self.y)

        option_mapping = {
            'random': lambda: np.random.choice(
                list(filter(lambda x: x != 'target', self.dataset.columns))
            ),
            'correlated': lambda: MathCalcs._find_correlation(
                self.X, self.y, 'target'
            ),
            'min': lambda: min(relevance)[1],
            'max': lambda: max(relevance)[1],
        }

        if not x_miss:
            self.x_miss = option_mapping[method]()
        else:
            self.x_miss = x_miss

    # ------------------------------------------------------------------------
    def random(self):
        """
        Function to randomly select locations in the feature (x_miss) to be missing.

        Returns:
            dataset (DataFrame): The dataset with missing values generated under
            the MCAR mechanism.

        Reference:
        [1] Santos, M. S., R. C. Pereira, A. F. Costa, J. P. Soares, J. Santos, and
        P. H. Abreu. 2019. Generating Synthetic Missing Data: A Review by Missing Mechanism.
        IEEE Access 7: 11651–67.

        """
        pos_xmiss = np.random.choice(
            self.dataset[self.x_miss].index, self.N, replace=False
        )
        self.dataset.loc[pos_xmiss, self.x_miss] = np.nan
        return self.dataset

    # ------------------------------------------------------------------------
    def binomial(self):
        """
        Function to choose the feature (x_miss) locations to be missing by Bernoulli distribution.

        Returns:
            dataset (DataFrame): The dataset with missing values generated under
            the MCAR mechanism.

        Reference:
        [1] Santos, M. S., R. C. Pereira, A. F. Costa, J. P. Soares, J. Santos, and
        P. H. Abreu. 2019. Generating Synthetic Missing Data: A Review by Missing Mechanism.
        IEEE Access 7: 11651–67.

        """
        pos_xmiss = np.random.binomial(
            n=1, p=self.missing_rate / 100, size=self.dataset.shape[0]
        ).astype(bool)
        self.dataset.loc[pos_xmiss, self.x_miss] = np.nan
        return self.dataset
