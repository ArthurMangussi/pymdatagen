import pandas as pd
import numpy as np
from utils.math_calcs import MathCalcs
from utils.feature_choice import FeatureChoice

import warnings

# ==========================================================================
class MAR:
    """
    Generate missing values in a dataset based on the MAR (Missing At Random) mechanism for multiple features simultaneously.

    Args:
        insertion_dataset (pd.DataFrame): The dataset to receive the missing data.
        missing_rate (int): The percentage of missing values to be generated.
        method (str, optional): The method to select the observed feature and the feature to receive the missing data. It can be one of the folowing: ["random", "correlated"]. Defaults to "random".
        n_xmiss (int): The number of features in the dataset that will receive missing values.

    Returns:
        pd.DataFrame: The modified dataset with missing values based on the MAR mechanism for multiple features.
    """

    def __init__(self, X: pd.DataFrame, y: np.array, n_xmiss: int = 2):
        if not isinstance(X, pd.DataFrame):
            raise TypeError('Dataset must be a Pandas Dataframe')
        if not isinstance(y, np.ndarray):
            raise TypeError('y must be a numpy array')

        self.n_xmiss = n_xmiss
        self.X = X
        self.y = y
        self.dataset = self.X.copy()
        self.dataset['target'] = y

    def random(self, missing_rate: int = 10) -> pd.DataFrame:
        if missing_rate >= 100:
            raise ValueError(
                'Missing Rate is too high, the whole feature will be deleted!'
            )
        elif missing_rate < 0:
            raise ValueError('Missing rate must be between [0,100]')

        n = self.dataset.shape[0]
        p = self.dataset.shape[1]
        mr = missing_rate / 100

        N = round((n * p * mr) / self.n_xmiss)

        if self.n_xmiss > p:
            raise ValueError(
                'n_xmiss must be lower than number of features in dataset'
            )
        elif N > n:
            raise ValueError(
                f'FEATURES will be all NaN ({N}>{n}), you should increase the number of features that will receive missing data'
            )

        xmiss_multiva = []
        cont = 0

        while cont < self.n_xmiss:
            x_obs = np.random.choice(self.dataset.columns, replace=False)
            x_miss = np.random.choice(
                [x for x in self.dataset.columns if x != x_obs]
            )

            if x_miss not in xmiss_multiva:
                pos_xmiss = self.dataset[x_obs].sort_values()[:N].index
                self.dataset.loc[pos_xmiss, x_miss] = np.nan

                xmiss_multiva.append(x_miss)
                cont += 1

        return self.dataset

    def correlated(self, missing_rate: int = 10) -> pd.DataFrame:
        """Não considera a classe"""

        if missing_rate >= 50:
            raise ValueError(
                'Features will be all NaN, you should decrease the missing rate'
            )

        mr = missing_rate / 100

        pairs = FeatureChoice._make_pairs(self.X, self.y)

        for pair in pairs:
            if len(pair) % 2 == 0:
                cutK = 2 * mr
                x_miss = FeatureChoice._find_most_correlated_feature_even(
                    self.X, self.y, pair
                )
                x_obs = next(elem for elem in pair if elem != x_miss)

            else:
                x_miss = FeatureChoice._find_most_correlated_feature_odd(
                    self.X, self.y, pair
                )
                x_obs = set(pair).difference(x_miss).pop()
                cutK = 1.5 * mr

            N = round(len(self.dataset) * cutK)
            pos_xmiss = self.dataset[x_obs].sort_values()[:N].index
            self.dataset.loc[pos_xmiss, x_miss] = np.nan

        return self.dataset

    def median(self, missing_rate: int = 10) -> pd.DataFrame:
        if missing_rate >= 50:
            raise ValueError(
                'Features will be all NaN, you should decrease the missing rate'
            )

        mr = missing_rate / 100

        pairs = FeatureChoice._make_pairs(self.X, self.y)

        for pair in pairs:
            x_obs = np.random.choice(pair)

            if len(pair) % 2 == 0:
                cutK = 2 * mr
                x_miss = next(elem for elem in pair if elem != x_obs)
            else:
                cutK = 1.5 * mr
                x_miss = list(pair)
                x_miss.remove(x_obs)

            N = round(len(self.dataset) * cutK)

            if len(self.dataset[x_obs].unique()) == 2:  # Binary feature
                g1, g2 = MathCalcs.define_groups(self.dataset, x_obs)

            else:  # Continuos or ordinal feature
                median_xobs = self.dataset[x_obs].median()

                g1 = self.dataset[x_obs][self.dataset[x_obs] <= median_xobs]
                g2 = self.dataset[x_obs][self.dataset[x_obs] > median_xobs]

                if len(g1) != len(
                    g2
                ):  # If median do not divide in equal-size groups
                    g1, g2 = MathCalcs.define_groups(self.dataset, x_obs)

            choice = np.random.choice([0, 1])
            if choice == 0:
                pos_xmiss = g1.sort_values()[:N].index

            else:
                pos_xmiss = g2.sort_values()[:N].index

            self.dataset.loc[pos_xmiss, x_miss] = np.nan

        return self.dataset
