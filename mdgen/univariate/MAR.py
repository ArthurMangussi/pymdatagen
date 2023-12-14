import pandas as pd
import numpy as np
import random
from utils.feature_choice import FeatureChoice


# ==========================================================================
class MAR:
    """
    Imputes missing values in the one feature from dataset based on the MAR (Missing At Random) mechanism.

    Args:
        X (DataFrame):
        y (array):
        x_miss (string): The name of feature to insert the missing data. If not informed, x_miss will be based on "method".


    Example Usage:
    # Create an instance of the MAR class
    imputer = MAR(X, y, missing_rate=20, x_miss='feature1')

    # Generate missing values using the lowest strategy
    data_md = imputer.lowest()
    """

    # ------------------------------------------------------------------------
    def __init__(
        self,
        X: pd.DataFrame,
        y: np.array,
        missing_rate: int,
        x_miss: str = None,
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

        self.x_miss, self.x_obs = FeatureChoice._define_xmiss(
            self.X, self.y, x_miss
        )
        self.N = round(self.missing_rate * self.dataset.shape[0] / 100)

    # ------------------------------------------------------------------------
    def lowest(self):
        pos_xmis = self.dataset[self.x_obs].sort_values()[: self.N].index
        self.dataset.loc[pos_xmis, self.x_miss] = np.nan
        return self.dataset

    # ------------------------------------------------------------------------
    def rank(self):
        ranks = self.dataset[self.x_obs].rank(
            method='average', na_option='bottom'
        )
        random_numbers = np.random.random(self.dataset.shape[0])
        lista_index = set()
        i, max_iter = 0, 0
        while i < self.N:
            if max_iter == 50:
                random_numbers = np.random.random(self.dataset.shape[0])

            lin = random.randint(0, self.dataset.shape[0] - 1)
            if (
                random_numbers[lin] <= ranks[lin] / (ranks.max() + 1)
                and lin not in lista_index
            ):
                lista_index.add(lin)
                max_iter = 0
                i += 1
            else:
                max_iter += 1
        pos_xmis = list(lista_index)
        self.dataset.loc[pos_xmis, self.x_miss] = np.nan
        return self.dataset

    # ------------------------------------------------------------------------
    def median(self):
        median_xobs = self.dataset[self.x_obs].median()
        pb = np.zeros(self.dataset.shape[0])

        grupos = np.where(self.dataset[self.x_obs] >= median_xobs, 1, 0)

        if len(pb[grupos == 0]) == 0 or len(pb[grupos == 1]) == 0:
            grupos = np.where(self.dataset[self.x_obs] <= median_xobs, 1, 0)

        # The group with values higher or equal to the median have 9 times more probability to be chosen
        pb[grupos == 0] = 0.1 / np.sum(grupos == 0)
        pb[grupos == 1] = 0.9 / np.sum(grupos == 1)
        pos_xmis = np.random.choice(
            np.arange(self.dataset.shape[0]), size=self.N, replace=False, p=pb
        )
        self.dataset.loc[pos_xmis, self.x_miss] = np.nan
        return self.dataset

    # ------------------------------------------------------------------------
    def highest(self):
        pos_xmis = (
            self.dataset[self.x_obs]
            .sort_values(ascending=False)[: self.N]
            .index
        )
        self.dataset.loc[pos_xmis, self.x_miss] = np.nan
        return self.dataset

    # ------------------------------------------------------------------------
    def mix(self):
        pos_xmis_lowest = (
            self.dataset[self.x_obs].sort_values()[: round(self.N / 2)].index
        )
        pos_xmis_highest = (
            self.dataset[self.x_obs]
            .sort_values(ascending=False)[: round(self.N / 2)]
            .index
        )
        pos_xmis = list(pos_xmis_lowest) + list(pos_xmis_highest)
        self.dataset.loc[pos_xmis, self.x_miss] = np.nan
        return self.dataset
