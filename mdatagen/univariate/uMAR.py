# -*- coding: utf-8 -*

# =============================================================================
# Aeronautics Institute of Technologies (ITA) - Brazil
# University of Coimbra (UC) - Portugal
# Arthur Dantas Mangussi - mangussiarthur@gmail.com
# =============================================================================

__author__ = 'Arthur Dantas Mangussi'


import random

import numpy as np
import pandas as pd
from mdatagen.utils.feature_choice import FeatureChoice


# ==========================================================================
class uMAR:
    """
    A class to generate missing values in a dataset based on the Missing At Random (MAR) univariate mechanism.

    Args:
        X (pd.DataFrame): The dataset to receive the missing data.
        y (np.array): The label values from dataset
        missing_rate (int, optional): The rate of missing data to be generated. Default is 10.
        x_miss (string): The name of feature to insert the missing data. If not informed, x_miss will be the feature most correlated with target

    Example Usage:
    ```python
    # Create an instance of the MAR class
    generator = MAR(X, y, missing_rate=20, x_miss='feature1')

    # Generate missing values using the lowest strategy
    data_md = generator.lowest()
    ```
    """

    # ------------------------------------------------------------------------
    def __init__(
        self,
        X: pd.DataFrame,
        y: np.array,
        missing_rate: int = 10,
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
        """Function to generate missing values in the feature (x_miss) using
        the lowest values from an observed feature.

        Returns:
            dataset (DataFrame): The dataset with missing values generated under
            the MAR mechanism.

        Reference:
        [1] Santos, M. S., R. C. Pereira, A. F. Costa, J. P. Soares, J. Santos, and
        P. H. Abreu. 2019. Generating Synthetic Missing Data: A Review by Missing Mechanism.
        IEEE Access 7: 11651–67.

        """
        pos_xmis = self.dataset[self.x_obs].sort_values()[: self.N].index
        self.dataset.loc[pos_xmis, self.x_miss] = np.nan
        return self.dataset

    # ------------------------------------------------------------------------
    def rank(self):
        """
        Function to generate missing values in the feature (x_miss) using
        a rank from an observed feature.

        Returns:
            dataset (DataFrame): The dataset with missing values generated under
            the MAR mechanism.

        Reference:
        [1] Santos, M. S., R. C. Pereira, A. F. Costa, J. P. Soares, J. Santos, and
        P. H. Abreu. 2019. Generating Synthetic Missing Data: A Review by Missing Mechanism.
        IEEE Access 7: 11651–67.

        """
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
        """
        Function to generate missing data in the feature (x_miss) using the median
        of an observed feature.

        Returns:
            dataset (DataFrame): The dataset with missing values generated under
            the MAR mechanism.

        Reference:
        [1] Santos, M. S., R. C. Pereira, A. F. Costa, J. P. Soares, J. Santos, and
        P. H. Abreu. 2019. Generating Synthetic Missing Data: A Review by Missing Mechanism.
        IEEE Access 7: 11651–67.

        """
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
        """Function to generate missing values in the feature (x_miss) using the
        highest values from an observed feature.

        Returns:
            dataset (DataFrame): The dataset with missing values generated under
            the MAR mechanism.

        Reference:
        [1] Santos, M. S., R. C. Pereira, A. F. Costa, J. P. Soares, J. Santos, and
        P. H. Abreu. 2019. Generating Synthetic Missing Data: A Review by Missing Mechanism.
        IEEE Access 7: 11651–67.

        """

        pos_xmis = (
            self.dataset[self.x_obs]
            .sort_values(ascending=False)[: self.N]
            .index
        )
        self.dataset.loc[pos_xmis, self.x_miss] = np.nan
        return self.dataset

    # ------------------------------------------------------------------------
    def mix(self):
        """
        Function to generate missing values in the feature (x_miss) using the
        N/2 lowest values and N/2 highest values from an observed feature.

        Returns:
            dataset (DataFrame): The dataset with missing values generated under
            the MAR mechanism.

        Reference:
        [1] Santos, M. S., R. C. Pereira, A. F. Costa, J. P. Soares, J. Santos, and
        P. H. Abreu. 2019. Generating Synthetic Missing Data: A Review by Missing Mechanism.
        IEEE Access 7: 11651–67.

        """
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
