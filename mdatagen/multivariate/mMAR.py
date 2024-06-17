# -*- coding: utf-8 -*

#  =============================================================================
# Aeronautics Institute of Technologies (ITA) - Brazil
# University of Coimbra (UC) - Portugal
# Arthur Dantas Mangussi - mangussiarthur@gmail.com
# =============================================================================

__author__ = 'Arthur Dantas Mangussi'


import warnings

import numpy as np
import pandas as pd
from mdatagen.utils.feature_choice import FeatureChoice
from mdatagen.utils.math_calcs import MathCalcs


# ==========================================================================
class mMAR:
    """
    A class to generate missing data in a dataset based on the Missing At Random (MAR) mechanism for multiple features simultaneously.

    Args:
        X (pd.DataFrame): The dataset to receive the missing data.
        y (np.array): The label values from dataset
        n_xmiss (int): The number of features in the dataset that will receive missing values. Default is 2.
        missTarget (bool, optional): A flag to generate missing into the target.

    Example Usage:

    ```python
    # Create an instance of the MAR class
    generator = MAR(X, y, n_xmiss=4)

    # Generate missing values using the random strategy
    data_md = generator.random(missing_rate = 20)
    ```
    """

    def __init__(self, X: pd.DataFrame, y: np.array, n_xmiss: int = 2, missTarget:bool=False):
        if not isinstance(X, pd.DataFrame):
            raise TypeError('Dataset must be a Pandas Dataframe')
        if not isinstance(y, np.ndarray):
            raise TypeError('y must be a numpy array')

        self.n_xmiss = n_xmiss
        self.X = X
        self.y = y
        self.dataset = self.X.copy()
        self.missTarget = missTarget

        if missTarget:
            self.dataset['target'] = y

    def random(self, missing_rate: int = 10) -> pd.DataFrame:
        """
        Function to generate arficial missing data in n_xmiss features chosen randomly.
        The lower values in observed feature for each feature x_miss will determine
        the miss locations in x_miss.

        Args:
            missing_rate (int, optional): The rate of missing data to be generated. Default is 10.

        Returns:
            dataset (DataFrame): The dataset with missing values generated under
            the MAR mechanism.

        Reference:
        [1] Santos, M. S., R. C. Pereira, A. F. Costa, J. P. Soares, J. Santos, and
        P. H. Abreu. 2019. Generating Synthetic Missing Data: A Review by Missing Mechanism.
        IEEE Access 7: 11651–67.

        """
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

        if not self.missTarget:
            self.dataset['target'] = self.y
        return self.dataset

    def correlated(self, missing_rate: int = 10) -> pd.DataFrame:
        """
        Function to generate missing data in features from dataset, except the class (target).
        The lower values in observed feature for each correlated pair will determine the
        miss locations in feature x_miss.

        Args:
            missing_rate (int, optional): The rate of missing data to be generated. Default is 10.

        Returns:
            dataset (DataFrame): The dataset with missing values generated under
            the MAR mechanism.

        Reference:
        [1] Santos, M. S., R. C. Pereira, A. F. Costa, J. P. Soares, J. Santos, and
        P. H. Abreu. 2019. Generating Synthetic Missing Data: A Review by Missing Mechanism.
        IEEE Access 7: 11651–67.

        """

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

        if not self.missTarget:
            self.dataset['target'] = self.y
        return self.dataset

    def median(self, missing_rate: int = 10) -> pd.DataFrame:
        """
        Function to generate missing data in features from dataset.
        The median in observed feature for each correlated pair will create two groups.
        The group is chosen randomly, and lower values will determine the miss locations
        in feature x_miss.

        Args:
            missing_rate (int, optional): The rate of missing data to be generated. Default is 10.

        Returns:
            dataset (DataFrame): The dataset with missing values generated under
            the MAR mechanism.

        Reference:
        [1] Santos, M. S., R. C. Pereira, A. F. Costa, J. P. Soares, J. Santos, and
        P. H. Abreu. 2019. Generating Synthetic Missing Data: A Review by Missing Mechanism.
        IEEE Access 7: 11651–67.

        """

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

        if not self.missTarget:
            self.dataset['target'] = self.y
        return self.dataset
