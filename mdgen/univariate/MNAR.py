# =============================================================================
# Aeronautics Institute of Technologies (ITA) - Brazil
# University of Coimbra (UC) - Portugal
# Arthur Dantas Mangussi - mangussiarthur@gmail.com
# =============================================================================

__author__ = 'Arthur Dantas Mangussi'
__version__ = '1.0.0'

import pandas as pd
import numpy as np
from utils.feature_choice import FeatureChoice

# ==========================================================================
class MNAR:
    """
    A class to generate missing values in a dataset based on the Missing Not At Random (MNAR) univariate mechanism.

    Args:
        X (pd.DataFrame): The dataset to receive the missing data.
        y (np.array): The label values from dataset
        missing_rate (int, optional): The rate of missing data to be generated. Default is 10.
        x_miss (string, optional): The name of feature to insert the missing data. If not informed, x_miss will be the feature most correlated with target
        threshold (float, optional): The threshold to select the locations in feature (xmiss) to receive missing values where 0 indicates de lowest and 1 highest values. Default p=0

    Example Usage:
    ```
    # Create an instance of the MNAR class
    generator = MNAR(X, y, missing_rate=20, x_miss='feature1', threshold = 0.5)

    # Generate missing values using the lowest strategy
    data_md = generator.run()
    ```
    """

    # ------------------------------------------------------------------------
    def __init__(
        self,
        X: pd.DataFrame,
        y: np.array,
        missing_rate: int=10,
        x_miss: str = None,
        threshold: float = 0,
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
        self.p = threshold
        self.N = round(self.missing_rate * self.dataset.shape[0] / 100)

        if not (0 <= self.p <= 1):
            raise ValueError('Threshold must be in range [0,1]')

        self.x_miss, self.x_obs = FeatureChoice._define_xmiss(
            self.X, self.y, x_miss, flag=False
        )

    # ------------------------------------------------------------------------
    def run(self):
        """
        Function to generate missing values in the feature (x_miss) using the
        threshold to choose values from a unobserved feature. 

        Returns:
            dataset (DataFrame): The dataset with missing values generated under the MNAR mechanism.

        Reference:
        [1] Santos, M. S., R. C. Pereira, A. F. Costa, J. P. Soares, J. Santos, and 
        P. H. Abreu. 2019. Generating Synthetic Missing Data: A Review by Missing Mechanism.
        IEEE Access 7: 11651–67.

        """

        x_f = self.dataset.loc[:, self.x_miss].values

        # Unobserved random feature
        ordered_id = np.lexsort((np.random.random(x_f.size), x_f))
        pos_xmiss = FeatureChoice.miss_locations(ordered_id, self.p, self.N)

        self.dataset.loc[pos_xmiss, self.x_miss] = np.nan

        return self.dataset
