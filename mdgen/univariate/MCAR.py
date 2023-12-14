import pandas as pd
import numpy as np
from utils.math_calcs import MathCalcs


# ==========================================================================
class MCAR:
    """
    Generate missing values in a dataset based on the MCAR (Missing Completely At Random) mechanism.

    Args:
        insertion_dataset (pd.DataFrame): The dataset to receive the missing data.
        flag (str): An string flag that determines the method to select locations on xmiss. It should be one of the values ['random', 'bernoulli'].
        method (str, optional): The method to select the feature (xmiss) with missing values. It should be one of the values ['random', 'correlated', 'min', 'max']. Defaults to 'random'.
        x_miss (string): The name of feature to insert the missing data. If not informed, x_miss will be based on "method".

    Returns:
        pd.DataFrame: The inserted dataset with missing values under the MCAR mechanism.

    Example Usage:
    ```python
    # Initialize the MissingDataGenerator object
    imputador = MissingDataGenerator(dados_completo=df, missing_rate=10)

    # Call the MCAR_univa method to impute missing values
    imputed_dataset = imputador.MCAR_univa(df)
    ```
    """

    # ------------------------------------------------------------------------
    def __init__(
        self,
        X: pd.DataFrame,
        y: np.array,
        missing_rate: int,
        x_miss: str = None,
        method: str = 'random',
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
        pos_xmiss = np.random.choice(
            self.dataset[self.x_miss].index, self.N, replace=False
        )
        self.dataset.loc[pos_xmiss, self.x_miss] = np.nan
        return self.dataset

    # ------------------------------------------------------------------------
    def binomial(self):
        pos_xmiss = np.random.binomial(
            n=1, p=self.missing_rate / 100, size=self.dataset.shape[0]
        ).astype(bool)
        self.dataset.loc[pos_xmiss, self.x_miss] = np.nan
        return self.dataset
