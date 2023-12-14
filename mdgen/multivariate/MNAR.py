import pandas as pd
import numpy as np
from utils.feature_choice import FeatureChoice
from scipy.stats import mannwhitneyu

import warnings


# ==========================================================================
class MNAR:
    """
    Generate missing values in a dataset based on the MNAR (Missing Not At Random) mechanism for multiple features simultaneously.

    Args:
        insertion_dataset (pd.DataFrame): The dataset in which missing values will be generated.
        method (str, optional): The method to be used for generating missing values. It can be "random", "correlated", or "median". Default is "random".
        n_xmiss (int, optional): The number of features that will receive missing data for random method. Default is the number of columns in the dataset.
        threshold (float, optional): The threshold for selecting the lowest (threshold=0) and highest (threshold=1) values to be replaced with missing values. Default is 0.

    Returns:
        pd.DataFrame: The dataset with missing values generated according to the specified method.

    Example Usage:
        # Initialize the MissingDataGenerator object
        generator = MissingDataGenerator(X, y, missing_rate)

        # Generate missing values using the MNAR_multiva method
        missing_data = generator.MNAR_multiva(insertion_dataset, method="random", n_xmiss=5, threshold=0.2)
    """

    def __init__(self, X: pd.DataFrame, y: np.array, **kwargs):
        if not isinstance(X, pd.DataFrame):
            raise TypeError('Dataset must be a Pandas Dataframe')
        if not isinstance(y, np.ndarray):
            raise TypeError('y must be a numpy array')
        self.X = X
        self.y = y
        self.dataset = self.X.copy()
        self.dataset['target'] = y

        self.n = self.dataset.shape[0]
        self.p = self.dataset.shape[1]

        self.n_xmiss = kwargs.get('n_xmiss', self.p)
        self.threshold = kwargs.get('threshold', 0)

    def random(self, missing_rate: int = 10):
        if missing_rate >= 100:
            raise ValueError(
                'Missing Rate is too high, the whole feature will be deleted!'
            )
        elif missing_rate < 0:
            raise ValueError('Missing rate must be between [0,100]')

        mr = missing_rate / 100
        cont = 0
        N = round((self.n * self.p * mr) / self.n_xmiss)
        xmiss_multiva = []

        if self.n_xmiss > self.p:
            raise ValueError(
                'n_xmiss must be lower than number of feature in dataset'
            )
        elif N > self.n:
            raise ValueError(
                f'FEATURES will be all NaN ({N}>{self.n}), you should increase the number of features that will receive missing data'
            )

        options_xmiss = list(self.dataset.columns.copy())

        while cont < self.n_xmiss:
            # Random choice for determining feature
            x_miss = np.random.choice(options_xmiss)

            if x_miss not in xmiss_multiva:
                x_f = self.dataset.loc[:, x_miss].values

                # Unobserved random feature
                ordered_id = np.lexsort((np.random.random(x_f.size), x_f))
                pos_xmiss = FeatureChoice.miss_locations(
                    ordered_id, self.threshold, N
                )

                self.dataset.loc[pos_xmiss, x_miss] = np.nan

                xmiss_multiva.append(x_miss)
                options_xmiss.remove(x_miss)
                cont += 1

        return self.dataset

    def correlated(self, missing_rate: int = 10):
        if missing_rate >= 50:
            raise ValueError(
                'Features will be all NaN, you should decrease the missing rate'
            )

        # Creation of pairs/triples
        pairs = FeatureChoice._make_pairs(self.X, self.y)
        mr = missing_rate / 100

        for pair in pairs:
            if len(pair) % 2 == 0:
                cutK = 2 * mr
                # Find the feature most correlated with the target
                x_miss = FeatureChoice._find_most_correlated_feature_even(
                    self.X, self.y, pair
                )

            else:
                cutK = 1.5 * mr
                # Find the feature most correlated with the target
                x_miss = FeatureChoice._find_most_correlated_feature_odd(
                    self.X, self.y, pair
                )

            N = round(len(self.dataset) * cutK)

            x_f = self.dataset.loc[:, x_miss].values

            # Unobserved random feature
            ordered_id = np.lexsort((np.random.random(x_f.size), x_f))
            pos_xmiss = FeatureChoice.miss_locations(
                ordered_id, self.threshold, N
            )

            self.dataset.loc[pos_xmiss, x_miss] = np.nan

        return self.dataset

    def median(self, missing_rate: int = 10):
        if missing_rate >= 50:
            raise ValueError(
                'Features will be all NaN, you should decrease the missing rate'
            )

        mr = missing_rate / 100

        # For each column in dataset
        for col in self.dataset.columns:
            cutK = 2 * mr
            N = round(len(self.dataset) * cutK)

            if len(self.dataset[col].unique()) == 2:  # Nominal feature
                g1_index = np.random.choice(
                    self.dataset[col].index,
                    round(len(self.dataset) / 2),
                    replace=False,
                )
                g2_index = np.random.choice(
                    self.dataset[col].index,
                    round(len(self.dataset) / 2),
                    replace=False,
                )

            else:  # Continuos or ordinal feature
                median_xobs = self.dataset[col].median()

                g1 = self.dataset[col][self.dataset[col] <= median_xobs]
                g2 = self.dataset[col][self.dataset[col] > median_xobs]

                if len(g1) != len(
                    g2
                ):  # If median do not divide in equal-size groups
                    g1_index = np.random.choice(
                        self.dataset[col].index,
                        round(len(self.dataset) / 2),
                        replace=False,
                    )
                    g2_index = np.random.choice(
                        self.dataset[col].index,
                        round(len(self.dataset) / 2),
                        replace=False,
                    )
                else:
                    g1_index = g1.index
                    g2_index = g2.index

            choice = np.random.choice([0, 1])
            if choice == 0:
                x_f = self.dataset.loc[g1_index, col].values
            else:
                x_f = self.dataset.loc[g2_index, col].values

            # Unobserved random feature
            ordered_id = np.lexsort((np.random.random(x_f.size), x_f))
            pos_xmiss = FeatureChoice.miss_locations(
                ordered_id, self.threshold, N
            )

            self.dataset.loc[pos_xmiss, col] = np.nan

        return self.dataset

    def MBOUV(
        self, missing_rate: int = 10, depend_on_external=None, ascending=True
    ):
        if depend_on_external is None:
            depend_on_external = []

        if missing_rate > 90:
            raise ValueError(
                'Maximum missing rate per feature must be clipped at 90%'
            )
        elif missing_rate < 0:
            raise ValueError('Missing rate must be between [0,100]')

        mr = missing_rate / 100
        N = round(mr * self.n * self.p)

        # With moderate missing rates (up to 60%), it never happens.
        max_mvs_feat = round(min(mr + (mr / 2.0), 0.9) * self.n)

        # Randomly distribute missing values by the features.
        num_mvs_per_feat = np.zeros(self.p)
        while N > 0:
            for i in range(self.p):
                if num_mvs_per_feat[i] < max_mvs_feat:
                    num_mvs_it = np.random.randint(
                        0, min(max_mvs_feat - num_mvs_per_feat[i] + 1, N + 1)
                    )
                    N -= num_mvs_it
                    num_mvs_per_feat[i] += num_mvs_it
                    if N == 0:
                        break

        np.random.shuffle(num_mvs_per_feat)
        hidden_f = np.random.normal(size=self.n)
        tie_breaker = np.random.random(hidden_f.size)

        # Amputate the values.
        for i, col in enumerate(self.dataset.columns.values):
            num_mv = round(num_mvs_per_feat[i])
            num_mv = num_mv if num_mv > 0 else 0

            if col in depend_on_external:
                start_n = end_n = int(num_mv / 2)
                if num_mv % 2 == 1:
                    end_n += 1

                indices_start = np.lexsort((tie_breaker, hidden_f))[:start_n]
                indices_end = np.lexsort((tie_breaker, -hidden_f))[:end_n]

                self.dataset.loc[indices_start, col] = np.nan
                self.dataset.loc[indices_end, col] = np.nan
            else:
                ordered_indices = FeatureChoice.get_ordered_indices(
                    col, self.dataset, ascending
                )
                self.dataset.loc[ordered_indices[:num_mv], col] = np.nan

        return self.dataset

    def MBOV_randomness(
        self,
        missing_rate: int = 10,
        randomness: float = 0,
        columns: list = None,
    ):
        if not (0 <= randomness <= 0.5):
            raise ValueError('Threshold must be in range [0,0.5]')

        if columns is None:
            raise TypeError(
                'You must inform columns from dataset to generate missing data'
            )

        if missing_rate >= 100:
            raise ValueError(
                'Missing Rate is too high, the whole feature will be deleted!'
            )
        elif missing_rate < 0:
            raise ValueError('Missing rate must be between [0,100]')

        mr = missing_rate / 100
        N = round(len(self.dataset) * mr)
        aux_index = list(self.dataset.index)

        for col in columns:
            pos_xmis_deterministic = (
                self.dataset[col]
                .sort_values()[: round((1 - randomness) * N)]
                .index
            )

            # Eliminate the lowest candidate from index
            np.delete(aux_index, pos_xmis_deterministic)

            pos_xmiss_randomness = np.random.choice(
                aux_index, round(randomness * N), replace=False
            )

            pos_xmis = np.concatenate(
                [pos_xmis_deterministic, pos_xmiss_randomness]
            )

            self.dataset.loc[pos_xmis, col] = np.nan

        return self.dataset

    def MBOV_median(self, missing_rate: int = 10, columns: list = None):
        if missing_rate >= 100:
            raise ValueError(
                'Missing Rate is too high, the whole feature will be deleted!'
            )
        elif missing_rate < 0:
            raise ValueError('Missing rate must be between [0,100]')

        if columns is None:
            raise TypeError(
                'You must inform columns from dataset to generate missing data'
            )

        mr = missing_rate / 100
        N = round(len(self.dataset) * mr)

        for col in columns:
            if col.dtype == 'object':
                raise TypeError(
                    'Only continuos or ordinal feature are accepted'
                )

            median_att = self.dataset[col].median()

            pos_xmis = np.argsort(np.abs(self.dataset[col] - median_att))[:N]
            self.dataset.loc[pos_xmis, col] = np.nan

        return self.dataset

    def MBIR(
        self,
        missing_rate: int = 10,
        columns: list = None,
        statistical_method: str = 'Mann-Whitney',
    ):
        if missing_rate >= 100:
            raise ValueError(
                'Missing Rate is too high, the whole feature will be deleted!'
            )
        elif missing_rate < 0:
            raise ValueError('Missing rate must be between [0,100]')

        if columns is None:
            raise TypeError(
                'You must inform columns from dataset to generate missing data'
            )

        elif set(columns) == set(self.dataset):
            warnings.warn('All dataset will be drop', UserWarning)

        mr = missing_rate / 100
        N = round(len(self.dataset) * mr)

        df_columns = self.dataset.columns.tolist()

        for x_miss in columns:
            most_significant_diff = {}
            for x_obs in df_columns:
                if self.dataset[x_obs].dtype != 'object':
                    
                    instances = self.dataset.copy()

                    if x_obs != x_miss:
                        # Find the instance with lower MR% values of x_obs
                        lower = instances[x_obs].sort_values()[:N].index

                        # Set the f_miss to be missing
                        instances.loc[lower, x_miss] = np.nan

                        # Create the missing indicator
                        auxiliary_ind = np.where(instances[x_miss].isna(), 1, 0)

                        # Statistical tests between values of f_obs and f_ind
                        match statistical_method:
                            case 'Mann-Whitney':
                                statistic, p_value = mannwhitneyu(
                                    instances[x_obs], auxiliary_ind
                                )

                            case 'Bayesian':
                                print('Implementar Bayesian')

                        if p_value < 0.05:
                            # There is evidence of a significant difference.
                            most_significant_diff[x_obs] = p_value
                            print('There is evidence of a significant difference')

            most_feature = min(
                most_significant_diff, key=most_significant_diff.get
            )
            pos_xmis = self.dataset[most_feature].sort_values()[:N].index
            self.dataset.loc[pos_xmis, x_miss] = np.nan
            df_columns.remove(most_feature)
            self.dataset = self.dataset.drop(columns=most_feature)

        return self.dataset
