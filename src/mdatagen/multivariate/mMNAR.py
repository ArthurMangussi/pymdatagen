# -*- coding: utf-8 -*-

# =============================================================================
# Aeronautics Institute of Technologies (ITA) - Brazil
# University of Coimbra (UC) - Portugal
# Arthur Dantas Mangussi - mangussiarthur@gmail.com
# =============================================================================

import warnings

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from mdatagen.utils.feature_choice import FeatureChoice
from mdatagen.utils.math_calcs import MathCalcs

from multiprocessing.pool import Pool

# ==========================================================================
class mMNAR:
    """
    A class to generate missing values in a dataset based on the Missing Not At Random (MNAR) mechanism for multiple features simultaneously.

    Args:
        X (pd.DataFrame): The dataset to receive the missing data.
        y (np.array): The label values from dataset
        missing_rate (int, optional): The rate of missing data to be generated. Default is 10.

    Keyword Args:
        n_xmiss (int, optional): The number of features in the dataset that will receive missing values. Default is the number of features in dataset.
        threshold (float, optional): The threshold to select the locations in feature (xmiss) to receive missing values where 0 indicates de lowest and 1 highest values. Default is 0
        missTarget (bool, optional): A flag to generate missing into the target.

    Example Usage:
    ```python
    # Create an instance of the MNAR class
    generator = MNAR(X, y)

    # Generate missing values using the random strategy
    data_md = generator.random()
    ```
    """

    def __init__(self, 
                 X: pd.DataFrame, 
                 y: np.ndarray, 
                 threshold:float=0, 
                 n_xmiss:int=2,
                 missTarget:bool = False,
                 n_Threads:int = 1):
        if not isinstance(X, pd.DataFrame):
            raise TypeError('Dataset must be a Pandas Dataframe')
        if not isinstance(y, np.ndarray):
            raise TypeError('y must be a numpy array')
        self.X = X
        self.y = y
        self.dataset = self.X.copy()
        self.missTarget = missTarget
        self.n_Threads = n_Threads

        if self.missTarget:
            self.dataset['target'] = y

        self.n_xmiss = n_xmiss
        self.threshold = threshold
        

    def _set_chuck_size(self, 
                        missing_rate, 
                        deterministic,
                        depend_on_external, 
                        ascending,
                        randomness,
                        columns,
                        statistical_method)->list:
        """
        Split the dataset into chunks for parallel processing.

        Args:
            missing_rate (int): The rate of missing data to be generated.

        Returns:
            list: A list of tuples, each containing a chunk of the dataset
                  and associated parameters for processing.
        """
        n_chunks = self.n_Threads
        chunk_size = len(self.dataset) // n_chunks
        chunks = [
            (
                self.dataset.iloc[i:i + chunk_size].copy(),
                self.n_xmiss,
                missing_rate,
                self.y[i:i + chunk_size],
                self.missTarget,
                self.threshold,
                deterministic,
                depend_on_external,
                ascending,
                randomness,
                columns,
                statistical_method
            )
            for i in range(0, len(self.dataset), chunk_size)
            ]
        return chunks        

    @staticmethod
    def _random_strategy_to_generate_md(args):
        """
        Function to randomly choose the feature (x_miss) in dataset for generate missing
        data. The miss locations on x_miss is the lower values based on unobserved feature
        or feature x_miss itself.

        Args:
            missing_rate (int, optional): The rate of missing data to be generated. Default is 10.
            deterministc (bool, optinal): A flag that determine if x_miss will have miss 
            locations based on itself or an unobserved feature. Default is False
            (i.e., an unobserved feature).

        Returns:
            dataset (DataFrame): The dataset with missing values generated under
            the MNAR mechanism.

        Reference:
        [1] Santos, M. S., R. C. Pereira, A. F. Costa, J. P. Soares, J. Santos, and
        P. H. Abreu. 2019. Generating Synthetic Missing Data: A Review by Missing Mechanism.
        IEEE Access 7: 11651–67.
        """
        dataset_chunk, n_xmiss, missing_rate, y_chunk, missTarget, threshold, deterministic,depend_on_external, ascending,randomness,columns,statistical_method = args
        mr = missing_rate / 100
        n = dataset_chunk.shape[0]
        p = dataset_chunk.shape[1]

        cont = 0
        N = round((n * p * mr) / n_xmiss)
        xmiss_multiva = []

        if n_xmiss > p:
            raise ValueError(
                'n_xmiss must be lower than number of feature in dataset'
            )
        elif N > n:
            raise ValueError(
                f'FEATURES will be all NaN ({N}>{n}), you should increase the number of features that will receive missing data'
            )

        options_xmiss = list(dataset_chunk.columns.copy())

        while cont < n_xmiss:
            # Random choice for determining feature
            x_miss = np.random.choice(options_xmiss)

            if x_miss not in xmiss_multiva:
                x_f = dataset_chunk.loc[:, x_miss]

                if deterministic:
                    # Observed feature
                    ordered_id = x_f.sort_values()
                    pos_xmiss = FeatureChoice.miss_locations(
                    ordered_id, threshold, N
                )

                else:
                    # Unobserved random feature
                    ordered_id = np.lexsort((np.random.random(x_f.size), x_f))
                    pos_xmiss = FeatureChoice.miss_locations(
                    ordered_id, threshold, N
                )

                dataset_chunk.loc[pos_xmiss, x_miss] = np.nan

                xmiss_multiva.append(x_miss)
                options_xmiss.remove(x_miss)
                cont += 1

        if not missTarget:
            dataset_chunk['target'] = y_chunk

        return dataset_chunk
    
    @staticmethod
    def _correlated_strategy_to_generate_md(args):
        """
        Function to generate missing data in dataset based on correlated pair.
        The feature (x_miss) most correlated with the class for each pair will
        receive the missing data based on lower values of an unobserved feature
        or feature x_miss itself.

        Args:
            missing_rate (int, optional): The rate of missing data to be generated. Default is 10.
            deterministc (bool, optinal): A flag that determine if x_miss will have miss 
            locations based on itself or an unobserved feature. Default is False
            (i.e., an unobserved feature).

        Returns:
            dataset (DataFrame): The dataset with missing values generated under
            the MNAR mechanism.

        Reference:
        [1] Santos, M. S., R. C. Pereira, A. F. Costa, J. P. Soares, J. Santos, and
        P. H. Abreu. 2019. Generating Synthetic Missing Data: A Review by Missing Mechanism.
        IEEE Access 7: 11651–67.
        """
        dataset_chunk, n_xmiss, missing_rate, y_chunk, missTarget, threshold, deterministic,depend_on_external, ascending,randomness,columns,statistical_method = args
        dataset_chunk = dataset_chunk.reset_index(drop=True)
        # Creation of pairs/triples
        pairs,correlation_matrix = FeatureChoice._make_pairs(dataset_chunk, y_chunk, missTarget)
        mr = missing_rate / 100

        for pair in pairs:
            if len(pair) % 2 == 0:
                cutK = 2 * mr
                # Find the feature most correlated with the target
                x_miss = FeatureChoice._find_most_correlated_feature_even(
                    correlation_matrix, pair,
                )

            else:
                cutK = 1.5 * mr
                # Find the feature most correlated with the target
                x_miss = FeatureChoice._find_most_correlated_feature_odd(
                    correlation_matrix, pair
                )

            N = round(len(dataset_chunk) * cutK)

            # Determine whether input is a single feature or a list of features
            x_miss_list = [x_miss] if isinstance(x_miss, str) else x_miss

            for val in x_miss_list:
                try:
                    x_f = dataset_chunk.loc[:, val]
                except KeyError as error:
                    print("Caiu aqui")

                # Sort values deterministically or randomly
                if deterministic:
                    ordered_id = x_f.sort_values()
                else:
                    ordered_id = np.lexsort((np.random.random(x_f.size), x_f))
                
                # Get positions for missing data
                pos_xmiss = FeatureChoice.miss_locations(ordered_id, threshold, N)
                
                # Introduce missing values
                dataset_chunk.loc[pos_xmiss, val] = np.nan

        return dataset_chunk
    
    @staticmethod
    def _median_strategy_to_generate_md(args):
        """
        Function to generate missing data in all dataset based on median from
        each feature. The miss locations are chosen by lower values from a unobserved feature
        or feature x_miss itself.

        Args:
            missing_rate (int, optional): The rate of missing data to be generated. Default is 10.
            deterministc (bool, optinal): A flag that determine if x_miss will have miss 
            locations based on itself or an unobserved feature. Default is False
            (i.e., an unobserved feature).

        Returns:
            dataset (DataFrame): The dataset with missing values generated under
            the MNAR mechanism.

        Reference:
        [1] Santos, M. S., R. C. Pereira, A. F. Costa, J. P. Soares, J. Santos, and
        P. H. Abreu. 2019. Generating Synthetic Missing Data: A Review by Missing Mechanism.
        IEEE Access 7: 11651–67.
        """
        dataset_chunk, n_xmiss, missing_rate, y_chunk, missTarget, threshold, deterministic,depend_on_external, ascending,randomness,columns,statistical_method = args
        dataset_chunk = dataset_chunk.reset_index(drop=True)
        mr = missing_rate / 100

        if not missTarget:
            dataset_chunk['target'] = y_chunk

        # For each column in dataset
        for col in dataset_chunk.columns:
            N = round(len(dataset_chunk) * mr)

            if len(dataset_chunk[col].unique()) == 2:  # Binary feature
                g1, g2 = MathCalcs.define_groups(dataset_chunk, col)

            else:  # Continuos or ordinal feature
                median_xobs = dataset_chunk[col].median()

                g1 = dataset_chunk[col][dataset_chunk[col] <= median_xobs]
                g2 = dataset_chunk[col][dataset_chunk[col] > median_xobs]

                # If median do not divide in equal-size groups
                if len(g1) != len(g2):  
                    g1, g2 = MathCalcs.define_groups(dataset_chunk, col)

            choice = np.random.choice([0, 1])
            if choice == 0:
                g1_index = g1.index
                x_f = dataset_chunk.loc[g1_index, col]
            else:
                g2_index = g2.index
                x_f = dataset_chunk.loc[g2_index, col]

            if deterministic:
                # Observed feature
                ordered_id = x_f.sort_values()
                pos_xmiss = FeatureChoice.miss_locations(
                ordered_id, threshold, N
            )

            else:
                # Unobserved random feature
                ordered_id = np.lexsort((np.random.random(x_f.size), x_f))
                pos_xmiss = FeatureChoice.miss_locations(
                ordered_id, threshold, N
            )

            dataset_chunk.loc[pos_xmiss, col] = np.nan
        return dataset_chunk
    
    @staticmethod
    def _MBOUV_strategy_to_generate_md(args):
        """
        Function to generate missing data based on Missigness Based on Own and Unobserved Values (MBOUV).

        Args:
            missing_rate (int, optional): The rate of missing data to be generated. Default is 10.

        Returns:
            dataset (DataFrame): The dataset with missing values generated under
            the MNAR mechanism.

        Reference:
        [2] Pereira, R. C., Abreu, P. H., Rodrigues, P. P., Figueiredo, M. A. T., (2024). 
        Imputation of data Missing Not at Random: Artificial generation and benchmark analysis. 
        Expert Systems with Applications, 249(B), 123654.

        """
        dataset_chunk, n_xmiss, missing_rate, y_chunk, missTarget, threshold, deterministic,depend_on_external, ascending,randomness,columns,statistical_method = args
        dataset_chunk = dataset_chunk.reset_index(drop=True)

        if depend_on_external is None:
            depend_on_external = []

        n = dataset_chunk.shape[0]
        p = dataset_chunk.shape[1]    
        
        mr = missing_rate / 100
        N = round(mr * n * p)

        # With moderate missing rates (up to 60%), it never happens.
        max_mvs_feat = round(min(mr + (mr / 2.0), 0.9) * n)

        # Randomly distribute missing values by the features.
        num_mvs_per_feat = np.zeros(p)
        while N > 0:
            for i in range(p):
                if num_mvs_per_feat[i] < max_mvs_feat:
                    num_mvs_it = np.random.randint(
                        0, min(max_mvs_feat - num_mvs_per_feat[i] + 1, N + 1)
                    )
                    N -= num_mvs_it
                    num_mvs_per_feat[i] += num_mvs_it
                    if N == 0:
                        break

        np.random.shuffle(num_mvs_per_feat)
        hidden_f = np.random.normal(size=n)
        tie_breaker = np.random.random(hidden_f.size)

        # Amputate the values.
        for i, col in enumerate(dataset_chunk.columns.values):
            num_mv = round(num_mvs_per_feat[i])
            num_mv = num_mv if num_mv > 0 else 0

            if col in depend_on_external:
                start_n = end_n = int(num_mv / 2)
                if num_mv % 2 == 1:
                    end_n += 1

                indices_start = np.lexsort((tie_breaker, hidden_f))[:start_n]
                indices_end = np.lexsort((tie_breaker, -hidden_f))[:end_n]

                dataset_chunk.loc[indices_start, col] = np.nan
                dataset_chunk.loc[indices_end, col] = np.nan
            else:
                ordered_indices = FeatureChoice.get_ordered_indices(
                    col, dataset_chunk, ascending
                )
                dataset_chunk.loc[ordered_indices[:num_mv], col] = np.nan

        return dataset_chunk
    
    @staticmethod
    def _MBOV_randomness_strategy_to_generate_md(args):
        """
        Function to generate missing data based on Missigness Based on Own Values (MBOV) using
        a randomess to choose miss locations in each feature.

        Args:
            missing_rate (int, optional): The rate of missing data to be generated. Default is 10.
            randomness (float, optional): The randomness rate for choose miss locations. Default is 0 that represents lower values
            columns (list): A list of strings containing columns names.

        Returns:
            dataset (DataFrame): The dataset with missing values generated under
            the MNAR mechanism.

        Reference:
        [2] Pereira, R. C., Abreu, P. H., Rodrigues, P. P., Figueiredo, M. A. T., (2024). 
        Imputation of data Missing Not at Random: Artificial generation and benchmark analysis. 
        Expert Systems with Applications, 249(B), 123654.

        """
        dataset_chunk, n_xmiss, missing_rate, y_chunk, missTarget, threshold, deterministic,depend_on_external, ascending,randomness,columns,statistical_method = args
        dataset_chunk = dataset_chunk.reset_index(drop=True)
        mr = missing_rate / 100
        N = round(len(dataset_chunk) * mr)
        aux_index = list(dataset_chunk.index)

        for col in columns:
            pos_xmis_deterministic = (
                dataset_chunk[col]
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

            dataset_chunk.loc[pos_xmis, col] = np.nan

        return dataset_chunk

    
    @staticmethod
    def _MBOV_median_strategy_to_generate_md(args):
        """
        Function to generate missing data based on Missigness Based on Own Values (MBOV) using
        a median to choose miss locations in each feature.

        Args:
            missing_rate (int, optional): The rate of missing data to be generated. Default is 10.
            columns (list): A list of strings containing columns names.

        Returns:
            dataset (DataFrame): The dataset with missing values generated under
            the MNAR mechanism.

        Reference:
        [2] Pereira, R. C., Abreu, P. H., Rodrigues, P. P., Figueiredo, M. A. T., (2024). 
        Imputation of data Missing Not at Random: Artificial generation and benchmark analysis. 
        Expert Systems with Applications, 249(B), 123654.

        """
        dataset_chunk, n_xmiss, missing_rate, y_chunk, missTarget, threshold, deterministic,depend_on_external, ascending,randomness,columns,statistical_method = args
        dataset_chunk = dataset_chunk.reset_index(drop=True)
        mr = missing_rate / 100
        N = round(len(dataset_chunk) * mr)

        for col in columns:
            if dataset_chunk[col].dtype == 'object':
                raise TypeError(
                    'Only continuos or ordinal feature are accepted'
                )

            median_att = dataset_chunk[col].median()

            pos_xmis = np.argsort(np.abs(dataset_chunk[col] - median_att))[:N]
            dataset_chunk.loc[pos_xmis, col] = np.nan

        return dataset_chunk
        
    
    @staticmethod
    def _MBIR_strategy_to_generate_md(args):
        """
        Function to generate missing data based on Missingness Based on Intra-Relation (MBIR).

        Args:
            missing_rate (int, optional): The rate of missing data to be generated. Default is 10.
            columns (list): A list of strings containing columns names.
            statistical_method (str, optional): A string to inform statistical method. 
            The options are ["Mann-Whitney", "Bayesian"]. Default is Mann-Whitney

        Returns:
            dataset (DataFrame): The dataset with missing values generated under
            the MNAR mechanism.

        Reference:
        [2] Pereira, R. C., Abreu, P. H., Rodrigues, P. P., Figueiredo, M. A. T., (2024). 
        Imputation of data Missing Not at Random: Artificial generation and benchmark analysis. 
        Expert Systems with Applications, 249(B), 123654.

        """
        dataset_chunk, n_xmiss, missing_rate, y_chunk, missTarget, threshold, deterministic,depend_on_external, ascending,randomness,columns,statistical_method = args
        dataset_chunk = dataset_chunk.reset_index(drop=True)
        mr = missing_rate / 100
        N = round(len(dataset_chunk) * mr)

        if not missTarget:
            dataset_chunk['target'] = y_chunk

        df_columns = dataset_chunk.columns.tolist()

        for x_miss in columns:
            most_significant_diff = {}
            for x_obs in df_columns:
                if dataset_chunk[x_obs].dtype != 'object':

                    instances = dataset_chunk.copy()

                    if x_obs != x_miss:
                        # Find the instance with lower MR% values of x_obs
                        lower = instances[x_obs].sort_values()[:N].index

                        # Set the f_miss to be missing
                        instances.loc[lower, x_miss] = np.nan

                        # Create the missing indicator
                        auxiliary_ind = np.where(
                            instances[x_miss].isna(), 1, 0
                        )

                        # Statistical tests between values of f_obs and f_ind
                        match statistical_method:
                            case 'Mann-Whitney':
                                statistic, p_value = mannwhitneyu(
                                    instances[x_obs], auxiliary_ind
                                )

                        if p_value < 0.05:
                            # There is evidence of a significant difference.
                            most_significant_diff[x_obs] = p_value

            most_feature = min(
                most_significant_diff, key=most_significant_diff.get
            )
            pos_xmis = dataset_chunk[most_feature].sort_values()[:N].index
            dataset_chunk.loc[pos_xmis, x_miss] = np.nan
            df_columns.remove(most_feature)
            dataset_chunk = dataset_chunk.drop(columns=most_feature)

        
        return dataset_chunk
    
    def random(self, missing_rate: int = 10,deterministic:bool = True):
        """Generate missing data using parallel processing."""
        if missing_rate >= 100:
            raise ValueError(
                'Missing Rate is too high, the whole feature will be deleted!'
            )
        elif missing_rate < 0:
            raise ValueError('Missing rate must be between [0,100]')

        chunks = self._set_chuck_size(missing_rate=missing_rate, 
                                      deterministic=deterministic,
                                      depend_on_external=None,
                                      ascending=None,
                                      randomness=None,
                                      columns=None,
                                      statistical_method=None)

        # Use multiprocessing Pool to parallelize
        with Pool(processes=self.n_Threads) as pool:
            results = pool.map(mMNAR._random_strategy_to_generate_md, chunks)

        # Combine the results into a single DataFrame
        final_dataset = pd.concat(results, axis=0)
        return final_dataset

    def correlated(self, missing_rate: int = 10, deterministic:bool = False):
        """Generate missing data using parallel processing."""

        if missing_rate >= 50:
            raise ValueError(
                'Features will be all NaN, you should decrease the missing rate'
            )
        
        chunks = self._set_chuck_size(missing_rate=missing_rate, 
                                      deterministic=deterministic,
                                      depend_on_external=None,
                                      ascending=None,
                                      randomness=None,
                                      columns=None,
                                      statistical_method=None)

        # Use multiprocessing Pool to parallelize
        with Pool(processes=self.n_Threads) as pool:
            results = pool.map(mMNAR._correlated_strategy_to_generate_md, chunks)

        # Combine the results into a single DataFrame
        final_dataset = pd.concat(results, axis=0)
        return final_dataset

    def median(self, missing_rate: int = 10, deterministic:bool = False):
        """Generate missing data using parallel processing."""
        
        if missing_rate >= 50:
            raise ValueError(
                'Features will be all NaN, you should decrease the missing rate'
            )

        chunks = self._set_chuck_size(missing_rate=missing_rate, 
                                      deterministic=deterministic,
                                      depend_on_external=None,
                                      ascending=None,
                                      randomness=None,
                                      columns=None,
                                      statistical_method=None)

        # Use multiprocessing Pool to parallelize
        with Pool(processes=self.n_Threads) as pool:
            results = pool.map(mMNAR._median_strategy_to_generate_md, chunks)

        # Combine the results into a single DataFrame
        final_dataset = pd.concat(results, axis=0)
        return final_dataset

    def MBOUV(
        self, missing_rate: int = 10, depend_on_external=None, ascending=True
    ):
        """Generate missing data using parallel processing."""
        
        if missing_rate > 90:
            raise ValueError(
                'Maximum missing rate per feature must be clipped at 90%'
            )
        elif missing_rate < 0:
            raise ValueError('Missing rate must be between [0,100]')
        
        chunks = self._set_chuck_size(missing_rate=missing_rate, 
                                      deterministic=None,
                                      depend_on_external=depend_on_external,
                                      ascending=ascending,
                                      randomness=None,
                                      columns=None,
                                      statistical_method=None)

        # Use multiprocessing Pool to parallelize
        with Pool(processes=self.n_Threads) as pool:
            results = pool.map(mMNAR._MBOUV_strategy_to_generate_md, chunks)

        # Combine the results into a single DataFrame
        final_dataset = pd.concat(results, axis=0)
        return final_dataset

    def MBOV_randomness(
        self,
        missing_rate: int = 10,
        randomness: float = 0,
        columns: list = None
    ):
        """Generate missing data using parallel processing."""
        
        if not (0 <= randomness <= 0.5):
            raise ValueError('randomness must be in range [0,0.5]')

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
        
        chunks = self._set_chuck_size(missing_rate=missing_rate, 
                                      deterministic=None,
                                      depend_on_external=None,
                                      ascending=None,
                                      randomness=randomness,
                                      columns=columns,
                                      statistical_method=None)

        # Use multiprocessing Pool to parallelize
        with Pool(processes=self.n_Threads) as pool:
            results = pool.map(mMNAR._MBOV_randomness_strategy_to_generate_md, chunks)

        # Combine the results into a single DataFrame
        final_dataset = pd.concat(results, axis=0)
        return final_dataset


    def MBOV_median(self, missing_rate: int = 10, columns: list = None):
        """Generate missing data using parallel processing."""
        

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
        
        chunks = self._set_chuck_size(missing_rate=missing_rate, 
                                      deterministic=None,
                                      depend_on_external=None,
                                      ascending=None,
                                      randomness=None,
                                      columns=columns,
                                      statistical_method=None)

        # Use multiprocessing Pool to parallelize
        with Pool(processes=self.n_Threads) as pool:
            results = pool.map(mMNAR._MBOV_median_strategy_to_generate_md, chunks)

        # Combine the results into a single DataFrame
        final_dataset = pd.concat(results, axis=0)
        return final_dataset


    def MBIR(
        self,
        missing_rate: int = 10,
        columns: list = None,
        statistical_method: str = 'Mann-Whitney',
    ):
        """Generate missing data using parallel processing."""
        
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

        chunks = self._set_chuck_size(missing_rate=missing_rate, 
                                      deterministic=None,
                                      depend_on_external=None,
                                      ascending=None,
                                      randomness=None,
                                      columns=columns,
                                      statistical_method=statistical_method)

        # Use multiprocessing Pool to parallelize
        with Pool(processes=self.n_Threads) as pool:
            results = pool.map(mMNAR._MBIR_strategy_to_generate_md, chunks)

        # Combine the results into a single DataFrame
        final_dataset = pd.concat(results, axis=0)
        return final_dataset


if __name__ == "__main__":
    import numpy as np 
    import pmlb
    import os

    # Function to help split data
    def split_data(data):
        df = data.copy()
        X = df.drop(columns=["target"])
        y = data["target"]

        return X,np.array(y)

    # The data from PMLB
    kddcup = pmlb.fetch_data('kddcup')
    X_, y_ = split_data(kddcup)


    generator = mMNAR(X=X_, y=y_, n_Threads=os.cpu_count())
    gen_md = generator.MBOV_randomness(missing_rate=35, columns=X_.columns)
    print(sum(gen_md.isna().sum()) / (np.shape(gen_md)[0] * np.shape(gen_md)[1]))