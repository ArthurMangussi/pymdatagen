# -*- coding: utf-8 -*

#  =============================================================================
# Aeronautics Institute of Technologies (ITA) - Brazil
# University of Coimbra (UC) - Portugal
# Arthur Dantas Mangussi - mangussiarthur@gmail.com
# =============================================================================

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from mdatagen.utils.feature_choice import FeatureChoice
from mdatagen.utils.math_calcs import MathCalcs
from multiprocessing.pool import Pool
from pyampute.ampute import MultivariateAmputation

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

    def __init__(self, X: pd.DataFrame, 
                 y: np.ndarray, 
                 n_xmiss: int = 2, 
                 missTarget:bool=False,
                 n_Threads:int = 1):
        if not isinstance(X, pd.DataFrame):
            raise TypeError('Dataset must be a Pandas Dataframe')
        if not isinstance(y, np.ndarray):
            raise TypeError('y must be a numpy array')

        self.n_xmiss = n_xmiss
        self.X = X
        self.y = y
        self.dataset = self.X.copy()
        self.missTarget = missTarget
        self.n_Threads = n_Threads

        if missTarget:
            self.dataset['target'] = y

    def _set_chuck_size(self, missing_rate:int)->list:
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
                self.missTarget
            )
            for i in range(0, len(self.dataset), chunk_size)
        ]
        return chunks
    
    @staticmethod
    def _random_strategy_to_generate_md(args):
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
        dataset_chunk, n_xmiss, missing_rate, y_chunk, missTarget = args
        
        n = dataset_chunk.shape[0]
        p = dataset_chunk.shape[1]
        mr = missing_rate / 100

        N = round((n * p * mr) / n_xmiss)

        if n_xmiss > p:
            raise ValueError('n_xmiss must be lower than number of features in dataset')
        elif N > n:
            raise ValueError(f'FEATURES will be all NaN ({N}>{n}), you should increase the number of features that will receive missing data')

        xmiss_multiva = []
        cont = 0

        while cont < n_xmiss:
            x_obs = np.random.choice(dataset_chunk.columns, replace=False)
            x_miss = np.random.choice([x for x in dataset_chunk.columns if x != x_obs])

            if x_miss not in xmiss_multiva:
                pos_xmiss = dataset_chunk[x_obs].sort_values()[:N].index
                dataset_chunk.loc[pos_xmiss, x_miss] = np.nan

                xmiss_multiva.append(x_miss)
                cont += 1

        if not missTarget:
            dataset_chunk['target'] = y_chunk
        return dataset_chunk
    
    @staticmethod
    def _correlated_strategy_to_generate_md(args):
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
        dataset_chunk, n_xmiss, missing_rate, y_chunk, missTarget = args
        dataset_chunk = dataset_chunk.reset_index(drop=True)
        mr = missing_rate / 100

        pairs, correlation_matrix = FeatureChoice._make_pairs(dataset_chunk, y_chunk, missTarget)

        for pair in pairs:
            if len(pair) % 2 == 0:
                cutK = 2 * mr
                x_miss = FeatureChoice._find_most_correlated_feature_even(
                    correlation_matrix, pair
                )
                x_obs = next(elem for elem in pair if elem != x_miss)

            else:
                x_miss = FeatureChoice._find_most_correlated_feature_odd(
                   correlation_matrix, pair
                )
                x_obs = set(pair).difference(x_miss).pop()
                cutK = 1.5 * mr

            N = round(len(dataset_chunk) * cutK)
            pos_xmiss = dataset_chunk[x_obs].sort_values()[:N].index
            dataset_chunk.loc[pos_xmiss, x_miss] = np.nan

        return dataset_chunk

    @staticmethod
    def _median_strategy_to_generate_md(args):
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
        dataset_chunk, n_xmiss, missing_rate, y_chunk, missTarget = args
        dataset_chunk = dataset_chunk.reset_index(drop=True)

        mr = missing_rate / 100

        pairs,_ = FeatureChoice._make_pairs(dataset_chunk, y_chunk, missTarget)

        for pair in pairs:
            x_obs = np.random.choice(pair)

            if len(pair) % 2 == 0:
                cutK = 2 * mr
                x_miss = next(elem for elem in pair if elem != x_obs)
            else:
                cutK = 1.5 * mr
                x_miss = list(pair)
                x_miss.remove(x_obs)

            N = round(len(dataset_chunk) * cutK)

            if len(dataset_chunk[x_obs].unique()) == 2:  # Binary feature
                g1, g2 = MathCalcs.define_groups(dataset_chunk, x_obs)

            else:  # Continuos or ordinal feature
                median_xobs = dataset_chunk[x_obs].median()

                g1 = dataset_chunk[x_obs][dataset_chunk[x_obs] <= median_xobs]
                g2 = dataset_chunk[x_obs][dataset_chunk[x_obs] > median_xobs]

                if len(g1) != len(g2):  # If median do not divide in equal-size groups
                    g1, g2 = MathCalcs.define_groups(dataset_chunk, x_obs)

            choice = np.random.choice([0, 1])
            if choice == 0:
                pos_xmiss = g1.sort_values()[:N].index

            else:
                pos_xmiss = g2.sort_values()[:N].index

            dataset_chunk.loc[pos_xmiss, x_miss] = np.nan

        if not missTarget:
            dataset_chunk['target'] = y_chunk
        return dataset_chunk
    
    @staticmethod
    def _translate_params(data: pd.DataFrame, incomplete_vars):
        """
        Function to transform the columns names into index
        """

        # Se os incomplete_vars forem nomes, converte para índices
        if isinstance(incomplete_vars[0], str):
            incomplete_vars_idx = [data.columns.get_loc(c) for c in incomplete_vars]
            
        else:
            incomplete_vars_idx = incomplete_vars


        return incomplete_vars_idx
    
    @staticmethod
    def _apply_pyampute_on_chunk(args):
        (dataset_chunk, 
         patterns, 
         missing_rate, 
         std, 
         verbose, 
         seed, 
         lower_range, 
         upper_range, 
         max_diff_with_target, 
         max_iter) = args

        try:
            ma = MultivariateAmputation(
                prop=missing_rate / 100,
                patterns=patterns,
                std=std,
                verbose=verbose,
                seed=seed,
                lower_range=lower_range,
                upper_range=upper_range,
                max_diff_with_target=max_diff_with_target,
                max_iter=max_iter
            )
            data_amputed = ma.fit_transform(dataset_chunk.values)
            return pd.DataFrame(data_amputed, columns=dataset_chunk.columns)
        except Exception as error:
            raise ValueError(error)
    
    def random(self, missing_rate: int = 10) -> pd.DataFrame:
        """Generate missing data using parallel processing."""
        
        if missing_rate >= 100:
            raise ValueError('Missing Rate is too high, the whole feature will be deleted!')
        elif missing_rate < 0:
            raise ValueError('Missing rate must be between [0,100]')

        chunks = self._set_chuck_size(missing_rate)

        # Use multiprocessing Pool to parallelize
        with Pool(processes=self.n_Threads) as pool:
            results = pool.map(mMAR._random_strategy_to_generate_md, chunks)

        # Combine the results into a single DataFrame
        final_dataset = pd.concat(results, axis=0)
        return final_dataset
    
    def correlated(self, missing_rate: int = 10) -> pd.DataFrame:
        """Generate missing data using parallel processing."""        
        if missing_rate >= 50:
            raise ValueError(
                'Features will be all NaN, you should decrease the missing rate'
            )
        
        chunks = self._set_chuck_size(missing_rate)

        # Use multiprocessing Pool to parallelize
        with Pool(processes=self.n_Threads) as pool:
            results = pool.map(mMAR._correlated_strategy_to_generate_md, chunks)

        # Combine the results into a single DataFrame
        final_dataset = pd.concat(results, axis=0)
        return final_dataset
    
    def median(self, missing_rate: int = 10) -> pd.DataFrame:
        """Generate missing data using parallel processing.""" 
        if missing_rate > 25:
            raise ValueError(
                'The missing rate must be in the range [0, 25]'
            )
                      
        chunks = self._set_chuck_size(missing_rate)

        # Use multiprocessing Pool to parallelize
        with Pool(processes=self.n_Threads) as pool:
            results = pool.map(mMAR._median_strategy_to_generate_md, chunks)

        # Combine the results into a single DataFrame
        final_dataset = pd.concat(results, axis=0)
        return final_dataset
    
    def pattern_missingness(self, 
                            patterns: List[Dict]=None,
                            missing_rate: int = 10,
                            std:bool = True,                  
                            verbose: bool = False,
                            seed: Optional[int] = None,
                            lower_range: float = -3,
                            upper_range: float = 3,
                            max_diff_with_target: float = 0.001,
                            max_iter: int = 100):
        """
        Generate missing data using pattern-based multivariate amputation.

        References:
        [2] van Buuren, S., J. P. L. Brand, C. G. M. Groothuis-Oudshoorn, and D. B. Rubin.
        Fully conditional specification in multivariate imputation.
        Journal of Statistical Computation and Simulation, 76(12):1049–1064, 2006.

        [3] Schouten, R. M., P. Lugtig, and G. Vink.
        Generating missing values for simulation purposes: a multivariate amputation procedure.
        Journal of Statistical Computation and Simulation, 88(15):2909–2930, 2018.

        """
        if missing_rate >= 100:
            raise ValueError(
                'Missing Rate is too high, the whole dataset will be deleted!'
            )
        
        if patterns is None:
            incomplete_f = np.random.choice(self.X.columns, 
                                            size=int(self.X.shape[1]*0.5), 
                                            replace=False)
            incomplete_features = self._translate_params(self.X, incomplete_f)

            patterns = [{
                "score_to_probability_func": "SIGMOID-RIGHT",
                "mechanism": "MAR",
                "incomplete_vars": incomplete_features,   
                "freq": 1
            }]

        
        chunks = self._set_chuck_size(missing_rate)
        chunk_args = [
            (
                chunk_data,
                patterns,
                missing_rate,
                std,
                verbose,
                seed,
                lower_range,
                upper_range,
                max_diff_with_target,
                max_iter
            )
            for (chunk_data, _, _, _, _) in chunks
        ]

        with Pool(processes=self.n_Threads) as pool:
            results = pool.map(mMAR._apply_pyampute_on_chunk, chunk_args)

        final_dataset = pd.concat(results, axis=0).reset_index(drop=True)
        return final_dataset
