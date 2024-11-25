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
class FeatureChoice:
    @staticmethod
    def _define_xmiss(
        X: pd.DataFrame, y: np.ndarray, x_miss: str = None, x_obs:str = None, flag:bool=True
    ):
        if not x_miss:
            x_miss = MathCalcs._find_correlation(X, y, 'target')

        if not x_obs:
            if flag:
                x_obs = MathCalcs._find_correlation(X, y, x_miss)

                if x_obs == 'target':
                    x_obs = MathCalcs._find_correlation(X, y, x_miss, flag=True)

        return x_miss, x_obs

    # ------------------------------------------------------------------------
    @staticmethod
    def get_ordered_indices(col, dataset, ascending):
        x_f = dataset.loc[:, col].values
        tie_breaker = np.random.random(x_f.size)
        if ascending:
            return np.lexsort((tie_breaker, x_f))
        else:
            return np.lexsort((tie_breaker, x_f))[::-1]

    # ------------------------------------------------------------------------
    @staticmethod
    def _delete(item: str, list_items: np.ndarray) -> np.ndarray:
        """
        Remove the specified item from the list of items.

        Args:
            item (str): The item to be deleted from the list.
            list_items (np.array): The list of items from which the specified item will be removed.

        Returns:
            np.array: The updated list of items with the specified item removed.
        """
        mask = list_items != item
        return list_items[mask]

    # ------------------------------------------------------------------------
    @staticmethod
    def _add_var_remaining(pairs: list, var: str, col: str) -> list:
        """
        Add the column to the pair if the variable is present in the pair.

        Args:
            pairs (list): A list of tuples representing pairs of variables.
            var (str): The variable to check for in the pairs.
            col (str): The column to add to the pairs.

        Returns:
            list: The updated list of pairs with the col added to the tuples where var is present.
        """
        for i, tpl in enumerate(pairs):
            if var in tpl:
                pairs[i] = tpl + (col,)

        return pairs

    # ------------------------------------------------------------------------
    @staticmethod
    def _make_pairs(X: pd.DataFrame, y: np.ndarray, missTarget:bool=False)->list[tuple]:
        df = X.copy()
        df['target'] = y
        
        # If correlation is NaN, we set to 0
        correlation_matrix = df.corr().fillna(0)       
        
        if missTarget:
            # Flatten the correlation matrix and exclude self-correlations
                correlations = (
                    correlation_matrix.where(correlation_matrix != 1.0)
                    .stack()
                    .reset_index()
                )
        else:
            correlations = (
                correlation_matrix.drop(columns="target", index="target").where(correlation_matrix != 1.0)
                .stack()
                .reset_index()
            )
            
        correlations.columns = ["Feature 1", "Feature 2", "Correlation"]

        # Sort by absolute correlation in descending order
        correlations = correlations.sort_values(by="Correlation", key=abs, ascending=False).reset_index(drop=True)

        # Initialize sets to track paired features and the final pairs
        paired_features = set()
        pairs = []

        # Track the feature that is left out (if any)
        unpaired_feature = None
        
        # Iterate through the sorted correlations to find the best pairs
        for _, row in correlations.iterrows():
            f1, f2 = row["Feature 1"], row["Feature 2"]
            if f1 != f2 and f1 not in paired_features and f2 not in paired_features:
                pairs.append((f1, f2))
                paired_features.update([f1, f2])

        # Identify the unpaired feature, if there is any
        all_features = set(correlations['Feature 1']).union(set(correlations['Feature 2']))
        unpaired_feature = list(all_features - paired_features)

        # If there is an unpaired feature, add it to the pair with the highest correlation
        if unpaired_feature:
            # The remaining highest correlation pair is the first in the sorted list
            remaining_pair = pairs[0]  # Pair with the highest correlation (already sorted)
            # Add the unpaired feature to this pair
            pairs[0] = (remaining_pair[0], remaining_pair[1], unpaired_feature[0])

        return pairs, correlation_matrix
            
    # ------------------------------------------------------------------------
    @staticmethod
    def _find_most_correlated_feature_even(
        correlation_matrix:pd.DataFrame, pair_features:list[tuple]
    ):
        
        corr_0 = abs(correlation_matrix['target'][pair_features[0]])
        corr_1 = abs(correlation_matrix['target'][pair_features[1]])

        return pair_features[0] if corr_0 > corr_1 else pair_features[1]

    # ------------------------------------------------------------------------
    @staticmethod
    def _find_most_correlated_feature_odd(
        correlation_matrix:pd.DataFrame, pair_features:list[tuple]
    ):
        
        corr_0 = abs(correlation_matrix['target'][pair_features[0]])
        corr_1 = abs(correlation_matrix['target'][pair_features[1]])
        corr_2 = abs(correlation_matrix['target'][pair_features[2]])

        correlations = [
            (pair_features[0], corr_0),
            (pair_features[1], corr_1),
            (pair_features[2], corr_2),
        ]

        sorted_correlations = sorted(
            correlations, key=lambda x: x[1], reverse=True
        )

        return [t[0] for t in sorted_correlations][:2]

    # ------------------------------------------------------------------------
    @staticmethod
    def miss_locations(ordered_id, threshold, N):

        lowest = ordered_id[: round((1 - threshold) * N)]

        highest = [] if threshold == 0 else ordered_id[round(-threshold * N) :]

        if isinstance(ordered_id, pd.Series):
            if len(lowest) == 0:
                pos_xmiss = np.hstack([lowest, highest.index])
            elif len(highest) == 0:
                pos_xmiss = np.hstack([lowest.index, highest])
            else:
                pos_xmiss = np.hstack([lowest.index, highest.index])

        elif isinstance(ordered_id, np.ndarray):
            pos_xmiss = np.hstack([lowest, highest])
        else:
            raise TypeError('ordered_id must be pd.Series or np.array')

        return pos_xmiss
