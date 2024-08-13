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
        X: pd.DataFrame, y: np.array, x_miss: str = None, x_obs:str = None, flag:bool=True
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
    def _delete(item: str, list_items: np.array) -> np.array:
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
    def _make_pairs(X: pd.DataFrame, y: np.array):

        (
            matriz_correlacao,
            matriz_correlacao_X,
        ) = MathCalcs._calculate_correlation(X, y)
        remaining_features = np.array(matriz_correlacao_X.columns)
        pairs = []

        cont = 0
        while len(remaining_features) > 0:
            col = remaining_features[cont]
            var_corr_id = matriz_correlacao_X[col].drop(col).abs().idxmax()
            try:
                if np.isnan(var_corr_id):
                   options = FeatureChoice._delete(col, remaining_features)
                   var_corr_id = np.random.choice(options)
            except TypeError as e:
                pass

            if col in remaining_features and var_corr_id in remaining_features:
                remaining_features = FeatureChoice._delete(
                    col, remaining_features
                )
                remaining_features = FeatureChoice._delete(
                    var_corr_id, remaining_features
                )
                matriz_correlacao_X = matriz_correlacao_X.drop(
                    index=col, columns=col
                )
                matriz_correlacao_X = matriz_correlacao_X.drop(
                    index=var_corr_id, columns=var_corr_id
                )
                pairs.append((col, var_corr_id))

                # Number of features is even
                if len(remaining_features) == 2:
                    pairs.append(
                        (remaining_features[0], remaining_features[1])
                    )
                    return pairs

                elif len(remaining_features) == 1:  # Number of features is odd
                    # Find which is most correlated with the remaining feature
                    col_remaining = remaining_features[cont]
                    var_remaining = MathCalcs._find_correlation(
                        X, y, col_remaining, flag=True
                    )
                    pairs = FeatureChoice._add_var_remaining(
                        pairs, var_remaining, col_remaining
                    )
                    return pairs
            
    # ------------------------------------------------------------------------
    @staticmethod
    def _find_most_correlated_feature_even(
        X: pd.DataFrame, y: np.array, pair_features
    ):
        (
            matriz_correlacao,
            matriz_correlacao_X,
        ) = MathCalcs._calculate_correlation(X, y)
        corr_0 = abs(matriz_correlacao['target'][pair_features[0]])
        corr_1 = abs(matriz_correlacao['target'][pair_features[1]])

        return pair_features[0] if corr_0 > corr_1 else pair_features[1]

    # ------------------------------------------------------------------------
    @staticmethod
    def _find_most_correlated_feature_odd(
        X: pd.DataFrame, y: np.array, pair_features
    ):
        (
            matriz_correlacao,
            matriz_correlacao_X,
        ) = MathCalcs._calculate_correlation(X, y)
        corr_0 = abs(matriz_correlacao['target'][pair_features[0]])
        corr_1 = abs(matriz_correlacao['target'][pair_features[1]])
        corr_2 = abs(matriz_correlacao['target'][pair_features[2]])

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
            pos_xmiss = np.hstack([lowest.index, highest.index])
        elif isinstance(ordered_id, np.ndarray):
            pos_xmiss = np.hstack([lowest, highest])
        else:
            raise TypeError('ordered_id must be pd.Series or np.array')

        return pos_xmiss
