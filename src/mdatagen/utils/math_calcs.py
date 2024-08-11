# -*- coding: utf-8 -*-

# =============================================================================
# Aeronautics Institute of Technologies (ITA) - Brazil
# University of Coimbra (UC) - Portugal
# Arthur Dantas Mangussi - mangussiarthur@gmail.com
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

# ==========================================================================
class MathCalcs:
    # ------------------------------------------------------------------------
    @staticmethod
    def _find_correlation(
        X: pd.DataFrame, y: np.array, colunm: str, flag:bool=False
    ):
        """
        Calculates the correlation matrix of the dataset and finds the feature that is most correlated with the given column.

        Parameters:
        - coluna: str
            The column name to find the most correlated feature with.
        - flag: bool, optional
            If True, find the new most correlated feature excluding the given column and the target feature. Default is False.

        Returns:
        - str
            The name of the most correlated feature.
        """
        (
            matriz_correlacao,
            matriz_correlacao_X,
        ) = MathCalcs._calculate_correlation(X, y)

        id_feature_mais_correlacionada = (
            matriz_correlacao[colunm].drop(colunm).abs().idxmax()
        )

        if flag:
            id_feature_mais_correlacionada = (
                matriz_correlacao_X[colunm].drop(colunm).abs().idxmax()
            )

        return id_feature_mais_correlacionada

    # ------------------------------------------------------------------------
    @staticmethod
    def _nmi(X: pd.DataFrame, y: np.array):
        """
        Calculates the normalized mutual information (NMI) between each feature in a dataset and the target variable.

        Args:
            dataset (pd.DataFrame): The dataset for which to calculate the NMI. It should contain the target variable named "target" and other features.

        Returns:
            List[tuple]: A list of tuples, where each tuple contains the NMI value and the name of the corresponding feature. The list is sorted in descending order of NMI values.
        """
        dataset = X.copy()
        X = X.loc[:, :].values
        dataset['target'] = y

        mi = mutual_info_classif(X, y)

        caracteristicas_ordenadas = sorted(
            zip(map(lambda x: round(x, 4), mi / sum(mi)), dataset.columns),
            reverse=True,
        )

        return [i for i in caracteristicas_ordenadas]

    # ------------------------------------------------------------------------
    @staticmethod
    def _calculate_correlation(X: pd.DataFrame, y: np.array):
        """
        Updates the correlation matrices used in the MissingDataGenerator class.

        This method recalculates the correlation matrix of the original dataset (self.X) and the complete dataset (self.complete_dataset).

        Inputs:
        - None

        Flow:
        1. Calculate the correlation matrix of the original dataset (self.X) and assign it to self.matriz_correlacao_X.
        2. Calculate the correlation matrix of the complete dataset (self.complete_dataset) and assign it to self.matriz_correlacao.

        Outputs:
        - None
        """
        complete_dataset = X.copy()
        complete_dataset['target'] = y

        matriz_correlacao = complete_dataset.corr()

        matriz_correlacao_X = X.corr()

        return matriz_correlacao, matriz_correlacao_X

    # ------------------------------------------------------------------------
    @staticmethod
    def define_groups(dataset, x_obs):
        g1_index = np.random.choice(
            dataset[x_obs].index, round(len(dataset) / 2), replace=False
        )
        g2_index = np.array([i for i in dataset[x_obs].index if i not in g1_index])

        g1 = dataset.loc[g1_index, x_obs]
        g2 = dataset.loc[g2_index, x_obs]

        return g1, g2
