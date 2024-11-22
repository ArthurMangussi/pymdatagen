# -*- coding: utf-8 -*

#  =============================================================================
# Aeronautics Institute of Technologies (ITA) - Brazil
# University of Coimbra (UC) - Portugal
# Arthur Dantas Mangussi - mangussiarthur@gmail.com
# =============================================================================

import pandas as pd
from typing import Optional
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error,
                             mean_absolute_percentage_error,
                             r2_score
                            )
from scipy.stats import kstest, pearsonr
import numpy as np

# ==========================================================================
class EvaluateImputation:
    """
    A class to evaluate the performance of imputed data against the 
    original data using various metrics.

    Attributes
    ----------
    original : pd.DataFrame
        The original data before imputation.
    data_imputed : pd.DataFrame
        The data after imputation.
    metric : str, optional. Default is "mean_absolute_error".
        The metric to use for evaluating the imputed data. Supported metrics are:
        - "mean_absolute_error"
        - "mean_squared_error"
        - "root_mean_squared_error"
        - "kolmogorov_smirnov_distance" 
        - "pearson_correlation_coefficient" 
        - "proportion_falsely_classified_entries"
        - "mean_absolute_percentage_error" 
        - "normalize_mean_absolute_error"
        - "percentage_of_correct_prediction"
        - "R2"
    data_missing : pd.DataFrame, optional
        The data with missing values. Currently supports as a parameter
        for "proportion_falsely_classified_entries". 
    
    Example:
    import pandas as pd
    from sklearn.datasets import load_iris

    from mdatagen.univariate.uMCAR import uMCAR

    # Load the data
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

    X = iris_df.copy()   # Features
    y = iris.target    # Label values

    # Create a instance with missing rate equal to 25% in dataset under MCAR mechanism
    generator = uMCAR(X=X, y=y, missing_rate=25, x_miss='petal length (cm)')

    # Generate the missing data under MNAR mechanism
    generate_data = generator.random()

    eval_metric = EvaluateImputation(data_imputed=generate_data.drop("target",axis=1).fillna(0),
                                     data_original=X)
    eval_metric.show()

    """
    def __init__(self, 
                 data_imputed:pd.DataFrame,
                 data_original:pd.DataFrame, 
                 metric:Optional[str] = "mean_absolute_error",
                 data_missing:pd.DataFrame = None
                 ) -> None:
        
        if data_imputed.shape != data_original.shape:
            raise ValueError("Different shapes for the original and imputed data")
        self.original = data_original
        self.data_imputed = data_imputed
        self.metric = metric
        self.data_missing = data_missing
        
    # ------------------------------------------------------------------------
    def _choose_metric(self):
        """
        Selects and executes the appropriate metric function based on the provided metric.

        Returns
        -------
        float
            The result of the selected error metric.

        """

        metrics = {
            "mean_absolute_error": self._mean_absolute_error,
            "mean_squared_error": self._mean_squared_error,
            "root_mean_squared_error": self._root_mean_squared_error,
            "kolmogorov_smirnov_distance": self._kolmogorov_smirnov_distance,
            "pearson_correlation_coefficient": self._pearson_correlation_coefficient,
            "proportion_falsely_classified_entries": self._proportion_falsely_classified_entries,
            "mean_absolute_percentage_error": self._mean_absolute_percentage_error,
            "normalize_mean_absolute_error": self._normalize_mean_absolute_error,
            "percentage_of_correct_prediction": self._percentage_of_correct_prediction,
            "R2": self._r2_Score,
            "matches":self._matches
        }
        
        self.metric_function = metrics.get(self.metric)

        if self.metric_function:
            error_metric_result = self.metric_function()
            return error_metric_result
        else:
            raise ValueError(f"Metric '{self.metric}' is not supported.")
    
    # ------------------------------------------------------------------------
    def show(self):
        """
        Calculates and returns the selected error metric.

        Returns
        -------
        float
            The result of the selected error metric.
        """
        return self._choose_metric()

    # ------------------------------------------------------------------------
    def _mean_absolute_error(self):
        """
        Calculates and returns the Mean Absolute Error (MAE).
        Values closer to 0 indicate more accurate imputations.

        Returns
        -------
        float
            The Mean Absolute Error (MAE) between the original and imputed data.
        
        Reference:
        [1] Hasan, Md. K., Alam, Md. A., Roy, S., Dutta, A., Jawad, Md. T., and Das, S. 2021. 
        Missing value imputation affects the performance of machine learning: A review and 
        analysis of the literature (2010–2021). Informatics in Medicine Unlocked. 27. 
        10.1016/j.imu.2021.100799. 
        """
        mae = mean_absolute_error(y_true=self.original,
                                  y_pred=self.data_imputed)
        return mae
    
    # ------------------------------------------------------------------------
    def _mean_squared_error(self):
        """
        Calculates and returns the Mean Squared Error (MSE).
        Values closer to 0 indicate more accurate imputations.

        Returns
        -------
        float
            The Mean Squared Error (MSE) between the original and imputed data.
        
        Reference:
        [1] Hasan, Md. K., Alam, Md. A., Roy, S., Dutta, A., Jawad, Md. T., and Das, S. 2021. 
        Missing value imputation affects the performance of machine learning: A review and 
        analysis of the literature (2010–2021). Informatics in Medicine Unlocked. 27. 
        10.1016/j.imu.2021.100799. 
        """
        mse = mean_squared_error(y_true=self.original,
                                  y_pred=self.data_imputed)
        return mse
    
    # ------------------------------------------------------------------------
    def _root_mean_squared_error(self):
        """
        Calculates and returns the Root Mean Squared Error (RMSE).
        Values closer to 0 indicate more accurate imputations.

        Returns
        -------
        float
            The Root Mean Squared Error (RMSE) between the original and imputed data.

        Reference:
        [2] Lin, W. C., and Tsai, C. F. 2020. Missing value imputation: a review and analysis of the 
        literature (2006–2017). Artif Intell Rev 53, 1487–1509. 
        https://doi.org/10.1007/s10462-019-09709-4
        """
        rmse = mean_squared_error(y_true=self.original,
                                  y_pred=self.data_imputed,
                                  squared=True)
        return rmse
    
    # ------------------------------------------------------------------------
    def _kolmogorov_smirnov_distance(self):
        """
        Calculates and returns the Kolmogorov-Smirnov Distance.

        Returns
        -------
        float
            The Kolmogorov-Smirnov Distance between the original and imputed data.
        
        Reference:
        [3] Nguyen, C. D., Carlin, J. B., and Lee, K.J. 2013. Diagnosing problems with 
        imputation models using the Kolmogorov-Smirnov test: a simulation study. 
        BMC Med Res Methodol 13, 144. 
        https://doi.org/10.1186/1471-2288-13-144
        """
        original = np.array(self.original).flatten()
        imputed = np.array(self.data_imputed).flatten()
        kolmogorov = kstest(rvs=original,
                            cdf=imputed)
        return kolmogorov
    
    # ------------------------------------------------------------------------
    def _pearson_correlation_coefficient(self):
        """
        Calculates and returns the Pearson Correlation Coefficient.

        Returns
        -------
        float
            The Pearson Correlation Coefficient between the original and imputed data.
        
        Reference:
        [4] 
        """
        original = np.array(self.original).flatten()
        imputed = np.array(self.data_imputed).flatten()
        pearson = pearsonr(original,
                           imputed)
        return pearson
    
    # ------------------------------------------------------------------------
    def _proportion_falsely_classified_entries(self):
        """
        Calculates and returns the Proportion of Falsely Classified Entries.

        Returns
        -------
        float
            The Proportion of Falsely Classified Entries between the original and imputed data.

        Reference:
        [5] Guo, C. Y., Yang, Y. C., and Chen, Y. H. 2021. The Optimal Machine Learning-Based Missing Data 
        Imputation for the Cox Proportional Hazard Model. Frontiers in public health, 9, 680054.
        https://doi.org/10.3389/fpubh.2021.680054

        """
        if self.data_missing is None:
            raise ValueError("You have to provide the data with missing values")
        
        original = np.array(self.original).flatten()
        imputed = np.array(self.data_imputed).flatten() 
        not_identical = sum(a != b for a, b in zip(original, imputed))
        pfc = not_identical/sum(np.isnan(np.array(self.data_missing).flatten()))

        return pfc
    
    # ------------------------------------------------------------------------
    def _mean_absolute_percentage_error(self):
        """
        Calculates and returns the Mean Absolute Percentage Error (MAPE).
        Values closer to 0 indicate more accurate imputations.

        Returns
        -------
        float
            The Mean Absolute Percentage Error (MAPE) between the original and imputed data.
        
        Reference:
        [2] Lin, W. C., and Tsai, C. F. 2020. Missing value imputation: a review and analysis of the 
        literature (2006–2017). Artif Intell Rev 53, 1487–1509. 
        https://doi.org/10.1007/s10462-019-09709-4.
        """
        mape = mean_absolute_percentage_error(y_true=self.original,
                                  y_pred=self.data_imputed)
        return mape
    
    # ------------------------------------------------------------------------
    def _normalize_mean_absolute_error(self):
        """
        Calculates and returns the Normalize Mean Absolute Error (NMAE).

        Returns
        -------
        float
            The Normalize Mean Absolute Error (NMAE) between the original and imputed data.
        
        Reference:
        [6] 
        """
        mae = mean_absolute_error(y_true=self.original,
                                  y_pred=self.data_imputed)
        
        max_xj = max(self.original)
        min_xj = min(self.original)
        nmae = mae / (max_xj - min_xj)
        
        return nmae
    
    # ------------------------------------------------------------------------
    def _percentage_of_correct_prediction(self):
        """
        Calculates and returns the Percentage of Correct Predictions (PCP).
        Values closer to 1 indicate more accurate imputations.

        Returns
        -------
        float
            The Percentage of Correct Predictions (PCP) between the original and imputed data.

        Reference:
        [2] Lin, W. C., and Tsai, C. F. 2020. Missing value imputation: a review and analysis of the 
        literature (2006–2017). Artif Intell Rev 53, 1487–1509. 
        https://doi.org/10.1007/s10462-019-09709-4
        """
        original = np.array(self.original).flatten()
        imputed = np.array(self.data_imputed).flatten() 
        correct = sum(a == b for a, b in zip(original, imputed))
        pcp = correct / len(self.original.shape[0])

        return round(100 * pcp, 2)
    
    # ------------------------------------------------------------------------
    def _r2_Score(self):
        """
        Calculates and returns the R2 score (coefficient of determination).

        Returns
        -------
        float
            The R2 (coefficient of determination) between the original and imputed data.
        
        Reference:
        [1] Hasan, Md. K., Alam, Md. A., Roy, S., Dutta, A., Jawad, Md. T., and Das, S. 2021. 
        Missing value imputation affects the performance of machine learning: A review and 
        analysis of the literature (2010–2021). Informatics in Medicine Unlocked. 27. 
        10.1016/j.imu.2021.100799. 
        """
        r2 = r2_score(y_true=self.original,
                      y_pred=self.data_imputed)

        return r2

    
    