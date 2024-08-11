# -*- coding: utf-8 -*-

"""
A basic example of generate artificial missing data with mdatagen library with the Iris dataset from scikit-learn.
The feature petal length will receive the missing values under Missing Completly at Random (MCAR) mechanism.
The simulated missing rate is 25%. The method to choose missing values is random. Our example fills the missing
values with zero and evaluted the imputation quality with Mean Squared Error (MSE).
"""

import pandas as pd
from sklearn.datasets import load_iris

from mdatagen.univariate.uMCAR import uMCAR
from mdatagen.metrics.metrics import EvaluateImputation

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
                                    data_original=X,
                                    metric="mean_squared_error")
print(eval_metric.show())