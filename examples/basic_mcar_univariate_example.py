# -*- coding: utf-8 -*-

"""
A basic example of generate artificial missing data with mdatagen library with the Iris dataset from scikit-learn.
The feature petal length will receive the missing values under Missing Completly at Random (MCAR) mechanism.
The simulated missing rate is 25%. The method to choose missing values is random.
"""

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
print(generate_data.isna().sum())
