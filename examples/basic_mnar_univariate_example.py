# -*- coding: utf-8 -*-

"""
A basic example of generate artificial missing data with mdatagen library with the Iris dataset from scikit-learn.
The most correlated feature with label will receive the missing values under Missing Not at Random (MNAR) mechanism.
The simulated missing rate is 10% (default). The method to choose missing values is lowest.
"""

import pandas as pd
from sklearn.datasets import load_iris

from mdatagen.univariate.uMNAR import uMNAR

# Load the data
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

X = iris_df.copy()   # Features
y = iris.target    # Label values

# Create a instance with missing rate equal to 10% in dataset under MNAR mechanism
generator = uMNAR(X=X, y=y, threshold=0)

# Generate the missing data under MNAR mechanism
generate_data = generator.run()
print(generate_data.isna().sum())