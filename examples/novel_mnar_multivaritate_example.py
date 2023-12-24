# -*- coding: utf-8 -*-

"""
A novel example of generate artificial missing data with mdatagen library with the Breast Cancer Wiscosin dataset
from scikit-learn. The features will receive the missing values under
Missing Not at Random (MNAR) mechanism. The simulated missing rate is 20%.
The method to choose missing values is Missingness Based on Own and Unobserved Values (MBOUV).
"""
import pandas as pd
from sklearn.datasets import load_breast_cancer

from mdatagen.multivariate.mMNAR import mMNAR

# Load the data
wiscosin = load_breast_cancer()
wiscosin_df = pd.DataFrame(data=wiscosin.data, columns=wiscosin.feature_names)

X = wiscosin_df.copy()   # Features
y = wiscosin.target    # Label values

# Create a instance with missing rate equal to 20% in dataset under MNAR mechanism
generator = mMNAR(X=X, y=y)

# Generate the missing data under MNAR mechanism
generate_data = generator.MBOUV(missing_rate=20)
print(generate_data.isna().sum())
