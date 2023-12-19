""""
A example of generate artificial missing data with mdgen library with the Breast Cancer Wiscosin dataset 
from scikit-learn, and use Multiple Imputation by Chained Equations (MICE) to imputation values into dataset
generated. 
"""

from mdgen.multivariate.MNAR import MNAR
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Load the data
wiscosin = load_breast_cancer()
wiscosin_df = pd.DataFrame(data=wiscosin.data, columns=wiscosin.feature_names)

X = wiscosin_df.copy() # Features
y = wiscosin.target    # Label values

# Create a instance with missing rate equal to 20% in dataset under MNAR mechanism
generator = MNAR(X=X, y=y)

# Generate the missing data under MNAR mechanism
generate_mddata = generator.random(missing_rate=20)

# Initialize the MICE imputer
imputer = IterativeImputer(max_iter=100)

# Training the Imputer
imputer.fit(generate_mddata)

df_imputate = pd.DataFrame(imputer.transform(generate_mddata), columns=X.columns)