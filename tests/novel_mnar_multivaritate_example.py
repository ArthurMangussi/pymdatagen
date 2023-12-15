""""
A novel example of generate artificial missing data with mdgen library with the Breast Cancer Wiscosin dataset 
from scikit-learn. The features with label will receive the missing values under 
Missing Not at Random (MNAR) mechanism. The simulated missing rate is 20%. 
The method to choose missing values is Missingness Based on Own and Unobserved Values (MBOUV).
"""

from mdgen.multivariate.MNAR import MNAR
from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load the data
wiscosin = load_breast_cancer()
wiscosin_df = pd.DataFrame(data=wiscosin.data, columns=wiscosin.feature_names)

X = wiscosin_df.copy() # Features
y = wiscosin.target    # Label values

# Create a instance with missing rate equal to 20% in dataset under MNAR mechanism
generator = MNAR(X=X, y=y)

# Generate the missing data under MNAR mechanism
generate_data = generator.MBOUV(missing_rate=20)
print(generate_data.isna().sum())

