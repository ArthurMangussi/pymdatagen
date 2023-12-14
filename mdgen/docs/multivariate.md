## Multivariate mechanisms

**missing-data-generator** possui a abordagem multivariada, a qual se refere vários atributos do dataset contendo os dados ausentes. A partir disso, a literatura apresenta 3 mecanismos: Missing Completly at Random (MCAR), Missing Not at Random (MNAR) e Missing at Random (MAR). Todos os 3 estão implementados na biblioteca Python.

As estratégias para gerar os dados ausentes são:

### MAR
- random:
- correlated:
- median:

### MNAR 
- random:
- correlated:
- median:
- MBOUV:
- MBOV_randomness:
- MBOV_median:
- MBIR:

### MCAR
- random:
- binomial:


## Usage example

Here is a simple example of how you can use the **missing-data-generator** in your own Python code:

```python
# Import the missing data generator 
from multivariate.multivariate import MAR, MNAR, MCAR
from sklearn.datasets import load_iris
import pandas as pd

# Load the data
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

X = iris_df.copy() # Features
y = iris.target    # Label values

# Create a instance with missing rate equal to 30% in dataset under MCAR mechanism
generator = MCAR(X=X,y = y, missing_rate = 30)
md_df = generator.random()

# Create a instance with missing rate equal to 10% in dataset under MNAR mechanism
generator = MNAR(X=X,y = y)
md_df = generator.MBOUV()

# Create a instance with missing rate equal to 50% in dataset under MAR mechanism
generator = MAR(X=X,y = y, , n_xmiss=4)
md_df = generator.correlated(missing_rate = 50)

``````
