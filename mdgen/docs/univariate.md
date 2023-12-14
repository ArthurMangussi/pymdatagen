## Univariate mechanisms

**missing-data-generator** possui a abordagem univariada, a qual se refere à apenas um atributo do dataset contendo os dados ausentes. A partir disso, a literatura apresenta 3 mecanismos: Missing Completly at Random (MCAR), Missing Not at Random (MNAR) e Missing at Random (MAR). Todos os 3 estão implementados na biblioteca Python.

As estratégias para gerar os dados ausentes são:

### MAR
- lowest:
- rank:
- median:
- highest:
- mix:

### MNAR 
- run:

### MCAR
- random:
- binomial:


## Usage example

Here is a simple example of how you can use the missing value generation class in your own Python code:

```python
# Import the generate missing data class
from univariate.univariate import MAR, MNAR, MCAR
from sklearn.datasets import load_iris
import pandas as pd

# Load the data
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

X = iris_df.copy() # Features
y = iris.target    # Label values

# Create a instance with missing rate equal to 10% in dataset under MNAR mechanism
generator = MNAR(X=X,y = y, missing_rate=10)

# Generate the missing data under MNAR mechanism
generate_data = generator.run()

# Generate the missing data under MAR mechanism with lowest strategy
generate_data = generator.lowest()

# Generate the missing data under MCAR mechanism with random strategy
generate_data = generator.random()

``````
