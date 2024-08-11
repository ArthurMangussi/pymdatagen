# mdatagen: A Python Library for the Generation of Artificial Missing Data

![Python3](https://img.shields.io/badge/Language-Python3-steelblue)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/Documentation-Link-green.svg)](docs/)
[![Latest PyPI Version](https://img.shields.io/pypi/v/mdatagen.svg)](https://pypi.org/project/mdatagen/)


## Overview
This package has been developed to address a gap in machine learning research, specifically the artificial generation of missing data. Santos et al. (2019) provided a survey that presents various strategies for both univariate and multivariate scenarios, but the Python community still needs implementations of these strategies. Our Python library **missing-data-generator** (mdatagen) puts forward a comprehensive set of implementations of missing data mechanisms, covering Missing Completely at Random (MCAR), Missing at Random (MAR), and Missing Not at Random (MNAR), allowing users to simulate several real-world scenarios comprising absent observations. The library is designed for easy integration with existing Python-based data analysis workflows, including well-established modules such as scikit-learn, and popular libraries for missing data visualization, such as missingno, enhancing its accessibility and usability for researchers.

This Python package is a collaboration between researchers at the Aeronautics Institute of Technologies (Brazil) and the University of Coimbra (Portugal).

## User Guide

Please refer to the [univariate docs](docs/univariate.md) or [multivariate docs](docs/multivariate.md) for more implementatios details.


### Installation
To install the package, please use the `pip` installation as follows:

```bash
pip install mdatagen
```

## API Usage

API usage is described in each of the following sections in the [document]().

- [Getting Started](https://moshi4.github.io/pyCirclize/getting_started/)


### Code examples
More examples on how to use the mdatagen package, follow these [examples](examples/).
Here, we provide a basic usage for MAR mechanism in both univariate and multivariate
scenarios. Also, we illustrate how to use the Histogram plot and evaluate the imputation
quality. 

### MAR univariate 
```python
import pandas as pd
from sklearn.datasets import load_iris

from mdatagen.univariate.uMAR import uMAR

# Load the data
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, 
                      columns=iris.feature_names)

X = iris_df.copy()   # Features
y = iris.target      # Label values

generator = uMAR(X=X, 
                  y=y, 
                  missing_rate=50, 
                  x_miss='sepal length (cm)',
                  x_obs = 'petal lenght (cm)')

# Generate the missing data under MAR mechanism univariate
generate_data = generator.rank()
print(generate_data.isna().sum())

```
### MAR multivariate

```python
import pandas as pd
from sklearn.datasets import load_iris

from mdatagen.multivariate.mMAR import mMAR

# Load the data
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, 
                      columns=iris.feature_names)

X = iris_df.copy()   # Features
y = iris.target      # Label values

generator = mMAR(X=X, 
                  y=y)

# Generate the missing data under MAR mechanism multivariate
generate_data = generator.correlated(missing_rate=25)
print(generate_data.isna().sum())
```
### Histogram plot
 
```python
import pandas as pd
from sklearn.datasets import load_iris

from mdatagen.univariate.uMCAR import uMCAR
from mdatagen.plots.plot import PlotMissingData

# Load the data
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, 
                        columns=iris.feature_names)

X = iris_df.copy()   # Features
y = iris.target      # Label values

# Create a instance with missing rate 
# equal to 25% in dataset under MCAR mechanism
generator = uMCAR(X=X, 
                  y=y, 
                  missing_rate=25, 
                  x_miss='petal length (cm)')

# Generate the missing data under MNAR mechanism
generate_data = generator.random()


miss_plot = PlotMissingData(data_missing=generate_data,
                            data_original=X
                            )
miss_plot.visualize_miss("histogram",
                         col_missing="petal length (cm)",
                         save=True,
                         path_save_fig = "MCAR_iris.png")
```
### Imputation Quality Evaluation: Mean Squared Error (MSE) 

```python
import pandas as pd
from sklearn.datasets import load_iris

from mdatagen.univariate.uMCAR import uMCAR
from mdatagen.metrics.metrics import EvaluateImputation

# Load the data
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, 
                        columns=iris.feature_names)

X = iris_df.copy()   # Features
y = iris.target      # Label values

# Create a instance with missing rate 
# equal to 25% in dataset under MCAR mechanism
generator = uMCAR(X=X, 
                  y=y, 
                  missing_rate=25, 
                  x_miss='petal length (cm)')

# Generate the missing data under MNAR mechanism
generate_data = generator.random()

# Calculate the metric: MSE
fill_zero = generate_data.drop("target",axis=1).fillna(0)
eval_metric = EvaluateImputation(
            data_imputed=fill_zero,
            data_original=X,
            metric="mean_squared_error")
print(eval_metric.show())
```


## Contribuitions
Contributions are welcome! Feel free to open issues, submit pull requests, or provide feedback.

## Citation
If you use **mdatagen** in your research, please cite the [original paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8605316)

Bibtex entry:
```bash
@ARTICLE{Santos2019,
  author={Santos, Miriam Seoane and Pereira, Ricardo Cardoso and Costa, Adriana Fonseca and Soares, Jastin Pompeu and Santos, João and Abreu, Pedro Henriques},
  journal={IEEE Access}, 
  title={Generating Synthetic Missing Data: A Review by Missing Mechanism}, 
  year={2019},
  volume={7},
  number={},
  pages={11651-11667},
  doi={10.1109/ACCESS.2019.2891360}}
```
## Acknowledgements
The authors gratefully acknowledge the Brazilian funding agencies FAPESP (Fundação Amparo à Pesquisa do Estado de São Paulo) under grants 2022/10553-6, 2023/13688-2, and 2021/06870-3. Moreover, this research was supported in part by the Coordenação de Aperfeiçoamento de Pessoalde Nível Superior - Brasil (CAPES) - Finance Code 001, and Portuguese Recovery and Resilience Plan (PRR) through project C645008882-00000055 Center for Responsable AI.
