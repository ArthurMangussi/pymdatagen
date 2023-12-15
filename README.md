# mdgen: Generation of artificial missing data in datasets

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/Documentation-Link-green.svg)](mdgen/docs/)

This package was developed to address a gap in machine learning research, specifically the artificial generation of missing data. Santos et al (2019) provided a survey that presents various strategies for both univariate and multivariate scenarios, but the Python community does not have implementations of these strategies. Besides, Pereira et al (2023) proposed new benchmark strategies for Missing Not At Random (MNAR), and these novel methods are not yet implemented in Python. Hence, **missing-data-generator** (mdgen) is a Python package that implements methods for generating missing values ​​for data, including MAR, MNAR, and MCAR mechanisms in both univariate and multivariate scenarios.

This Python package is a collaboration between researchers at the Aeronautics Institute of Technologies and the University of Coimbra.

## User Guide

Please refer to the [univariate docs](mdgen/docs/univariate.md) or [multivariate docs](mdgen/docs/multivariate.md) for more details.


### Installation
To install the package, please use the `pip` installation as follows:

```bash
pip3.11 install mdgen
```

### Usage examples
For examples on how to use the mdgen package, from basic examples that generate artificial missing data under a mechanism to complete examples using Multiple Imputation by Chained Equations (MICE) from scikit-learn for imputation, follow this [examples](tests/).


## Contribuitions
Contributions are welcome! Feel free to open issues, submit pull requests, or provide feedback

## Citation
If you use **mdgen** in your research, please cite the **mdgen paper**

Bibtex entry:
```bash
@article{mdgen2024,
  author  = {Arthur D Mangussi and Ana Carolina Lorena and Filipe Loyola and Pedro Henriques Abreu},
  title   = {mdgen: A Python library for Generate Artifical Missing Data},
  journal = {Journal of Machine Learning Research},
  year    = {},
  volume  = {},
  number  = {},
  pages   = {},
  url     = {}
}
```
## Acknowledgements
The authors gratefully acknowledge the Brazilian funding agencies FAPESP (Fundação Amparo à Pesquisa do Estado de São Paulo) process number 2022/10553-6, bolsa FILIPE, LORENA e ver se Pedro tem.