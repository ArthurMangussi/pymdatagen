# mdatagen: A Python Library for Generate of Artificial Missing Data

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/Documentation-Link-green.svg)](mdatagen/docs/)
[![Version](https://img.shields.io/badge/Version-1.0.0-brightgreen.svg)](https://github.com/ArthurMangussi/pymdatagen/releases/tag/v1.0.0)

This package has been developing to address a gap in machine learning research, specifically the artificial generation of missing data. Santos et al (2019) provided a survey that presents various strategies for both univariate and multivariate scenarios, but the Python community does not have implementations of these strategies. Besides, Pereira et al (2023) proposed new benchmark strategies for Missing Not At Random (MNAR), and these novel methods are not yet implemented in Python. Hence, **missing-data-generator** (mdatagen) is a Python package that implements methods for generating missing values ​​for data, including Missing At Random (MAR), Missing Not At Random (MNAR), and Missing Completly At Random (MCAR) mechanisms in both univariate and multivariate scenarios.

This Python package is a collaboration between researchers at the Aeronautics Institute of Technologies (Brazil) and the University of Coimbra (Portugal).

## User Guide

Please refer to the [univariate docs](docs/univariate.md) or [multivariate docs](docs/multivariate.md) for more details.


### Installation
To install the package, please use the `pip` installation as follows:

```bash
pip install mdatagen
```

### Usage examples
For examples on how to use the mdatagen package, from basic examples that generate artificial missing data under a mechanism to complete examples using Multiple Imputation by Chained Equations (MICE) from scikit-learn for imputation, follow this [examples](examples/).


## Contribuitions
Contributions are welcome! Feel free to open issues, submit pull requests, or provide feedback

## Citation
If you use **mdatagen** in your research, please cite the [mdatagen paper]()

Bibtex entry:
```bash
@article{mdatagen2024,
  author  = {Arthur D Mangussi and Ana Carolina Lorena and Filipe Loyola Lopes and Miriam Seone Santos and Pedro Henriques Abreu and  Ricardo Cardoso Pereira},
  title   = {mdatagen: A Python Library for Generate Artifical Missing Data},
  journal = {Journal of Machine Learning Research},
  year    = {},
  volume  = {},
  number  = {},
  pages   = {},
  url     = {}
}
```
## Acknowledgements
The authors gratefully acknowledge the Brazilian funding agencies FAPESP (Fundação Amparo à Pesquisa do Estado de São Paulo) process numbers 2022/10553-6, and LORENA. Moreover, this study was financed in part by the Coordenação de Aperfeiçoamento de Pessoalde Nível Superior - Brasil (CAPES) - Finance Code 001, PEDRO.