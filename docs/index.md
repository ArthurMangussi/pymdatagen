# mdatagen: A Python Library for the Generation of Artificial Missing Data

![Python3](https://img.shields.io/badge/Language-Python3-steelblue)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](pymdatagen/LICENSE)
[![Documentation](https://img.shields.io/badge/Documentation-Link-green.svg)](docs/)
[![Latest PyPI Version](https://img.shields.io/pypi/v/mdatagen.svg)](https://pypi.org/project/mdatagen/)


## Overview
This package has been developed to address a gap in machine learning research, specifically the artificial generation of missing data. Santos et al. (2019) provided a survey that presents various strategies for both univariate and multivariate scenarios, but the Python community still needs implementations of these strategies. Our Python library **missing-data-generator** (mdatagen) puts forward a comprehensive set of implementations of missing data mechanisms, covering Missing Completely at Random (MCAR), Missing at Random (MAR), and Missing Not at Random (MNAR), allowing users to simulate several real-world scenarios comprising absent observations. The library is designed for easy integration with existing Python-based data analysis workflows, including well-established modules such as scikit-learn, and popular libraries for missing data visualization, such as missingno, enhancing its accessibility and usability for researchers.

This Python package is a collaboration between researchers at the Aeronautics Institute of Technologies (Brazil) and the University of Coimbra (Portugal).

### Installation
To install the package, please use the `pip` installation as follows:

```bash
pip install mdatagen
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