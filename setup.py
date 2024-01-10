# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='mdatagen',
    version='0.0.87',
    keywords=['machine learning', 'preprocessing data'],
    license='MIT',
    author='Arthur Dantas Mangussi',
    author_email='mangussiarthur@gmail.com',
    url='https://github.com/ArthurMangussi/pymdatagen',
    
    packages=find_packages(),
    description='mdatagen: A Python Library for the Generation of Artificial Missing Data',
    long_description=long_description,
    long_description_content_type='text/markdown',

    python_requires='>=3.11', 
    install_requires=[
        'numpy >= 1.25.0',
        'pandas >= 2.0.3',
        'scikit-learn == 1.3.0']
)