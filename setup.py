# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='mdatagen',
    version='0.0.8',
    keywords=['machine learning', 'preprocessing data'],
    license='MIT',
    author='Arthur Dantas Mangussi',
    author_email='mangussiarthur@gmail.com',
    url='https://github.com/ArthurMangussi/pymdatagen',
    
    packages=find_packages(),
    description='mdatagen: A Python library to Generate Artifical Missing Data',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',

    install_requires=[
        'python >= 3.11',
        'numpy >= 1.25.0',
        'pandas >= 2.0.3',
        'scikit-learn == 1.3.0',
    ],
)


