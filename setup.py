from setuptools import setup

setup(
    name="mdgen",
    version="1.0.0",
    packages=["mdgen"],
    license="MIT",
    author="Arthur Dantas Mangussi",
    author_email="mangussiarthur@gmail.com",
    description="mdgen: Missing Data Generator",
    requires=["python >= 3.11",
              "numpy >= 1.25.0",
              "pandas >= 2.0.3",
              "scikit-learn == 1.3.0"]
)