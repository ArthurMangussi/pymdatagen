from setuptools import find_packages, setup

setup(
    name="mdatagen",
    version="1.0.0",
    packages=["mdatagen"],
    license="MIT",
    author="Arthur Dantas Mangussi",
    author_email="mangussiarthur@gmail.com",
    url= "https://github.com/ArthurMangussi/mdatagen",
    keywords=["machine learning", "preprocessing data"],
    description="mdatagen: A Python library to Generate Artifical Missing Data",
    requires=["python >= 3.11",
              "numpy >= 1.25.0",
              "pandas >= 2.0.3",
              "scikit-learn == 1.3.0"],
    download_url="https://github.com/ArthurMangussi/mdatagen/archive/refs/tags/v1.0.0.tar.gz",
    classifiers=[
    'Development Status :: 4 - Alpha',      
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',     
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   
    'Programming Language :: Python :: 3',      
    'Programming Language :: Python :: 3.11'],
    
)