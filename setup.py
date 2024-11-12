from setuptools import setup, find_packages

import os
import sys

with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()

setup(name='qcgnoms',
      version='0.1',
      description='GNN for MSMS Prediction',
      url='https://github.com/PNNL-m-q/qcgnoms',
      author='Richard Overstreet',
      author_email='richard.overstreet@pnnl.gov',
      license='BSD-2',
      package_data={'qcgnoms': ['weights/*']},
      include_package_data=True,
      packages=find_packages(),
      zip_safe=False)
