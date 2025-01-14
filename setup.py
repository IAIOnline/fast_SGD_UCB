# -*- coding: utf-8 -*-
from setuptools import setup

packages = ["sgdbandit", "sgdbandit.agents", "sgdbandit.environments", "sgdbandit.utils"]

package_data = {"": ["*"]}

install_requires = [
    "dill",
    "joblib",
    "matplotlib",
    "numpy",
    "pre-commit",
    "seaborn",
    "tqdm",
    "scipy",    
    "ipykernel",
    "Jinja2",
    "numba"
]

setup_kwargs = {
    "name": "SGDBandit",
    "version": "0.1.0",
    "description": "Code for algorithm SGDBandit to deal with MAB with heavy tails",
    "author": "None",
    "author_email": "None",
    "maintainer": "None",
    "maintainer_email": "None",
    "url": "None",
    "packages": packages,
    "package_data": package_data,
    "install_requires": install_requires,
    "python_requires": ">=3.9",
}


setup(**setup_kwargs)
