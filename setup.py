#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    description="Implementation of 'Fast multi-language LSTM-based online handwriting recognition' paper by Carbune et al.",
    author="Martin Lellep",
    author_email="konten.ma.le@gmail.com",
    url="https://github.com/PellelNitram/carbune2020_implementation",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
