"""
ifera - A financial data processing library

This is a setuptools-based package for processing financial data with support for
AWS S3 storage, pandas data manipulation, and PyTorch tensor operations.
"""
from setuptools import setup, find_packages

setup(
    name="ifera",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "torch",
        "boto3",
        "pydantic",
        "pydantic-settings",
        "tqdm"
    ],
    author="ifera",
    description="A library for processing financial instrument data",
    python_requires=">=3.9",
)
