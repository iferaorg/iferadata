"""
ifera - A financial data processing library

This is a setuptools-based package for processing financial data with support for
AWS S3 storage, Polars data manipulation, and PyTorch tensor operations.
"""

from setuptools import find_packages, setup

setup(
    name="ifera",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        "ifera.strategy_optimization": [
            "default_features.yaml",
            "feature_templates.yaml",
        ]
    },
    install_requires=[
        "polars",
        "numpy",
        "torch",
        "holidays",
        "matplotlib",
        "boto3",
        "pydantic",
        "pydantic-settings",
        "tqdm",
        "SQLAlchemy>=2.0,<3",
        "oracledb>=3.0,<5",
        "yfinance>=0.2.66,<2",
        "pandas>=1.5,<4",
        "PyYAML",
        "requests>=2.32,<3",
        "beautifulsoup4>=4.12,<5",
        "pypdf>=5,<7",
    ],
    author="ifera",
    description="A library for processing financial instrument data",
    python_requires=">=3.9",
)
