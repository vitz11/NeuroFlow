"""
Setup configuration for NeuroFlow package
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="neuroflow",
    version="1.0.0",
    author="vitz11",
    author_email="vitthalgautam11@gmail.com",
    description="An intelligent automated machine learning system with Kaggle integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vitz11/neuroflow",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "streamlit>=1.28.0",
        "kaggle>=1.5.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "xgboost>=1.7.0",
        "joblib>=1.3.0",
        "openpyxl>=3.1.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov"],
    },
)
