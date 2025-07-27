from setuptools import setup, find_packages

setup(
    name="math_modeling_2024",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "ortools>=9.3.0",
        "networkx>=2.6.0",
        "plotly>=5.3.0",
        "streamlit>=1.0.0",
        "pyvista>=0.34.0",
        "pytest>=6.2.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0"
    ],
    python_requires=">=3.8",
) 