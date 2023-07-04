from setuptools import setup

setup(name="mltlk",
    version="0.1.39",
    author="Johan Hagelbäck",
    author_email="johan.hagelback@gmail.com",
    description="Toolkit for making machine learning easier",
    packages=["mltlk"],
    install_requires=["termcolor", "scikit-learn", "pandas", "imblearn", "gensim", "customized_table", "matplotlib", "nltk"],
    )
