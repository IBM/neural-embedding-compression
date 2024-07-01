from setuptools import setup, find_packages

setup(
    name='NEC',
    version='0.1',
    description='Neural Embedding Compression',
    author='Carlos Gomes',
    author_email='carlos.gomes@ibm.com',
    packages=find_packages(include=["MAEPretrain_SceneClassification", "MAEPretrain_SceneClassification.*"]),
    install_requires=[
        'numpy',
        'pandas',
    ],
)