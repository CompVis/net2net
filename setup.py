from setuptools import setup, find_packages

setup(
    name='net2net',
    version='0.0.1',
    description='Translate between networks through their latent representations.',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)