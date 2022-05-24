from setuptools import setup

setup(
    name='OTAnalysis',
    version='0.2.72',
    packages=['otanalysis'],
    install_requires=[
        'requests',
        'importlib-metadata; python_version == "3.8"',
        'PyQt5'
    ],
)