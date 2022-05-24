from setuptools import setup

setup(
    name='mypackage',
    version='0.0.1',
    packages=['mypackage'],
    install_requires=[
        'requests',
        'importlib-metadata; python_version == "3.8"',
        'PyQt5'
    ],
)