from setuptools import setup, find_namespace_packages

classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3'
]

setup(
    name = 'OTAnalysis',
    version = '0.0.1',
    author = 'Thierry GALLIANO',
    author_email = 'thierry.galliano@etu.univ-amu.fr',
    description = "Tools for extracting, analyzing and classifying optical tweezer data curves",
    long_description = open('README.md').read() + '\n\n' + open('CHANGELOG.txt').read(),
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/TigiGln/OTAnalysis.git',
    License= 'GPLv3+',
    classifiers=classifiers,
    keywords='optical tweezers',
    packages=find_namespace_packages(),
    install_requires=['']

)


