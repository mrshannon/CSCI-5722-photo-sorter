import os
import re
from setuptools import setup, find_packages


def read_version(filename):
    return re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        read(filename), re.MULTILINE).group(1)


def read(filename):
    with open(os.path.join(os.path.dirname(__file__), filename)) as infile:
        text = infile.read()
    return text


setup(
    name='phos',
    version=read_version('phos/__init__.py'),
    author='Michael R. Shannon',
    author_email='mrshannon.aerospace@gmail.com',
    description='Sort photos based on scene.',
    url='https://github.com/mrshannon/CSCI-5722-photo-sorter',
    packages=find_packages(),
    scripts=['bin/phos'],
    install_requires=[
        'numpy',
        'opencv-contrib-python==4.0.1',
        'pillow',
        'progressbar2',
        'rtree',
        'scikit-learn',
        'scipy',
        'sqlalchemy'
    ],
    data_files=[('phos', ['wordlist']),
                ('phos/keywords', [
                    'keywords/cars',
                    'keywords/city',
                    'keywords/coast',
                    'keywords/forest',
                    'keywords/garden',
                    'keywords/highway',
                    'keywords/home',
                    'keywords/industrial',
                    'keywords/mountain',
                    'keywords/office',
                    'keywords/open_country',
                    'keywords/people',
                    'keywords/residential',
                    'keywords/river',
                    'keywords/store',
                    'keywords/street',
                    'keywords/text'])],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    zip_safe=False
)