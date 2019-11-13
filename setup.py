from distutils.core import setup
from os import path
import sys

file_dir = path.abspath(path.dirname(__file__))

# Get the long description from the README file.
try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
    long_description = long_description.replace('\r', '')
except ImportError:
    print('Pandoc not found. Long_description conversion failure.')
    with open(path.join(file_dir, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()

# To be filled in later
REQUIRES = []

setup(
    name='rivnet',
    version='0.1dev',

    description='A python library for generating a river network from a water mask',
    long_description=long_description,
    url='tobefilledout_later_githubthingy',

    author='Charlie Marshak',
    author_email='charlie.z.marshak@jpl.nasa.gov',


    keywords='fast marching method centerlines river network',

    packages=['rivnet'],  # setuptools.find_packages(exclude=['doc']),

    # Required Packages
    install_requires=REQUIRES,

)
