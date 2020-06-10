from distutils.core import setup
from os import path

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


def parse_requirements_file(filename):
    with open(filename, encoding='utf-8') as fid:
        requires = [l.strip() for l in fid.readlines() if l]

    return requires


INSTALL_REQUIRES = parse_requirements_file('requirements.txt')

setup(name='orinoco',
      version='0.1dev',

      description='A python library for generating a connected channel network from a water mask',
      long_description=long_description,
      url='tobefilledoutlatergithubthingy',

      author='Charlie Marshak',
      author_email='charlie.z.marshak@jpl.nasa.gov',

      keywords='fast marching method centerlines river network',

      packages=['orinoco'],  # setuptools.find_packages(exclude=['doc']),

      # Required Packages
      install_requires=INSTALL_REQUIRES,
      )
