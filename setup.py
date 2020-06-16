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

desc = '''A python library for generating a connected channel network from a
water mask'''

setup(name='orinoco',
      version='0.1dev',

      d =
      description=desc,
      long_description=long_description,
      url='tobefilledoutlatergithubthingy',

      author='Charlie Marshak',
      author_email='charlie.z.marshak@jpl.nasa.gov',

      keywords='fast marching method, centerlines, delta channel network',

      packages=['orinoco'],  # setuptools.find_packages(exclude=['doc']),

      # Required Packages
      # We assume an environment specified by requirements.txt is provided We
      # could take this approach:
      # https://github.com/scikit-image/scikit-image/blob/master/setup.py#L117-L131
      # but rather use the requirements.txt to specify a valid environment and
      # not muddle the installation with pip and possibly conda.
      install_requires=[],
      )
