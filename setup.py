#!/usr/bin/env python

import os


from setuptools import setup, find_packages

# This provides the variable `__version__`.
# execfile('opensim/version.py')
__version__ = 1.2

setup(name='osim-rl',
      version=__version__,
      description='OpenSim Reinforcement Learning Framework',
      author='Lukasz Kidzinski',
      author_email='lukasz.kidzinski@stanford.edu',
      url='http://opensim.stanford.edu/',
      license='Apache 2.0',
      packages=find_packages(),
      package_data={'osim': ['models/Geometry/*.vtp', 'models/*.osim']},
      include_package_data=True,
      install_requires=['numpy','gym'],
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3.5',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          ],
      )
