'''
Created on 08 December 2020
@author: Sambit Giri
Setup script
'''

# from setuptools import setup, find_packages
from distutils.core import setup


setup(name='pySZe',
      version='0.1',
      author='Sambit Giri',
      author_email='sambit.giri@ics.uzh.ch',
      package_dir = {'pySZe' : 'src'},
      packages=['pySZe'],
      # package_data={'share':['*'],},
      install_requires=['numpy','scipy','astropy','tqdm'],
      url=""
      #include_package_data=True,
)
