#!/usr/bin/env python

"""The setup script."""

from os.path import exists

from setuptools import find_packages, setup

if exists('requirements.txt'):
    with open('requirements.txt') as f:
        install_requires = f.read().strip().split('\n')
else:
    install_requires = ['dask', 'xarray', 'numpy', 'tensorflow', 'pandas',
                        'scikit-learn', 'netcdf4', 'scipy']

if exists('README.rst'):
    with open('README.rst') as f:
        long_description = f.read()
else:
    long_description = ''


setup(
    name='ml-ocean-bl',
    description='Machine Learning Ocean Boundary Layer',
    long_description=long_description,
    maintainer='David John Gagne, Dallas Foster, Dan Whitt',
    maintainer_email='dgagne@ucar.edu',
    url='https://github.com/NCAR/ml-ocean-bl',
    packages=find_packages(),
    include_package_data=True,
    install_requires=install_requires,
    license='open_source_license',
    zip_safe=False,
    keywords='project_name',
    use_scm_version=True,
    setup_requires=['setuptools_scm', 'setuptools>=30.3.0', 'setuptools_scm_git_archive'],
)
