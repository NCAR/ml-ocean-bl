===============================
Machine Learning Ocean Boundary Layer (ML-Ocean-BL)
===============================

.. image:: https://img.shields.io/circleci/project/github/NCAR/package-name/master.svg?style=for-the-badge&logo=circleci
    :target: https://circleci.com/gh/NCAR/package-name/tree/master

.. image:: https://img.shields.io/codecov/c/github/NCAR/package-name.svg?style=for-the-badge
    :target: https://codecov.io/gh/NCAR/package-name


.. image:: https://img.shields.io/readthedocs/package-name/latest.svg?style=for-the-badge
    :target: https://package-name.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/pypi/v/package-name.svg?style=for-the-badge
    :target: https://pypi.org/project/package-name
    :alt: Python Package Index

.. image:: https://img.shields.io/conda/vn/conda-forge/package-name.svg?style=for-the-badge
    :target: https://anaconda.org/conda-forge/package-name
    :alt: Conda Version


Installation
------------

ml-olcean-bl can be installed from PyPI with pip:

.. code-block:: bash

    pip install ml-ocean-bl


It is also available from `conda-forge` for conda installations:

.. code-block:: bash

    conda install -c conda-forge package-name

Download and Preprocess Data
----------------------------

Prerequisite data must be downloaded and preprocessed into a specific format. 
Users can either download the original satellite sea surface data and argo-based mixed layer depth data and preprocess themselves, or
users can download already preprocessed data corresponding to the paper 'Probabilistic Machine Learning Estimation of Ocean Mixed Layer
Depth from Dense Satellite and Sparse In-Situ Observations' Foster et al. (ex. 2021). 

For the former, see the jupyter notebook ./notebooks/download_preprocess_data.ipynb and run the corresponding lines. Users can also run 

.. code-block:: bash

    sh wget_unprocessed_surface_argo_data.sh

and see the python files ./mloceanbl/preprocess_sss_sst_ssh.py and ./mloceanbl/preprocess_mld.py. Note that in order to download the 
satellite sea surface data, you will need to register at https://urs.earthdata.nasa.gov/ and get a corresponding password from the podaac
drive. 

To simply download the already preprocessed data used in the paper mentioned above, simply run

.. code-block:: bash

    sh wget_preprocessed_surface_argo_data.sh


Training and Usage
----------------------------

For details on the training and testing procedures, see the notebook ./notebooks/model_train_Example.ipynb. For information about how the
data should be organized, see ./mloceanbl/data.py. 

For detail on the model class, see ./mloceanbl/models.py. Individual NN models: Linear, 
ANN, ANN with Dropout, ANN with parameterized output distribution, and VAE can be found at ./mloceanbl/linear.py, ann.py, ann_dropout.py, 
vann.py, vae.py. 

Details on the training proceedure can be found in ./mloceanbl/train.py. 
