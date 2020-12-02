#!/bin/bash
# This script will download preprocessed surface and argo-based mixed layer depth data to notebooks/data/.
# For more information, see the zenodo record https://doi.org/10.5281/zenodo.4301074


wget "https://zenodo.org/record/4301074/files/mldb_climatology_climatologystd_binned.nc?download=1" ../notebooks/data/
wget "https://zenodo.org/record/4301074/files/mldb_full_anomalies_stdanomalies_climatology_stdclimatology.nc?download=1" ../notebooks/data/
wget "https://zenodo.org/record/4301074/files/sss_sst_ssh_anomalies.nc?download=1" ../notebooks/data