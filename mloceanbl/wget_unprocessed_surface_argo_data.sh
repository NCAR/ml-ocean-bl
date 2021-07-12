#!/bin/bash
# This script will download sea-surface temperature, salinity, and height data from NASA repositories to the current path.
# It is necessary to register and get a username and password at https://urs.earthdata.nasa.gov/
# and replace "username" and "PASSWORD" below with your username and password from this registration.
# To optionally modify the target path for the data, modify "./" after "-P"
# This script also downloads argo-based mixed layer depth, see https://doi.org/10.5281/zenodo.4291175


wget  --user=USERNAME --password=PASSWORD -r -np -nH -nd "*.nc" "https://podaac-tools.jpl.nasa.gov/drive/files/allData/ghrsst/data/GDS2/L4/GLOB/REMSS/mw_OI/v5.0/" -P ../notebooks/data/sst 
wget  --user=USERNAME --password=PASSWORD -nH -nd -r -np "*.nc" "https://podaac-tools.jpl.nasa.gov/drive/files/allData/merged_alt/L4/cdr_grid" -P ../notebooks/data/ssh 
wget  --user=USERNAME --password=PASSWORD -nH -nd -r -np ".*nc" "https://podaac-tools.jpl.nasa.gov/drive/files/SalinityDensity/aquarius/L4/IPRC/v5/7day" -P ../notebooks/data/sss 
wget "https://zenodo.org/record/4291175/files/mldbmax.nc?download=1" ../notebooks/data