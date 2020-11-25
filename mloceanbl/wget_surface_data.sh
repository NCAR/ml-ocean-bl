#!/bin/bash
# This script will download sea-surface temperature, salinity, and height data from NASA repositories to the current path.
# It is necessary to register and get a username and password at https://urs.earthdata.nasa.gov/
# and replace "username" and "PASSWORD" below with your username and password from this registration.
# To optionally modify the target path for the data, modify "./" after "-P"

wget  --user=username --password=PASSWORD -nH -nd -r -np "*.nc" "https://podaac-tools.jpl.nasa.gov/drive/files/allData/ghrsst/data/GDS2/L4/GLOB/REMSS/mw_OI/v5.0/" -P ./data/sst 
wget  --user=username --password=PASSWORD -nH -nd -r -np "*.nc" "https://podaac-tools.jpl.nasa.gov/drive/files/SeaSurfaceTopography/merged_alt/L4/cdr_grid" -P ./data/ssh 
wget  --user=username --password=PASSWORD -nH -nd -r -np "*.nc" "https://podaac-tools.jpl.nasa.gov/drive/files/SalinityDensity/aquarius/L4/IPRC/v5/7day" -P ./data/sss 
