####################################################################################
####################################################################################
########################### File Description #######################################
####################################################################################
####################################################################################

# This file imports sss data, sst data, and ssh data that are on separate temporal
# and spatial grids and interpolates them to fit on the same grid (sss grid is master).
# SSS master grid is 2011-2015 W-FRI (time, lat, lon) in (weeks, half degree) which
# results in data approximately (200, 316, 720) arrays for each sss, sst, ssh
# lat upper and lower bounds are -79 and 79 because of ssh data constraints.
# Then the file saves the new data set to file at sss_sst_ssh_weekly_halfdegree.nc'

## TO DO
# Potentially update interpolation method from linear interpolant to gaussian process
# rethink and be more careful with temporal interpolation of 5 day ssh averaging. 

# add units and metadata to new array
# add section for mldb data

####################################################################################
########################### Import libraries #######################################
####################################################################################

import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from netCDF4 import Dataset
from glob import glob
import scipy as sp

####################################################################################
############################# Import data ##########################################
####################################################################################
print('Beginning to Import Data')
## Import SSS Dataset
SSS_path = '/glade/work/dwhitt/OISSS/'
SSS_files = sorted(glob(SSS_path+'*.nc'))
SSS_path = '/glade/work/dwhitt/OISSS/'
SSS_files = sorted(glob(SSS_path+'*.nc'))
sss = xr.open_mfdataset(SSS_files, decode_times=False, combine='by_coords')

# Convert to time variable to dateTime range
sss = sss.assign_coords(time = pd.date_range('2011-08-26T12:00:00', periods=200, freq = 'W-FRI'))

# Rename coordinate names for unity
sss = sss.rename({'longitude':'lon', 'latitude':'lat'})


## Import SST Dataset
SST_path = '/glade/work/dwhitt/GHRSST_OIMW/'
SST_files = sorted(glob(SST_path+'*.nc'))
sst = xr.open_mfdataset(SST_files, combine='by_coords')


## Import SSH Dataset
SSH_path = '/glade/scratch/dwhitt/measures_ssh/'
SSH_files = sorted(glob(SSH_path+'*.nc'))
ssh = xr.open_mfdataset(SSH_files, combine='by_coords')

# Rename coordinates for unity
ssh = ssh.rename({'Longitude':'lon', 'Latitude':'lat', 'Time': 'time'})

# Rearrange coordinates for (time, lat, lon)
ssh = ssh.transpose('time', 'lat', 'lon', 'nv')

# Convert (0,360) lon grid to (-180, 180) lon grid
ssh = ssh.assign_coords(lon=(((ssh.lon + 180) % 360) - 180)).sortby('lon')

####################################################################################
############################# Interpolate Data #####################################
####################################################################################
print('Beginning to Interpolate Data')
## Create masks for latitude (needed because ssh data is only defined (-79, 79) )
sst_lat_mask = (sst.lat>-79) & (sst.lat < 79)
sss_lat_mask = (sss.lat>-79) & (sss.lat < 79)
sss_nan_mask = ~np.isnan(sss.sss) 

## Convert SST data from daily to weekly
time_interp_sst = sst.analysed_sst.chunk((7896, 720, 1440)).rolling(time=7).reduce(np.mean).chunk((1,720, 1440)).sel(time = pd.date_range('2011-08-26T12:00:00', periods=200, freq = 'W-FRI'))

## Convert SST data onto 1/2 degree grid (same as sss data, which is the coarsest available )
sp_t_inter_sst = time_interp_sst.interp(lon = sss.lon, lat = sss.lat[sss_lat_mask])
sp_t_inter_sst = sp_t_inter_sst.interpolate_na(dim = 'lon', use_coordinate='lon')
sp_t_inter_sst = sp_t_inter_sst.interpolate_na(dim='lat', use_coordinate='lat')
sp_t_inter_sst = sp_t_inter_sst.where(sss_nan_mask)
print(sp_t_inter_sst)


## Convert SSH data
## Requires a lot of manual work because of issues with built in xarray interpolation.

# Create necessary date time indexes (found manually, need to encompass entire sss data timeframe + 1 on each end)
indexes = np.arange(1360, 1680)
times = ssh.time[indexes]

# Convert ssh data to numpy arrays for scipy interpolation
sla_array = np.array(ssh.SLA.sel(time=times))
lat_array = np.array(ssh.lat)
lon_array = np.array(ssh.lon)
time_array = np.array(ssh.time.sel(time=times))
nan_mask = np.isnan(sla_array)
[TIME, LAT, LON] = np.meshgrid(time_array, lat_array, lon_array, indexing='ij');

# Convert sss data to numpy arrays, this will be the grid to interpolate to
xlat_array = np.array(sss.lat[(sss.lat>-79) & (sss.lat<79)])
xlon_array = np.array(sss.lon)
xtime_array = np.array(sss.time)
[xTIME, xLAT, xLON] = np.meshgrid(xtime_array, xlat_array, xlon_array, indexing='ij');

# mask lats and lons (see note above)
# xnan_mask = np.isnan(np.array(sss.sss[:, sss_nan_mask, :]))
# points = (TIME[nan_mask], LAT[nan_mask], LON[nan_mask])
# values = sla_array[nan_mask]

# Arrange interpolation grid as required by scipy.interpolate.interpn
interp_mesh = np.array(np.meshgrid(xtime_array.astype('float'), xlat_array, xlon_array, indexing='ij'))
interp_points = np.rollaxis(interp_mesh, 0, 4)

# Interpolate
gridded = sp.interpolate.interpn((time_array.astype('float'), lat_array, lon_array), sla_array, interp_points)

####################################################################################
############################# Combine Data #########################################
####################################################################################

ds = xr.Dataset(
        {'salinity': (['time', 'lat', 'lon'], sss.sss[:, sss_lat_mask, :]), 
         'temperature': (['time', 'lat', 'lon'], sp_t_inter_sst), 
         'height': (['time', 'lat', 'lon'], gridded)},
        coords={'time': sss.time,
                'lat': sss.lat[sss_lat_mask],
                'lon': sss.lon}
)
print(ds)

####################################################################################
################################ Save Data #########################################
####################################################################################
print('Saving Data')
ds.to_netcdf(path='sss_sst_ssh_weekly_halfdegree.nc')

####################################################################################
################################ Updates ###########################################
####################################################################################

# 05/26/2020
# Added rolling average to SST interpolation.
# Removed NaN's from SST by interpolating over a mask