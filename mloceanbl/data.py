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
# add units and metadata to new array
# add section for mldb data

####################################################################################
########################### Import libraries #######################################
####################################################################################

import xarray as xr
import pandas as pd
import numpy as np
from netCDF4 import Dataset
from glob import glob
from scipy.interpolate import interpn
from mpl_toolkits.basemap import maskoceans

####################################################################################
############################# Import data ##########################################
####################################################################################

print('Beginning to Import Data')

# 1. Import SSS Data from OISSS and SMAP (for climatology)
# 2. Import SST Data
# 3. Import SSH Data, transform lon grid to (-180, 180)

############################### Salinity ###### ####################################

## Import SSS Dataset
SSS_path = '/glade/work/dwhitt/OISSS/'
SSS_files = sorted(glob(SSS_path+'*.nc'))
sss = xr.open_mfdataset(SSS_files, decode_times=False, combine='by_coords')

# Convert to time variable to dateTime range
sss = sss.assign_coords(time = pd.date_range(
                        '2011-08-26T12:00:00', periods=200, freq = 'W-FRI'))

# Rename coordinate names for unity
sss = sss.rename({'longitude':'lon', 'latitude':'lat'})

# Create Mask
sss_lat_mask = (sss.lat>-55) & (sss.lat < 60)

## Import SSS data 2015-2020 to calculate climatologies
SSS_clim_path = '/glade/scratch/dwhitt/SMAP_RSS_L3_8d/'
SSS_clim_files = sorted(glob(SSS_clim_path+'*.nc'))
with xr.open_mfdataset(SSS_clim_files, combine='by_coords') as sss_clim:
    # Change from (0, 360) -> (-180, 180)
    sss_clim = sss_clim.assign_coords(lon=(((sss_clim.lon + 180) % 360) - 180)).sortby('lon')
    
    # Get on to sss 1/2 degree lat lon grid, select 7/2015 - 2020
    sss_climatology = sss_clim.sss_smap[81:].interp(lat = sss.lat, lon = sss.lon)
    
    # Take monthly rolling average
    sss_climatology = sss_climatology.chunk((45, 720,1440)).rolling(
                        time=28,center=True).reduce(np.nanmean)
    
    # Group by day of year and take average over 4 years
    sss_climatology = sss_climatology[28:-28].groupby('time.dayofyear').mean('time')
    
    # Rechunk
    sss_climatology = sss_climatology.chunk((1,sss.lat.size, sss.lon.size))
    

########################### Sea Level Temperatures ####################################

## Import SST Dataset
SST_path = '/glade/work/dwhitt/GHRSST_OIMW/'
SST_files = sorted(glob(SST_path+'*.nc'))
sst = xr.open_mfdataset(SST_files, combine='by_coords')

# Create Mask
sst_lat_mask = (sst.lat>-55) & (sst.lat < 60)

############################### Sea Level Height ####################################

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

# 1. Interpolate Sea level height onto spatial and temporal grid. Due to difficulties with
# the netcdf file, this has to be done by conversion to numpy array.

# 2. Remove climatology of sss data and interpolate nan values. 

# 3. Remove climatology of sst data and interpolate to spatial and temporal grid.


## Create a fine ocean mask for interpolation purposes
Array = np.ma.zeros((ssh.lat.size,ssh.lon.size))
Lons,Lats = np.meshgrid(np.array(ssh.lon),np.array(ssh.lat))
MaskedArray = maskoceans(Lons,Lats,np.array(ssh.SLA[0]),resolution='f',grid=1.25)

############################### Sea Level Height ####################################

## Convert SSH data
## Requires a lot of manual work because of issues with built in xarray interpolation.

# Create necessary date time indexes (found manually, need to encompass entire sss data timeframe + extra on each end)
indexes = np.arange(1360, 1680)
times = ssh.time[indexes]

# Convert ssh data to numpy arrays for scipy interpolation
sla = ssh.SLA.sel(time=times).where(MaskedArray.mask).interpolate_na(dim='lon')
sla = sla.where(MaskedArray.mask).interpolate_na(dim='lat')
sla = sla.where(MaskedArray.mask)

sla_array = np.array(sla)
lat_array = np.array(ssh.lat)
lon_array = np.array(ssh.lon)
time_array = np.array(ssh.time.sel(time=times))
nan_mask = np.isnan(sla_array)
[TIME, LAT, LON] = np.meshgrid(time_array, lat_array, lon_array, indexing='ij');

# Convert sss data to numpy arrays, this will be the grid to interpolate to
xlat_array = np.array(sss.lat[sss_lat_mask])
xlon_array = np.array(sss.lon)
xtime_array = np.array(sss.time)
[xTIME, xLAT, xLON] = np.meshgrid(xtime_array, xlat_array, xlon_array, indexing='ij');

# Arrange interpolation grid as required by scipy.interpolate.interpn
interp_mesh = np.array(np.meshgrid(xtime_array.astype('float'), xlat_array, xlon_array, indexing='ij'))
interp_points = np.rollaxis(interp_mesh, 0, 4)

# Interpolate
gridded = interpn((time_array.astype('float'), lat_array, lon_array), sla_array, interp_points)
ssh_nan_mask = ~np.isnan(gridded)

################################ Salinity #####################################################
# Remove climatology
anomalies = (sss.sss.groupby("time.dayofyear") - sss_climatology)
anomalies = anomalies.drop('dayofyear')

# Interpolate nans in SSS data
sss_lat_interp = anomalies[:, sss_lat_mask, :].interpolate_na(dim='lat', use_coordinate='lat')
sss_lon_interp = anomalies[:, sss_lat_mask, :].interpolate_na(dim='lon', use_coordinate='lon')

sss_lon_interp = sss_lon_interp.interpolate_na(dim='lat') 
sss_lat_interp = sss_lat_interp.interpolate_na(dim='lon')
sss_lat_interp = sss_lat_interp.where(ssh_nan_mask)
sss_lon_interp = sss_lon_interp.where(ssh_nan_mask)
sss_interp = 0.5*np.array(sss_lat_interp) + 0.5*np.array(sss_lon_interp)

################################ Temperature #####################################################

# Remove computed climatology from data.
climatology_mean = sst.analysed_sst.sel(time=pd.date_range('2002-06-01T12:00:00', periods=365.25*8, freq="D").union(
                    pd.date_range('2016-01-01T12:00:00', periods=366+365+365+180, freq="D"))).chunk((50, 720, 1440))

climatology_mean = climatology_mean.rolling(time=28,center=True).reduce(np.nanmean)
climatology_mean = climatology_mean[28:-28].groupby('time.dayofyear').mean('time').chunk((1, 720, 1440))
anomalies = (sst.analysed_sst.sel(time=pd.date_range('2011-01-01T12:00:00', periods = 365+366+365+365+365, freq="D")).groupby("time.dayofyear") - climatology_mean) 
anomalies = anomalies.drop('dayofyear')

## Convert SST data from daily to weekly
time_interp_sst = anomalies.chunk((15, 720, 1440)).rolling(time=7).reduce(np.nanmean).chunk((1,720, 1440)).sel(time = pd.date_range('2011-08-26T12:00:00', periods=200, freq = 'W-FRI'))

## Convert SST data onto 1/2 degree grid (same as sss data, which is the coarsest available )
sp_t_inter_sst = time_interp_sst.interp(lon = sss.lon, lat = sss.lat[sss_lat_mask])
sp_t_inter_sst = sp_t_inter_sst.interpolate_na(dim = 'lon', use_coordinate='lon')
sp_t_inter_sst = sp_t_inter_sst.interpolate_na(dim='lat', use_coordinate='lat')
sp_t_inter_sst = sp_t_inter_sst.where(ssh_nan_mask)

####################################################################################
############################# Combine Data #########################################
####################################################################################

ds = xr.Dataset(
        {'salinity': (['time', 'lat', 'lon'], sss_interp), 
         'temperature': (['time', 'lat', 'lon'], sp_t_inter_sst), 
         'height': (['time', 'lat', 'lon'], gridded)},
        coords={'time': sss.time,
                'lat': sss.lat[sss_lat_mask],
                'lon': sss.lon}
)

############################## Normalize ###########################################

sal = (ds.salinity - np.nanmean(ds.salinity, axis=0))/np.nanstd(ds.salinity, axis=0)
ds = ds.assign(norm_salinity = sal)
temp = (ds.temperature - np.nanmean(ds.temperature, axis=0))/np.nanstd(ds.temperature, axis=0)
ds = ds.assign(norm_temperature = temp)
height = (ds.height - np.nanmean(ds.height, axis=0))/np.nanstd(ds.height, axis=0)
ds = ds.assign(norm_height = height)

####################################################################################
################################ Save Data #########################################
####################################################################################
print('Saving Data')
print(ds)
ds.to_netcdf(path='sss_sst_ssh_normed_anomalies_weekly.nc')

####################################################################################
################################ Updates ###########################################
####################################################################################

# 05/26/2020
# Added rolling average to SST interpolation.
# Removed NaN's from SST by interpolating over a mask