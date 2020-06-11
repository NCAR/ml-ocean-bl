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

with xr.open_dataset('sss_sst_ssh_normed_anomalies_weekly.nc') as ds:
    print(ds)
    sal = ds.salinity
    sal_anom = ds.salinity_anomaly
    temp = ds.temperature
    temp_anom = ds.temperature_anomaly
    height = ds.height
    lat = ds.lat.values
    lon = ds.lon.values
    time = ds.time.values
    
# Import MLDB data -> convert to dataframe
with xr.open_dataset('/glade/work/dwhitt/magda_argo_data/mldbmax.nc') as ds:
    print(ds)
    mldb = ds.dropna('time')
    mldb_ds = pd.DataFrame(index=mldb.time.values, columns = ['lat', 'lon', 'mldb'])
    mldb_ds.lat = mldb.LATITUDE.values
    mldb_ds.lon = mldb.LONGITUDE.values
    mldb_ds.mldb = mldb.MLDBMAX.values
    print(mldb_ds)
    
# Bin Data
_, lat_bins = np.histogram(lat, 50)
_, lon_bins = np.histogram(lon, 80)
m1 = mldb_ds.assign(lat_cut = pd.cut(mldb_ds.lat, lat_bins, labels=lat_bins[1:]), 
                    lon_cut = pd.cut(mldb_ds.lon, lon_bins, labels=lon_bins[1:]))
m1 = m1.dropna()
m1 = m1.assign(cartesian=pd.Categorical(m1.filter(regex='_cut').apply(tuple, 1)))
bins = m1.cartesian.unique()


####################################################################################
######################### Calculate Climatology ####################################
####################################################################################

# Refine daterange for climatology
time_mask =  ((m1.index < '2011-08-01') & (m1.index > '2002-08-01')) ^ (m1.index > '2015-08-01')
time_mask1 = ((m1.index > '2011-08-01') & (m1.index < '2015-08-01'))
m_clim = m1[time_mask].copy()

m_full = m1[time_mask1].copy()
m_anom = m1[time_mask1].copy()
m_std_anom = m1[time_mask1].copy()
bins = m_anom.cartesian.unique()


for j in range(bins.size):
    if j % 100 == 0: print(100*j/bins.size)
    m_temp = m_clim[ (m_clim.lat_cut == bins[j][0]) & (m_clim.lon_cut == bins[j][1])]
    m_temp = m_temp[ (m_temp.mldb - m_temp.mldb.mean()) / (m_temp.mldb.std()) < 4]
    m_temp1 = m_temp.mldb.resample('D').mean().interpolate('time').rolling(28).mean()
    m_temp1[(m_temp1.index < '2015-08-01') & (m_temp1.index > '2011-08-01')] = np.nan
    monthly_averages = m_temp1.groupby(by=m_temp1.index.month).mean().values
    monthly_std = m_temp1.groupby(by=m_temp1.index.month).std().values
    if monthly_averages.size != 12:
        print('Dropping index', j)
        m_full = m_full[ ~ ((m_full.lat_cut == bins[j][0]) & (m_full.lon_cut == bins[j][1])) ]
        m_anom = m_anom[ ~ ((m_anom.lat_cut == bins[j][0]) & (m_anom.lon_cut == bins[j][1])) ]
        m_std_anom = m_std_anom[ ~ ((m_std_anom.lat_cut == bins[j][0]) & (m_std_anom.lon_cut == bins[j][1])) ]
        continue
    for k in range(12):
        m_anom.loc[(m_anom.lat_cut == bins[j][0]) & (m_anom.lon_cut == bins[j][1]) & (m_anom.index.month == (k+1)), 'mldb'] = \
        m_anom.loc[(m_anom.lat_cut == bins[j][0]) & (m_anom.lon_cut == bins[j][1]) & (m_anom.index.month == (k+1)), 'mldb'].sub(monthly_averages[k])
        m_std_anom.mldb.loc[(m_std_anom.lat_cut == bins[j][0]) & (m_std_anom.lon_cut == bins[j][1]) & (m_std_anom.index.month == (k+1))] = \
        m_anom.mldb[(m_anom.lat_cut == bins[j][0]) & (m_anom.lon_cut == bins[j][1]) & (m_anom.index.month == (k+1))]/monthly_std[k]

        
# Shape Data
times = pd.date_range('2011-08-19T12:00:00', periods=201, freq = 'W-FRI')

m_full = m_full.assign(week = pd.cut(m_full.index, times, labels=times[1:]) ).dropna()
m_full.index = m_full.index.set_names(['time'])
m_full = m_full.reset_index()

m_anom = m_anom.assign(week = pd.cut(m_anom.index, times, labels=times[1:]) ).dropna()
m_anom.index = m_anom.index.set_names(['time'])
m_anom = m_anom.reset_index()

m_std_anom = m_std_anom.assign(week = pd.cut(m_std_anom.index, times, labels=times[1:]) ).dropna()
m_std_anom.index = m_std_anom.index.set_names(['time'])
m_std_anom = m_std_anom.reset_index()

m_anom['mldb_full'] = m_full.mldb.copy()
m_anom['mldb_anom_std'] = m_std_anom.mldb.copy()

# Combine Data

rng = np.random.default_rng(10)
m_time = np.zeros((200, 1790))
m_lat = np.ones_like(m_time)
m_lon = np.ones_like(m_time)
m_mldb_full = np.ones_like(m_time)
m_mldb_anom = np.ones_like(m_time)
m_mldb_std_anom = np.ones_like(m_time)
for i in range(m_anom.week.unique().size):
    mask = m_anom.week == m_anom.week.unique()[i]
    t_temp = np.array(m_anom.time[mask])
    lat_temp = np.array(m_anom.lat[mask])
    lon_temp = np.array(m_anom.lon[mask])
    mldb_anom_temp = np.array(m_anom.mldb[mask])
    mldb_full_temp = np.array(m_anom.mldb_full[mask])
    mldb_std_anom_temp = np.array(m_anom.mldb_anom_std[mask])
    index = np.arange(mask.sum())
    rng.shuffle(index)
    m_time[i, :] = t_temp[index[:1790]]

    m_lat[i, :] = lat_temp[index[:1790]]

    m_lon[i, :] = lon_temp[index[:1790]]
    
    m_mldb_full[i, :] = mldb_full_temp[index[:1790]]

    m_mldb_anom[i, :] = mldb_anom_temp[index[:1790]]
    
    m_mldb_std_anom[i, :] = mldb_std_anom_temp[index[:1790]]



####################################################################################
############################# Combine Data #########################################
####################################################################################

ds = xr.Dataset(
        {'mldbmax': (['time', 'index'], m_mldb_full),
         'mldbmax_anomaly' : (['time', 'index'], m_mldb_anom),
         'mldbmax_std_anomaly': (['time', 'index'], m_mldb_std_anom),
         'lat': (['time', 'index'], m_lat),
         'lon': (['time', 'index'], m_lon),
         'date': (['time', 'index'], np.array(pd.to_datetime(m_time.flatten())).reshape(200, -1))
        },
        coords={'time': time,
                'index': np.arange(1790)}
)

####################################################################################
################################ Save Data #########################################
####################################################################################
print('Saving Data')
print(ds)
ds.to_netcdf(path='mldbmax_full_anomalies_weekly.nc')

####################################################################################
################################ Updates ###########################################
####################################################################################
