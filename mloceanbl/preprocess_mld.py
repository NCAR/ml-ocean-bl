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


####################################################################################
########################### Import libraries #######################################
####################################################################################

import xarray as xr
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from math import ceil

####################################################################################
############################# Import data ##########################################
####################################################################################
# 'sss_sst_ssh_normed_anomalies_weekly.nc'
# '/glade/work/dwhitt/magda_argo_data/mldbmax.nc'
  
def preprocess_argo_data(sss_sst_ssh_dataset_input,
               mldb_argo_dataset_input,
                    argo_output,
                    clim_output):
    print('Beginning to Import Data')

    with xr.open_dataset( sss_sst_ssh_dataset_input) as ds:
        lat = ds.lat.values
        lon = ds.lon.values
        time = ds.time.values


    with xr.open_dataset(mldb_argo_dataset_input) as ds:
        mldb = ds.dropna('time')
        mldb_ds = pd.DataFrame(index=mldb.time.values, columns = ['lat', 'lon', 'mldb'])
        mldb_ds.lat = mldb.LATITUDE.values
        mldb_ds.lon = mldb.LONGITUDE.values
        mldb_ds.mldb = mldb.MLDBMAX.values

    # Bin Data
    lat_bins =  lat[::8]
    lon_bins =  lon[::8]
    m1 = mldb_ds.assign(lat_cut = pd.cut(mldb_ds.lat, lat_bins, labels=lat_bins[1:]), 
                        lon_cut = pd.cut(mldb_ds.lon, lon_bins, labels=lon_bins[1:]))
    m1 =  m1.dropna()
    m1 =  m1.assign(cartesian=pd.Categorical( m1.filter(regex='_cut').apply(tuple, 1)))
    bins =  m1.cartesian.unique()


    ####################################################################################
    ######################### Calculate Climatology ####################################
    ####################################################################################

    # Refine daterange for climatology
    time_mask =  (( m1.index < '2011-08-01') & ( m1.index > '2002-08-01')) ^ ( m1.index > '2015-08-01')
    time_mask1 = (( m1.index > '2011-08-01') & ( m1.index < '2015-08-01'))
    m_clim =  m1[time_mask].copy()
    m_full =  m1[time_mask1].copy()
    m_full['anomaly'] =  m1[time_mask1].mldb.copy()
    m_full['std_anomaly'] =  m1[time_mask1].mldb.copy()
    m_full['climatology'] =  m1[time_mask1].mldb.copy()
    m_full['climatology_std'] =  m1[time_mask1].mldb.copy()
    bins =  m_full.cartesian.unique()
    times = pd.date_range('2011-08-19T12:00:00', periods=201, freq = 'W-FRI')
    m_full =  m_full.assign(week = pd.cut( m_full.index, times, labels=times[1:]) ).dropna()

    m_clim_ndarray = np.empty( (2,  bins.size, 12) )
    m_full_ndarray = np.empty( (3,  bins.size, 12) )

    epoch =  bins.size//10
    for j in range( bins.size):
        if j % epoch == 0: print('Calculating Anomalies: {:2d}%'.format(ceil(100*j/ bins.size)))      
        m_full.mldb.loc[( m_full.lat_cut ==  bins[j][0]) & ( m_full.lon_cut ==  bins[j][1])] = \
             m_full.mldb.loc[( m_full.lat_cut ==  bins[j][0]) & ( m_full.lon_cut ==  bins[j][1])].rolling('7D').mean().dropna().rolling('14D').mean().dropna()
        m_full.anomaly.loc[( m_full.lat_cut ==  bins[j][0]) & ( m_full.lon_cut ==  bins[j][1])] = \
             m_full.mldb.loc[( m_full.lat_cut ==  bins[j][0]) & ( m_full.lon_cut ==  bins[j][1])]
        m_temp = m_clim[ (m_clim.lat_cut ==  bins[j][0]) & (m_clim.lon_cut ==  bins[j][1])]
        m_temp1 = m_temp.mldb.rolling('D').mean().ffill().interpolate('time').rolling('7D').mean().rolling('14D').mean()
        m_temp1[(m_temp1.index < '2015-08-01') & (m_temp1.index > '2011-08-01')] = np.nan
        monthly_averages = m_temp1.dropna().groupby(by=m_temp1.index.month).mean().dropna().values
        monthly_std = m_temp1.dropna().groupby(by=m_temp1.index.month).std().dropna().values+1
        if np.min([monthly_std.size, monthly_averages.size]) < 12:
            m_full =  m_full[ ~ (( m_full.lat_cut ==  bins[j][0]) & ( m_full.lon_cut ==  bins[j][1])) ]
            continue
        first_last = 0.5*monthly_averages[0]+0.5*monthly_averages[-1]
        monthly_averages[0] = first_last
        monthly_averages[-1] = first_last
        first_last = 0.5*monthly_std[0]+0.5*monthly_std[-1]
        monthly_std[0] = first_last
        monthly_std[-1] = first_last

        m_clim_ndarray[0, j] = monthly_averages
        m_clim_ndarray[1, j] = monthly_std
        clim_spline = CubicSpline(np.arange(1,13), monthly_averages)
        clim_std_spline = CubicSpline(np.arange(1,13), monthly_std)

        m_full.climatology.loc[( m_full.lat_cut ==  bins[j][0]) & ( m_full.lon_cut ==  bins[j][1])] = \
             m_full.mldb.loc[( m_full.lat_cut ==  bins[j][0]) & ( m_full.lon_cut ==  bins[j][1])].transform(
                lambda x: 0.*x + clim_spline(x.index.month)).rolling('14D').mean().dropna()
        m_full.climatology_std.loc[( m_full.lat_cut ==  bins[j][0]) & ( m_full.lon_cut ==  bins[j][1])] = \
             m_full.mldb.loc[( m_full.lat_cut ==  bins[j][0]) & ( m_full.lon_cut ==  bins[j][1])].transform(
                lambda x: 0.*x + clim_std_spline(x.index.month)).rolling('14D').mean().dropna()
        m_full.anomaly.loc[( m_full.lat_cut ==  bins[j][0]) & ( m_full.lon_cut ==  bins[j][1])] = \
             m_full.mldb.loc[( m_full.lat_cut ==  bins[j][0]) & ( m_full.lon_cut ==  bins[j][1])].sub(
                 m_full.climatology.loc[( m_full.lat_cut ==  bins[j][0]) & ( m_full.lon_cut ==  bins[j][1])])

        m_full.std_anomaly.loc[( m_full.lat_cut ==  bins[j][0]) & ( m_full.lon_cut ==  bins[j][1])] = \
             m_full.anomaly.loc[( m_full.lat_cut ==  bins[j][0]) & ( m_full.lon_cut ==  bins[j][1])].divide(
                 m_full.climatology_std.loc[( m_full.lat_cut ==  bins[j][0]) & ( m_full.lon_cut ==  bins[j][1])])

    # Shape Data
    m_full.index =  m_full.index.set_names(['time'])
    m_full =  m_full.reset_index()

    ####################################################################################
    ############################# Combine Data #########################################
    ####################################################################################
    # print('Saving MLD and MLD Anomalies')
    ds =  m_full.to_xarray()
    ds_new = ds.drop('lat_cut').drop('lon_cut').drop('cartesian')
    ds_new.to_netcdf(path=argo_output)
    print(ds_new)


    ####################################################################################
    ################################ Process Climatology ###############################
    ####################################################################################
    print('Saving Climatologies')
    lats =  lat[::8][1:]
    lons =  lon[::8][1:]
    times = pd.date_range('2011-08-19T12:00:00', periods=201, freq = 'W-FRI')[1:]
    LON, LAT = np.meshgrid(lons, lats, indexing='ij')
    clim = np.zeros((lons.size, lats.size, times.size))
    clim_std = np.zeros_like(clim)
    for j in range( bins.size):
        index1, index2 = np.where((LAT== bins[j][0]) & (LON==( bins[j][1])) )
        for k in range(times.size):
            month = times[k].month
            clim[index1[0], index2[0], k] =  m_clim_ndarray[0, j, month-1]
            clim_std[index1[0], index2[0], k] =  m_clim_ndarray[1, j, month-1]

    d_clim = xr.Dataset(
        {
            "climatology": (["lon", "lat", "time"], clim),
            "climatology_std": (["lon", "lat", "time"], clim_std),
        },
        coords={
            "lon": lons,
            "lat": lats,
            "time": times,
        },
    )

    d_clim.to_netcdf(path=clim_output)