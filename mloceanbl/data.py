####################################################################################
####################################################################################
########################### File Description #######################################
####################################################################################
####################################################################################
# This file creates a class that imports preprocessed sss data, sst data, and ssh data 
# as well as mld data using the dataset class and methods. Using this class, users 
# can treat the 'get_index' method as an iterator to draw individual weeks from the
# data set for use in training various machine learning models.
####################################################################################
########################### Import libraries #######################################
####################################################################################
import xarray as xr
import numpy as np
from preprocess_mld import preprocess_argo_data
from preprocess_sss_sst_ssh import preprocess_surface_data
####################################################################################
########################### Import Data ############################################
####################################################################################


def preprocess_data(sss_path, sst_path, ssh_path, 
                       sss_sst_ssh_output_path,
                       mld_argo_dataset_path,
                       mld_argo_output_path,
                       mld_clim_output_path):
 
    
        preprocess_surface_data(sss_path, 
                                           sst_path, 
                                           ssh_path, 
                                           sss_sst_ssh_output_path)
        preprocess_argo_data(sss_sst_ssh_output_path,
                           mld_argo_dataset_path,
                           mld_argo_output_path,
                           mld_clim_output_path)


class dataset():
    def __init__(self, lat_bounds, lon_bounds, 
                 sss_sst_ssh_data = './data/sss_sst_ssh_anomalies.nc',
                 mld_data = './data/mldb_full_anomalies_stdanomalies_climatology_stdclimatology.nc',
                 anomalies=True, 
                 CNN = False,
                 half_data = False):
        self.half_data = half_data
        with xr.open_dataset(sss_sst_ssh_data) as ds:
            print(ds)
            sal = ds.salinity.values.astype(np.float64)
            sal_anom = ds.salinity_anomaly.values.astype(np.float64)
            temp = ds.temperature.values.astype(np.float64)
            temp_anom = ds.temperature_anomaly.values.astype(np.float64)
            height = ds.height.values.astype(np.float64)
            lat = ds.lat.values.astype(np.float64)
            lon = ds.lon.values.astype(np.float64)
            self.time = ds.time.values
            self.i_train = ds.training_index.values
            self.i_test = ds.testing_index.values
            self.i_val  = ds.validation_index.values

        with xr.open_dataset(mld_data) as ds:
            print(ds)
            mldb = ds.copy()
            self.y_week = mldb['week']

        if anomalies:
            self.X_data = np.stack((sal_anom, temp_anom, height), axis=-1)
            self.y_data = mldb['std_anomaly']
        else:
            self.X_data = np.stack((sal, temp, height), axis=-1)
            self.y_data = mldb['mldb']


        lat_mask = (lat < lat_bounds[1]) & (lat > lat_bounds[0])
        lon_mask = (lon < lon_bounds[1]) & (lon > lon_bounds[0])
        self.mask = np.isfinite(self.X_data[:, :, :, 0].mean(axis=0))
        if CNN:
            self.X_data[:, ~self.mask, :] = 0.0
            self.mask = np.isfinite(self.X_data[:, :, :, 0].mean(axis=0))
        self.mask[~lat_mask, :] = False
        self.mask[:, ~lon_mask] = False
        self.X_data = self.X_data[:, self.mask, :]

        [LAT, LON] = np.meshgrid(lat, lon, indexing='ij')
        self.X_loc = np.stack((LON[self.mask], LAT[self.mask]), axis=-1)

        self.y_data = self.y_data.where(
            (mldb.lat < lat_bounds[1]) & (mldb.lat > lat_bounds[0]) & \
            (mldb.lon < lon_bounds[1]) & (mldb.lon > lon_bounds[0])
            ).dropna('index')
        self.y_week = self.y_week.where(
            (mldb.lat < lat_bounds[1]) & (mldb.lat > lat_bounds[0]) & \
            (mldb.lon < lon_bounds[1]) & (mldb.lon > lon_bounds[0])
            ).dropna('index')
        self.y_lat = mldb['lat'].where(
            (mldb.lat < lat_bounds[1]) & (mldb.lat > lat_bounds[0]) & \
            (mldb.lon < lon_bounds[1]) & (mldb.lon > lon_bounds[0])
            ).dropna('index')
        self.y_lon = mldb['lon'].where(
            (mldb.lat < lat_bounds[1]) & (mldb.lat > lat_bounds[0]) & \
            (mldb.lon < lon_bounds[1]) & (mldb.lon > lon_bounds[0])
            ).dropna('index')

        self.lat_bounds = lat_bounds
        self.lon_bounds = lon_bounds

        if self.half_data == True:
            from numpy.random import RandomState
            prng = RandomState(24)
            self.test_indices = []
            self.train_indices = []
            for i in range(self.time.size):
                size = self.y_data.where(np.in1d(self.y_week, self.time[i])).dropna('index').values.size
                self.test_indices.append(prng.choice(np.arange(size), int(size//2), replace = False))
                self.train_indices.append(np.setdiff1d(np.arange(size), self.test_indices[i]))

    def normalize(self):

        for i in range(self.X_data.shape[-1]):
            self.X_data[:, :, i] = normalize(self.X_data[:, :, i], self.i_train)

        self.y_mean = self.y_data.where(np.in1d(self.y_week, self.time[self.i_train])).dropna('index').mean().values
        self.y_std = self.y_data.where(np.in1d(self.y_week, self.time[self.i_train])).dropna('index').std().values
        self.y_data = ( self.y_data - self.y_mean)/self.y_std

    def get_index(self, index):
        if self.half_data == True:
            y = self.y_data.where(np.in1d(self.y_week, self.time[index])).dropna('index').values
            y_loc = np.stack( (self.y_lon.where(np.in1d(self.y_week, self.time[index])).dropna('index'), 
                                self.y_lat.where(np.in1d(self.y_week, self.time[index])).dropna('index')),
                                axis=-1)
            self.test_y = y[self.test_indices[index]].astype(np.float64)
            self.test_y_loc = y_loc[self.test_indices[index]].astype(np.float64)

            return self.X_data[index].astype(np.float64), self.X_loc.astype(np.float64), y[self.train_indices[index]].astype(np.float64), y_loc[self.train_indices[index]].astype(np.float64)
        else:
            y = self.y_data.where(np.in1d(self.y_week, self.time[index])).dropna('index').values
            y_loc = np.stack( (self.y_lon.where(np.in1d(self.y_week, self.time[index])).dropna('index'), 
                                self.y_lat.where(np.in1d(self.y_week, self.time[index])).dropna('index')),
                                axis=-1)
            return self.X_data[index].astype(np.float64), self.X_loc.astype(np.float64), y.astype(np.float64), y_loc.astype(np.float64)


def normalize(inputs, train_index):
    std = np.nanstd(inputs[train_index], axis=0)
    return np.true_divide(inputs - np.nanmean(inputs[train_index], axis=0), std, where = (std > 0) )
