####################################################################################
####################################################################################
########################### File Description #######################################
####################################################################################
####################################################################################

# This file creates a class that imports preprocessed sss data, sst data, and ssh data 
# It currently hosts a number of plot creation functions. It will hold all of the code
# for the potential models that we will test.

## TO DO
# Add argo data
# Add models
# Add plot creation


####################################################################################
########################### Import libraries #######################################
####################################################################################
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from netCDF4 import Dataset
import seaborn as sns
import cartopy
import cartopy.crs as ccrs
from mpl_toolkits.basemap import maskoceans
import tensorflow as tf
import tensorflow.keras as keras
import gpflow as gpf

####################################################################################
########################### Import Data ############################################
####################################################################################

            
class dataset:
    def __init__(self, x_data, lon, lat, y_data, m_lon, m_lat, lat_bounds, lon_bounds):
        self.X_data = np.stack(x_data, axis=-1)
        lat_mask = (lat < lat_bounds[1]) & (lat > lat_bounds[0])
        lon_mask = (lon < lon_bounds[1]) & (lon > lon_bounds[0])
        self.mask = np.isfinite(x_data[0].mean(axis=0))
        self.mask[~lat_mask, :] = False
        self.mask[:, ~lon_mask] = False
        self.X_data = np.stack(x_data, axis=-1)[:, self.mask, :]
        
        [LAT, LON] = np.meshgrid(lat, lon, indexing='ij')
        self.X_loc = np.stack((LON[self.mask], LAT[self.mask]), axis=-1)
        self.y_data = y_data
        self.y_loc = np.stack((m_lon, m_lat), axis=-1)
        self.lat_bounds = lat_bounds
        self.lon_bounds = lon_bounds
        
    def normalize(self, train_index):
        
        for i in range(self.X_data.shape[-1]):
            self.X_data[:, :, i] = normalize(self.X_data[:, :, i], train_index)
        
        self.y_data = normalize(self.y_data, train_index)
        
    def get_index(self, index):
        mask = (self.y_loc[index, :, 1]<self.lat_bounds[1]) & (self.y_loc[index, :, 1]>self.lat_bounds[0]) & \
                (self.y_loc[index, :, 0]<self.lon_bounds[1]) & (self.y_loc[index, :, 0]>self.lon_bounds[0])
        y = self.y_data[index, mask].reshape(-1, 1)
        y_loc = self.y_loc[index, mask, :]
        return self.X_data[index], self.X_loc, y, y_loc

####################################################################################
########################### Preprocess Models ######################################
####################################################################################

def train_test_val_indices(n, n_train, n_test):
    n_validate = n - n_train -  n_test
    assert n_validate > 0, print('Validation size must be positive')
    i = np.arange(n)
    rng = np.random.default_rng(10)
    np.random.shuffle(i)
    return i[:n_train], i[n_train:n_train+n_test], i[n_train+n_test:]



####################################################################################
########################### Define Models ##########################################
####################################################################################

class HaversineMatern(gpf.kernels.Kernel):
    def __init__(self):
        super().__init__()
        self.amplitude = gpf.Parameter(10.0, transform=positive())
        self.length_scale = gpf.Parameter(1.0, transform=positive())
        
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        dist = tf.linalg.matmul(X, X2, transpose_b=True)
        dist = tf.clip_by_value(dist, -1, 1)
        dist = tf.math.acos(dist)
        return self.matern(dist)  # this returns a 2D tensor

    def K_diag(self, X):
        dist = tf.math.acos(tf.clip_by_value(tf.linalg.matmul(X, X, transpose_b=True), -1, 1))
        return tf.reshape(tf.linalg.diag_part(self.matern(dist)), (-1,)) # this returns a 1D tensor
    
    def matern(self, dist):
        Kl = dist * tf.cast(tf.math.sqrt(3.0), dtype='float64')*self.length_scale
        Kl = self.amplitude*(1. + Kl) * tf.exp(-Kl)
        return Kl


class Linear(keras.Model):
    def __init__(self, input_dim, n_features, x_l, dtype='float64', **kwargs):
        super(Linear, self).__init__(name='linear_projection', dtype='float64', **kwargs)
        self.input_dim = input_dim
        self.n_features = n_features
        
        self.x_l = np.stack( (np.sin(np.deg2rad(90.0-x_l[:,1]))*np.cos(np.deg2rad(x_l[:,0]+180.0)),  
                     np.sin(np.deg2rad(90.0-x_l[:,1]))*np.sin(np.deg2rad(x_l[:,0]+180.0)),
                     np.cos(np.deg2rad(90.0-x_l[:,1]))), axis=-1)
        self.x_l = tf.cast(self.x_l, dtype='float64')
        
        w_init = tf.initializers.GlorotNormal()
        self.w = tf.Variable(initial_value=w_init(shape=(self.input_dim, self.n_features),
                                    dtype='float64'),trainable=True, 
                            name = 'linear')
        b_init = tf.initializers.GlorotNormal()
        self.b = tf.Variable(initial_value=b_init(shape=(self.input_dim,),
                                    dtype='float64'), trainable=True, 
                            name = 'bias')

        self.k = HaversineMatern()
        self.m = gpf.models.GPR(data=(self.x_l, self.x_l[:,0]), kernel=self.k, mean_function=None)
        
    def call(self, y_l, X_d, y_d):
        # Linear Map
        x = tf.math.multiply(X_d, self.w)
        x = tf.math.reduce_sum(x, axis=1) + self.b
        x = tf.reshape(x, (-1, 1))
        self.m.data = data_input_to_tensor((self.x_l, x))
        
        # Gaussian Process
        self.y_l = np.stack( (np.sin(np.deg2rad(90.0-y_l[:,1]))*np.cos(np.deg2rad(y_l[:,0]+180.0)),  
                     np.sin(np.deg2rad(90.0-y_l[:,1]))*np.sin(np.deg2rad(y_l[:,0]+180.0)),
                     np.cos(np.deg2rad(90.0-y_l[:,1]))), axis=-1)
        self.y_l = tf.cast(self.y_l, dtype='float64')
        mean, var = self.m.predict_f(self.y_l)
        
#         loss_mse = 0.5*tf.linalg.matmul((y_d - mean), (y_d - mean)/var, transpose_a = True)
        loss_mse = tf.math.reduce_mean( (y_d-mean)*(y_d-mean)/var)
        self.add_loss(loss_mse)
        return x, mean, var
    
    def predict(self, X_d):
        x = tf.math.multiply(X_d, self.w)
        x = tf.math.reduce_sum(x, axis=1) + self.b
        x = tf.reshape(x, (-1, 1))
        return x




####################################################################################
########################### Plotting functions #####################################
####################################################################################
def plot_stats(var, time, lats, lons, text, quantile = 0.5):
    import matplotlib.gridspec as gridspec
    # Plot the 'stat' statistic (either 'mean', or 'std') of variable 'var'
    # Along with temporal cdf's of randomly chosen points
    # Pass lat and lon values

    assert lats.size == var.shape[1], 'lat dimension does not match'
    assert lons.size == var.shape[2], 'lon dimension does not match'
    means = np.nanmean(var, axis=(1, 2))
    sort_i = np.argsort(means)
    sort_means = means[sort_i]
    q = np.quantile(means, quantile)
    print(q)
    q_i = np.argwhere(sort_means > q)[0]
    print(q_i)
    t_i = sort_i[q_i][0]
    print(t_i)

    fig = plt.figure(figsize=(18,5), constrained_layout = True)
    gs = gridspec.GridSpec(nrows=1, ncols=2, figure = fig, width_ratios = [2, 1])

    ax1 = fig.add_subplot(gs[0, :-1], projection=ccrs.Mollweide())

    cs = ax1.contourf(lons, lats, var[t_i], cmap = 'Spectral', transform=ccrs.PlateCarree())
    plt.colorbar(cs, ax = ax1, aspect=10, fraction = 0.05)
    ax1.coastlines()
    ax1.set_aspect('auto', adjustable=None)
    tt = pd.to_datetime(str(time[t_i])).strftime('%Y.%m.%d')
    ax1.set_title(str(quantile) + ' Quantile of ' + text + '\n time :' + tt )

    ax2 = fig.add_subplot(gs[0, -1])
    i = np.random.choice(np.arange(10, lats.size-10), 80)
    j = np.random.choice(np.arange(10, lons.size-10), 80)
    for ik, jk in zip(i, j):
        if np.isnan(var[0, ik, jk]):
            continue
        values, base = np.histogram(var[:, ik, jk], bins=40)
        cumulative = np.cumsum(values)/var.shape[0]
        ax2.plot(base[:-1], cumulative, c='blue', linewidth=0.2)
        ax1.scatter(lons[jk], lats[ik], c= 'k', s = 80, marker='*', transform=ccrs.PlateCarree())
    values, base = np.histogram(np.random.randn(var.shape[0]), bins = 40)
    cumulative = np.cumsum(values)/var.shape[0]
    ax2.plot(base[:-1], cumulative, 'k--', linewidth=2.0, label='Normal CDF')
    ax2.set_title('Temporal CDF of Starred Locations')
    ax2.grid()
    ax2.legend()
    ax2.set_xlim(-3, 3)
    plt.savefig(text+'_' + str(int(100*quantile)) + '_' + 'quantile.png')
    plt.show()
    
def plot_eof(var, lats, lons, text, n = 2):
    from sklearn.decomposition import PCA
    import matplotlib.gridspec as gridspec
    # Plot leading 'n' EOF's of variable 'var' in one of the two
    # following axes set up,   limit n to 5      
    # if n is even:                if n is odd:
    #    EOF1 EOF2                  EOF1 EOF2
    #      ...                        ...
    #    EOF-2 EOF-1                EOF-1 sing. values
    #     sing. values
    n = min(n, 5)

    pca = PCA(n_components=15)
    mask = np.isfinite(var.values.mean(axis=0))
    pca.fit(var.values[:, mask])
    v = np.NaN*np.ones_like(var[0])

    fig = plt.figure(figsize = (16,4))
    
    gs = gridspec.GridSpec(nrows=n//2, ncols=3, figure = fig, width_ratios = [10, 10, 1])
    levs = np.linspace(np.nanmin(pca.components_[:n]), np.nanmax(pca.components_[:n]), 15)
    for j in range(n):
        row = j // 2
        col = j % 2
        axes = fig.add_subplot(gs[row, col], projection=ccrs.Robinson())
        v[mask] = pca.components_[j]
        cs = axes.contourf(lons, lats, v, levels = levs, cmap='Spectral', transform=ccrs.PlateCarree())
#         plt.colorbar(cs, ax = axes)
        axes.coastlines()
        axes.set_title('EOF # ' + str(j+1) + ' with explained variance ratio = {:1.2f}'.format(pca.explained_variance_ratio_[j]))
        axes.set_aspect('auto', adjustable=None)
    
    f3_ax4 = fig.add_subplot(gs[:, -1])
    plt.colorbar(cs, cax = f3_ax4, aspect = 5, fraction = 0.01);
    plt.suptitle(text)
    plt.savefig(text+'_'+str(n)+'_'+'EOF.png')
    plt.show()

def plot_hovmoller(var, time, lats, lons, string):
    import matplotlib.gridspec as gridspec
    import metpy.calc as mpcalc
    import cartopy.feature as cfeature
    pd.plotting.register_matplotlib_converters()
    # Start figure
    ## Make Hovmuller Diagram

    # Compute weights and take weighted average over latitude dimension
    weights = np.cos(np.deg2rad(lats))
    avg_data = (var * weights[None, :, None]).sum(axis=1) / np.sum(weights)

    # Get times and make array of datetime objects
    vtimes = time.astype('datetime64[ms]').astype('O')

    fig = plt.figure(figsize=(15, 20))

    # Use gridspec to help size elements of plot; small top plot and big bottom plot
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[1, 6], hspace=0.03)

    # Tick labels
    x_tick_labels = [u'180\N{DEGREE SIGN}E', u'90\N{DEGREE SIGN}W',
                     u'0\N{DEGREE SIGN}E', u'90\N{DEGREE SIGN}E',
                     u'180\N{DEGREE SIGN}E']

    # Top plot for geographic reference (makes small map)
    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax1.set_extent([-180, 180, np.round(lats.min())-5, np.round(lats.max())+5], ccrs.PlateCarree())
    ax1.set_yticks([np.round(lats.min()), np.round(lats.max())])
    ax1.set_yticklabels([u'{:2.1f}\N{DEGREE SIGN}S'.format(np.round(lats.min())), u'{:2.1f}\N{DEGREE SIGN}S'.format(np.round(lats.max()))])
    ax1.set_xticks([-180, -90, 0, 90, 180])
    ax1.set_xticklabels(x_tick_labels)
    ax1.grid(linestyle='dotted', linewidth=2)

    # Add geopolitical boundaries for map reference
    ax1.add_feature(cfeature.COASTLINE.with_scale('50m'))
    ax1.add_feature(cfeature.LAKES.with_scale('50m'), color='black', linewidths=0.5)

    # Set some titles
    plt.title('Hovmoller Diagram', loc='left')
    plt.title(string, loc='right')

    # Bottom plot for Hovmoller diagram
    ax2 = fig.add_subplot(gs[1, 0])
    # ax2.invert_yaxis()  # Reverse the time order to do oldest first

    # Plot of chosen variable averaged over latitude and slightly smoothed
    # clevs = np.arange(-5, 5, 10)
    cf = ax2.contourf(lons, vtimes, mpcalc.smooth_n_point(
        avg_data, 9, 2), cmap=plt.cm.bwr, extend='both')
    cs = ax2.contour(lons, vtimes, mpcalc.smooth_n_point(
        avg_data, 9, 2), colors='k', linewidths=1)
    cbar = plt.colorbar(cf, orientation='horizontal', pad=0.04, aspect=50, extendrect=True)
    cbar.set_label('m $s^{-1}$')

    # Make some ticks and tick labels
    ax2.set_xticks([-177.5, -90, 0, 90, 177.5])
    ax2.set_xticklabels(x_tick_labels)
    ax2.set_yticks(pd.date_range(start=vtimes[0], end=vtimes[-1], freq="3MS").strftime('%Y.%m.%d'))
    ax2.set_yticklabels(pd.date_range(start=vtimes[0], end=vtimes[-1], freq="3MS").strftime('%Y.%m.%d'))

    # Set some titles
    plt.title(string, loc='left', fontsize=10)
    plt.title('Time Range: {0:%Y%m} - {1:%Y%m}'.format(vtimes[0], vtimes[-1]),
              loc='right', fontsize=10)
    
    if lats[0] < 0:
        min_lat_string = str(int(-lats[0]))+'S'
    else:
        min_lat_string = str(int(lats[0]))+'N'
    if lats[-1] <0:
        max_lat_string = str(int(-lats[-1]))+'S'
    else:
        max_lat_string = str(int(lats[-1]))+'N'
    plt.savefig(string+'_'+min_lat_string+'_'+max_lat_string+'_'+'hovmoller.png')

    plt.show()
