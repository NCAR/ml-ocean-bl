# Import Necessary Libraries
import numpy as np
import tensorflow as tf
import pandas as pd
import properscoring as ps
import xarray as xr
from scipy.stats import pearsonr
# Import Model module
from sys import path
path.append('../mloceanbl/')
import data
import models

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from numpy.random import default_rng
rng = default_rng(0)
from scipy.stats import entropy
from scipy.special import ndtri
t = [1e-4, 1e-3, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99, 1-1e-3, 1-1e-4]
bins = ndtri(t)


df_weekly = []
df_individual = []
save_locations = {'LINEAR': 'VLinear/vlinear', 'VCNN':'VCNN/vcnn', 'VLCNN':'VLCNN/vlcnn', 'DEEPCNN':'DEEPCNN/deepcnn'}
for loc in ["EPO", "SIO"]:
    if loc == "EPO":
        lat_bounds = np.array([-10, 10])
        lon_bounds = np.array([-150, -120])
    else:
        lat_bounds = np.array([-45, -35])
        lon_bounds = np.array([55, 115])

    dataset = data.dataset(lat_bounds, lon_bounds)
    
    for m_type in ['GP', 'LINEAR', 'VCNN', 'VLCNN', 'DEEPCNN']:
        
        if m_type == 'GP':
            data_series = ['train', 'test', 'val']
            dataset_cont = [dataset.i_train, dataset.i_test, dataset.i_val]
            for (series, iterator) in zip(data_series, dataset_cont):
                for i in iterator:
                    X_d, X_l, y_d, y_l = dataset.get_index(i)
                    m = y_d.shape[0]
                    
                    mae = 0.0
                    corr = 0.0
                    cal = 0.0
                    kldiv = 0.0
                    for j in range(5):
                        ind = rng.permutation(m)
                        X_train = y_l[ind[:m//2]]
                        X_test = y_l[ind[m//2:]]
                        y_train = y_d[ind[:m//2]]
                        y_test = y_d[ind[m//2:]]
                        kernel = 1.0 * RBF(length_scale=1.0)
                        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer = 5, random_state=0)
                        gpr = gpr.fit(X_train, y_train)
                        pred = gpr.predict(X_test, return_std = True)
                        mean = pred[0]
                        std = pred[1]
                        if np.abs(mean - mean.mean()).mean() < 1e-3:
                            mean += 1e-4*rng.standard_normal(mean.shape[0])
                        d = np.abs(mean - y_test)
                        mae += d.mean()/5.0
                        corr += pearsonr(mean.flatten(), y_test.flatten())[0]/5.0
                        d_norm = (mean - y_test)/std
                        cal += ( d_norm < 1.0 ).mean()/5.0
                        
                        # KL Div
                        hist, bin_edges = np.histogram(d_norm, bins, density=True)
                        bin_mids = 0.5*bins[1:] + 0.5*bins[:-1]
                        obs = np.exp(-bin_mids**2/2.)/np.sqrt(2*np.pi)
                        kldiv += entropy(hist, qk = obs)/5.0
                        # Individual Metrics: Normed Error
                        for j in range(d.shape[0]):
                            df_ind_temp = [loc, 'std_anomalies', series, dataset.time[i], m_type, d[j]/np.abs(y_test[j]), d_norm[j]]
                            df_individual.append(df_ind_temp)
                    df_weekly_temp = [loc, 'std_anomalies', series, dataset.time[i],  m_type, mae/np.abs(y_d).mean(), corr, cal, kldiv]
                    df_weekly.append(df_weekly_temp)
            
            df_to_save = pd.DataFrame(df_weekly, columns = ['Location', 'DataType', 'Dataset', 'Time', 'Model', 'RelativeMAEError', 'Correlation', 'Calibration', 'KLDivergence'])
            df_to_save.to_csv('Model_Calibration_Results_Weekly.csv')

            df_to_save = pd.DataFrame(df_individual, columns = ['Location', 'DataType', 'Dataset', 'Time', 'Model', 'RelativeMAEError', 'NormedError'])
            df_to_save.to_csv('Model_Calibration_Results_Individual.csv')

                    
                    
                    
        else:
            m = models.MLD_ML(dataset, m_type)
            m.model.load_weights('../mloceanbl/saved_models/' + loc + '/std_anomalies/' + save_locations[m_type])

            data_series = ['train', 'test', 'val']
            dataset_cont = [dataset.i_train, dataset.i_test, dataset.i_val]
            for (series, iterator) in zip(data_series, dataset_cont):
                for i in iterator:
                    X_d, X_l, y_d, y_l = dataset.get_index(i)     
                    y_pred = m.model(X_d)
                    y_pred = tf.reshape(y_pred, (-1, 1))
                    mean, var = m.model.gp(y_pred, y_l, m.model.var)
    
                    d = tf.abs(mean - y_d)
                    d_norm = (mean - y_d)/tf.sqrt(var)


                    # Weekly Metrics: MAE, Calibration, Correlation, KL Div
                    mae = tf.reduce_mean(d).numpy() / np.abs(y_d).mean()
                    cal = tf.reduce_mean( tf.cast(d_norm < 1, dtype = 'float64')).numpy()
                    corr = pearsonr(mean.numpy().flatten(), y_d.flatten())[0]
                    # KL Div
                    hist, bin_edges = np.histogram(d_norm, bins, density=True)
                    bin_mids = 0.5*bins[1:] + 0.5*bins[:-1]
                    obs = np.exp(-bin_mids**2/2.)/np.sqrt(2*np.pi)
                    kldiv = entropy(hist, qk = obs)

                    df_weekly_temp = [loc, 'std_anomalies', series, dataset.time[i],  m_type, mae, corr, cal, kldiv]
                    df_weekly.append(df_weekly_temp)

                    # Individual Metrics: Normed Error
                    for j in range(y_d.shape[0]):
                        df_ind_temp = [loc, 'std_anomalies', series, dataset.time[i], m_type, d[j].numpy()/np.abs(y_d[j]), d_norm[j].numpy()]
                        df_individual.append(df_ind_temp)
                print(df_weekly[-1])
                print(df_individual[-1])

            tf.keras.backend.clear_session()
    
            df_to_save = pd.DataFrame(df_weekly, columns = ['Location', 'DataType', 'Dataset', 'Time', 'Model', 'RelativeMAEError', 'Correlation', 'Calibration', 'KLDivergence'])
            df_to_save.to_csv('Model_Calibration_Results_Weekly.csv')

            df_to_save = pd.DataFrame(df_individual, columns = ['Location', 'DataType', 'Dataset', 'Time', 'Model', 'RelativeMAEError', 'NormedError'])
            df_to_save.to_csv('Model_Calibration_Results_Individual.csv')
            
            