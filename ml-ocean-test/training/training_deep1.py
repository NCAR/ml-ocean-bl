import numpy as np
import xarray as xr
import tensorflow as tf

from sys import path
path.append('../data/')
from data import data
path.append('..')

for loc in ["EPO"]:
    print(loc)
    for d_type in ["std_anomalies"]:
        print(d_type)
        file = "../data/" + d_type + "_" + loc + ".nc"
        d = data(file)
        d_train, d_test, d_val = d.get_data()
        if loc == "EPO":
            input_dim = (40,60)
        else:
            input_dim = (20, 120)
        n_features = 3
        
        from models.DEEP_CNN import DEEP_CNN
        l = DEEP_CNN(input_dim, n_features, variational = True, location = loc)
        losses = l.train(d_train, d_test, loc, d_type, num_epochs = 40, lr = 1e-4)
        l.save_weights('../models/saved_models/'+loc+'/'+d_type+'/VDEEPCNN/vdeep_cnn')
        losses = l.train(d_train, d_test, loc, d_type, num_epochs = 20, lr = 1e-4)
        l.save_weights('../models/saved_models/'+loc+'/'+d_type+'/VDEEPCNN/vdeep_cnn')
        tf.keras.backend.clear_session()
        
        
        
        



        

        