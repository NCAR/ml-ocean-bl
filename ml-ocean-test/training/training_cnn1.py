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
        
                
        
        from models.CNN import CNN
        for j in range(50):
            l = CNN(input_dim, n_features, batch_norm = True, num_layers = 5, Dropout = 0.1, num_dense_layers = 0, location = loc)
            losses = l.train(d_train, d_test, num_epochs = 80, lr = 1e-3)
            losses = l.train(d_train, d_test, num_epochs = 80, lr = 1e-4)
            l.save_weights('../models/saved_models/'+loc+'/'+d_type+'/CNN_ens/ens_'+str(j)+'/cnn')
            tf.keras.backend.clear_session()
        
        
        
        



        

        