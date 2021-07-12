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
        
        from models.GAN import GAN
        l = GAN(input_dim, n_features, batch_norm = True, Dropout = 0.1, latent_dim = 80, location = loc)
        losses = l.train(d_train, d_test, num_epochs = 200, lr = 1e-3)
        l.save_weights('../models/saved_models/'+loc+'/'+data_type+'/GAN/gan')
        tf.keras.backend.clear_session()