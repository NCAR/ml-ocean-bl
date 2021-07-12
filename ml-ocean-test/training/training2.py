import numpy as np
import xarray as xr
import tensorflow as tf

from data import data

for loc in ["EPO"]:
    print(loc)
    for d_type in ["std_anomalies"]:
        print(d_type)
        file = "./data/" + d_type + "_" + loc + ".nc"
        d = data(file)
        d_train, d_test, d_val = d.get_data()
        if loc == "EPO":
            input_dim = (40,60)
        else:
            input_dim = (20, 120)
        n_features = 3

        from models.RESNET import RESNET
        for j in range(30, 50):
            print(j)
            resnet = RESNET( input_dim, n_features, num_layers = 6, Dropout = 0.05, location = loc)
            resnet_losses = resnet.train(d_train, d_test, num_epochs = 40, lr = 1e-3)
            resnet_losses = resnet.train(d_train, d_test, num_epochs = 20, lr = 2e-4)
            resnet.save_weights('./models/saved_models/'+loc+'/'+d_type+'/RESNET_ens/ens_'+str(j)+'/resnet')  
#         from models.CVAE import CVAE
#         print('CVAE')
#         cvae = CVAE(input_dim, n_features, batch_norm = True, Dropout = 0.1, latent_dim = 100, location = loc)
#         cvae_losses = cvae.train(d_train, d_test, num_epochs = 100, num_epochs_split = 40, lr = 1e-3)
#         cvae_losses = cvae.train(d_train, d_test, num_epochs = 40, num_epochs_split = 10, lr = 1e-4)
#         cvae.save_weights('./models/saved_models/'+loc+'/'+d_type+'/CVAE/cvae')
#         tf.keras.backend.clear_session()
        
#         from models.GAN import GAN
#         print('GAN')
#         gan = GAN(input_dim, n_features, batch_norm = True, Dropout = 0.1, latent_dim = 100, location = loc)
#         gan_losses = gan.train(d_train, d_test, num_epochs = 200, lr = 8e-4)
#         gan.save_weights('./models/saved_models/'+loc+'/'+d_type+'/GAN/gan')
#         tf.keras.backend.clear_session()


        