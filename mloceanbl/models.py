####################################################################################
####################################################################################
########################### File Description #######################################
####################################################################################
####################################################################################

# There are a number of machine learning models coded in this documents. 

#  Machine learning models take sss, sst, ssh data as inputs and produce dense
#  grid predictions. Using the log_prob method of each model class we link the dense
#  grid to the sparse mld argo observation network via the gaussian process regression
#  class. The available models are:
#           1. Linear
#           2. ANN
#           3. Variational ANN (ANN_distribution)
#           4. Dropout ANN
#           5. Variational AutoEncoder (VAE)
#  For full details, see associated text with each method. 
# 



####################################################################################
########################### Import libraries #######################################
####################################################################################
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tf.keras.backend.set_floatx('float64')
tfd = tfp.distributions
tfb = tfp.bijectors

from linear import linear
from ann import ANN
from vann import VANN
from ann_dropout import ANN_dropout
from vae import VAE

from train import train
####################################################################################
########################### Define Models ##########################################
####################################################################################
 
MODELS = {
    'LINEAR': linear,
    'ANN': ANN,
    'VANN': VANN,
    'DROPOUT': ANN_dropout,
    'VAE': VAE,
}

class MLD_ML():
    def __init__(self, data, model='LINEAR', **kwargs):
        super(MLD_ML, self).__init__()
        
        # Normalize data according to training data
        self.data = data.normalize()
        
        X_d, X_l, y_d, y_l = self.data.get_index(i_train[0])

        # Call of methods is models.METHOD( input dimensions, number of features, input data locations )
        input_dim = X_d.shape[0]
        n_features = X_d.shape[1]
        self.model_text = model
        self.model = MODELS[model](input_dim, n_features, X_l)
    
    
    def train(self, num_epochs = 200, print_epoch = 1, lr = 2e-4, num_early_stopping = 15, num_restarts = 4, mini_batch_size = 25):
        loss  = train(self.data, 
                        self.model, 
                        epochs=num_epochs, 
                        print_epoch = print_epoch, 
                        lr = lr, 
                        num_early_stopping = num_early_stopping, 
                        num_restarts = num_restarts,
                        mini_batch_size = mini_batch_size)
        lp.save_weights('./saved_model/finished/'+self.model_text+'_test_corr_{:.2f}'.format(np.max(loss[-1])))
        return loss
    
    def evaluate(self, week_index):
        X_d, X_l, y_d, y_l = self.data.get_index(week_index)
        return self.model(X_d)
    
    def reanalysis(self, week_index):
        X_d, X_l, y_d, y_l = self.data.get_index(week_index)
        x = self.model(X_d)
        assert self.model_text is in ['VANN', 'DROPOUT', 'VAE'], 'Model does not have model error'
        
        u = self.model.gp.reanalysis(x, y_d, model.noise)
        return u
        
    