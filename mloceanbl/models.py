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
#           1. Linear:                                'LINEAR'
#           2. ANN:                                   'ANN'
#           3. Variational ANN (ANN_distribution):    'VANN'
#           4. Dropout ANN:                           'DROPOUT'
#           5. Variational AutoEncoder (VAE):         'VAE'
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

from linear import Linear
from ann import ANN
from vann import VANN
from ann_dropout import ANN_dropout
from vae import VAE
from cvae import CVAE
from vcnn import VCNN
from vlcnn import VLCNN
from deepcnn import DEEPCNN
import gp
from train import train
from time import time, strftime
####################################################################################
########################### Define Models ##########################################
####################################################################################
 
MODELS = {
    'LINEAR': Linear,
    'ANN': ANN,
    'VANN': VANN,
    'DROPOUT': ANN_dropout,
    'VAE': VAE,
    'CVAE':CVAE,
    'VCNN': VCNN,
    'VLCNN': VLCNN,
    'DEEPCNN': DEEPCNN,
}

class MLD_ML():
    def __init__(self, data, model='LINEAR', **kwargs):
        super(MLD_ML, self).__init__()
        
        # Normalize data according to training data
        self.data = data
        self.data.normalize()
        
        X_d, X_l, y_d, y_l = self.data.get_index(self.data.i_train[0])

        # Call of methods is models.METHOD( input dimensions, number of features, input data locations )
        x_dim = X_d.shape[0]
        y_dim = X_d.shape[1]
        input_dim = (x_dim, y_dim)
        n_features = X_d.shape[2]
        self.model_text = model
        self.model = MODELS[model](input_dim, n_features, X_l)
            
    
    def train(self, num_epochs = 200, print_epoch = 1, lr = 2e-4, num_early_stopping = 15, num_restarts = 4, mini_batch_size = 25, 
             save_location = 'TEMP', save_name = 'temp'):
        loss  = train(self.data, 
                        self.model, 
                        epochs=num_epochs, 
                        print_epoch = print_epoch, 
                        lr = lr, 
                        num_early_stopping = num_early_stopping, 
                        num_restarts = num_restarts,
                        mini_batch_size = mini_batch_size,
                        save_location = save_location,
                        save_name = save_name)
        return loss
    
    def evaluate(self, week_index, return_loss = False):
        self.X_d, self.X_l, self.y_d, self.y_l = self.data.get_index(week_index)
        x = self.model(self.X_d, training = False)
        assert isinstance(return_loss, bool), 'return_loss must be True/False!'
        
        if return_loss:
            self.model.gp.fit( x, noise = self.model.var )
            loss = self.model.log_prob(x, self.y_l, self.y_d, training = False)
            return x, loss
        else:
            return x
    
    def reanalysis(self, week_index, num_iters = 100, u0=None, var = None, y = None, silent = True):
        self.X_d, self.X_l, self.y_d, self.y_l = self.data.get_index(week_index)
        if y is not None:
            self.y_l, self.y_d = y
        x = self.model(self.X_d)
        x = tf.reshape(x, (-1,1))
        u0 = tf.reshape(u0, (-1,1))
        if u0 is None:
            u = tf.Variable(x)
        else:
            assert x.shape.as_list() == u0.shape.as_list(), 'Shape of u0 should be consistent with the model.'
            u = tf.Variable(u0)
        if var is None:
            var = self.model.var
        var = tf.Variable(var)
        gpr = gp.GPR(self.X_l)
        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        opt_var = tf.keras.optimizers.Adam(learning_rate=0.01)
        
        def optimize(u, var):
            with tf.GradientTape(persistent = True) as tape:
                m, v = gpr(u, self.y_l, noise = var)
                loss = -gpr.log_prob( self.y_d )
                loss += tf.reduce_mean( var - tf.math.log(var) )
                z = (tf.reshape(u, (-1,1)) - tf.reshape(x, (-1,1)))
            u_grads = tape.gradient(loss, [u])
            opt.apply_gradients(zip(u_grads, [u]))
            return loss
        
        assert isinstance(silent, bool), 'silence must be True/False!'
        if not silent: print('Beginning Reanalysis')
        total = 0
        for j in range(num_iters):
            start = time()
            loss = optimize(u, var)
            end = time()
            diff = end-start
            total += diff
            remaining = ( total/(j+1) )*num_iters - total
            hours, rem = divmod(remaining, 3600)
            minutes, seconds = divmod(rem, 60)
            if (j%10 == 0) and (not silent): print('Epoch: ',j,'/',num_iters,'\t Loss: {:.2f}'.format(loss.numpy()), '\t Time Remaining:', "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
        return u, var
        
    