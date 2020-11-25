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
}

class MLD_ML():
    def __init__(self, data, model='LINEAR', **kwargs):
        super(MLD_ML, self).__init__()
        
        # Normalize data according to training data
        self.data = data
        self.data.normalize()
        
        X_d, X_l, y_d, y_l = self.data.get_index(self.data.i_train[0])

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
        self.model.save_weights('./saved_model/finished/'+self.model_text+ \
                                    '_test_corr_{:.2f}'.format(np.max(loss[-1])))
        return loss
    
    def evaluate(self, week_index, return_loss = False):
        self.X_d, self.X_l, self.y_d, self.y_l = self.data.get_index(week_index)
        x = self.model(self.X_d)
        assert isinstance(return_loss, bool), 'return_loss must be True/False!'
        
        if return_loss:
            loss = self.model.log_prob(x, self.y_l, self.y_d)
            return x, loss
        else:
            return self.model(self.X_d)
    
    def reanalysis(self, week_index, num_iters = 100, u0=None):
        self.X_d, self.X_l, self.y_d, self.y_l = self.data.get_index(week_index)
        x = self.model(self.X_d)
        if u0 is None:
            u = tf.Variable(x)
        else:
            assert x.shape.as_list() == u0.shape.as_list(), 'Shape of u0 should be consistent with the model.'
            u = tf.Variable(u0)
        gpr = gp.GPR(self.X_l)
        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        
        def optimize(u):
            with tf.GradientTape() as tape:
                m, v = gpr(u, self.y_l, noise = self.model.noise, number_of_opt_steps = 50)
                loss = -gpr.log_prob( self.y_d )
                z = (tf.reshape(u, (-1,1)) - tf.reshape(x, (-1,1)))
                V = tf.linalg.diag(1./self.model.noise)
                loss += 0.5 * tf.matmul(z, tf.matmul(V,z),
                                    transpose_a = True)/tf.cast(self.model.noise.shape, dtype='float64')
            grads = tape.gradient(loss, [u])
            # Ask the optimizer to apply the processed gradients.
            opt.apply_gradients(zip(grads, [u]))
            return loss
        print('Beginning Reanalysis')
        total = 0
        for j in range(num_iters):
            start = time()
            loss = optimize(u)
            end = time()
            diff = end-start
            total += diff
            remaining = ( total/(j+1) )*num_iters - total
            hours, rem = divmod(remaining, 3600)
            minutes, seconds = divmod(rem, 60)
            if j%10 == 0: print('Epoch: ',j,'/',num_iters,'\t Loss: {:.2f}'.format(loss.numpy()[0][0]), '\t Time Remaining:', "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
        return u
        
    