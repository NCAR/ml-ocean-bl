import tensorflow as tf
import numpy as np
import xarray as xr

class data():
    def __init__(self, file_path, with_time = False):
        with xr.open_dataset(file_path) as df:
            MLD = df.MLD.values
            SALT = df.SALT.values
            TEMP = df.TEMP.values
            SSH = df.SSH.values
            time = df.time.values
            lat = df.lat.values
            lon = df.lon.values
        self.time = time
        self.lat = lat
        self.lon = lon
        self.with_time = with_time
        index = np.arange(300)
        rng = np.random.default_rng(42)
        train_test_index = rng.permutation(index[:215]) 
        train_index = train_test_index[:200]
        test_index = np.append(train_test_index[200:], index[215:230])
        val_index = index[230:]

        X = np.stack( (SALT, TEMP, SSH), axis = -1)
        y = MLD

        mX = X[train_index].mean(axis = (0, 1,2))
        sX = X[train_index].std(axis = (0, 1,2))
        self.mX = mX
        self.sX = sX
        X = (X-mX)/sX

        my = y[train_index].mean(axis = (0, 1,2))
        sy = y[train_index].std(axis = (0, 1,2))
        self.my = my
        self.sy = sy
        y = (y - my)/sy
        
        X_train, X_test, X_val = X[train_index], X[test_index], X[val_index]
        y_train, y_test, y_val = y[train_index], y[test_index], y[val_index]
        t_train, t_test, t_val = time[train_index], time[test_index], time[val_index]
        
        if with_time:
            self.d_train = tf.data.Dataset.from_tensor_slices( (X_train, y_train, train_index)).shuffle(len(train_index), reshuffle_each_iteration=True)
            self.d_test = tf.data.Dataset.from_tensor_slices( (X_test, y_test, test_index)).shuffle(len(test_index), reshuffle_each_iteration=True)
            self.d_val = tf.data.Dataset.from_tensor_slices( (X_val, y_val, val_index)).shuffle(len(val_index), reshuffle_each_iteration=True)
        else:
            self.d_train = tf.data.Dataset.from_tensor_slices( (X_train, y_train)).shuffle(len(train_index), reshuffle_each_iteration=True)
            self.d_test = tf.data.Dataset.from_tensor_slices( (X_test, y_test)).shuffle(len(test_index), reshuffle_each_iteration=True)
            self.d_val = tf.data.Dataset.from_tensor_slices( (X_val, y_val)).shuffle(len(val_index), reshuffle_each_iteration=True)
    
    def get_data(self, batch_size = 10):
        if self.with_time:
            return self.d_train.batch(batch_size), self.d_test.batch(batch_size), self.d_val.batch(batch_size), self.time
        else:
            return self.d_train.batch(batch_size), self.d_test.batch(batch_size), self.d_val.batch(batch_size)
        
        
        