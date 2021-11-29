import numpy as np
import tensorflow as tf
from random import shuffle
# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.

class JIGSAWS_DataGen(tf.keras.utils.Sequence):
    def __init__(self, batch_size, x_set, y_set, max_len):
        self.X = x_set
        self.Y = y_set
        self.batch_size = batch_size
        self.max_len = max_len
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, i):
        x = self.X[i * self.batch_size:(i + 1) * self.batch_size]
        y = self.Y[i * self.batch_size:(i + 1) * self.batch_size]
        print(i)
        return self.mask_data(x, y)

    def load_split(self):


    def mask_data(self, X, Y, mask_value=-1):
        if self.max_len is None:
            self.max_len = np.max([x.shape[0] for x in X])
        X_ = np.zeros([len(X), self.max_len, X[0].shape[1]]) + mask_value
        Y_ = np.zeros([len(X), self.max_len, Y[0].shape[1]]) + mask_value
        #print(X_.shape)
        #print(Y_.shape)
        mask = np.zeros([len(X), self.max_len])
        for i in range(len(X)):
            l = X[i].shape[0]
            X_[i, :l] = X[i]
            Y_[i, :l] = Y[i]
            mask[i, :l] = 1
        print(X_.shape, Y_.shape)
        return X_, Y_, mask[:, :]

    def on_epoch_end(self):
        temp = list(zip(self.X, self.Y))
        shuffle(temp)
        self.X, self.Y = zip(*temp)