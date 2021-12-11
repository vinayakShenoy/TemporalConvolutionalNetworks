import numpy as np
import tensorflow as tf
from random import shuffle
import os
import scipy.io as sio

class JIGSAWS_DataLoader(tf.keras.utils.Sequence):
    def __init__(self,
                 batch_size,
                 dataset,
                 base_dir,
                 features_from,
                 feature_len,
                 max_len,
                 sample_rate,
                 is_train=True):
        self.sample_rate = sample_rate
        self.max_len = max_len
        self.batch_size = batch_size
        self.dataset = dataset
        self.base_dir = base_dir
        if is_train:
            self.features_from = features_from + "/train"
        else:
            self.features_from = features_from + "/val"
        self.filenames = os.listdir(self.base_dir+"features/{}/{}/".format(self.dataset, self.features_from))
        print(self.filenames)
        self.n_classes = 10
        self.feature_len = feature_len
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.filenames) / self.batch_size))

    def __getitem__(self, idx):
        main_dir = "{}/{}/{}/".format(self.base_dir, self.dataset, self.features_from)
        batch_filenames = self.filenames[idx * self.batch_size:(idx + 1) *self.batch_size]
        X_ = np.empty((self.batch_size, self.max_len, self.feature_len))
        Y_ = np.empty((self.batch_size, self.max_len, 1))
        M_ = np.empty((self.batch_size, self.max_len))
        for i, filename in enumerate(batch_filenames):
            data = self.load_mat(self.base_dir+"features/{}/{}/{}".format(self.dataset, self.features_from, filename))
            x, y = data['S'], data['Y']
            x_, y_, m_ = self.mask_data(x, y)
            if self.sample_rate > 1:
                x_, y_ = self.subsample(x_, y_, self.sample_rate, dim=0)
            X_[i, ], Y_[i, ], M_[i, ] = x_, y_, m_

        Y_ = tf.keras.utils.to_categorical(Y_, num_classes=self.n_classes)
        return X_, Y_, M_

    def on_epoch_end(self):
        shuffle(self.filenames)

    def load_mat(self, filename):
        return sio.loadmat(filename)

    def mask_data(self, X, Y, mask_value=-1):
        if self.max_len is None:
            self.max_len = np.max([x.shape[0] for x in X])
        x_ = np.zeros([self.max_len, X.shape[1]]) + mask_value
        y_ = np.zeros([self.max_len, 1]) + mask_value
        mask = np.zeros([self.max_len])
        l = X.shape[0]
        x_[:l] = X
        y_[:l] = Y.transpose()
        mask[:l] = 1
        return x_, y_, mask[:]

    def subsample(self, X, Y, rate=1, dim=0):
        if dim == 0:
            X_ = [x[::rate] for x in X]
            Y_ = [y[::rate] for y in Y]
        elif dim == 1:
            X_ = [x[:, ::rate] for x in X]
            Y_ = [y[::rate] for y in Y]
        else:
            print("Subsample not defined for dim={}".format(dim))
            return None, None

        return X_, Y_