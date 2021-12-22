"""
This script takes features from a set of .mat files, trains a model, and outputs a set of action predictions for all time steps in the input.

I suggest working with this file interactively in an iPython shell. 
The input dataset, features, model parameters, etc should be specified below. 

Available models:
* SVM: Support Vector machine applied to each frame independently
* LSTM: Typical causal or acausal/birdirectional RNN+LSTM baseline from Keras
* LC-SC-CRF: Latent Convolutional Skip Chain Conditional Random Field. 
-- From "Learning Convolutional Action Primitives for Fine-grained Action Recognition" C Lea, R Vidal, G Hager (ICRA 2016)
* DTW: Dynamic Time Warping baseline. 
* tCNN: The temporal component of a Spatiotemporal CNN
-- From "Segmental Spatiotemporal CNNs for Fine-grained Action Segmentation" C Lea, A Reiter, R Vidal, G Hager (ECCV 2016)
* ED-TCN: Encoder Decoder Temporal Convolutional Network
* DilatedTCN: Variation on WaveNet
-- Both of these are from "Temporal Convolutional Networks for Action Segmentation and Detection" C Lea, M Flynn, R Vidal, A Reiter, H Hager (arXiv 2016)
* TDNN: Simple variation on the Time-Delay Neural Network (not fully tested)

Note: the LC-SC-CRF and DTW code can be obtained here: https://github.com/colincsl/LCTM

Updated December 2016
Colin Lea
"""

import os
from collections import OrderedDict

import numpy as np
import matplotlib.pylab as plt


from scipy import io as sio
import sklearn.metrics as sm
from sklearn.svm import LinearSVC

import tensorflow as tf
#from tf.keras.utils import np_utils

# TCN imports 
import tf2_models, tf_models, datasets, utils, metrics, jigsaws_dataloader
from utils import imshow_

import warnings

# ---------- Directories & User inputs --------------
# Location of data/features folder
base_dir = os.path.expanduser("~/temporal_action_detection/TCN/")

save_predictions = [False, True][1]
viz_predictions = [False, True][1]
viz_weights = [False, True][0]

# Set dataset and action label granularity (if applicable)
dataset = ["50Salads", "JIGSAWS", "MERL", "GTEA"][1]
granularity = ["eval", "mid"][1]
sensor_type = ["video", "sensors"][0]

# Set model and parameters
model_type = ["SVM", "LSTM", "LC-SC-CRF", "tCNN",  "DilatedTCN", "ED-TCN", "TDNN"][-2]
# causal or acausal? (If acausal use Bidirectional LSTM)
causal = [False, True][0]

# How many latent states/nodes per layer of network
# Only applicable to the TCNs. The ECCV and LSTM  model suses the first element from this list.
n_nodes = [64, 128]#[128, 256, 512]
nb_epoch = 200
video_rate = 3
conv = {'50Salads':25, "JIGSAWS":20, "MERL":5, "GTEA":25}[dataset]

# Which features for the given dataset
features_from = "VideoSwin_K600_CrossVal"
bg_class = 0 if dataset is not "JIGSAWS" else None

if dataset == "50Salads":
    features_from = "SpatialCNN_" + granularity

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# Evaluate using different filter lengths
if 1:
# for conv in [5, 10, 15, 20]:
    # Initialize dataset loader & metrics

    trial_metrics = metrics.ComputeMetrics(overlap=.1, bg_class=bg_class)

    lengths = []
    n_feat = None
    path = "{}/{}/{}/{}".format(base_dir, "features",dataset, features_from)
    for f in os.listdir(path):
        f_name = "{}/{}".format(path, f)
        t = sio.loadmat(f_name)
        lengths.append(t['S'].shape[0])
        if not n_feat:
            n_feat = t['S'].shape[1]


    n_classes = 10

    # ------------------ Models ----------------------------
    if model_type == "SVM":
        svm = LinearSVC()
        svm.fit(np.vstack(X_train), np.hstack(y_train))
        P_test = [svm.predict(x) for x in X_test]

        # AP_x contains the per-frame probabilities (or class) for each class
        AP_train = [svm.decision_function(x) for x in X_train]
        AP_test = [svm.decision_function(x) for x in X_test]
        param_str = "SVM"

    # --------- CVPR model ----------
    elif model_type in ["tCNN", "ED-TCN", "DilatedTCN", "TDNN", "LSTM"]:
        # Go from y_t = {1...C} to one-hot vector (e.g. y_t = [0, 0, 1, 0])
        #Y_train = [tf.keras.utils.to_categorical(y, n_classes) for y in y_train]
        #Y_test = [tf.keras.utils.to_categorical(y, n_classes) for y in y_test]

        # In order process batches simultaneously all data needs to be of the same length
        # So make all same length and mask out the ends of each.
        n_layers = len(n_nodes)
        max_len = np.max(lengths)
        max_len = int(np.ceil(max_len / (2**n_layers)))*2**n_layers
        print("Max length:", max_len)

        train_accuracies = []
        val_accuracies = []
        number_splits = 8
        files = os.listdir(base_dir + "features/" + dataset + "/" + features_from)
        splits = []
        users = ['B','C','D','E','F','G','H','I']
        for split_num in users:
            splits.append(list(filter(lambda x: split_num in x, files)))
        for split_num in range(number_splits):
            print("Split :", split_num)
            print("#########################################")
            test_files = splits[split_num]
            train_files_splits = splits[:split_num] + splits[split_num+1:]
            train_files = [x for l in train_files_splits for x in l]
            train_data_generator = jigsaws_dataloader.JIGSAWS_DataLoader(batch_size=1, dataset=dataset, base_dir=base_dir,
                                                                         features_from=features_from, feature_len=n_feat,
                                                                         max_len=max_len, sample_rate=1, filenames=train_files)
            val_data_generator = jigsaws_dataloader.JIGSAWS_DataLoader(batch_size=1, dataset=dataset, base_dir=base_dir,
                                                                        features_from=features_from, feature_len=n_feat,
                                                                        max_len=max_len, sample_rate=1, filenames=test_files, is_train=False)

            model = tf2_models.ED_TCN(n_nodes, conv, n_classes, n_feat, max_len, causal=causal,
                                                activation='norm_relu', return_param_str=True)

            history = model.fit(x=train_data_generator,
                      validation_data=val_data_generator,
                      epochs=50,
                      verbose=1)

            train_accuracies.append(history.history['accuracy'][-1])
            val_accuracies.append(history.history['val_accuracy'][-1])

            AP_train = model.predict(train_data_generator, verbose=0)
            AP_test = model.predict(val_data_generator, verbose=0)
            #print(AP_train)
            #print(AP_test)
            AP_train = utils.unmask(AP_train, )
            AP_test = utils.unmask(AP_test, M_test)

            #P_train = [p.argmax(1) for p in AP_train]
            #P_test = [p.argmax(1) for p in AP_test]
            break
        print("Average Train Accuracy: ", np.mean(train_accuracies))
        print("Average Val Accuracy: ", np.mean(val_accuracies))

