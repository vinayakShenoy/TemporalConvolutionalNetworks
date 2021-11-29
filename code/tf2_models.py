import tensorflow as tf
#from tf.keras.models import Sequential, Model
#from keras.layers import Input, Dense, TimeDistributed, merge, Lambda
#from keras.layers.core import *
#from keras.layers.convolutional import *
#from keras.layers.recurrent import *


#from keras.activations import relu
from functools import partial

clipped_relu = partial(tf.keras.activations.relu, max_value=5)

def fix_gpu():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)


fix_gpu()

def channel_normalization(x):
    # Normalize by the highest activation
    max_values = tf.keras.backend.max(tf.keras.backend.abs(x), 2, keepdims=True)+1e-5
    out = x / max_values
    return out

def WaveNet_activation(x):
    tanh_out = tf.keras.layers.Activation('tanh')(x)
    sigm_out = tf.keras.layers.Activation('sigmoid')(x)
    return tf.keras.layers.Merge(mode='mul')([tanh_out, sigm_out])

def ED_TCN(n_nodes, conv_len, n_classes, n_feat, max_len,
            loss='categorical_crossentropy', causal=False,
            optimizer="rmsprop", activation='norm_relu',
            return_param_str=False):

    n_layers = len(n_nodes)
    inputs = tf.keras.layers.Input(shape=(max_len, n_feat))
    model = inputs

    for i in range(n_layers):
        #if causal: model = tf.keras.layers.ZeroPadding1D(padding=(conv_len//2))
        model = tf.keras.layers.Conv1D(n_nodes[i], conv_len)(model)    #n_nodes[i], conv_len, padding='same')(model)
        #if causal: model = tf.keras.layers.Cropping1D(())
        model = tf.keras.layers.SpatialDropout1D(0.3)(model)

        model = tf.keras.layers.Activation('relu')(model)
        model = tf.keras.layers.Lambda(channel_normalization, name="encoder_Norm_{}".format(i))(model)

    model = tf.keras.layers.MaxPooling1D(2)(model)

    for i in range(n_layers):
        model = tf.keras.layers.UpSampling1D(2)(model)
        model = tf.keras.layers.Convolution1D(n_nodes[-i - 1], conv_len, padding='same')(model)
        model = tf.keras.layers.SpatialDropout1D(0.3)(model)

    model = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_classes, activation="softmax"))(model)

    model = tf.keras.models.Model(inputs=inputs, outputs=model)
    model.compile(loss=loss, optimizer=optimizer, sample_weight_mode="temporal")#, metrics=['accuracy'])

    return model
    """
    for i in range(n_layers):
        if causal: model = tf.keras.layers.ZeroPadding1D(padding=(conv_len//2))
        #if causal: model = tf.keras.layers.ZeroPadding1D(padding=(conv_len//2, 0))(model)
        model = tf.keras.layers.Convolution1D(n_nodes[i], conv_len, padding='same')(model)
        if causal: model = tf.keras.layers.Cropping1D((0, conv_len//2))(model)
        model = tf.keras.layers.SpatialDropout1D(0.3)(model)

        if activation=='norm_relu':
            model = tf.keras.layers.Activation('relu')(model)
            model = tf.keras.layers.Lambda(channel_normalization, name="encoder_norm_{}".format(i))(model)
        elif activation=='wavenet':
            model = WaveNet_activation(model)
        else:
            model = tf.keras.layers.Activation(activation)(model)

        model = tf.keras.layers.MaxPooling1D(2)(model)


    for i in range(n_layers):
        model = tf.keras.layers.UpSampling1D(2)(model)
        if causal: model = tf.keras.layers.ZeroPadding1D((conv_len//2, 0))(model)
        model = tf.keras.layers.Convolution1D(n_nodes[-i - 1], conv_len, padding='same')(model)
        if causal: model = tf.keras.layers.Cropping1D((0,conv_len//2))(model)

        model = tf.keras.layers.SpatialDropout1D(0.3)(model)

        if activation=='norm_relu':
            model = tf.keras.layers.Activation('relu')(model)
            model = tf.keras.layers.Lambda(channel_normalization, name="decoder_norm_{}".format(i))(model)
        elif activation=='wavenet':
            model = WaveNet_activation(model)
        else:
            model = tf.keras.layers.Activation(activation)(model)
    """