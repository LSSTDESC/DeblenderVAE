# Import necessary librairies

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
import sys
import os
import logging
import galsim
import random
import cmath as cm
import math
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Add, Multiply, Reshape, Flatten, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Input, Dense, Dropout, MaxPool2D, Flatten,  Reshape, UpSampling2D, Cropping2D, Conv2DTranspose, PReLU, Concatenate, Lambda, BatchNormalization, concatenate, LeakyReLU


#### Create encooder
def build_encoder(latent_dim, hidden_dim, filters, kernels,nb_of_bands, conv_activation='softplus', dense_activation='softplus'):
    """
    Return encoder as model
    latent_dim : dimension of the latent variable
    hidden_dim : dimension of the dense hidden layer
    filters: list of the sizes of the filters used for this model
    list of the size of the kernels used for each filter of this model
    conv_activation: type of activation layer used after the convolutional layers
    dense_activation: type of activation layer used after the dense layers
    nb_of bands : nb of band-pass filters needed in the model
    """
    input_layer = Input(shape=(64,64,nb_of_bands))

    r = Reshape((64,64,nb_of_bands))(input_layer)
    h = BatchNormalization()(r)
    for i in range(len(filters)):
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same')(h)
        h = PReLU()(h)
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same', strides=(2,2))(h)
        h = PReLU()(h)
    h = Flatten()(h)
    h = Dense(hidden_dim, activation=dense_activation)(h)
    h = PReLU()(h)
    mu = Dense(latent_dim)(h)
    sigma = Dense(latent_dim, activation='softplus')(h)
    return Model(input_layer, [mu, sigma])


#### Create encooder
def build_decoder(input_shape, latent_dim, hidden_dim, filters, kernels, conv_activation='softplus', dense_activation='softplus'):
    """
    Return decoder as model
    input_shape: shape of the input data
    latent_dim : dimension of the latent variable
    hidden_dim : dimension of the dense hidden layer
    filters: list of the sizes of the filters used for this model
    list of the size of the kernels used for each filter of this model
    conv_activation: type of activation layer used after the convolutional layers
    dense_activation: type of activation layer used after the dense layers
    """
    input_layer = Input(shape=(latent_dim,))
    h = Dense(hidden_dim, activation=dense_activation)(input_layer)
    h = PReLU()(h)
    w = int(np.ceil(input_shape[0]/2**(len(filters))))
    h = Dense(w*w*filters[-1], activation=dense_activation)(h)
    h = PReLU()(h)
    h = Reshape((w,w,filters[-1]))(h)
    for i in range(len(filters)-1,-1,-1):
        h = Conv2DTranspose(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same', strides=(2,2))(h)
        h = PReLU()(h)
        h = Conv2DTranspose(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same')(h)
        h = PReLU()(h)
    h = Conv2D(input_shape[-1], (3,3), activation='sigmoid', padding='same')(h)
    cropping = int(h.get_shape()[1]-input_shape[0])
    if cropping>0:
        print('in cropping')
        if cropping % 2 == 0:
            h = Cropping2D(cropping/2)(h)
        else:
            h = Cropping2D(((cropping//2,cropping//2+1),(cropping//2,cropping//2+1)))(h)

    return Model(input_layer, h)



# Function to define model

def vae_model(nb_of_bands):
    """
    Function to create VAE model
    nb_of bands : nb of band-pass filters needed in the model
    """

    #### Parameters to fix
    # batch_size : size of the batch given to the network
    # epsilon_std : standard deviation given to compute the latent variable z
    # input_shape: shape of the input data
    # latent_dim : dimension of the latent variable
    # hidden_dim : dimension of the dense hidden layer
    # filters: list of the sizes of the filters used for this model
    # kernels: list of the size of the kernels used for each filter of this model

    batch_size = 100 
    epsilon_std = 1.0
    
    input_shape = (64,64,nb_of_bands)
    latent_dim = 10
    hidden_dim = 256
    filters = [32,64, 128, 256]
    kernels = [3,3,3,3]

    # Build the encoder
    encoder = build_encoder(latent_dim, hidden_dim, filters, kernels, nb_of_bands)
    # Build the decoder
    decoder = build_decoder(input_shape, latent_dim, hidden_dim, filters, kernels, conv_activation=None, dense_activation=None)

    #### Create an input for the lambda function to compute the latent variable z
    input_vae = Input(shape=(64,64, nb_of_bands))
    output_encoder = encoder(input_vae)

    # Lambda function to compute latent variable z
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    # Build the latent variable z
    z = Lambda(sampling, output_shape=(latent_dim,))(output_encoder)

    #### Build the model
    vae = Model(input_vae, decoder(z))  
    
    return encoder, decoder