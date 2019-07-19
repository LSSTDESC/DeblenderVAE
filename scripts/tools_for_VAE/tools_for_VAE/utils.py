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
from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Add, Multiply, BatchNormalization, Reshape, Flatten, Conv2D,  PReLU,Conv2DTranspose
from tensorflow.keras.models import Model, Sequential
from scipy.stats import norm
import tensorflow as tf

from . import model, vae_functions

############# Normalize data ############# 
def norm(x, bands, channel_last=False):
    I = [6.48221069e+05, 4.36202878e+05, 2.27700000e+05, 4.66676013e+04,2.91513800e+02, 2.64974100e+03, 4.66828170e+03, 5.79938030e+03,5.72952590e+03, 3.50687710e+03]
    beta = 5.
    if channel_last:
        assert x.shape[-1] == len(bands)
        for i in range (len(x)):
            for ib, b in enumerate(bands):
                x[i,:,:,ib] = np.tanh(np.arcsinh(x[i,:,:,ib]/(I[b]/beta)))
    else:
        assert x.shape[1] == len(bands)
        for i in range (len(x)):
            for ib, b in enumerate(bands):
                x[i,ib] = np.tanh(np.arcsinh(x[i,ib]/(I[b]/beta)))
    return x

def denorm(x, bands, channel_last=False):
    I = [6.48221069e+05, 4.36202878e+05, 2.27700000e+05, 4.66676013e+04,2.91513800e+02, 2.64974100e+03, 4.66828170e+03, 5.79938030e+03,5.72952590e+03, 3.50687710e+03]
    beta = 5.
    if channel_last:
        assert x.shape[-1] == len(bands)
        for i in range (len(x)):
            for ib, b in enumerate(bands):
                x[i,:,:,ib] = np.sinh(np.arctanh(x[i,:,:,ib]))*(I[b]/beta)
    else:
        assert x.shape[1] == len(bands)
        for i in range (len(x)):
            for ib, b in enumerate(bands):
                x[i,ib] = np.sinh(np.arctanh(x[i,ib]))*(I[b]/beta)
    return x

############# LOAD MODEL ##################
def load_vae_conv(path,nb_of_bands,folder = False):
    """
    Return the loaded VAE located at the path given when the function is called
    """        
    batch_size = 100 
    epsilon_std = 1.0
    
    input_shape = (64,64,nb_of_bands)
    latent_dim = 32
    hidden_dim = 256
    filters = [32,64, 128, 256]
    kernels = [3,3,3,3]

    # Build the encoder
    encoder = model.build_encoder(latent_dim, hidden_dim, filters, kernels, nb_of_bands)
    # Build the decoder
    decoder = model.build_decoder(input_shape, latent_dim, hidden_dim, filters, kernels, conv_activation=None, dense_activation=None)

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
    vae_loaded = Model(input_vae, decoder(z)) 

    if folder == False: 
        vae_loaded.load_weights(path)
    else:
        latest = tf.train.latest_checkpoint(path)
        vae_loaded.load_weights(latest)

    return vae_loaded , encoder, output_encoder


def load_vae_decoder(path,nb_of_bands,folder = False):
    """
    Return the decoder of the VAE located at the path given when the function is called
    """
    batch_size = 100 
    epsilon_std = 1.0
    
    input_shape = (64,64,nb_of_bands)
    latent_dim = 32
    hidden_dim = 256
    filters = [32,64, 128, 256]
    kernels = [3,3,3,3]

    # Build the encoder
    encoder = model.build_encoder(latent_dim, hidden_dim, filters, kernels, nb_of_bands)
    # Build the decoder
    decoder = model.build_decoder(input_shape, latent_dim, hidden_dim, filters, kernels, conv_activation=None, dense_activation=None)

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
    vae_loaded = Model(input_vae, decoder(z)) 

    if folder == False: 
        vae_loaded.load_weights(path)
    else:
        latest = tf.train.latest_checkpoint(path)
        vae_loaded.load_weights(latest)

    return decoder


# DEBLENDER
def load_deblender(path_deblender, path_encoder,nb_of_bands, folder = False):
    """
    Return the loaded deblender located at the path given when the function is called
    """
    folder = folder

    decoder = load_vae_decoder(path_encoder,nb_of_bands,folder = folder)
    decoder.trainable = False

    # Deblender model
    batch_size = 100
    latent_dim = 32
    hidden_dim = 256
    filters = [32,64, 128, 256]
    kernels = [3,3,3,3]
    epsilon_std = 1.0

    # Deblender model
    input_deblender = Input(shape=(64,64,6))
    # Lambda function to compute latent variable z
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    encoder_d = model.build_encoder(latent_dim, hidden_dim, filters, kernels, nb_of_bands)

    output_encoder_d = encoder_d(input_deblender)
    z_d = Lambda(sampling, output_shape=(latent_dim,))(output_encoder_d)

    # Use the encoder of the trained VAE
    # Definition of deblender
    #deblender_loaded = Model(input_deblender, decoder(z_d))
    deblender_loaded, deblender_utils, Dkl = vae_functions.build_vanilla_vae(encoder_d, decoder, full_cov=False, coeff_KL = 0)

    if folder == False: 
        deblender_loaded.load_weights(path_deblender)
    else:
        latest = tf.train.latest_checkpoint(path_deblender)
        deblender_loaded.load_weights(latest)

    return deblender_loaded, encoder_d, Dkl

def load_alpha(path_alpha):
    return np.load(path_alpha+'alpha.npy')



##############   LOAD DATA    ############
def delta_r_min(shift_path):
    """
    Function to compute the delta_r from the shift saved
    """
    shift =np.load(shift_path)
    
    # set lists
    deltas_r= np.zeros((len(shift),4))
    delta_r= np.zeros((len(shift)))
    
    # compute the delta r for each couple of galaxies
    for i in range (4):
        deltas_r[:,i] = np.sqrt(np.square(shift[:,i,0])+np.square(shift[:,i,1]))

    # Take the min of the non zero delta r
    for j in range (len(shift)):
        if (deltas_r[j,:].any() == 0):
            delta_r[j] = 0
        else:
            x = np.where(deltas_r[j] == 0)[0]
            deltas = np.delete(deltas_r[j],x)
            delta_r[j] = np.min(deltas)
    
    return delta_r




######## COMPARISON OF VAE

def compare_vae(vae1, vae2,nb_bands_1,nb_bands_2, input_vae_1, input_vae_2, expected_1, expected_2):
    N = len(input_vae_1)#100
    
    list_galsim_1 = input_vae_1
    list_galsim_2 = input_vae_2
    
    if nb_bands_1 == 1 :
        list_gal_in_1 = expected_1
        list_gal_in_1 = list_gal_in_1.reshape(N,64,64)
        
        list_gal_out_1 = vae1.predict(list_galsim_1, batch_size= 100)
        list_gal_out_1 = list_gal_out_1.reshape(N,64,64)
        
        list_galsim_1 = list_galsim_1.reshape(N,64,64)
    else: 
        list_gal_in_1 = expected_1
        list_gal_in_1 = list_gal_in_1.reshape(N,nb_bands_1,64,64)

        list_gal_out_1 = vae1.predict(list_galsim_1, batch_size= 100)
        list_gal_out_1 = list_gal_out_1.reshape(N,nb_bands_1,64,64)

        list_galsim_1 = list_galsim_1.reshape(N,nb_bands_1,64,64)
        
    if nb_bands_2 == 1 :
        list_gal_in_2 = expected_2
        list_gal_in_2 = list_gal_in_2.reshape(N,64,64)
        
        list_gal_out_2 = vae2.predict(list_galsim_2, batch_size= 100)
        list_gal_out_2 = list_gal_out_2.reshape(N,64,64)
        
        list_galsim_2 = list_galsim_2.reshape(N,64,64)
    
    else: 
        list_gal_in_2 = expected_2
        list_gal_in_2 = list_gal_in_2.reshape(N,nb_bands_2,64,64)

        list_gal_out_2 = vae2.predict(list_galsim_2, batch_size= 100)
        list_gal_out_2 = list_gal_out_2.reshape(N,nb_bands_2,64,64)

        list_galsim_2 = list_galsim_2.reshape(N,nb_bands_2,64,64)
    

    res_galsim_1 = np.empty([N,], dtype='float32')
    res_galsim_2 = np.empty([N,], dtype='float32')

    res_out_1 = np.empty([N,], dtype='float32')
    res_out_2 = np.empty([N,], dtype='float32')
    
    err_count = 0

    # Ajout de la PSF utilisée pour pouvoir faire l'estimation du shear
    PSF = galsim.Moffat(fwhm=0.1, beta=2.5)
    final_epsf_image = PSF.drawImage(scale=0.2)

    for i in range (N):
        try :
            if nb_bands_1 ==1 :
                img_in_1 = galsim.Image(list_gal_in_1[i])
                img_out_1 = galsim.Image(list_gal_out_1[i])
            else:
                img_in_1 = galsim.Image(list_gal_in_1[i][nb_bands_1-4])
                img_out_1 = galsim.Image(list_gal_out_1[i][nb_bands_1-4])
                
            if nb_bands_2 ==1:
                img_in_2 = galsim.Image(list_gal_in_2[i])
                img_out_2 = galsim.Image(list_gal_out_2[i])
            else:
                img_in_2 = galsim.Image(list_gal_in_2[i][nb_bands_2-4])
                img_out_2 = galsim.Image(list_gal_out_2[i][nb_bands_2-4])
            
            res_galsim_1[i] = galsim.hsm.EstimateShear(img_in_1,final_epsf_image).observed_shape.g
            res_out_1[i] = galsim.hsm.EstimateShear(img_out_1,final_epsf_image).observed_shape.g
            res_galsim_2[i] = galsim.hsm.EstimateShear(img_in_2,final_epsf_image).observed_shape.g
            res_out_2[i] = galsim.hsm.EstimateShear(img_out_2,final_epsf_image).observed_shape.g
        except :
            err_count +=1
            print('erreur')
            pass
        continue

    return res_galsim_1, res_out_1, res_galsim_2, res_out_2




def compare_deblender(deb1, deb2,nb_1, nb_2, input_deb_1, input_deb_2, expected):
    ######
    # deb1, deb2 : the 2 deblenders which are being compared
    # nb_1, nb_2 : number of bands in the images processed respectively by deb1 and deb2
    # input_deb_1, input_deb_2 : the noisy blended images to process respectively for deb1 and deb2
    # expected: the noiseless centered galaxy images which are the targerts for the deblender 1
    ######
    # List of noisy blended images
    list_blended_1 = input_deb_1.reshape(len(input_deb_1),64,64,nb_1)
    list_blended_2 = input_deb_2.reshape(len(input_deb_2),64,64,nb_2)
    print(list_blended_1.shape)
    # List of noiseless centered galaxy
    list_simple = expected
    
    # Use deblenders on list of blended images
    list_output_1 = deb1.predict(list_blended_1, batch_size= 100)
    list_output_2 = deb2.predict(list_blended_2, batch_size= 100)
    
    # Reshape the lists so that it can be used for measurement
    list_output_1 = list_output_1.reshape(len(input_deb_1),nb_1,64,64)
    list_output_2 = list_output_2.reshape(len(input_deb_2),nb_2,64,64)
    print(list_output_1.shape)
    list_simple = list_simple.reshape(len(expected),nb_1,64,64)

    # Create empty lists for the futur measurements
    g_in = np.empty([len(expected),], dtype='float32')
    g1_in= np.empty([len(expected),], dtype='float32')
    g2_in= np.empty([len(expected),], dtype='float32')
    
    g_out_1 = np.empty([len(input_deb_1),], dtype='float32')
    g1_out_1= np.empty([len(input_deb_1),], dtype='float32')
    g2_out_1= np.empty([len(input_deb_1),], dtype='float32')

    g_out_2 = np.empty([len(input_deb_2),], dtype='float32')
    g1_out_2= np.empty([len(input_deb_2),], dtype='float32')
    g2_out_2= np.empty([len(input_deb_2),], dtype='float32')

    err_count = 0
    for i in range (len(input_deb_1)):
        try :
            # Add a PSF to be able to do an estimation of the shear
            PSF = galsim.Moffat(fwhm=0.1, beta=2.5)
            final_epsf_image = PSF.drawImage(scale=0.2)

            # Define the images
            img_in = galsim.Image(list_simple[i,nb_1-4,16:48,16:48])
            img_out_1 = galsim.Image(list_output_1[i,nb_1-4,16:48,16:48])
            img_out_2 = galsim.Image(list_output_2[i,nb_2-4,16:48,16:48])
            # Measurements 
            ## for the input image
            g_in[i] = galsim.hsm.EstimateShear(img_in,final_epsf_image).observed_shape.g
            g1_in[i] = galsim.hsm.EstimateShear(img_in,final_epsf_image).observed_shape.g1
            g2_in[i] = galsim.hsm.EstimateShear(img_in,final_epsf_image).observed_shape.g2

            ## for the output image for the deblender deb1
            g_out_1[i] = galsim.hsm.EstimateShear(img_out_1,final_epsf_image).observed_shape.g
            g1_out_1[i] = galsim.hsm.EstimateShear(img_out_1,final_epsf_image).observed_shape.g1
            g2_out_1[i] = galsim.hsm.EstimateShear(img_out_1,final_epsf_image).observed_shape.g2
            ## for the output image for the deblender deb2
            g_out_2[i] = galsim.hsm.EstimateShear(img_out_2,final_epsf_image).observed_shape.g
            g1_out_2[i] = galsim.hsm.EstimateShear(img_out_2,final_epsf_image).observed_shape.g1
            g2_out_2[i] = galsim.hsm.EstimateShear(img_out_2,final_epsf_image).observed_shape.g2
        except :
            err_count +=1
            print('erreur')
            pass
        continue
    print(err_count)
    
    return g_in, g1_in, g2_in, g_out_1, g1_out_1, g2_out_1, g_out_2, g1_out_2, g2_out_2