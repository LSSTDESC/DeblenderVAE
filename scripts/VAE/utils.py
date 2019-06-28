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

from model import build_decoder, vae_model
from vae_functions import build_vanilla_vae

############# LOAD MODEL ##################
def load_vae_conv(path,nb_of_bands,folder = False):
    """
    Return the loaded VAE located at the path given when the function is called
    """
    encoder, decoder = vae_model(nb_of_bands)
    vae_loaded, vae_utils, output_encoder = build_vanilla_vae(encoder, decoder, full_cov=False, coeff_KL = 0)
  
    if folder == False: 
        vae_loaded.load_weights(path)
    else:
        latest = tf.train.latest_checkpoint(path)
        vae_loaded.load_weights(latest)

    return vae_loaded , output_encoder


def load_vae_decoder(path,nb_of_bands,folder = False):
    """
    Return the decoder of the VAE located at the path given when the function is called
    """
    encoder, decoder = vae_model(nb_of_bands)
    vae_loaded, vae_utils, output_encoder = build_vanilla_vae(encoder, decoder, full_cov=False, coeff_KL = 0)

    if folder == False: 
        vae_loaded.load_weights(path)
    else:
        latest = tf.train.latest_checkpoint(path)
        vae_loaded.load_weights(latest)

    return decoder


# DEBLENDER
def load_deblender(path_deblender, path_encoder,nb_of_bands, decoder, folder = False):
    """
    Return the loaded deblender located at the path given when the function is called
    """
    folder = folder

    decoder = load_vae_decoder(path_encoder,nb_of_bands,folder = folder)
    decoder.trainable = False

    # Deblender model
    deb_encoder, deb_decoder = vae_model(nb_of_bands)

    # Use the encoder of the trained VAE
    deblender_loaded, deblender_utils, output_encoder_deb = build_vanilla_vae(deb_encoder, decoder, full_cov=False, coeff_KL = 0)
    
    if folder == False: 
        vae_loaded.load_weights(path_deblender)
    else:
        latest = tf.train.latest_checkpoint(path_deblender)
        vae_loaded.load_weights(latest)

    return deblender_loaded



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


###############    PLOTS     ####################

def plot_rgb_lsst(ugrizy_img, ax=None):
    RGB_img = np.zeros((int(stamp_size/2),int(stamp_size/2),3))
    if ax is None:
        _, ax = plt.subplots(1,1)
    max_img = np.max(ugrizy_img)
    ugrizy_img = ugrizy_img[:,int(stamp_size/4):int(stamp_size*3/4),int(stamp_size/4):int(stamp_size*3/4)].reshape((6,int(stamp_size/2),int(stamp_size/2)))
    RGB_img[:,:,0] = ugrizy_img[1][:,:]
    RGB_img[:,:,1] = ugrizy_img[2][:,:]
    RGB_img[:,:,2] = ugrizy_img[4][:,:]
    ax.imshow(np.clip(RGB_img[:,:,[2,1,0]], a_min=0.0, a_max=None) / max_img)    
    
    
def plot_rgb_lsst_euclid(ugrizy_img, ax=None):
    RGB_img = np.zeros((int(stamp_size/2),int(stamp_size/2),3))
    if ax is None:
        _, ax = plt.subplots(1,1)
    max_img = np.max(ugrizy_img[4:])
    ugrizy_img = ugrizy_img[:,int(stamp_size/4):int(stamp_size*3/4),int(stamp_size/4):int(stamp_size*3/4)].reshape(10,int(stamp_size/2),int(stamp_size/2))
    RGB_img[:,:,0] = ugrizy_img[5][:,:]
    RGB_img[:,:,1] = ugrizy_img[6][:,:]
    RGB_img[:,:,2] = ugrizy_img[8][:,:]
    ax.imshow(np.clip(RGB_img[:,:,[2,1,0]], a_min=0.0, a_max=None) / max_img)    
    
    
def mean_var(x,y,bins):
    """
    Return mean and variance in each bins of the histogram
    """
    n,_ = np.histogram(x,bins=bins, weights=None)
    ny,_ = np.histogram(x,bins=bins, weights=y)
    mean_y = ny/n
    ny2,_ = np.histogram(x,bins=bins, weights=y**2)
    var_y = (ny2/n - mean_y**2)/n
    
    return (mean_y, var_y)



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