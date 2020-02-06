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

from . import model, vae_functions, plot

I_lsst = np.array([255.2383, 2048.9297, 3616.1757, 4441.0576, 4432.7823, 2864.145])
I_euclid = np.array([5925.8097, 3883.7892, 1974.2465,  413.3895])
beta = 2.5


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

############# Normalize data ############# 
def norm(x, bands,n_years, channel_last=False, inplace=True):
    I = np.concatenate([I_euclid,n_years*I_lsst])
    if not inplace:
        y = np.copy(x)
    else:
        y = x
    if channel_last:
        assert y.shape[-1] == len(bands)
        for i in range (len(y)):
            for ib, b in enumerate(bands):
                y[i,:,:,ib] = np.tanh(np.arcsinh(y[i,:,:,ib]/(I[b]/beta)))
    else:
        assert y.shape[1] == len(bands)
        for i in range (len(y)):
            for ib, b in enumerate(bands):
                y[i,ib] = np.tanh(np.arcsinh(y[i,ib]/(I[b]/beta)))
    return y

def denorm(x, bands,n_years, channel_last=False, inplace=True):
    I = np.concatenate([I_euclid,n_years*I_lsst])
    if not inplace:
        y = np.copy(x)
    else:
        y = x
    if channel_last:
        assert y.shape[-1] == len(bands)
        for i in range (len(y)):
            for ib, b in enumerate(bands):
                y[i,:,:,ib] = np.sinh(np.arctanh(y[i,:,:,ib]))*(I[b]/beta)
    else:
        assert y.shape[1] == len(bands)
        for i in range (len(y)):
            for ib, b in enumerate(bands):
                y[i,ib] = np.sinh(np.arctanh(x[i,ib]))*(I[b]/beta)
    return y

# Here we do the detection in R band of LSST
def SNR_peak(gal_noiseless, sky_background_pixel, band=6, snr_min=2):
    # Make sure images have shape [nband, nx, ny] and sky_background_pixel has length nband
    assert len(sky_background_pixel) == gal_noiseless.shape[0]
    assert gal_noiseless.shape[1] == gal_noiseless.shape[2]
    
    snr = np.max(gal_noiseless[band])/sky_background_pixel[band]
    return (snr>snr_min), snr


def SNR(gal_noiseless, sky_background_pixel, band=6, snr_min=5):
    # Make sure images have shape [nband, nx, ny] and sky_background_pixel has length nband
    assert len(sky_background_pixel) == gal_noiseless.shape[0]
    assert gal_noiseless.shape[1] == gal_noiseless.shape[2]
    
    signal = gal_noiseless[band]
    variance = signal+sky_background_pixel[band] # for a Poisson process, variance=mean
    snr = np.sqrt(np.sum(signal**2/variance))
    return (snr>snr_min), snr


############# COMPUTE BLENDEDNESS #############
def compute_blendedness_single(image1, image2):
    """
    Return blendedness computed with two images of single galaxy created with GalSim

    Parameters
    ----------
    img, img_new : GalSim images convolved with its PSF and drawn in its filter
    """
    if isinstance(image1, galsim.image.Image):
        im1 = np.array(image1.array.data)
        im2 = np.array(image2.array.data)
    else:
        im1 = image1
        im2 = image2
    # print(image,image_new)
    blnd = np.sum(im1*im2)/np.sqrt(np.sum(im1**2)*np.sum(im2**2))
    return blnd

def compute_blendedness_total(img_central, img_others):
    if isinstance(img_central, galsim.image.Image):
        ic = np.array(img_central.array.data)
        io = np.array(img_others.array.data)
    else :
        ic = img_central
        io = img_others
    itot = ic + io
    # print(image,image_new)
    # blnd = np.sum(ic*io)/np.sum(io**2)
    # blnd = 1. - compute_blendedness_single(itot,io)
    blnd = 1. - np.sum(ic*ic)/np.sum(itot*ic)
    return blnd

def compute_blendedness_aperture(img_central, img_others, radius):
    if isinstance(img_central, galsim.image.Image):
        ic = np.array(img_central.array.data)
        io = np.array(img_others.array.data)
    else :
        ic = img_central
        io = img_others
    h, w = ic.shape
    mask = plot.createCircularMask(h, w, center=None, radius=radius)
    flux_central = np.sum(ic*mask.astype(float))
    flux_others = np.sum(io*mask.astype(float))
    return flux_others / (flux_central+flux_others)

############ DELTA_R and DELTA_MAG COMPUTATION FOR MOST BLENDED GALAXY WITH THE CENTERED ONE ##########
def compute_deltas_for_most_blended(shift,mag,blendedness):#(shift_path, mag_path):
    #mag =np.load(mag_path)
    #shift =np.load(shift_path)

    # Create an array of minimum magnitude and maximum blendedness for each image
    mag_min = np.zeros(len(mag))
    blend_max = np.zeros(len(blendedness))
    for k in range (len(mag)):
        mag_min[k] = np.min(mag[k])
        if (len(blendedness[k])>=1):
            blend_max[k] = np.max(blendedness[k])
        else:
            blend_max[k] = 0

    # set lists
    deltas_r= np.zeros((len(shift),3))
    delta_r= np.zeros((len(shift)))
    delta_mag = np.zeros((len(shift)))
    deltas_mag= np.zeros((len(shift),4))

    for i in range (len(shift)):
        for j in range (len(shift[i])):
            deltas_r[i][j] = np.sqrt(np.square(shift[i][j][0])+np.square(shift[i][j][1]))
        for j in range (len(mag[i])):
            deltas_mag [i][j] = mag[i][j] - mag_min[i]
            
    # Create a deltas_mag liste without all zeros: place of the centered galaxy when generated
    deltas_mag_3= np.zeros((len(deltas_mag),3))
    counter = 0
    for k in range (len(deltas_mag)):
        No_zero = True
        for l in range (len(deltas_mag[k])):
            if deltas_mag[k][l] == 0 and No_zero:
                counter +=1
                No_zero = False
            elif No_zero == False:
                deltas_mag_3[k][l-1] = deltas_mag[k][l]
            else:
                deltas_mag_3[k][l] = deltas_mag[k][l]
    
    # Return delta_mag and delta_r for most blended galaxies
    for i in range (len(blendedness)):
        for k in range (len(blendedness[i])):
            if blendedness[i][k] == blend_max[i]:
                delta_mag[i] = deltas_mag_3[i,k]
                delta_r[i ]=  deltas_r[i,k]

    return delta_r, delta_mag, blend_max

############ DELTA_R and DELTA_MAG COMPUTATION FOR DELTA_R MIN ##########
def delta_min(shift,mag):#(shift_path, mag_path):
    #mag =np.load(mag_path)
    #shift =np.load(shift_path)

    # Create an array of minimum magnitude for each image
    mag_min = np.zeros(len(mag))
    for k in range (len(mag)):
        mag_min[k] = np.min(mag[k])

    # set lists
    deltas_r= np.zeros((len(shift),3))
    delta_r= np.zeros((len(shift)))
    delta_mag = np.zeros((len(shift)))
    deltas_mag= np.zeros((len(shift),4))

    for i in range (len(shift)):
        for j in range (len(shift[i])):
            deltas_r[i][j] = np.sqrt(np.square(shift[i][j][0])+np.square(shift[i][j][1]))
        for j in range (len(mag[i])):
            deltas_mag [i][j] = mag[i][j] - mag_min[i]
            
    # Create a deltas_mag liste without all zeros: place of the centered galaxy when generated
    deltas_mag_3= np.zeros((len(deltas_mag),3))
    counter = 0
    for k in range (len(deltas_mag)):
        No_zero = True
        for l in range (len(deltas_mag[k])):
            if deltas_mag[k][l] == 0 and No_zero:
                counter +=1
                No_zero = False
            elif No_zero == False:
                deltas_mag_3[k][l-1] = deltas_mag[k][l]
            else:
                deltas_mag_3[k][l] = deltas_mag[k][l]
                    
    # Take the min of the non zero delta r
    c = 0
    for j in range (len(shift)):
        # If all the deta_r are equals to 0 (there is only on galaxy on the image) then write 0
        if (deltas_r[j,:].any() == 0):
            delta_r[j] = 0
            delta_mag[j] = 0
            c+=1
        else:
            x = np.where(deltas_r[j] == 0)[0]
            deltas = np.delete(deltas_r[j],x)
            delta_r[j] = np.min(deltas)
            y = np.where(deltas == np.min(deltas))[0]
            delta_mag[j] = deltas_mag_3[j,y]
        
    return delta_r, delta_mag

############# LOAD MODEL ##################
def load_vae_conv(path,nb_of_bands,folder = False):
    """
    Return the loaded VAE located at the path given when the function is called
    """        
    latent_dim = 32
    
    # Build the encoder and decoder
    encoder, decoder = model.vae_model(latent_dim, nb_of_bands)

    #### Build the model
    vae_loaded, vae_utils,  Dkl = vae_functions.build_vanilla_vae(encoder, decoder, full_cov=False, coeff_KL = 0)

    if folder == False: 
        vae_loaded.load_weights(path)
    else:
        latest = tf.train.latest_checkpoint(path)
        vae_loaded.load_weights(latest)

    return vae_loaded, vae_utils, encoder, Dkl

def load_vae_conv_2(path,nb_of_bands,folder = False):
    """
    Return the loaded VAE located at the path given when the function is called
    """        
    latent_dim = 32
    
    # Build the encoder and decoder
    encoder, decoder = model.vae_model_2(latent_dim, nb_of_bands)

    #### Build the model
    vae_loaded, vae_utils,  Dkl = vae_functions.build_vanilla_vae(encoder, decoder, full_cov=False, coeff_KL = 0)

    if folder == False: 
        vae_loaded.load_weights(path)
    else:
        latest = tf.train.latest_checkpoint(path)
        vae_loaded.load_weights(latest)

    return vae_loaded, vae_utils, encoder, Dkl

def load_vae_full(path, nb_of_bands, folder=False):
    """
    Return the loaded VAE located at the path given when the function is called
    """        
    latent_dim = 32
    
    # Build the encoder and decoder
    encoder, decoder = model.vae_model(latent_dim, nb_of_bands)

    #### Build the model
    vae_loaded, vae_utils,  Dkl = vae_functions.build_vanilla_vae(encoder, decoder, full_cov=False, coeff_KL = 0)

    if folder == False: 
        vae_loaded.load_weights(path)
    else:
        latest = tf.train.latest_checkpoint(path)
        vae_loaded.load_weights(latest)

    return vae_loaded, vae_utils, encoder, decoder, Dkl

def load_vae_decoder(path,nb_of_bands,folder = False):
    """
    Return the decoder of the VAE located at the path given when the function is called
    """
    latent_dim = 32
    
    # Build the encoder and decoder
    encoder, decoder = model.vae_model(latent_dim, nb_of_bands)

    #### Build the model
    vae_loaded, vae_utils,  Dkl = vae_functions.build_vanilla_vae(encoder, decoder, full_cov=False, coeff_KL = 0)

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
    deblender_loaded, deblender_utils, Dkl = vae_functions.build_vanilla_vae(encoder_d, decoder, full_cov=False, coeff_KL = 0)

    if folder == False: 
        deblender_loaded.load_weights(path_deblender)
    else:
        latest = tf.train.latest_checkpoint(path_deblender)
        deblender_loaded.load_weights(latest)

    return deblender_loaded, deblender_utils, encoder_d, Dkl

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


##############   MULTIPROCESSING    ############
import multiprocessing
import time
from tqdm import tqdm, trange

def apply_ntimes(func, n, args, verbose=True, timeout=None):
    """
    Applies `n` times the function `func` on `args` (useful if, eg, `func` is partly random).
    Parameters
    ----------
    func : function
        func must be pickable, see https://docs.python.org/2/library/pickle.html#what-can-be-pickled-and-unpickled .
    n : int
    args : any
    timeout : int or float
        If given, the computation is cancelled if it hasn't returned a result before `timeout` seconds.
    Returns
    -------
    type
        Result of the computation of func(iter).
    """
    pool = multiprocessing.Pool()

    multiple_results = [pool.apply_async(func, args) for _ in range(n)]

    pool.close()
    
    return [res.get(timeout) for res in tqdm(multiple_results, desc='# castor.parallel.apply_ntimes', disable = True)]




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
