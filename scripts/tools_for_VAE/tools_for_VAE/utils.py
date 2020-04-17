# Import packages
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
import pathlib
from pathlib import Path

from . import model, vae_functions, plot



def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

############## NORMALIZATION OF IMAGES
# Value used for normalization
beta = 2.5

def norm(x, bands, path , channel_last=False, inplace=True):
    '''
    Return image x normalized

    Parameters:
    -----------
    x: image to normalize
    bands: filter number
    path: path to the normalization constants
    channel_last: is the channels (filters) in last in the array shape
    inplace: boolean: change the value of array itself
    '''
    full_path = pathlib.PurePath(path)
    isolated_or_blended = full_path.parts[6][0:len(full_path.parts[6])-9]

    test_dir = str(Path(path).parents[0])+'/test/'
    I = np.load(test_dir+'galaxies_'+isolated_or_blended+'_20191024_0_I_norm.npy', mmap_mode = 'c')
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

def denorm(x, bands ,path , channel_last=False, inplace=True):
    '''
    Return image x denormalized

    Parameters:
    ----------
    x: image to denormalize
    bands: filter number
    path: path to the normalization constants
    channel_last: is the channels (filters) in last in the array shape
    inplace: boolean: change the value of array itself
    '''
    full_path = pathlib.PurePath(path)
    isolated_or_blended = full_path.parts[6][0:len(full_path.parts[6])-9]
    #print(isolated_or_blended)
    test_dir = str(Path(path).parents[0])+'/test/'
    I = np.load(test_dir+'galaxies_'+isolated_or_blended+'_20191024_0_I_norm.npy', mmap_mode = 'c')#I = np.concatenate([I_euclid,n_years*I_lsst])
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



############### SNR COMPUTATION IN R-BAND FILTER

def SNR_peak(gal_noiseless, sky_background_pixel, band=6, snr_min=2):
    '''
    Return SNR computed with brightest peak value

    Parameters:
    ---------
    gal_noiseless: noiseless image of the isolated galaxy
    sky_background_pixel: sky background level per pixel
    band: filter number of r-band filter
    snr_min: minimum snr to keep the image
    '''
    # Make sure images have shape [nband, nx, ny] and sky_background_pixel has length nband
    assert len(sky_background_pixel) == gal_noiseless.shape[0]
    assert gal_noiseless.shape[1] == gal_noiseless.shape[2]
    
    snr = np.max(gal_noiseless[band])/sky_background_pixel[band]
    return (snr>snr_min), snr


def SNR(gal_noiseless, sky_background_pixel, band=6, snr_min=5):
    '''
    Return SNR

    Parameters:
    ---------
    gal_noiseless: noiseless image of the isolated galaxy
    sky_background_pixel: sky background level per pixel
    band: filter number of r-band filter
    snr_min: minimum snr to keep the image
    '''
    # Make sure images have shape [nband, nx, ny] and sky_background_pixel has length nband
    assert len(sky_background_pixel) == gal_noiseless.shape[0]
    assert gal_noiseless.shape[1] == gal_noiseless.shape[2]
    
    signal = gal_noiseless[band]
    variance = signal+sky_background_pixel[band] # for a Poisson process, variance=mean
    snr = np.sqrt(np.sum(signal**2/variance))
    return (snr>snr_min), snr




################## COMPUTE BLENDEDNESS 

def compute_blendedness_single(image1, image2):
    """
    Return blendedness computed with two images of single galaxy created with GalSim

    Parameters
    ----------
    image1, image2: GalSim images convolved with PSF
    """
    if isinstance(image1, galsim.image.Image):
        im1 = np.array(image1.array.data)
        im2 = np.array(image2.array.data)
    else:
        im1 = image1
        im2 = image2

    blnd = np.sum(im1*im2)/np.sqrt(np.sum(im1**2)*np.sum(im2**2))
    return blnd

def compute_blendedness_total(img_central, img_others):
    """
    Return blendedness computed with image of target galaxy and all neighbour galaxies

    Parameters
    ----------
    img_central, img_others: GalSim images convolved with PSF
    """
    if isinstance(img_central, galsim.image.Image):
        ic = np.array(img_central.array.data)
        io = np.array(img_others.array.data)
    else :
        ic = img_central
        io = img_others
    itot = ic + io
    blnd = 1. - np.sum(ic*ic)/np.sum(itot*ic)
    return blnd

def compute_blendedness_aperture(img_central, img_others, radius):
    """
    Return blendedness computed with image of target galaxy and all neighbour galaxies on a circle of radius "radius"

    Parameters
    ----------
    img_central, img_others: GalSim images convolved with PSF
    radius: radius of the circle used to compute blend rate
    """
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



################ LOAD MODEL FUNCTIONS

def load_vae_conv(path,nb_of_bands,folder = False):
    """
    Return the loaded VAE, outputs for plotting evlution of training, the encoder and the Kullback-Leibler divergence 

    Parameters:
    ----------
    path: path to saved weights
    nb_of_bands: number of filters to use
    folder: boolean, change the loading function
    """        
    latent_dim = 32
    
    # Build the encoder and decoder
    encoder, decoder = model.vae_model(latent_dim, nb_of_bands)

    # Build the model
    vae_loaded, vae_utils,  Dkl = vae_functions.build_vanilla_vae(encoder, decoder, full_cov=False, coeff_KL = 0)

    if folder == False: 
        vae_loaded.load_weights(path)
    else:
        latest = tf.train.latest_checkpoint(path)
        vae_loaded.load_weights(latest)

    return vae_loaded, vae_utils, encoder, Dkl


def load_vae_full(path, nb_of_bands, folder=False):
    """
    Return the loaded VAE, outputs for plotting evlution of training, the encoder, the decoder and the Kullback-Leibler divergence 

    Parameters:
    ----------
    path: path to saved weights
    nb_of_bands: number of filters to use
    folder: boolean, change the loading function
    """        
    latent_dim = 32
    
    # Build the encoder and decoder
    encoder, decoder = model.vae_model(latent_dim, nb_of_bands)

    # Build the model
    vae_loaded, vae_utils,  Dkl = vae_functions.build_vanilla_vae(encoder, decoder, full_cov=False, coeff_KL = 0)

    if folder == False: 
        vae_loaded.load_weights(path)
    else:
        print(path)
        latest = tf.train.latest_checkpoint(path)
        vae_loaded.load_weights(latest)

    return vae_loaded, vae_utils, encoder, decoder, Dkl


def load_alpha(path_alpha):
    return np.load(path_alpha+'alpha.npy')



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

