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
import pandas as pd

from tqdm.auto import trange

from . import utils, plot


def processing(network, data_dir,root,test_sample,bands,r_band,im_size,batch_size, psf, pix_scale, cut_mag):
    """
    Returns measured magnitudes and ellipticities on a galaxy

    Parameters:
    ---------
    network: network to test
    data_dir: directory to load data
    root: root of the files name
    test_sample: file name
    bands: band-pass filters used for this VAE
    r_band: R band-pass filter number in the filters list.
    im_size: size of the stamp for the creation of images
    batch_size: size of batchs for training
    psf: PSF to deconvolve output images
    pix_scale: scale of pixel
    cut_mag: magnitude cut
    """
    
    psf_image = psf.drawImage(nx=im_size, ny=im_size, scale=pix_scale)

    ellipticities = []
    e_obs = []
    e = []
    indices = []
    flux_in = []
    flux_out= []

    # Load data saved during sample generation
    dfs = []
    dfs.append(pd.read_csv(os.path.join(data_dir, root+'_0_data.csv')))
    df = dfs[0]
    #df = pd.DataFrame()

    # Load test sample
    input_sample_len = len(np.load(test_sample, mmap_mode = 'c'))

    for j in trange(int(input_sample_len/batch_size)):
        # Resize and reshape images
        input_images = np.load(test_sample, mmap_mode = 'c')
        input_noisy = utils.norm(input_images[j*batch_size:(j+1)*batch_size,1,bands].transpose([0,2,3,1]), bands,data_dir, channel_last = True, inplace=False)
        input_noiseless = input_images[j*batch_size:(j+1)*batch_size,0,bands].transpose([0,2,3,1])

        # Random flip of images to do data augmentation
        for k in range (1):
            if k == 1: 
                input_noiseless = np.flip(input_noiseless, axis=-2)
                input_noisy = np.flip(input_noisy, axis=-2)
            elif k == 2 : 
                input_noiseless = np.swapaxes(input_noiseless, -2, -3)
                input_noisy = np.swapaxes(input_noisy, -2, -3)
            elif k == 3:
                input_noiseless = np.flip(input_noiseless, axis=-2)
                input_noisy = np.flip(input_noisy, axis=-2)
            
            # Compute output of the network
            output_vae = utils.denorm(network.predict(input_noisy), bands,data_dir, channel_last = True, inplace=False)
            
            # Ellipticity measurement
            for i in range (len(output_vae)):
                try: 
                    gal_image = galsim.Image(input_noiseless[i][:,:,r_band])
                    gal_image.scale = pix_scale
                    shear_est = 'KSB'
                    res_in = galsim.hsm.EstimateShear(gal_image, psf_image, shear_est=shear_est, strict=True)
                    if shear_est != 'KSB':
                        e_in = [res_in.corrected_e1, res_in.corrected_e2]
                    else:
                        e_in = [res_in.corrected_g1, res_in.corrected_g2]
                        e_in_obs = [res_in.observed_shape.e]

                    gal_image_out = galsim.Image(output_vae[i][:,:,r_band])
                    gal_image_out.scale = pix_scale

                    res_out = galsim.hsm.EstimateShear(gal_image_out, psf_image, shear_est=shear_est, strict=True)
                    if shear_est != 'KSB':
                        e_out = [res_out.corrected_e1, res_out.corrected_e2]
                    else:
                        e_out = [res_out.corrected_g1, res_out.corrected_g2]
                        e_out_obs = [res_out.observed_shape.e]

                except :
                        print('error for galaxy '+str(j*400+k*100+i))
                        e_in = [np.nan, np.nan]
                        e_out_obs = [np.nan]
                        e_out = [np.nan, np.nan]
                
                e_obs.append([e_in_obs,e_out_obs])
                ellipticities.append([e_in, e_out])
                    
                # Measurement of fluxes
                flux_in.append(np.sum(input_noiseless[i][:,:,r_band]))
                flux_out.append(np.sum(output_vae[i][:,:,r_band]))
    
    # Save results into dataframe
    print(np.array(ellipticities).shape)
    ellipticities = np.array(ellipticities)
    df['e1_in'] = ellipticities[:,0,0]
    df['e1_out'] = ellipticities[:,1,0]
    df['e1_error'] = df['e1_out'] - df['e1_in']

    df['e2_in'] = ellipticities[:,0,1]
    df['e2_out'] = ellipticities[:,1,1]
    df['e2_error'] = df['e2_out'] - df['e2_in']

    df['e_in'] = np.sqrt(df['e1_in']**2+df['e2_in']**2)
    df['e_out'] = np.sqrt(df['e1_out']**2+df['e2_out']**2)
    df['e_error'] = df['e_out'] - df['e_in']

    e_obs = np.array(e_obs)
    df['e_in_obs'] = e_obs[:,0]
    df['e_out_obs'] = e_obs[:,1]
    df['e_obs_error'] = df['e_out_obs'] - df['e_in_obs']

    df['flux_in'] = flux_in
    df['flux_out']= flux_out

    flux_for_cst = []
    for z in range (len(input_images)):
        flux_for_cst.append(np.sum(input_images[z,0,6,:,:]))
    cst = cut_mag - (np.max(-2.5*np.log10(flux_for_cst)))
    df['mag_in'] = -2.5*np.log10(flux_in)+cst
    df['mag_out']= -2.5*np.log10(flux_out)+cst
    df['delta_mag'] = df['mag_out'] - df['mag_in']

    return df

