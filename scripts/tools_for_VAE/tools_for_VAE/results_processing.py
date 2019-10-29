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

from . import utils, plot

######### VAE #########

def VAE_processing(vae, generator, bands, r_band, im_size, N, batch_size, psf, pix_scale):
    """
    Returns 

    Paramters:
    ---------
    vae: vae to test
    generator: generator of input images (input of network and target) and parameters for processing results (SNR, scale radius)
    bands: band-pass filters used for this VAE
    r_band: R band-pass filter number in the bands list.
    im_size= size of the stamp for the creation of images
    N: number of batch to test
    batch_size: size of the batches generated by generator
    """

    # see LSST Science Book
    # pix_scale = 0.2 #arcseconds
    # PSF_fwhm = 0.1
    # PSF_beta = 2.5

    # psf = galsim.Moffat(fwhm=PSF_fwhm, beta=PSF_beta)
    psf_image = psf.drawImage(nx=im_size, ny=im_size, scale=pix_scale)

    ellipticities = []
    redshift_R=[]
    e=[]
    SNR = []
    scale_radius = []

    #flux_in = np.zeros((N,batch_size))#([N,batch_size,],dtype='float32')
    #flux_out= np.zeros((N,batch_size))#([N,batch_size,], dtype='float32')
    flux_in = []
    flux_out = []

    for j in range(N):
        input_vae = generator.__getitem__(2)
        output_vae = vae.predict(input_vae[0], batch_size = batch_size)
        #input_noiseless = input_vae[1]
        input_noiseless = utils.denorm(input_vae[1], bands, channel_last = True)
        output_vae = utils.denorm (output_vae, bands, channel_last = True)

        for i in range (len(input_vae[0])):
            try: 
                gal_image = galsim.Image(input_noiseless[i][:,:,r_band])
                gal_image.scale = pix_scale

                gal_image_out = galsim.Image(output_vae[i][:,:,r_band])
                gal_image_out.scale = pix_scale

                # Measurements of shapes
                shear_est = 'REGAUSS'#'KSB'
                
                res = galsim.hsm.EstimateShear(gal_image, psf_image)#, shear_est=shear_est, strict=True)
                if shear_est != 'KSB':
                    e_in = [res.corrected_e1, res.corrected_e2] 
                else:
                    e_in = [res.corrected_g1, res.corrected_g2]
                #e_in = [res.corrected_e1, res.corrected_e2]
                e_beta_in = [res.observed_shape.e, res.observed_shape.beta.rad]

                res_out = galsim.hsm.EstimateShear(gal_image_out, psf_image)#, shear_est=shear_est, strict=True)
                if shear_est != 'KSB':
                    e_out = [res_out.corrected_e1, res_out.corrected_e2] 
                else:
                    e_out = [res_out.corrected_g1, res_out.corrected_g2]
                #e_out = [res_out.corrected_e1, res_out.corrected_e2]
                e_beta_out = [res_out.observed_shape.e, res_out.observed_shape.beta.rad]

                ellipticities.append([e_in, e_out])
                e.append([e_beta_in, e_beta_out])

                # Measurement of fluxes
                mask = plot.createCircularMask(im_size,im_size,None,5)
                masked_img_in_simple = input_noiseless[i][:,:,r_band].copy()
                masked_img_in_simple[~mask] = 0  

                masked_img_out_simple = output_vae[i][:,:,r_band].copy()
                masked_img_out_simple[~mask] = 0

                # Calculate the luminosity by substracting the noise
                #flux_in[j,i] = np.sum(masked_img_in_simple)
                flux_in.append(np.sum(masked_img_in_simple))
                #flux_out[j,i] = np.sum(masked_img_out_simple)
                flux_out.append(np.sum(masked_img_out_simple))
            
            # Save scale radius and SNR
            #scale_radius.append(input_vae[2][i])
            #SNR.append(input_vae[3][i])
    
            except :
                print('erreur')
                pass
            continue

    ellipticities = np.array(ellipticities)
    e_beta = np.array(e)
    scale_radius = np.array(scale_radius)
    SNR = np.array(SNR)

    flux_in = np.array(flux_in)#np.concatenate(flux_in)
    flux_out = np.array(flux_out)#np.concatenate(flux_out)

    return ellipticities, e_beta, flux_in, flux_out, scale_radius, SNR 



######### Deblender #########

def deblender_processing(deblender, generator,bands,r_band,im_size, N, batch_size):
    """
    Returns 

    Paramters:
    ---------
    deblender: deblender to test
    generator: generator of input images (input of network and target) and parameters for processing results (SNR, scale radius)
    bands: band-pass filters used for this VAE
    r_band: R band-pass filter number in the bands list.
    im_size= size of the stamp for the creation of images
    N: number of batch to test
    batch_size: size of the batches generated by generator
    """
    # see LSST Science Book
    pix_scale = 0.2 #arcseconds
    PSF_fwhm = 0.65

    psf = galsim.Kolmogorov(fwhm=PSF_fwhm)
    #psf= galsim.Moffat(fwhm = 0.1, beta = 2.5)
    psf_image = psf.drawImage(nx=im_size, ny=im_size, scale=pix_scale)

    ellipticities = []
    e = []
    indices = []

    flux_in = []#np.empty([N,batch_size,],dtype='float32')
    flux_out= []#np.empty([N,batch_size,], dtype='float32')
    for j in range(N):
        #print(j)
        input_vae = generator.__getitem__(2)
        output_vae = deblender.predict(input_vae[0], batch_size = batch_size)
        output_vae = utils.denorm(output_vae, bands, channel_last = True)
        input_noiseless = utils.denorm(input_vae[1], bands, channel_last = True)

        for i in range (len(input_vae[0])):
            #print(i)
            try: 
                gal_image = galsim.Image(input_noiseless[i][:,:,r_band])
                gal_image.scale = pix_scale

                shear_est = 'KSB'
                res = galsim.hsm.EstimateShear(gal_image, psf_image, shear_est=shear_est, strict=True)
                if shear_est != 'KSB':
                    e_in = [res.corrected_e1, res.corrected_e2] 
                else:
                    e_in = [res.corrected_g1, res.corrected_g2]
                #e_in = [res.corrected_e1, res.corrected_e2]
                e_beta_in = [res.observed_shape.e, res.observed_shape.beta.rad]

                gal_image_out = galsim.Image(output_vae[i][:,:,r_band])
                gal_image_out.scale = pix_scale

                res = galsim.hsm.EstimateShear(gal_image_out, psf_image, shear_est=shear_est, strict=True)
                if shear_est != 'KSB':
                    e_out = [res.corrected_e1, res.corrected_e2] 
                else:
                    e_out = [res.corrected_g1, res.corrected_g2]
                #e_out = [res.corrected_e1, res.corrected_e2]
                e_beta_out = [res.observed_shape.e, res.observed_shape.beta.rad]

                ellipticities.append([e_in, e_out])
                e.append([e_beta_in, e_beta_out])

                    
                # Measurement of fluxes
                #mask = plot.createCircularMask(im_size,im_size,None,5)
                #masked_img_in_simple = input_noiseless[i][:,:,r_band].copy()
                #masked_img_in_simple[~mask] = 0  

                #masked_img_out_simple = output_vae[i][:,:,r_band].copy()
                #masked_img_out_simple[~mask] = 0

                # Calculate the flux
                #flux_in[j,i] = np.sum(input_noiseless[i][:,:,r_band])
                flux_in.append(np.sum(input_noiseless[i][:,:,r_band]))
                #flux_out[j,i] = np.sum(output_vae[i][:,:,r_band])
                flux_out.append(np.sum(output_vae[i][:,:,r_band]))

            except :
                print('error for galaxy '+str(j*100+i))
                #ellipticities.append([np.nan, np.nan])
                #e.append([np.nan, np.nan])
                #flux_in.append(np.nan)
                #flux_out.append(np.nan)
                pass
            continue
        
        indices.append(input_vae[3])

    ellipticities = np.array(ellipticities)
    e_beta = np.array(e)
    indices = np.concatenate(indices)

    flux_in = np.array(flux_in)#np.concatenate(flux_in)
    flux_out = np.array(flux_out)#np.concatenate(flux_out)

    return ellipticities,e_beta, flux_in, flux_out, indices



# def deblender_processing(deblender, generator,bands,r_band,im_size, N, batch_size):
#     """
#     Returns 

#     Paramters:
#     ---------
#     deblender: deblender to test
#     generator: generator of input images (input of network and target) and parameters for processing results (SNR, scale radius)
#     bands: band-pass filters used for this VAE
#     r_band: R band-pass filter number in the bands list.
#     im_size= size of the stamp for the creation of images
#     N: number of batch to test
#     batch_size: size of the batches generated by generator
#     """
#     # see LSST Science Book
#     pix_scale = 0.2 #arcseconds
#     PSF_fwhm = 0.1
#     PSF_beta = 2.5

#     psf = galsim.Moffat(fwhm=PSF_fwhm, beta=PSF_beta)
#     psf_image = psf.drawImage(nx=im_size, ny=im_size, scale=pix_scale)

#     ellipticities = []
#     e = []
#     magnitudes = []
#     deltas_r = []
#     deltas_m = []
#     max_blendedness = []
#     SNR_list = []
#     blend_total = []

#     flux_in = np.empty([N,batch_size,],dtype='float32')
#     flux_out= np.empty([N,batch_size,], dtype='float32')
#     for j in range(N):
#         input_vae = generator.__getitem__(2)
#         output_vae = deblender.predict(input_vae[0], batch_size = batch_size)
#         output_vae = utils.denorm(output_vae, bands, channel_last = True)
#         input_noiseless = utils.denorm(input_vae[1], bands, channel_last = True)

#         for i in range (len(input_vae[0])):
#                 try: 
#                     gal_image = galsim.Image(input_noiseless[i][:,:,r_band])
#                     gal_image.scale = pix_scale

#                     #shear_est = 'KSB'
#                     res = galsim.hsm.EstimateShear(gal_image, psf_image)#, shear_est=shear_est, strict=True)
#                     e_in = [res.corrected_e1, res.corrected_e2]
#                     e_beta_in = [res.observed_shape.e, res.observed_shape.beta.rad]

#                     gal_image_out = galsim.Image(output_vae[i][:,:,r_band])
#                     gal_image_out.scale = pix_scale

#                     res = galsim.hsm.EstimateShear(gal_image_out, psf_image)#, shear_est=shear_est, strict=True)
#                     e_out = [res.corrected_e1, res.corrected_e2]
#                     e_beta_out = [res.observed_shape.e, res.observed_shape.beta.rad]

#                     ellipticities.append([e_in, e_out])
#                     e.append([e_beta_in, e_beta_out])

#                     magnitudes.append(input_vae[2])
                        
#                     # Measurement of fluxes
#                     mask = plot.createCircularMask(im_size,im_size,None,5)
#                     masked_img_in_simple = input_noiseless[i][:,:,r_band].copy()
#                     masked_img_in_simple[~mask] = 0  

#                     masked_img_out_simple = output_vae[i][:,:,r_band].copy()
#                     masked_img_out_simple[~mask] = 0

#                     # Calculate the luminosity by substracting the noise
#                     flux_in[j,i] = np.sum(masked_img_in_simple)
#                     flux_out[j,i] = np.sum(masked_img_out_simple)

#                 except :
#                     print('error for galaxy '+str(j*100+i))
#                     pass
#                 continue
                
#         blend_total.append(input_vae[8])
#         deltas_r.append(input_vae[4])
#         deltas_m.append(input_vae[5])
#         max_blendedness.append(input_vae[6])
#         SNR_list.append(input_vae[10])
        
#     ellipticities = np.array(ellipticities)
#     e_beta = np.array(e)
#     magnitudes = np.array(magnitudes)
#     delta_r_arr = np.array(deltas_r)
#     delta_mag_arr = np.array(deltas_m)
#     max_blendedness_arr = np.array(max_blendedness)
#     blend_total_arr = np.array(blend_total)
#     SNR = np.array(SNR_list)

#     blend_total_arr = np.concatenate(blend_total_arr)
#     delta_r_arr = np.concatenate(delta_r_arr)
#     delta_mag_arr = np.concatenate(delta_mag_arr)
#     max_blendedness_arr = np.concatenate(max_blendedness_arr)
#     SNR = np.concatenate(SNR)
#     flux_in = np.concatenate(flux_in)
#     flux_out = np.concatenate(flux_out)

#     return ellipticities,e_beta, flux_in, flux_out, magnitudes, delta_r_arr, delta_mag_arr, max_blendedness_arr, blend_total_arr, SNR