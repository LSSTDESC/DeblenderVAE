# Import packages

import numpy as np
import matplotlib.pyplot as plt
# import keras
import sys
import os
import logging
import galsim
import cmath as cm
import math
import random
import scipy
from scipy.stats import norm
from astropy.io import fits

# Import all functions AND VARIABLES from cosmos_generation.py
from cosmos_generation import *

center_brightest = True

def get_scale_radius(gal):
    """
    Return the scale radius of the created galaxy
    
    Parameter:
    ---------
    gal: galaxy from which the scale radius is needed
    """
    try:
        return gal.obj_list[1].original.scale_radius
    except:
        return gal.original.scale_radius


shift_method='uniform'
#shift_method='lognorm_rad'
#shift_method='annulus'

def shift_gal(gal, gal_to_add, method='uniform'):
    """
    Return galaxy shifted according to the chosen shifting method
    
    Parameters:
    ----------
    gal: galaxy to shift (GalSim object)
    method: method to use for shifting
    """
    scale_radius = get_scale_radius(gal)
    if method == 'uniform':
        shift_x = np.random.uniform(-1,1)
        shift_y = np.random.uniform(-1,1)
    elif method == 'lognorm_rad':
        sample_x = np.random.lognormal(mean=0.2*scale_radius,sigma=1*scale_radius,size=None)
        shift_x = np.random.choice((sample_x, -sample_x), 1)[0]
        sample_y = np.random.lognormal(mean=0.2*scale_radius,sigma=1*scale_radius,size=None)
        shift_y = np.random.choice((sample_y, -sample_y), 1)[0]
    elif method == 'annulus':
        min_r = 0.1
        max_r = 2.
        r = np.sqrt(np.random.uniform(min_r**2, max_r**2))
        theta = np.random.uniform(0., 2*np.pi)
        shift_x = r * np.cos(theta)
        shift_y = r * np.sin(theta)
    else:
        raise ValueError
    return gal_to_add.shift((shift_x,shift_y)), (shift_x,shift_y)

# Generation function
def blend_generator(cosmos_cat, training_or_test, used_idx=None, nmax_blend=4, max_try=3, mag_cut = 100.): 
    """
    Return numpy arrays: noiseless and noisy image of single galaxy and of blended galaxies

    Parameters:
    ----------
    cosmos_cat: COSMOS catalog
    nb_blended_gal: number of galaxies to add to the centered one on the blended image
    training_or_test: choice for generating a training or testing dataset
    """

    counter = 0
    np.random.seed() # important for multiprocessing !

    while counter < max_try:
        try:
            ############## GENERATION OF THE GALAXIES ##################
            ud = galsim.UniformDeviate()

            nb_blended_gal = np.random.randint(nmax_blend)+1
            
            data = {}

            # Pick galaxies in catalog
            galaxies = []
            mag=[]
            mag_ir=[]
            for j in range (nb_blended_gal):
                if used_idx is not None:
                    idx = np.random.choice(used_idx)
                else:
                    idx = np.random.randint(cosmos_cat.nobject)
                gal = cosmos_cat.makeGalaxy(idx, gal_type='parametric', chromatic=True, noise_pad_size=0)
                if gal.calculateMagnitude(filters['r'].withZeropoint(28.13)) > mag_cut:
                    #print(gal.calculateMagnitude(filters['r'].withZeropoint(28.13)), mag_cut)
                    return blend_generator(cosmos_cat, training_or_test, used_idx, nmax_blend, max_try, 28.)
                    #raise RuntimeError
                #print(gal.calculateMagnitude(filters['r'].withZeropoint(28.13)))
                gal = gal.rotate(ud() * 360. * galsim.degrees)
                galaxies.append(gal)
                mag.append(gal.calculateMagnitude(filters['r'].withZeropoint(28.13)))
                mag_ir.append(gal.calculateMagnitude(filters['H'].withZeropoint(24.92-22.35*coeff_noise_h)))

            # Optionally, find the brightest and put it first in the list
            if center_brightest:
                #print(len(mag))
                _idx = np.argmin(mag)
                galaxies.insert(0, galaxies.pop(_idx))
                mag.insert(0, mag.pop(_idx))
                mag_ir.insert(0, mag_ir.pop(_idx))

            # Draw shifts for other galaxies (shift has the same shape to make it simpler to save as numpy array)
            shift = np.zeros((nmax_blend,2))
            for j,gal in enumerate(galaxies[1:]):
                galaxies[j+1], shift[j+1] = shift_gal(galaxies[0], gal, method=shift_method)
            
            # Compute distances to find the closest blended galaxy
            if nb_blended_gal>1:
                distances = [shift[j][0]**2+shift[j][1]**2 for j in range(1,nb_blended_gal)]
                idx_closest = np.argmin(distances)+1
            else:
                idx_closest = 0
            

            # Draw images
            galaxy_noiseless = np.zeros((10,max_stamp_size,max_stamp_size))
            blend_noisy = np.zeros((10,max_stamp_size,max_stamp_size))
            for i, filter_name in enumerate(filter_names_all):
                galaxies_psf = [galsim.Convolve([gal*coeff_exp[i], PSF[i]]) for gal in galaxies]
                blend_img = galsim.ImageF(max_stamp_size, max_stamp_size, scale=pixel_scale[i])

                # draw all galaxy images
                images = []
                for j, gal in enumerate(galaxies_psf):
                    temp_img = galsim.ImageF(max_stamp_size, max_stamp_size, scale=pixel_scale[i])
                    gal.drawImage(filters[filter_name], image=temp_img)
                    images.append(temp_img)
                    if j>0: # at first, add only other galaxies
                        blend_img += temp_img

                # we can already save the central galaxy
                galaxy_noiseless[i] = images[0].array.data

                # get data for the test sample (LSST stuff)
                if training_or_test == 'test' and filter_name == 'r':
                    # need psf to compute ellipticities
                    psf_image = PSF[i].drawImage(nx=max_stamp_size, ny=max_stamp_size, scale=pixel_scale[i])
                    data['redshift'], data['moment_sigma'], data['e1'], data['e2'], data['mag']= get_data(galaxies[0], images[0], psf_image)
                    data['closest_redshift'], data['closest_moment_sigma'], data['closest_e1'], data['closest_e2'], data['closest_mag'] = get_data(galaxies[idx_closest], images[idx_closest], psf_image)
                    #data['ellipticity_weights'] = 
                    
                    if nb_blended_gal > 1:
                        data['blendedness_total_lsst'] = utils.compute_blendedness_total(images[0], blend_img)
                        data['blendedness_closest_lsst'] = utils.compute_blendedness_single(images[0], images[idx_closest])
                        data['blendedness_aperture_lsst'] = utils.compute_blendedness_aperture(images[0], blend_img, data['moment_sigma']/pixel_scale_lsst)
                    else:
                        data['blendedness_total_lsst'] = np.nan
                        data['blendedness_closest_lsst'] = np.nan
                        data['blendedness_aperture_lsst'] = np.nan
                # get data for the test sample (Euclid stuff)
                if training_or_test == 'test' and filter_name == 'V':
                    if nb_blended_gal > 1:
                        data['blendedness_total_euclid'] = utils.compute_blendedness_total(images[0], blend_img)
                        data['blendedness_closest_euclid'] = utils.compute_blendedness_single(images[0], images[idx_closest])
                        #data['blendedness_aperture_euclid'] = utils.compute_blendedness_aperture(images[0], blend_img, data['moment_sigma'],4)
                    else:
                        data['blendedness_total_euclid'] = np.nan
                        data['blendedness_closest_euclid'] = np.nan
                        #data['blendedness_aperture_euclid'] = np.nan
                
                # now add the central galaxy to the full scene
                blend_img += images[0]

                # add noise
                poissonian_noise = galsim.PoissonNoise(rng, sky_level=sky_level_pixel[i])
                blend_img.addNoise(poissonian_noise)
                blend_noisy[i] = blend_img.array.data
        
            break

        except RuntimeError as e:
            print(e)

    # For training/validation, return normalized images only
    if training_or_test in ['training', 'validation']:
        galaxy_noiseless = utils.norm(galaxy_noiseless[None,:], bands=range(10))[0]
        blend_noisy = utils.norm(blend_noisy[None,:], bands=range(10))[0]
        return galaxy_noiseless, blend_noisy

    # For testing, return unormalized images and data
    elif training_or_test == 'test':
        data['nb_blended_gal'] = nb_blended_gal
        data['mag'] = mag[0]
        data['mag_ir'] = mag_ir[0]
        data['closest_mag'] = mag[idx_closest]
        data['closest_mag_ir'] = mag_ir[idx_closest]
        data['closest_x'] = shift[idx_closest][0]
        data['closest_y'] = shift[idx_closest][1]
        data['SNR'] = utils.SNR(galaxy_noiseless, sky_level_pixel, band=6)[1]
        data['SNR_peak'] = utils.SNR_peak(galaxy_noiseless, sky_level_pixel, band=6)[1]

        return galaxy_noiseless, blend_noisy, data, shift
    else:
        raise ValueError