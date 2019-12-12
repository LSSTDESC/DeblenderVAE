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

from cosmos_params import *

import photutils
from photutils.centroids import centroid_com

sys.path.insert(0,'../tools_for_VAE/')
from tools_for_VAE import utils

# Coefficients to take the exposure time and the telescope size into account
# coeff_exp_lsst = (15. * (6.68**2)/((2.4**2)*(1.-0.33**2))) * N_exposures_lsst
# coeff_exp_euclid = (1800. * ((1.25)**2 - (0.37)**2)/((2.4**2)*(1.-0.33**2))) * N_exposures_euclid

rng = galsim.BaseDeviate(None)

center_brightest = True
print(n_years)
beta_prime_parameters = (14.022429614276358, 6.922508843325913, -0.0247188726955977, 0.04994196562063914)

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

def get_data(gal, gal_image, psf_image, param_or_real='param'):
    shear_est = 'KSB' #'REGAUSS' for e (default) or 'KSB' for g
    res = galsim.hsm.EstimateShear(gal_image, psf_image, shear_est=shear_est, strict=True)
    if param_or_real == 'param':
        mag = gal.calculateMagnitude(filters['r'].withZeropoint(28.13))
    else:
        mag = np.nan
    if res.error_message == "":
        if shear_est != 'KSB':
            return [gal.SED.redshift, res.moments_sigma, res.corrected_e1, res.corrected_e2, mag] #, res.observed_shape.e1, res.observed_shape.e2]
        else:
            return [gal.SED.redshift, res.moments_sigma, res.corrected_g1, res.corrected_g2, mag] #, res.observed_shape.e1, res.observed_shape.e2]
    else:
        return [gal.SED.redshift, np.nan, np.nan, np.nan, mag]
    # return [gal.SED.redshift, res.moments_sigma, res.observed_shape.e1, res.observed_shape.e2] #this is wrong!


# shift_method='uniform'
#shift_method='lognorm_rad'
# shift_method='annulus'

def shift_gal(gal, method='uniform', shift_x0=0., shift_y0=0., max_dx=0.1, min_r = 0.65/2., max_r = 2.):
    """
    Return galaxy shifted according to the chosen shifting method
    
    Parameters:
    ----------
    gal: galaxy to shift (GalSim object)
    method: method to use for shifting
    """
    # scale_radius = get_scale_radius(gal)
    if method == 'noshift':
        shift_x = 0.
        shift_y = 0.
    elif method == 'uniform':
        shift_x = np.random.uniform(-max_dx,+max_dx)#(-2.5,2.5)  #(-1,1)
        shift_y = np.random.uniform(-max_dx,+max_dx)#(-2.5,2.5)  #(-1,1)
    elif method == 'lognorm_rad':
        raise NotImplementedError
        # sample_x = np.random.lognormal(mean=0.2*scale_radius,sigma=1*scale_radius,size=None)
        # shift_x = np.random.choice((sample_x, -sample_x), 1)[0]
        # sample_y = np.random.lognormal(mean=0.2*scale_radius,sigma=1*scale_radius,size=None)
        # shift_y = np.random.choice((sample_y, -sample_y), 1)[0]
    elif method == 'annulus':
        r = np.sqrt(np.random.uniform(min_r**2, max_r**2))
        theta = np.random.uniform(0., 2*np.pi)
        shift_x = r * np.cos(theta)
        shift_y = r * np.sin(theta)
    elif method == 'uniform+betaprime':
        r = np.clip(scipy.stats.betaprime.rvs(*beta_prime_parameters), 0., 0.6)
        theta = np.random.uniform(0., 2*np.pi)
        shift_x = r * np.cos(theta) + np.random.uniform(-max_dx,+max_dx)
        shift_y = r * np.sin(theta) + np.random.uniform(-max_dx,+max_dx)
    else:
        raise ValueError
    shift_x += shift_x0
    shift_y += shift_y0
    return gal.shift((shift_x,shift_y)), (shift_x,shift_y)


def peak_detection(denormed_img, band, shifts, img_size, npeaks, nb_blended_gal, training_or_test, dist_cut):
    gal = denormed_img
    df_temp = photutils.find_peaks(gal, threshold=5*np.sqrt(sky_level_pixel[band]), npeaks=npeaks, centroid_func=centroid_com)
    if df_temp is not None:
        df_temp['x_peak'] = (df_temp['x_centroid']-((img_size/2.)-0.5))*pixel_scale[band]
        df_temp['y_peak'] = (df_temp['y_centroid']-((img_size/2.)-0.5))*pixel_scale[band]
        df_temp.sort('peak_value', reverse=True)
        # Distances of true centers to brightest peak
        qq = [np.sqrt(float((shifts[j,0]-df_temp['x_peak'][0])**2+ (shifts[j,1]-df_temp['y_peak'][0])**2)) for j in range(nb_blended_gal)]
        idx_closest = np.argmin(qq)
        if nb_blended_gal>1:
            # Distance from peak galaxy to others
            qq_prime = [np.sqrt(float((shifts[idx_closest,0]-shifts[j,0])**2+ (shifts[idx_closest,1]-shifts[j,1])**2)) if j!=idx_closest else np.inf for j in range(nb_blended_gal)]
            idx_closest_to_peak_galaxy = np.argmin(qq_prime)
            if training_or_test != 'test':
                if not np.all(np.array(qq_prime) > dist_cut):
                    print('TRAINING CUT: closest is not central and others are too close')
                    return False
        else:
            idx_closest_to_peak_galaxy = np.nan
        return idx_closest, idx_closest_to_peak_galaxy, df_temp[0]['x_centroid'], df_temp[0]['y_centroid'], df_temp[0]['x_peak'], df_temp[0]['y_peak'], len(df_temp)
    else:
        return False

def draw_images(galaxies_psf, band, img_size, filter_name,sky_level_pixel, real_or_param = 'param'):
    # Create image in r bandpass filter to do the peak detection
    blend_img = galsim.ImageF(img_size, img_size, scale=pixel_scale[band])

    images = []
    
    for j, gal in enumerate(galaxies_psf):
        temp_img = galsim.ImageF(img_size, img_size, scale=pixel_scale[band])
        # Parametric image
        if real_or_param == 'param':
            gal.drawImage(filters[filter_name], image=temp_img)
        # Real image
        elif real_or_param == "real":
            gal.drawImage(image=temp_img)
        images.append(temp_img)
        #if j>0: # at first, add only other galaxies
        blend_img += temp_img
    # add noise
    poissonian_noise = galsim.PoissonNoise(rng, sky_level=sky_level_pixel)
    blend_img.addNoise(poissonian_noise)

    return images, blend_img



def image_generator(cosmos_cat_dir, training_or_test, isolated_or_blended, used_idx=None, nmax_blend=4, max_try=3, mag_cut=28., method_first_shift='noshift', do_peak_detection=True):
    """
    Return numpy arrays: noiseless and noisy image of single galaxy and of blended galaxies
    Parameters:
    ----------
    cosmos_cat: COSMOS catalog
    nb_blended_gal: number of galaxies to add to the centered one on the blended image
    training_or_test: choice for generating a training or testing dataset
    """
    cosmos_cat = galsim.COSMOSCatalog('real_galaxy_catalog_25.2.fits', dir=cosmos_cat_dir)
    counter = 0
    np.random.seed() # important for multiprocessing !
    
    assert training_or_test in ['training', 'validation', 'test']
    #assert isinstance(n_years, int)
    assert isolated_or_blended in ['blended', 'isolated']
    
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
            j = 0
            while j < nb_blended_gal:
            # for j in range (nb_blended_gal):
                if used_idx is not None:
                    idx = np.random.choice(used_idx)
                else:
                    idx = np.random.randint(cosmos_cat.nobject)
                gal = cosmos_cat.makeGalaxy(idx, gal_type='parametric', chromatic=True, noise_pad_size=0)
                _mag_temp = gal.calculateMagnitude(filters['r'].withZeropoint(28.13))
                # Magnitude cut
                if _mag_temp < mag_cut:
                    gal = gal.rotate(ud() * 360. * galsim.degrees)
                    galaxies.append(gal)
                    mag.append(_mag_temp)
                    mag_ir.append(gal.calculateMagnitude(filters['H'].withZeropoint(24.92-22.35*coeff_noise_h)))
                    j += 1

            # Optionally, find the brightest and put it first in the list
            if center_brightest:
                _idx = np.argmin(mag)
                galaxies.insert(0, galaxies.pop(_idx))
                mag.insert(0,mag.pop(_idx))
                mag_ir.insert(0,mag_ir.pop(_idx))

            # Draw shifts for other galaxies (shift has the same shape to make it simpler to save as numpy array)
            shift = np.zeros((nmax_blend,2))
            # Shift centered
            galaxies[0], shift[0] = shift_gal(galaxies[0], method=method_first_shift, max_dx=0.1)
            for j,gal in enumerate(galaxies[1:]):
                galaxies[j+1], shift[j+1] = shift_gal(gal, shift_x0=shift[0,0], shift_y0=shift[0,1], min_r=0.65/2., max_r=1.5, method='annulus')
            
            if nb_blended_gal>1:
                distances = [shift[j][0]**2+shift[j][1]**2 for j in range(1,nb_blended_gal)]
                idx_closest_to_peak_galaxy = np.argmin(distances)+1
            else:
                idx_closest_to_peak_galaxy = 0
            
            galaxy_noiseless = np.zeros((10,max_stamp_size,max_stamp_size))
            blend_noisy = np.zeros((10,max_stamp_size,max_stamp_size))

            if do_peak_detection:
                band = 6
                galaxies_psf = [galsim.Convolve([gal*coeff_exp[band], PSF[band]]) for gal in galaxies]

                images, blend_img = draw_images(galaxies_psf, band, max_stamp_size*2, 'r', sky_level_pixel[band])
                blend_noisy_temp = blend_img.array.data
                # if isolated_or_blended == 'blended':
                peak_detection_output = peak_detection(blend_noisy_temp, band, shift, max_stamp_size*2, 4,nb_blended_gal, training_or_test, dist_cut=0.65/2.)
                if not peak_detection_output:
                    print('No peak detected')
                    raise RuntimeError
                else:
                    idx_closest_to_peak, idx_closest_to_peak_galaxy, center_pix_x, center_pix_y, center_arc_x, center_arc_y, n_peak = peak_detection_output

                # Modify galaxies and shift accordingly
                galaxies = [gal.shift(-center_arc_x, -center_arc_y) for gal in galaxies]
                shift[:nb_blended_gal] -= np.array([center_arc_x, center_arc_y])
            
            # Now draw image in all other bands
            for i, filter_name in enumerate(filter_names_all):
                galaxies_psf = [galsim.Convolve([gal*coeff_exp[i], PSF[i]]) for gal in galaxies]
                #if i != 6:
                images, blend_img = draw_images(galaxies_psf, i, max_stamp_size, filter_name, sky_level_pixel[i])
                if isolated_or_blended == 'isolated' or not do_peak_detection:
                    idx_closest_to_peak = 0
                    n_peak = 1
                galaxy_noiseless[i] = images[idx_closest_to_peak].array.data
                blend_noisy[i] = blend_img.array.data

                # get data for the test sample (LSST stuff)
                if training_or_test == 'test' and filter_name == 'r':
                    # need psf to compute ellipticities
                    psf_image = PSF[i].drawImage(nx=max_stamp_size, ny=max_stamp_size, scale=pixel_scale[i])
                    data['redshift'], data['moment_sigma'], data['e1'], data['e2'], data['mag'] = get_data(galaxies[idx_closest_to_peak], images[idx_closest_to_peak], psf_image)

                    # Compute data and blendedness
                    if nb_blended_gal > 1:
                        data['closest_redshift'], data['closest_moment_sigma'], data['closest_e1'], data['closest_e2'], data['closest_mag'] = get_data(galaxies[idx_closest_to_peak_galaxy], images[idx_closest_to_peak_galaxy], psf_image)
                        img_central = images[idx_closest_to_peak].array
                        img_others = np.zeros_like(img_central)
                        for _h, image in enumerate(images):
                            if _h!=idx_closest_to_peak:
                                img_others += image.array
                        #img_others = np.array([image.array.data for _h, image in enumerate(images) if _h!=idx_closest_to_peak]).sum(axis = 0)
                        img_closest_neighbour =images[idx_closest_to_peak_galaxy].array# np.array(images[idx_closest_to_peak_galaxy].array.data)
                        data['blendedness_total_lsst'] = utils.compute_blendedness_total(img_central, img_others)
                        data['blendedness_closest_lsst'] = utils.compute_blendedness_single(img_central, img_closest_neighbour)
                        data['blendedness_aperture_lsst'] = utils.compute_blendedness_aperture(img_central, img_others, data['moment_sigma'])
                    else:
                        data['closest_redshift'] = np.nan
                        data['closest_moment_sigma'] = np.nan
                        data['closest_e1'] = np.nan
                        data['closest_e2'] = np.nan
                        data['closest_mag'] = np.nan
                        data['blendedness_total_lsst'] = np.nan
                        data['blendedness_closest_lsst'] = np.nan
                        data['blendedness_aperture_lsst'] = np.nan
            break

        except RuntimeError as e:
            print(e)

    # For training/validation, return normalized images only
    if training_or_test in ['training', 'validation']:
        galaxy_noiseless = utils.norm(galaxy_noiseless[None,:], bands=range(10), n_years=n_years)[0]
        blend_noisy = utils.norm(blend_noisy[None,:], bands=range(10), n_years=n_years)[0]
        return galaxy_noiseless, blend_noisy

    # For testing, return unormalized images and data
    elif training_or_test == 'test':
        data['nb_blended_gal'] = nb_blended_gal
        data['mag'] = mag[0]
        data['mag_ir'] = mag_ir[0]
        if nb_blended_gal>1:
            data['closest_mag'] = mag[idx_closest_to_peak_galaxy]
            data['closest_mag_ir'] = mag_ir[idx_closest_to_peak_galaxy]
            data['closest_x'] = shift[idx_closest_to_peak_galaxy][0]
            data['closest_y'] = shift[idx_closest_to_peak_galaxy][1]
        else:
            data['closest_mag'] = np.nan
            data['closest_mag_ir'] = np.nan
            data['closest_x'] = np.nan
            data['closest_y'] = np.nan
        data['idx_closest_to_peak'] = idx_closest_to_peak
        data['n_peak_detected'] = n_peak
        data['SNR'] = utils.SNR(galaxy_noiseless, sky_level_pixel, band=6)[1]
        data['SNR_peak'] = utils.SNR_peak(galaxy_noiseless, sky_level_pixel, band=6)[1]
        return galaxy_noiseless, blend_noisy, data, shift
    else:
        raise ValueError

def image_generator_real(cosmos_cat_dir, training_or_test, isolated_or_blended, used_idx=None, nmax_blend=4, max_try=3, mag_cut=28., method_first_shift='noshift', do_peak_detection=True):
    """
    Return numpy arrays: noiseless and noisy image of single galaxy and of blended galaxies
    Parameters:
    ----------
    cosmos_cat: COSMOS catalog
    nb_blended_gal: number of galaxies to add to the centered one on the blended image
    training_or_test: choice for generating a training or testing dataset
    """
    
    cosmos_cat = galsim.COSMOSCatalog('real_galaxy_catalog_25.2.fits', dir=cosmos_cat_dir)
    counter = 0
    np.random.seed() # important for multiprocessing !
    
    assert training_or_test in ['training', 'validation', 'test']
    #assert isinstance(n_years, int)
    assert isolated_or_blended in ['blended', 'isolated']
    
    while counter < max_try:
        try:
            ############## GENERATION OF THE GALAXIES ##################
            ud = galsim.UniformDeviate()
            real_gal_list = []

            nb_blended_gal = np.random.randint(nmax_blend)+1
            data = {}

            # Pick galaxies in catalog
            galaxies = []
            mag=[]
            mag_ir=[]
            j = 0
            while j < nb_blended_gal:
            # for j in range (nb_blended_gal):
                if used_idx is not None:
                    idx = np.random.choice(used_idx)
                else:
                    idx = np.random.randint(cosmos_cat.nobject)
                gal = cosmos_cat.makeGalaxy(idx, gal_type='parametric', chromatic=True, noise_pad_size=0)
                _mag_temp = gal.calculateMagnitude(filters['r'].withZeropoint(28.13))

                # Magnitude cut
                if _mag_temp < mag_cut:
                    gal = gal.rotate(ud() * 360. * galsim.degrees)
                    galaxies.append(gal)
                    mag.append(_mag_temp)
                    mag_ir.append(gal.calculateMagnitude(filters['H'].withZeropoint(24.92-22.35*coeff_noise_h)))
                    j += 1
                    
                ## Take the real galaxy image only if parametric galaxy is actually created
                if  len(galaxies) == (len(real_gal_list)+1):
                    bp_file = os.path.join(galsim.meta_data.share_dir, 'wfc_F814W.dat.gz')
                    bandpass = galsim.Bandpass(bp_file, wave_type='ang').thin().withZeropoint(25.94)
                    real_gal = cosmos_cat.makeGalaxy(idx, gal_type='real',
                                                    noise_pad_size=max_stamp_size*pixel_scale_lsst)
                    real_gal_list.append(real_gal)

            # Optionally, find the brightest and put it first in the list
            if center_brightest:
                _idx = np.argmin(mag)
                galaxies.insert(0, galaxies.pop(_idx))
                real_gal_list.insert(0, real_gal_list.pop(_idx))
                mag.insert(0,mag.pop(_idx))
                mag_ir.insert(0,mag_ir.pop(_idx))

            # Draw shifts for other galaxies (shift has the same shape to make it simpler to save as numpy array)
            shift = np.zeros((nmax_blend,2))
            # Shift centered
            galaxies[0], shift[0] = shift_gal(galaxies[0], method=method_first_shift, max_dx=0.1)
            for j,gal in enumerate(galaxies[1:]):
                galaxies[j+1], shift[j+1] = shift_gal(gal, shift_x0=shift[0,0], shift_y0=shift[0,1], min_r=0.65/2., max_r=1.5, method='annulus')
            
#             # real galaxies
#             real_gal_list[0], shift[0] = shift_gal(real_gal_list[0], method=method_first_shift, max_dx=0.1)
#             print(nb_blended_gal, len(galaxies), len(real_gal_list))
#             for k,gal in enumerate(real_gal_list[1:]):
#                 real_gal_list[k+1], shift[k+1] = shift_gal(gal, shift_x0=shift[0,0], shift_y0=shift[0,1], min_r=0.65/2., max_r=1.5, method='annulus')



            if nb_blended_gal>1:
                distances = [shift[j][0]**2+shift[j][1]**2 for j in range(1,nb_blended_gal)]
                idx_closest_to_peak_galaxy = np.argmin(distances)+1
            else:
                idx_closest_to_peak_galaxy = 0
            
            galaxy_noiseless = np.zeros((10,max_stamp_size,max_stamp_size))
            blend_noisy = np.zeros((10,max_stamp_size,max_stamp_size))
            galaxy_noiseless_real = np.zeros((10,max_stamp_size,max_stamp_size))
            blend_noisy_real = np.zeros((10,max_stamp_size,max_stamp_size))

            if do_peak_detection:
                band = 6
                galaxies_psf = [galsim.Convolve([gal*coeff_exp[band], PSF[band]]) for gal in galaxies]

                images, blend_img = draw_images(galaxies_psf, band, max_stamp_size*2, 'r', sky_level_pixel[band])
                blend_noisy_temp = blend_img.array.data
                # if isolated_or_blended == 'blended':
                peak_detection_output = peak_detection(blend_noisy_temp, band, shift, max_stamp_size*2, 4,nb_blended_gal, training_or_test, dist_cut=0.65/2.)
                if not peak_detection_output:
                    print('No peak detected')
                    raise RuntimeError
                else:
                    idx_closest_to_peak, idx_closest_to_peak_galaxy, center_pix_x, center_pix_y, center_arc_x, center_arc_y, n_peak = peak_detection_output

                # Modify galaxies and shift accordingly
                galaxies = [gal.shift(-center_arc_x, -center_arc_y) for gal in galaxies]
                shift[:nb_blended_gal] -= np.array([center_arc_x, center_arc_y])

            # real galaxies
            for k,gal in enumerate(real_gal_list):
                real_gal_list[k], shift[k] = shift_gal(gal, shift_x0=shift[k,0], shift_y0=shift[k,1], method='noshift')
         
            # Draw real images
            galaxies_real_psf = [galsim.Convolve([real_gal*coeff_exp[6], PSF_lsst]) for real_gal in real_gal_list]
            images_real, _ = draw_images(galaxies_real_psf, 6, max_stamp_size, 'r', sky_level_pixel[6], real_or_param = 'real')
            
       
            # Now draw image in all other bands
            for i, filter_name in enumerate(filter_names_all):
                galaxies_psf = [galsim.Convolve([gal*coeff_exp[i], PSF[i]]) for gal in galaxies]
                #if i != 6:
                images, blend_img = draw_images(galaxies_psf, i, max_stamp_size, filter_name, sky_level_pixel[i])
                if isolated_or_blended == 'isolated' or not do_peak_detection:
                    idx_closest_to_peak = 0
                    n_peak = 1
                galaxy_noiseless[i] = images[idx_closest_to_peak].array.data
                blend_noisy[i] = blend_img.array.data

                # Rescale real images by flux
                for jj, image_real in enumerate(images_real):
                    images_real[jj] = image_real.array.data * np.sum(images[jj].array.data)/np.sum(image_real.array.data)
                    
                # real galaxies
                galaxy_noiseless_real[i] = images_real[idx_closest_to_peak]
                for image_real in images_real:
                    blend_noisy_real[i] += image_real\
                # Add noise
                poissonian_noise = galsim.PoissonNoise(rng, sky_level=sky_level_pixel[i])
                blend_noisy_real[i].addNoise(poissonian_noise)

                # get data for the test sample (LSST stuff)
                if training_or_test == 'test' and filter_name == 'r':
                    # need psf to compute ellipticities
                    psf_image = PSF[i].drawImage(nx=max_stamp_size, ny=max_stamp_size, scale=pixel_scale[i])
                    data['redshift'], data['moment_sigma'], data['e1'], data['e2'], data['mag'] = get_data(real_gal_list[idx_closest_to_peak], images_real[idx_closest_to_peak], psf_image, param_or_real = 'real')

                    # Compute data and blendedness
                    if nb_blended_gal > 1:
                        data['closest_redshift'], data['closest_moment_sigma'], data['closest_e1'], data['closest_e2'], data['closest_mag'] = get_data(real_gal_list[idx_closest_to_peak_galaxy], images_real[idx_closest_to_peak_galaxy], psf_image, param_or_real = 'real')
                        img_central = images[idx_closest_to_peak].array
                        img_others = np.zeros_like(img_central)
                        for _h, image in enumerate(images):
                            if _h!=idx_closest_to_peak:
                                img_others += image.array
                        #img_others = np.array([image.array.data for _h, image in enumerate(images) if _h!=idx_closest_to_peak]).sum(axis = 0)
                        img_closest_neighbour =images[idx_closest_to_peak_galaxy].array# np.array(images[idx_closest_to_peak_galaxy].array.data)
                        data['blendedness_total_lsst'] = utils.compute_blendedness_total(img_central, img_others)
                        data['blendedness_closest_lsst'] = utils.compute_blendedness_single(img_central, img_closest_neighbour)
                        data['blendedness_aperture_lsst'] = utils.compute_blendedness_aperture(img_central, img_others, data['moment_sigma'])
                    else:
                        data['closest_redshift'] = np.nan
                        data['closest_moment_sigma'] = np.nan
                        data['closest_e1'] = np.nan
                        data['closest_e2'] = np.nan
                        data['closest_mag'] = np.nan
                        data['blendedness_total_lsst'] = np.nan
                        data['blendedness_closest_lsst'] = np.nan
                        data['blendedness_aperture_lsst'] = np.nan
            break

        except RuntimeError as e:
            print(e)

    # For training/validation, return normalized images only
    if training_or_test in ['training', 'validation']:
        galaxy_noiseless_real = utils.norm(galaxy_noiseless_real[None,:], bands=range(10), n_years=n_years)[0]
        blend_noisy_real = utils.norm(blend_noisy_real[None,:], bands=range(10), n_years=n_years)[0]
        return galaxy_noiseless_real, blend_noisy_real

    # For testing, return unormalized images and data
    elif training_or_test == 'test':
        data['nb_blended_gal'] = nb_blended_gal
        data['mag'] = mag[0]
        data['mag_ir'] = mag_ir[0]
        if nb_blended_gal>1:
            data['closest_mag'] = mag[idx_closest_to_peak_galaxy]
            data['closest_mag_ir'] = mag_ir[idx_closest_to_peak_galaxy]
            data['closest_x'] = shift[idx_closest_to_peak_galaxy][0]
            data['closest_y'] = shift[idx_closest_to_peak_galaxy][1]
        else:
            data['closest_mag'] = np.nan
            data['closest_mag_ir'] = np.nan
            data['closest_x'] = np.nan
            data['closest_y'] = np.nan
        data['idx_closest_to_peak'] = idx_closest_to_peak
        data['n_peak_detected'] = n_peak
        data['SNR'] = utils.SNR(galaxy_noiseless, sky_level_pixel, band=6)[1]
        data['SNR_peak'] = utils.SNR_peak(galaxy_noiseless, sky_level_pixel, band=6)[1]
        return galaxy_noiseless_real, blend_noisy_real, data, shift
    else:
        raise ValueError
