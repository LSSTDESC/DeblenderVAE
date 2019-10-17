
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

# Coefficients to take the exposure time and the telescope size into account
# coeff_exp_lsst = (15. * (6.68**2)/((2.4**2)*(1.-0.33**2))) * N_exposures_lsst
# coeff_exp_euclid = (1800. * ((1.25)**2 - (0.37)**2)/((2.4**2)*(1.-0.33**2))) * N_exposures_euclid

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
        shift_x = np.random.uniform(-1,1)#(-2.5,2.5)  #(-1,1)
        shift_y = np.random.uniform(-1,1)#(-2.5,2.5)  #(-1,1)
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
def blend_generator(cosmos_cat, training_or_test, used_idx=None, nmax_blend=4, max_try=3):
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
                gal = gal.rotate(ud() * 360. * galsim.degrees)
                galaxies.append(gal)
                mag.append(gal.calculateMagnitude(filters['r'].withZeropoint(28.13)))
                mag_ir.append(gal.calculateMagnitude(filters['H'].withZeropoint(24.92-22.35*coeff_noise_h)))

            # Optionally, find the brightest and put it first in the list
            if center_brightest:
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
                    data['redshift'], data['moment_sigma'], data['e1'], data['e2'] = get_data(galaxies[0], images[0], psf_image)
                    data['closest_redshift'], data['closest_moment_sigma'], data['closest_e1'], data['closest_e2'] = get_data(galaxies[idx_closest], images[idx_closest], psf_image)
                    #data['ellipticity_weights'] = 
                    
                    if nb_blended_gal > 1:
                        data['blendedness_total_lsst'] = utils.compute_blendedness_total(images[0], blend_img)
                        data['blendedness_closest_lsst'] = utils.compute_blendedness_single(images[0], images[idx_closest])
                    else:
                        data['blendedness_total_lsst'] = np.nan
                        data['blendedness_closest_lsst'] = np.nan
                # get data for the test sample (Euclid stuff)
                if training_or_test == 'test' and filter_name == 'V':
                    if nb_blended_gal > 1:
                        data['blendedness_total_euclid'] = utils.compute_blendedness_total(images[0], blend_img)
                        data['blendedness_closest_euclid'] = utils.compute_blendedness_single(images[0], images[idx_closest])
                    else:
                        data['blendedness_total_euclid'] = np.nan
                        data['blendedness_closest_euclid'] = np.nan
                
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


# def create_images(i,filter_, sky_level_pixel, stamp_size, pixel_scale, nb_blended_gal, psf, bdgal, add_gal):
#     """
#     Return images of the noiseless and noisy single galaxy and blend of galaxies
    
#     Parameters:
#     ----------
#     i: number of bandpass filter (from 0 to 9: [0,1,2] are NIR, [3] is VIS, [4,5...,9] are LSST)
#     filter_: filter to use to create the image
#     sky_level_pixel: sky background
#     stamp_size: size of the image
#     pixel_scale: pixel scale corresponding to the used instrument
#     nb_blended_gal: number of galaxies to add on the blended images
#     PSF: PSF to use to create the image
#     bdgal: galaxy to add on the center of the image
#     add_gal: galaxies to add on the blended images
#     """
#     # Create poissonian noise
#     poissonian_noise = galsim.PoissonNoise(rng, sky_level=sky_level_pixel)
    
#     # Create images to be filled with galaxies and noise
#     img = galsim.ImageF(stamp_size,stamp_size, scale=pixel_scale)
#     img_noisy = galsim.ImageF(stamp_size,stamp_size, scale=pixel_scale)
#     img_blend_neighbours = galsim.ImageF(stamp_size,stamp_size, scale=pixel_scale)
#     img_blend_noiseless = galsim.ImageF(stamp_size,stamp_size, scale=pixel_scale)
#     img_blend_noisy = galsim.ImageF(stamp_size,stamp_size, scale=pixel_scale)

#     # Create noiseless image of the centered galaxy
#     bdfinal = galsim.Convolve([bdgal, psf])

#     # Initialize each image: mandatory or allocation problems
#     bdfinal.drawImage(filter_, image=img)
#     bdfinal.drawImage(filter_, image=img_noisy)
#     bdfinal.drawImage(filter_, image=img_blend_noiseless)
#     bdfinal.drawImage(filter_, image=img_blend_noisy)

#     # Initialize blendedness lists
#     Blendedness_lsst = np.zeros((nb_blended_gal-1))
#     Blendedness_euclid = np.zeros((nb_blended_gal-1))
#     Blendedness_total_euclid = 0
#     Blendedness_total_lsst = 0

#     # Save noiseless galaxy
#     galaxy_noiseless = img.array.data
#     #img_noiseless = img

#     # Add noise and save noisy centered galaxy
#     img_noisy.addNoise(poissonian_noise)
#     galaxy_noisy = img_noisy.array.data

#     # Choose value to create blended image in NIR, VIS or LSST filter
#     if i < 3:
#         rank_img = 0
#     elif i == 3:
#         rank_img = 1
#     else:
#         rank_img = 2

#     # Create blended image
#     for k in range (nb_blended_gal-1):
#         img_new = galsim.ImageF(stamp_size, stamp_size, scale=pixel_scale)
#         bdfinal_new = galsim.Convolve([add_gal[k][rank_img], psf])
#         if k == 0:
#             bdfinal_new.drawImage(filter_, image=img_blend_neighbours)
#         bdfinal_new.drawImage(filter_, image=img_new)

#         img_blend_noiseless += img_new
#         if k !=0:
#             img_blend_neighbours += img_new
#         img_blend_noisy += img_new
#         if i == 3 :
#             Blendedness_euclid[k]= utils.compute_blendedness(img,img_new)
#         elif i ==6 :
#             Blendedness_lsst[k]= utils.compute_blendedness(img,img_new)
    
#     if nb_blended_gal>1: 
#         if i == 3 :
#             Blendedness_total_euclid = utils.compute_blendedness(img,img_blend_neighbours)
#         elif i == 6:
#             Blendedness_total_lsst = utils.compute_blendedness(img,img_blend_neighbours)

#     # Save noiseless blended image
#     blend_noiseless = img_blend_noiseless.array.data
    
#     # Add noise and save noisy blended image
#     img_blend_noisy.addNoise(poissonian_noise)
#     blend_noisy = img_blend_noisy.array.data

#     return galaxy_noiseless, galaxy_noisy, blend_noiseless, blend_noisy, Blendedness_lsst, Blendedness_euclid, Blendedness_total_lsst, Blendedness_total_euclid



# # Generation function
# def blend_generator(cosmos_cat, training_or_test, used_idx=None, max_try=3):
#     """
#     Return numpy arrays: noiseless and noisy image of single galaxy and of blended galaxies

#     Parameters:
#     ----------
#     cosmos_cat: COSMOS catalog
#     nb_blended_gal: number of galaxies to add to the centered one on the blended image
#     training_or_test: choice for generating a training or testing dataset
#     """

#     counter = 0
#     np.random.seed() # important for multiprocessing !

#     while counter < max_try:
#         try:
#             ############## GENERATION OF THE GALAXIES ##################
#             ud = galsim.UniformDeviate()

#             nb_blended_gal = np.random.randint(4)+1
            
#             galaxies = []
#             mag=[]
#             # scale_radius = []
#             redshift = []
#             for i in range (nb_blended_gal):
#                 if used_idx is not None:
#                     idx = np.random.choice(used_idx)
#                 else:
#                     idx = np.random.randint(cosmos_cat.nobject)
#                 galaxies.append(cosmos_cat.makeGalaxy(idx, gal_type='parametric', chromatic=True, noise_pad_size = 0))
#                 mag.append(galaxies[i].calculateMagnitude(filters['r'].withZeropoint(28.13)))
#                 # scale_radius.append(get_scale_radius(galaxies[i]))
#                 redshift.append(galaxies[i].SED.redshift)


#             if center_brightest:
#                 gal = galaxies[np.where(mag == np.min(mag))[0][0]]
#             else:
#                 gal = galaxies[0]
            
#             galaxies.remove(gal)
                
#             ############ LUMINOSITY ############# 
#             # The luminosity is multiplied by the ratio of the noise in the LSST R band and the assumed cosmos noise             
#             # bdgal_lsst =  gal * coeff_exp_lsst
#             # bdgal_euclid_nir =  coeff_exp_euclid * gal
#             # bdgal_euclid_vis =  coeff_exp_euclid * gal


#             # Generate the blends
#             add_gal = []
#             shift=np.zeros((nb_blended_gal-1, 2))
#             for i in range (len(galaxies)):
#                 # gal_new = None
#                 ud = galsim.UniformDeviate()
#                 gal_new = galaxies[i].rotate(ud() * 360. * galsim.degrees)

#                 gal_new, shift[i]  = shift_gal(gal, gal_new, method=shift_method)
                    
#                 # bdgal_new_lsst = None
#                 # bdgal_new_euclid_nir =None
#                 # bdgal_new_euclid_vis =None
#                 bdgal_new_lsst = coeff_exp_lsst * gal_new
#                 bdgal_new_euclid_nir = coeff_exp_euclid * gal_new
#                 bdgal_new_euclid_vis = coeff_exp_euclid * gal_new
#                 add_gal.append([bdgal_new_euclid_nir, bdgal_new_euclid_vis, bdgal_new_lsst])

#             # Initialize some lists
#             galaxy_noiseless = np.zeros((10,max_stamp_size,max_stamp_size))
#             galaxy_noisy = np.zeros((10,max_stamp_size,max_stamp_size))

#             blend_noiseless = np.zeros((10,max_stamp_size,max_stamp_size))
#             blend_noisy = np.zeros((10,max_stamp_size,max_stamp_size))

#             Blendedness_lsst = np.zeros((10,nb_blended_gal-1))
#             Blendedness_euclid = np.zeros((10,nb_blended_gal-1))
#             Blendedness_total_lsst = np.zeros((10))
#             Blendedness_total_euclid = np.zeros((10))

#             # Generate LSST PSF
#             # fwhm_lsst = lsst_PSF()
#             # PSF_lsst = galsim.Kolmogorov(fwhm=fwhm_lsst)

#             # Generate images
#             # i = 0
#             # for filter_name, filter_ in filters.items():
#             for i, filter_name in enumerate(filter_names_all):
#                 # if (i < 3):
#                 #     galaxy_noiseless[i], galaxy_noisy[i], blend_noiseless[i], blend_noisy[i], Blendedness_lsst[i], Blendedness_euclid[i], Blendedness_total_lsst[i], Blendedness_total_euclid[i] = create_images(i, filter_,sky_level_pixel_nir[i], max_stamp_size, pixel_scale_euclid_nir, nb_blended_gal, PSF_euclid_nir, bdgal_euclid_nir, add_gal)
#                 # elif (i==3):
#                 #     galaxy_noiseless[i], galaxy_noisy[i], blend_noiseless[i], blend_noisy[i], Blendedness_lsst[i], Blendedness_euclid[i], Blendedness_total_lsst[i], Blendedness_total_euclid[i] = create_images(i, filter_,sky_level_pixel_vis, max_stamp_size, pixel_scale_euclid_vis, nb_blended_gal, PSF_euclid_vis, bdgal_euclid_vis, add_gal)
#                 # else:
#                 #     galaxy_noiseless[i], galaxy_noisy[i], blend_noiseless[i], blend_noisy[i], Blendedness_lsst[i], Blendedness_euclid[i], Blendedness_total_lsst[i], Blendedness_total_euclid[i] = create_images(i, filter_,sky_level_pixel_lsst[i-4], max_stamp_size, pixel_scale_lsst, nb_blended_gal, PSF_lsst, bdgal_lsst, add_gal)
#                 # i+=1
#                 galaxy_noiseless[i], galaxy_noisy[i], blend_noiseless[i], blend_noisy[i], Blendedness_lsst[i], Blendedness_euclid[i], Blendedness_total_lsst[i], Blendedness_total_euclid[i] = create_images(i, filters[filter_name], sky_level_pixel[i], max_stamp_size, pixel_scale[i], nb_blended_gal, PSF[i], gal*coeff_exp[i], add_gal)

#                 if training_or_test == 'test' and filter_name == 'r':
#                     # hlr = img.calculateHLR()
#                     psf_image = PSF[i].drawImage(nx=max_stamp_size, ny=max_stamp_size, scale=pixel_scale[i])
#                     data = get_data(gal, img, psf_image)
            
#             break

#         except RuntimeError as e:
#             print(e)

#     # Return outputs depending on the kind of generated dataset
#     if training_or_test in ['training', 'validation']:
#         galaxy_noiseless = utils.norm(galaxy_noiseless[None,:], bands=range(10))[0]
#         blend_noisy = utils.norm(blend_noisy[None,:], bands=range(10))[0]
#         return galaxy_noiseless, blend_noisy, None
#     elif training_or_test == 'test':
#         # return galaxy_noiseless, galaxy_noisy, blend_noiseless, blend_noisy, redshift, shift, mag, Blendedness_euclid[3], Blendedness_lsst[6], Blendedness_total_euclid[3], Blendedness_total_lsst[6] #, scale_radius
#         return galaxy_noiseless, blend_noisy, (redshift, shift, mag, Blendedness_euclid[3], Blendedness_lsst[6], Blendedness_total_euclid[3], Blendedness_total_lsst[6]) #, scale_radius)
#     else:
#         raise ValueError

##################################
##################################
##################################

# # Generation function
# def Gal_generator_noisy_test(cosmos_cat, nb_blended_gal):
#     count = 0
#     galaxy = np.zeros((10))
#     while (galaxy.all() == 0):
#         try:
#             ############## GENERATION OF THE GALAXIES ##################
#             ud = galsim.UniformDeviate()
            
#             galaxies = []
#             mag=[]
#             for i in range (nb_blended_gal):
#                 galaxies.append(cosmos_cat.makeGalaxy(random.randint(0,cosmos_cat.nobjects-1), gal_type='parametric', chromatic=True, noise_pad_size = 0))
#                 mag.append(galaxies[i].calculateMagnitude(filters['r'].withZeropoint(28.13)))

#             gal = galaxies[np.where(mag == np.min(mag))[0][0]]
#             redshift = gal.SED.redshift
            
#             galaxies.remove(gal)
                
#             ############ LUMINOSITY ############# 
#             # The luminosity is multiplied by the ratio of the noise in the LSST R band and the assumed cosmos noise             
#             bdgal_lsst =  gal * coeff_exp_lsst
#             bdgal_euclid_nir =  coeff_exp_euclid * gal
#             bdgal_euclid_vis =  coeff_exp_euclid * gal


#             add_gal = []
#             shift=np.zeros((nb_blended_gal-1, 2))
#             for i in range (len(galaxies)):
#                 gal_new = None
#                 ud = galsim.UniformDeviate()
#                 gal_new = galaxies[i].rotate(ud() * 360. * galsim.degrees)

#                 gal_new, shift[i]  = shift_gal(gal_new, method=shift_method)
                   
#                 bdgal_new_lsst = None
#                 bdgal_new_euclid_nir =None
#                 bdgal_new_euclid_vis =None
#                 bdgal_new_lsst =  coeff_exp_lsst * gal_new
#                 bdgal_new_euclid_nir =  coeff_exp_euclid * gal_new
#                 bdgal_new_euclid_vis =  coeff_exp_euclid * gal_new
#                 add_gal.append([bdgal_new_euclid_nir,bdgal_new_euclid_vis, bdgal_new_lsst])

#             #print(len(add_gal),len(add_gal[0]))
#             ########### PSF #####################
#             # convolve with PSF to make final profil : profil from LSST science book and (https://arxiv.org/pdf/0805.2366.pdf)
#             # mu = -0.43058681997903414
#             # sigma = 0.3404334041976153
#             # p_unnormed = lambda x : (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
#             #                    / (x * sigma * np.sqrt(2 * np.pi)))#((1/(2*z0))*((z/z0)**2)*np.exp(-z/z0))
#             # p_normalization = scipy.integrate.quad(p_unnormed, 0., np.inf)[0]
#             # p = lambda z : p_unnormed(z) / p_normalization

#             # from scipy import stats
#             # class PSF_distribution(stats.rv_continuous):
#             #     def __init__(self):
#             #         super(PSF_distribution, self).__init__()
#             #         self.a = 0.
#             #         self.b = 10.
#             #     def _pdf(self, x):
#             #         return p(x)

#             # pdf = PSF_distribution()
#             # fwhm_lsst = 0.65 #pdf.rvs() #Fixed at median value : Fig 1 : https://arxiv.org/pdf/0805.2366.pdf

#             # fwhm_euclid_nir = 0.22 # EUCLID PSF is supposed invariant (no atmosphere) despite the optical and wavelengths variations
#             # fwhm_euclid_vis = 0.18 # EUCLID PSF is supposed invariant (no atmosphere) despite the optical and wavelengths variations
#             # beta = 2.5
#             # PSF_lsst = galsim.Kolmogorov(fwhm=fwhm_lsst)#galsim.Moffat(fwhm=fwhm_lsst, beta=beta)
#             # PSF_euclid_nir = galsim.Moffat(fwhm=fwhm_euclid_nir, beta=beta)
#             # PSF_euclid_vis = galsim.Moffat(fwhm=fwhm_euclid_vis, beta=beta)

#             galaxy_noiseless = np.zeros((10,max_stamp_size,max_stamp_size))
#             galaxy_noisy = np.zeros((10,max_stamp_size,max_stamp_size))

#             blend_noiseless = np.zeros((10,max_stamp_size,max_stamp_size))
#             blend_noisy = np.zeros((10,max_stamp_size,max_stamp_size))

#             Blendedness_lsst = np.zeros((nb_blended_gal-1))
#             Blendedness_euclid = np.zeros((nb_blended_gal-1))


#             #image = np.zeros((64,64))
#             #image_new = np.zeros ((64,64))

#             i = 0
#             for filter_name, filter_ in filters.items():
#                 #print('i = '+str(i))
#                 if (i < 3):
#                     #print('in NIR')
#                     poissonian_noise_nir = galsim.PoissonNoise(rng, sky_level=sky_level_pixel_nir[i])
#                     img = galsim.ImageF(max_stamp_size,max_stamp_size, scale=pixel_scale_euclid_nir)  
#                     bdfinal = galsim.Convolve([bdgal_euclid_nir, PSF_euclid_nir])
#                     bdfinal.drawImage(filter_, image=img)
#                     # Noiseless galaxy
#                     galaxy_noiseless[i] = img.array.data
#                     img_noiseless = img

#                     #### Bended image
#                     img_blended = galsim.ImageF(max_stamp_size,max_stamp_size, scale=pixel_scale_euclid_nir)
#                     img_blended = img_noiseless
#                     for k in range (nb_blended_gal-1):
#                         img_new = galsim.ImageF(max_stamp_size, max_stamp_size, scale=pixel_scale_euclid_nir)
#                         bdfinal_new = galsim.Convolve([add_gal[k][0], PSF_euclid_nir])
#                         bdfinal_new.drawImage(filter_, image=img_new)
#                         img_blended += img_new
#                     # Noiseless blended image
#                     blend_noiseless[i] = img_blended.array.data


#                     # Noisy centered galaxy
#                     img_blended.addNoise(poissonian_noise_nir)
#                     blend_noisy[i] = img_blended.array.data

#                     # Noisy galaxy
#                     img.addNoise(poissonian_noise_nir)
#                     galaxy_noisy[i] = img.array.data
#                 else:
#                     if (i==3):
#                         poissonian_noise_vis = galsim.PoissonNoise(rng, sky_level=sky_level_pixel_vis)
#                         img = galsim.ImageF(max_stamp_size,max_stamp_size, scale=pixel_scale_euclid_vis)  
#                         bdfinal = galsim.Convolve([bdgal_euclid_vis, PSF_euclid_vis])
#                         bdfinal.drawImage(filter_, image=img)
#                         # Noiseless galaxy
#                         galaxy_noiseless[i] = img.array.data
#                         img_noiseless = img

#                         #### Bended image
#                         img_blended = galsim.ImageF(max_stamp_size,max_stamp_size, scale=pixel_scale_euclid_vis)
#                         img_blended = img_noiseless
#                         for k in range (nb_blended_gal-1):
#                             img_new = galsim.ImageF(max_stamp_size, max_stamp_size, scale=pixel_scale_euclid_vis)
#                             bdfinal_new = galsim.Convolve([add_gal[k][1], PSF_euclid_vis])
#                             bdfinal_new.drawImage(filter_, image=img_new)
#                             img_blended += img_new
#                             Blendedness_euclid[k]= utils.blendedness(img,img_new)
#                         # Noiseless blended image 
#                         blend_noiseless[i] = img_blended.array.data
#                         # Noisy centered galaxy
#                         img_blended.addNoise(poissonian_noise_vis)
#                         blend_noisy[i] = img_blended.array.data

#                         # Noisy galaxy
#                         img.addNoise(poissonian_noise_vis)
#                         galaxy_noisy[i] = img.array.data

#                     else:
#         #                print('passage a LSST')
#                         poissonian_noise_lsst = galsim.PoissonNoise(rng, sky_level_pixel_lsst[i-4])
#                         img = galsim.ImageF(max_stamp_size,max_stamp_size, scale=pixel_scale_lsst)  
#                         bdfinal = galsim.Convolve([bdgal_lsst, PSF_lsst])
#                         bdfinal.drawImage(filter_, image=img)
#                         #bdfinal.drawImage(filter_, image=img_noiseless)
#                         # Noiseless galaxy
#                         galaxy_noiseless[i] = img.array.data
#                         img_noiseless = img

#                         # Noisy galaxy
#                         img.addNoise(poissonian_noise_lsst)
#                         galaxy_noisy[i]= img.array.data

#                         #### Bended image
#                         img_blended = galsim.ImageF(max_stamp_size,max_stamp_size, scale=pixel_scale_lsst)
#                         img_blended = img_noiseless
#                         for k in range (nb_blended_gal-1):
#                             img_new = galsim.ImageF(max_stamp_size, max_stamp_size, scale=pixel_scale_lsst)
#                             bdfinal_new = galsim.Convolve([add_gal[k][2], PSF_lsst])
#                             bdfinal_new.drawImage(filter_, image=img_new)
#                             img_blended += img_new
#                             if (i==6):
#                                 Blendedness_lsst[k]= utils.blendedness(img,img_new)
#                         # Noiseless blended image
#                         blend_noiseless[i] = img_blended.array.data
#                         # Noisy centered galaxy
#                         img_blended.addNoise(poissonian_noise_lsst)
#                         blend_noisy[i] = img_blended.array.data




#                 i+=1
#             return galaxy_noiseless, galaxy_noisy, blend_noiseless, blend_noisy, shift, mag, Blendedness_euclid, Blendedness_lsst
#         except RuntimeError: 
#             count +=1
#     print("nb of error : "+(count))
    
    
# # Generation function
# def Gal_generator_noisy_training(cosmos_cat, nb_blended_gal):
#     cosmos_cat = cosmos_cat
#     count = 0
#     galaxy = np.zeros((10))
#     while (galaxy.all() == 0):
#         try:
#             ############## GENERATION OF THE GALAXIES ##################
#             ud = galsim.UniformDeviate()
            
#             galaxies = []
#             mag=[]
#             for i in range (nb_blended_gal):
#                 galaxies.append(cosmos_cat.makeGalaxy(random.randint(0,cosmos_cat.nobjects-1), gal_type='parametric', chromatic=True, noise_pad_size = 0))
#                 mag.append(galaxies[i].calculateMagnitude(filters['r'].withZeropoint(28.13)))

#             gal = galaxies[np.where(mag == np.min(mag))[0][0]]
#             redshift = gal.SED.redshift
            
#             galaxies.remove(gal)
                
#             ############ LUMINOSITY ############# 
#             # The luminosity is multiplied by the ratio of the noise in the LSST R band and the assumed cosmos noise             
#             bdgal_lsst =  (15. * (6.68**2)/((2.4**2)*(1.-0.33**2))) * gal * N_exposures_lsst
#             bdgal_euclid_nir =  (1800. * ((1.25)**2 - (0.37)**2)/((2.4**2)*(1.-0.33**2))) * gal * N_exposures_euclid
#             bdgal_euclid_vis =  (1800. * ((1.25)**2 - (0.37)**2)/((2.4**2)*(1.-0.33**2))) * gal * N_exposures_euclid         


#             add_gal = []
#             shift=np.zeros((nb_blended_gal-1, 2))
#             for i in range (len(galaxies)):
#                 gal_new = None
#                 ud = galsim.UniformDeviate()
#                 gal_new = galaxies[i].rotate(ud() * 360. * galsim.degrees)
#                 shift_x = np.random.uniform(-2.5,2.5)
#                 shift_y = np.random.uniform(-2.5,2.5)

#                 gal_new = gal_new.shift((shift_x,shift_y))
                   
#                 bdgal_new_lsst = None
#                 bdgal_new_euclid_nir =None
#                 bdgal_new_euclid_vis =None
#                 bdgal_new_lsst =  (15. * (6.68**2)/((2.4**2)*(1.-0.33**2))) * gal_new * N_exposures_lsst
#                 bdgal_new_euclid_nir =  (1800. * ((1.25)**2 - (0.37)**2)/((2.4**2)*(1.-0.33**2))) * gal_new * N_exposures_euclid
#                 bdgal_new_euclid_vis =  (1800. * ((1.25)**2 - (0.37)**2)/((2.4**2)*(1.-0.33**2))) * gal_new * N_exposures_euclid
#                 add_gal.append([bdgal_new_euclid_nir,bdgal_new_euclid_vis, bdgal_new_lsst])
#                 shift[i]=(shift_x,shift_y)

#             galaxy_noiseless = np.zeros((10,max_stamp_size,max_stamp_size))
#             blend_noisy = np.zeros((10,max_stamp_size,max_stamp_size))

#             i = 0
#             for filter_name, filter_ in filters.items():
#                 #print('i = '+str(i))
#                 if (i < 3):
#                     #print('in NIR')
#                     poissonian_noise_nir = galsim.PoissonNoise(rng, sky_level=sky_level_pixel_nir[i])
#                     img = galsim.ImageF(max_stamp_size,max_stamp_size, scale=pixel_scale_euclid_nir)  
#                     bdfinal = galsim.Convolve([bdgal_euclid_nir, PSF_euclid_nir])
#                     bdfinal.drawImage(filter_, image=img)
#                     # Noiseless galaxy
#                     galaxy_noiseless[i] = img.array.data
#                     img_noiseless = img
#                     #print('single galaxy finished')

#                     #### Bended image
#                     img_blended = galsim.ImageF(max_stamp_size,max_stamp_size, scale=pixel_scale_euclid_nir)
#                     img_blended = img_noiseless
#                     for k in range (nb_blended_gal-1):
#                         img_new = galsim.ImageF(max_stamp_size, max_stamp_size, scale=pixel_scale_euclid_nir)
#                         bdfinal_new = galsim.Convolve([add_gal[k][0], PSF_euclid_nir])
#                         bdfinal_new.drawImage(filter_, image=img_new)
#                         img_blended = img_blended + img_new
#                     # Noisy centered galaxy
#                     img_blended.addNoise(poissonian_noise_nir)
#                     blend_noisy[i] = img_blended.array.data

#                 else:
#                     if (i==3):
#                         poissonian_noise_vis = galsim.PoissonNoise(rng, sky_level=sky_level_pixel_vis)
#                         img = galsim.ImageF(max_stamp_size,max_stamp_size, scale=pixel_scale_euclid_vis)  
#                         bdfinal = galsim.Convolve([bdgal_euclid_vis, PSF_euclid_vis])
#                         bdfinal.drawImage(filter_, image=img)
#                         # Noiseless galaxy
#                         galaxy_noiseless[i] = img.array.data
#                         img_noiseless = img

#                         #### Bended image
#                         img_blended = galsim.ImageF(max_stamp_size,max_stamp_size, scale=pixel_scale_euclid_vis)
#                         img_blended = img_noiseless
#                         for k in range (nb_blended_gal-1):
#                             img_new = galsim.ImageF(max_stamp_size, max_stamp_size, scale=pixel_scale_euclid_vis)
#                             bdfinal_new = galsim.Convolve([add_gal[k][1], PSF_euclid_vis])
#                             bdfinal_new.drawImage(filter_, image=img_new)
#                             img_blended = img_blended + img_new
#                         # Noisy centered galaxy
#                         img_blended.addNoise(poissonian_noise_vis)
#                         blend_noisy[i] = img_blended.array.data


#                     else:
#         #                print('passage a LSST')
#                         poissonian_noise_lsst = galsim.PoissonNoise(rng, sky_level_pixel_lsst[i-4])
#                         img = galsim.ImageF(max_stamp_size,max_stamp_size, scale=pixel_scale_lsst)  
#                         bdfinal = galsim.Convolve([bdgal_lsst, PSF_lsst])
#                         bdfinal.drawImage(filter_, image=img)
#                         bdfinal.drawImage(filter_, image=img_noiseless)
#                         # Noiseless galaxy
#                         galaxy_noiseless[i] = img.array.data
#                         img_noiseless = img

#                         #### Bended image
#                         img_blended = galsim.ImageF(max_stamp_size,max_stamp_size, scale=pixel_scale_lsst)
#                         img_blended = img_noiseless
#                         for k in range (nb_blended_gal-1):
#                             img_new = galsim.ImageF(max_stamp_size, max_stamp_size, scale=pixel_scale_lsst)
#                             bdfinal_new = galsim.Convolve([add_gal[k][2], PSF_lsst])
#                             bdfinal_new.drawImage(filter_, image=img_new)
#                             img_blended = img_blended + img_new
#                         # Noisy centered galaxy
#                         img_blended.addNoise(poissonian_noise_lsst)
#                         blend_noisy[i] = img_blended.array.data

#                 i+=1
#             return galaxy_noiseless, blend_noisy
#         except RuntimeError: 
#             count +=1
#     print("nb of error : "+(count))
 
    