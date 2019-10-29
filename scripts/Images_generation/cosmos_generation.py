# Import packages

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import sys
import os
import logging
import galsim
import matplotlib.pyplot as plt

sys.path.insert(0,'../tools_for_VAE/')
import tools_for_VAE
from tools_for_VAE import utils

############# SIZE OF STAMPS ################
#################### PSF ###################
### If a varying PSF is needed, uncomment this part. ####
###--------------------------------------------------####
# convolve with PSF to make final profil : profil from LSST science book and (https://arxiv.org/pdf/0805.2366.pdf)
# mu = -0.43058681997903414
# sigma = 0.3404334041976153
# p_unnormed = lambda x : (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
#                     / (x * sigma * np.sqrt(2 * np.pi)))#((1/(2*z0))*((z/z0)**2)*np.exp(-z/z0))
# p_normalization = scipy.integrate.quad(p_unnormed, 0., np.inf)[0]
# p = lambda z : p_unnormed(z) / p_normalization

# from scipy import stats
# class PSF_distribution(stats.rv_continuous):
#     def __init__(self):
#         super(PSF_distribution, self).__init__()
#         self.a = 0.
#         self.b = 10.
#     def _pdf(self, x):
#         return p(x)
#
# def lsst_PSF():
#     #Fig 1 : https://arxiv.org/pdf/0805.2366.pdf
#     mu = -0.43058681997903414 # np.log(0.65)
#     sigma = 0.3404334041976153  # Fixed to have corresponding percentils as in paper
#     p_unnormed = lambda x : (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
#                     / (x * sigma * np.sqrt(2 * np.pi)))#((1/(2*z0))*((z/z0)**2)*np.exp(-z/z0))
#     p_normalization = scipy.integrate.quad(p_unnormed, 0., np.inf)[0]
#     p = lambda z : p_unnormed(z) / p_normalization

#     from scipy import stats
#     class PSF_distribution(stats.rv_continuous):
#         def __init__(self):
#             super(PSF_distribution, self).__init__()
#             self.a = 0.
#             self.b = 10.
#         def _pdf(self, x):
#             return p(x)

#     pdf = PSF_distribution()
#     return pdf.rvs()

# LSST
# The PSF is fixed since we stack here 100 exposures
fwhm_lsst = 0.65 ## Fixed at median value : Fig 1 : https://arxiv.org/pdf/0805.2366.pdf
PSF_lsst = galsim.Kolmogorov(fwhm=fwhm_lsst)

# Euclid
fwhm_euclid_nir = 0.22 # EUCLID PSF is supposed invariant (no atmosphere) despite the optical and wavelengths variations
fwhm_euclid_vis = 0.18 # EUCLID PSF is supposed invariant (no atmosphere) despite the optical and wavelengths variations
beta = 2.5
PSF_euclid_nir = galsim.Moffat(fwhm=fwhm_euclid_nir, beta=beta)
PSF_euclid_vis = galsim.Moffat(fwhm=fwhm_euclid_vis, beta=beta)

PSF = [PSF_euclid_vis]*3 + [PSF_euclid_vis] + [PSF_lsst]*6

#################### EXPOSURE AND LUMISOITY ###################
# The luminosity is multiplied by the ratio of the noise in the LSST R band and the assumed cosmos noise             
coeff_exp_euclid =  (1800. * ((1.25)**2 - (0.37)**2)/((2.4**2)*(1.-0.33**2))) * N_exposures_euclid
coeff_exp_lsst =  (15. * (6.68**2)/((2.4**2)*(1.-0.33**2))) * N_exposures_lsst
coeff_exp = [coeff_exp_euclid]*4 + [coeff_exp_lsst]*6

#random_seed = 1234567
rng = galsim.BaseDeviate(None)

def get_data(gal, gal_image, psf_image):
    shear_est = 'KSB' #'REGAUSS' for e (default) or 'KSB' for g
    res = galsim.hsm.EstimateShear(gal_image, psf_image, shear_est=shear_est, strict=True)
    mag = gal.calculateMagnitude(filters['r'].withZeropoint(28.13))
    if res.error_message == "":
        if shear_est != 'KSB':
            return [gal.SED.redshift, res.moments_sigma, res.corrected_e1, res.corrected_e2, mag] #, res.observed_shape.e1, res.observed_shape.e2]
        else:
            return [gal.SED.redshift, res.moments_sigma, res.corrected_g1, res.corrected_g2, mag] #, res.observed_shape.e1, res.observed_shape.e2]
    else:
        return [gal.SED.redshift, np.nan, np.nan, np.nan, mag]
    # return [gal.SED.redshift, res.moments_sigma, res.observed_shape.e1, res.observed_shape.e2] #this is wrong!


# Generation function
def cosmos_galaxy_generator(cosmos_cat_filename, training_or_test, used_idx=None, max_try=3):
    counter = 0
    np.random.seed() # important for multiprocessing !
    cosmos_cat = galsim.COSMOSCatalog(file_name=cosmos_cat_filename)

    while counter < max_try:
        try:
            ############## SHAPE OF THE GALAXY ##################
            ud = galsim.UniformDeviate()#cosmos_cat.nobjects -10000-1

            if used_idx is not None:
                idx = np.random.choice(used_idx)
            else:
                idx = np.random.randint(cosmos_cat.nobject)
            gal = cosmos_cat.makeGalaxy(idx, gal_type='parametric', chromatic=True, noise_pad_size = 0)

            gal = gal.rotate(ud() * 360. * galsim.degrees)
            
            galaxy_noiseless = np.zeros((10,max_stamp_size,max_stamp_size))
            galaxy_noisy = np.zeros((10,max_stamp_size,max_stamp_size))

            for i, filter_name in enumerate(filter_names_all):
                poissonian_noise = galsim.PoissonNoise(rng, sky_level=sky_level_pixel[i])
                img = galsim.ImageF(max_stamp_size, max_stamp_size, scale=pixel_scale[i])  
                bdfinal = galsim.Convolve([gal*coeff_exp[i], PSF[i]])
                bdfinal.drawImage(filters[filter_name], image=img)

                # Noiseless galaxy
                galaxy_noiseless[i] = img.array.data

                # Get data for test sample
                if training_or_test == 'test' and filter_name == 'r':
                    # hlr = img.calculateHLR()
                    psf_image = PSF[i].drawImage(nx=max_stamp_size, ny=max_stamp_size, scale=pixel_scale[i])
                    data = get_data(gal, img, psf_image)

                # Noisy galaxy
                img.addNoise(poissonian_noise)
                galaxy_noisy[i]= img.array.data
            
            break

        except RuntimeError as e:
            print(e)
            counter += 1
    
    if counter == max_try:
        raise RuntimeError
    
    # For training/validation, return normalized images only
    if training_or_test in ['training', 'validation']:
        galaxy_noiseless = utils.norm(galaxy_noiseless[None,:], bands=range(10))[0]
        galaxy_noisy = utils.norm(galaxy_noisy[None,:], bands=range(10))[0]
        return galaxy_noiseless, galaxy_noisy
    # For testing, return unormalized images and data
    else:
        SNR = utils.SNR(galaxy_noiseless, sky_level_pixel, band=6)[1]
        SNR_peak = utils.SNR_peak(galaxy_noiseless, sky_level_pixel, band=6)[1]
        data += [SNR, SNR_peak]
        return galaxy_noiseless, galaxy_noisy, data

