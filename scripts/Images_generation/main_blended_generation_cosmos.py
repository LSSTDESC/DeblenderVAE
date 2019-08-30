import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import sys
import os
import logging
import galsim
import cmath as cm
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import WMAP9 as cosmo
import scipy
import scipy.integrate 
from astropy.convolution import Gaussian1DKernel, Gaussian2DKernel, convolve
from blended_images import blend_generator
from multiprocess import *

sys.path.insert(0,'../tools_for_VAE/')
from tools_for_VAE import utils

# Parameters to fix
phys_stamp_size = 6.4 # arcsec
pixel_scale_euclid_vis = 0.1 # arcsec/pixel

stamp_size = int(phys_stamp_size/pixel_scale_euclid_vis)


### Definition of sky background
pixel_scale_lsst = 0.2 # arcseconds # LSST Science book
pixel_scale_euclid_nir = 0.3 # arcseconds # Euclid Science book
pixel_scale_euclid_vis = 0.1 # arcseconds # Euclid Science book
pixel_scale = [pixel_scale_euclid_nir]*3 + [pixel_scale_euclid_vis] + [pixel_scale_lsst]*6

#################### NOISE ###################
# Poissonian noise according to sky_level
N_exposures_lsst = 100
N_exposures_euclid = 1

sky_level_lsst_u = (2.512 **(26.50-22.95)) * N_exposures_lsst # in e-.s-1.arcsec_2
sky_level_lsst_g = (2.512 **(28.30-22.24)) * N_exposures_lsst # in e-.s-1.arcsec_2
sky_level_lsst_r = (2.512 **(28.13-21.20)) * N_exposures_lsst # in e-.s-1.arcsec_2
sky_level_lsst_i = (2.512 **(27.79-20.47)) * N_exposures_lsst # in e-.s-1.arcsec_2
sky_level_lsst_z = (2.512 **(27.40-19.60)) * N_exposures_lsst # in e-.s-1.arcsec_2
sky_level_lsst_y = (2.512 **(26.58-18.63)) * N_exposures_lsst # in e-.s-1.arcsec_2
sky_level_pixel_lsst = [int((sky_level_lsst_u * 15 * pixel_scale_lsst**2)),
                        int((sky_level_lsst_g* 15 * pixel_scale_lsst**2)),
                        int((sky_level_lsst_r* 15 * pixel_scale_lsst**2)),
                        int((sky_level_lsst_i* 15 * pixel_scale_lsst**2)),
                        int((sky_level_lsst_z* 15 * pixel_scale_lsst**2)),
                        int((sky_level_lsst_y* 15 * pixel_scale_lsst**2))]# in e-/pixel/15s

# average background level for Euclid observations : 22.35 mAB.arcsec-2 in VIS (Consortium book) ##
# For NIR bands, a coefficient is applied : it is calculated by comparing magnitudes AB of one point in the
# sky to the magnitude AB in VIS on this point. The choosen point is (-30;30) in galactic coordinates 
# (EUCLID and LSST overlap on this point).
coeff_noise_y = (22.57/21.95)
coeff_noise_j = (22.53/21.95)
coeff_noise_h = (21.90/21.95)

sky_level_nir_Y = (2.512 **(24.25-22.35*coeff_noise_y)) * N_exposures_euclid # in e-.s-1.arcsec_2
sky_level_nir_J = (2.512 **(24.29-22.35*coeff_noise_j)) * N_exposures_euclid # in e-.s-1.arcsec_2
sky_level_nir_H = (2.512 **(24.92-22.35*coeff_noise_h)) * N_exposures_euclid # in e-.s-1.arcsec_2
sky_level_vis = (2.512 **(25.58-22.35)) * N_exposures_euclid # in e-.s-1.arcsec_2
sky_level_pixel_nir = [int((sky_level_nir_Y * 1800 * pixel_scale_euclid_nir**2)),
                        int((sky_level_nir_J* 1800 * pixel_scale_euclid_nir**2)),
                        int((sky_level_nir_H* 1800 * pixel_scale_euclid_nir**2))]# in e-/pixel/1800s
sky_level_pixel_vis = int((sky_level_vis * 1800 * pixel_scale_euclid_vis**2))# in e-/pixel/1800s

sky_level = sky_level_pixel_nir + [sky_level_pixel_vis] + sky_level_pixel_lsst


# Loading the COSMOS catalog
cosmos_cat = galsim.COSMOSCatalog('real_galaxy_catalog_25.2.fits', dir='/sps/lsst/users/barcelin/COSMOS_25.2_training_sample')

# function to check if S/N > 2
# Here we do the detection in R band of LSST
def SNR_peak_old(gal_noiseless,gal_noisy, band=6, snr_min=2):
    noise = np.std(gal_noisy[band]-gal_noiseless[band])
    max_img_noiseless = np.max(gal_noiseless[band])
    snr = np.abs(max_img_noiseless/noise)
    return (snr>snr_min), snr


#def SNR(gal_noiseless,gal_noisy, band=6, snr_min=5):
#    snr = np.sum(gal_noiseless[band]) / (np.std(gal_noisy[band]-gal_noiseless[band]) * np.prod(gal_noisy[band].shape))
#    return (snr>snr_min), snr


import multiprocessing
import time

def map_f(args):
    f, i, v = args
    v.value += 1
    return f(i)
    
def map(func, iter, verbose=True, timesleep=15.0, timeout=None):
    """
    Maps the function func over the iterator iter in a multi-threaded way using the multiprocessing package
    
    func must be pickable, see https://docs.python.org/2/library/pickle.html#what-can-be-pickled-and-unpickled
    
    """
    pool = multiprocessing.Pool()
    m = multiprocessing.Manager()
    v = m.Value(int, 0)
    
    inputs = ((func,i,v) for i in iter) #use a generator, so that nothing is computed before it's needed :)
    
    res = pool.map_async(map_f, inputs)
    
    try :
        n = len(iter)
    except TypeError : # if iter is a generator
        n = None

    if verbose :
        while (True):
            if (res.ready()): break
            # remaining = res._number_left
            # print "Waiting for", remaining, "task chunks to complete..."
            print("# castor.parallel.map : tasks accomplished out of {0} : {1}".format(n, v.get()))
            time.sleep(timesleep)

    pool.close()
    m.shutdown()

    return res.get(timeout)

count = 0
N_cosmo = 10000
N_per_gal = 5

ud = galsim.UniformDeviate()
rng = galsim.BaseDeviate()

counter = 0

itr = np.arange(N_cosmo)

img_cube_list = []

import time


######## TRAINING SAMPLE 


def func(ind):
    nb_blended_gal = np.random.randint(1,5)
    galaxy_noiseless, galaxy_noisy, blend_noisy = blend_generator(cosmos_cat, nb_blended_gal, 'training')
    if (SNR_peak(galaxy_noiseless, galaxy_noisy) == True):
        return np.array((galaxy_noiseless, blend_noisy))
    else:
        return func(ind+1) 

# debut = time.time()

# img_cube_list = map(func, itr,timesleep = 10.0)# 
# print(len(img_cube_list, len(img_cube_list[0])))

# fin = time.time()
# print('time : '+ str(fin-debut))

# np.save('/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni25/galaxies_blended_val_v5.npy', img_cube_list)



######## TEST SAMPLE 
i = 0
galaxies = []
shift_list = []
redshift_list  = []
magnitude_list = []
blendedness_lsst_list = []
blendedness_euclid_list = []
blendedness_total_euclid_list = []
blendedness_total_lsst_list = []
scale_radius_list = []
SNR_peak_list = []
SNR_peak_old_list = []
SNR_list = []

while (i < N_cosmo):
    print(i)
    nb_blended_gal = np.random.randint(1,N_per_gal)
    galaxy_noiseless, galaxy_noisy, blend_noiseless, blend_noisy, redshift, shift, mag, Blendedness_euclid, Blendedness_lsst,Blendedness_total_euclid, Blendedness_total_lsst, scale_radius = blend_generator(cosmos_cat, nb_blended_gal, 'test')
    if (SNR_peak_old(galaxy_noiseless, galaxy_noisy)[0] == True):
        galaxies.append((galaxy_noiseless, blend_noisy))
        shift_list.append((shift))
        redshift_list.append((redshift))
        magnitude_list.append((mag))
        blendedness_lsst_list.append((Blendedness_lsst))
        blendedness_euclid_list.append((Blendedness_euclid))
        blendedness_total_lsst_list.append((Blendedness_total_lsst))
        blendedness_total_euclid_list.append((Blendedness_total_euclid))
        scale_radius_list.append(scale_radius)
        SNR_peak_old_list.append(SNR_peak_old(galaxy_noiseless,galaxy_noisy)[1])
        SNR_peak_list.append(utils.SNR_peak(galaxy_noiseless,sky_level)[1])
        SNR_list.append(utils.SNR(galaxy_noiseless,sky_level)[1])
        i+=1

np.save('/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_COSMOS_test_v5.npy', galaxies)
np.save('/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_COSMOS_test_shift_v5.npy', shift_list)
np.save('/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_COSMOS_test_redshift_v5.npy', redshift_list)
np.save('/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_COSMOS_test_magnitude_v5.npy', magnitude_list)
np.save('/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_COSMOS_test_blendedness_lsst_v5.npy', blendedness_lsst_list)
np.save('/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_COSMOS_test_blendedness_euclid_v5.npy', blendedness_euclid_list)
np.save('/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_COSMOS_test_blendedness_total_lsst_v5.npy', blendedness_total_lsst_list)
np.save('/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_COSMOS_test_blendedness_total_euclid_v5.npy', blendedness_total_euclid_list)
np.save('/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_COSMOS_test_scale_radius_v5.npy', scale_radius_list)
np.save('/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_COSMOS_test_SNR_peak_old_v5.npy', SNR_peak_old_list)
np.save('/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_COSMOS_test_SNR_peak_v5.npy', SNR_peak_list)
np.save('/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_COSMOS_test_SNR_v5.npy', SNR_list)
