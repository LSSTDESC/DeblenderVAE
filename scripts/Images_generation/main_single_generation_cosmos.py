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
from cosmos_generation import Gal_generator_noisy_pix_same
from multiprocess import *

sys.path.insert(0,'../tools_for_VAE/')
import tools_for_VAE
from tools_for_VAE import utils

# Parameters to fix
phys_stamp_size = 6.4 # arcsec
pixel_scale_euclid_vis = 0.1 # arcsec/pixel

stamp_size = int(phys_stamp_size/pixel_scale_euclid_vis)

# Loading the COSMOS catalog
cosmos_cat = galsim.COSMOSCatalog('real_galaxy_catalog_25.2.fits', dir='/sps/lsst/users/barcelin/COSMOS_25.2_training_sample')


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
N_per_gal = 1

counter = 0

itr = np.arange(N_cosmo)

img_cube_list = []

import time
debut = time.time()

print( 'test')
def func(ind):
    gal_noiseless, gal_noisy, redshift, scale_radius=Gal_generator_noisy_pix_same(cosmos_cat)
    if (SNR_peak_old(gal_noiseless,gal_noisy)[0] == True):
        return np.array((gal_noiseless,gal_noisy))
    else:
        return func(ind+1) 



#img_cube_list = map(func, itr,timesleep = 10.0)# 

# fin = time.time()
# print('time : '+ str(fin-debut))
i=0

galaxies = []
scale_radius_list = []
SNR_peak_old_list = []
SNR_peak_list = []
SNR_list = []

while (i < N_cosmo):
    print(i)
    gal_noiseless, gal_noisy, redshift, scale_radius=Gal_generator_noisy_pix_same(cosmos_cat)
    if (SNR_peak_old(gal_noiseless,gal_noisy)[0] == True):
        galaxies.append((gal_noiseless,gal_noisy))
        scale_radius_list.append((scale_radius))
        SNR_peak_old_list.append(SNR_peak_old(gal_noiseless,gal_noisy)[1])
        SNR_peak_list.append(utils.SNR_peak(gal_noiseless,sky_level)[1])
        SNR_list.append(utils.SNR(gal_noiseless,sky_level)[1])
        i+=1

fin = time.time()
print('time : '+ str(fin-debut))

np.save('/sps/lsst/users/barcelin/data/single/PSF_lsst_O.65/independant/SNR_7/galaxies_test_v5', galaxies)
np.save('/sps/lsst/users/barcelin/data/single/PSF_lsst_O.65/independant/SNR_7/scale_radius_test_v5', scale_radius_list)
np.save('/sps/lsst/users/barcelin/data/single/PSF_lsst_O.65/independant/SNR_7/SNR_peak_old_test_v5', SNR_peak_old_list)
np.save('/sps/lsst/users/barcelin/data/single/PSF_lsst_O.65/independant/SNR_7/SNR_peak_test_v5', SNR_peak_list)
np.save('/sps/lsst/users/barcelin/data/single/PSF_lsst_O.65/independant/SNR_7/SNR_test_v5', SNR_list)
#np.save('/sps/lsst/users/barcelin/data/single/PSF_lsst_O.65/independant/SNR_7/galaxies_COSMOS_val_SNR_7_test.npy', img_cube_list)
