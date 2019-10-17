import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import logging
import galsim
import cmath as cm
import matplotlib.pyplot as plt
from multiprocess import *
import pandas as pd
from tqdm.auto import tqdm, trange

from blended_images import blend_generator

sys.path.insert(0,'../tools_for_VAE/')
import tools_for_VAE
from tools_for_VAE import utils

# The script is used as, eg,
# # python main_blended_generation_cosmos.py training 10 1000
# to produce 10 files in the training sample with 1000 images each.
training_or_test = 'test' #str(sys.argv[1])
N_files = 1 #int(sys.argv[2])
N_per_file = 10000 #int(sys.argv[3])
assert training_or_test in ['training', 'validation', 'test']

# where to save images and data
save_dir = '/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/test/' #+ training_or_test
# what to call those files
root = 'galaxies_blended_20191004_test_01'

# Loading the COSMOS catalog
cosmos_cat = galsim.COSMOSCatalog('real_galaxy_catalog_25.2.fits', dir='/sps/lsst/users/barcelin/COSMOS_25.2_training_sample') #dir=os.path.join(galsim.meta_data.share_dir,'COSMOS_25.2_training_sample'))#
# Select galaxies to keep for the test sample
if training_or_test == 'test':
    used_idx = np.arange(5000)
else:
    used_idx = np.arange(5000,cosmos_cat.nobjects)

nmax_blend = 4 # total number of galaxies per image

# keys for data objects
keys = ['nb_blended_gal', 'SNR', 'SNR_peak', 'redshift', 'moment_sigma', 'e1', 'e2', 'mag', 'mag_ir', 'closest_x', 'closest_y', 'closest_redshift', 'closest_moment_sigma', 'closest_e1', 'closest_e2', 'closest_mag','closest_mag_ir', 'blendedness_total_lsst', 'blendedness_total_euclid', 'blendedness_closest_lsst', 'blendedness_closest_euclid']

for icat in trange(N_files):
    # Run params
    root_i = root+str(icat)

    galaxies = []
    shifts = []
    if training_or_test == 'test':
        df = pd.DataFrame(index=np.arange(N_per_file), columns=keys)

    res = utils.apply_ntimes(blend_generator, N_per_file, (cosmos_cat, training_or_test, used_idx, nmax_blend, 100))

    for i in trange(N_per_file):
        if training_or_test == 'test':
            gal_noiseless, blend_noisy, data, shift = res[i]
            assert set(data.keys()) == set(keys)
            df.loc[i] = [data[k] for k in keys]
            shifts.append(shift)
        else:
            gal_noiseless, blend_noisy = res[i]
        galaxies.append((gal_noiseless, blend_noisy))

    np.save(os.path.join(save_dir, root_i+'_images.npy'), galaxies)
    if training_or_test == 'test':
        df.to_csv(os.path.join(save_dir, root_i+'_data.csv'), index=False)
        np.save(os.path.join(save_dir, root_i+'_shifts.npy'), shifts)

# # while (i < N_per_file):
# for i in trange(N_per_file):
#     # try:
#     gal_noiseless, gal_noisy, redshift, hlr, SNR, SNR_peak = Gal_generator_noisy_pix_same(cosmos_cat, used_idx=used_idx, max_try=100)
#     # if (SNR_peak_old(gal_noiseless,gal_noisy)[0] == True):
#     galaxies.append((gal_noiseless,gal_noisy))
#     # hlr_list.append((hlr))
#     # # SNR_peak_old_list.append(SNR_peak_old(gal_noiseless,gal_noisy)[1])
#     # SNR_peak_list.append(SNR_peak)
#     # SNR_list.append(SNR)
#     # i+=1
#     df.loc[i] = np.array([redshift, hlr, SNR, SNR_peak])

# ######## TEST SAMPLE 
# i = 0
# galaxies = []
# shift_list = []
# redshift_list  = []
# magnitude_list = []
# blendedness_lsst_list = []
# blendedness_euclid_list = []
# blendedness_total_euclid_list = []
# blendedness_total_lsst_list = []
# scale_radius_list = []
# SNR_peak_list = []
# SNR_peak_old_list = []
# SNR_list = []

# while (i < N_per_file):
#     print(i)
#     nb_blended_gal = np.random.randint(1,N_per_gal)
#     galaxy_noiseless, galaxy_noisy, blend_noiseless, blend_noisy, redshift, shift, mag, Blendedness_euclid, Blendedness_lsst,Blendedness_total_euclid, Blendedness_total_lsst, scale_radius = blend_generator(cosmos_cat, nb_blended_gal, 'test')
#     if (SNR_peak_old(galaxy_noiseless, galaxy_noisy)[0] == True):
#         galaxies.append((galaxy_noiseless, blend_noisy))
#         shift_list.append((shift))
#         redshift_list.append((redshift))
#         magnitude_list.append((mag))
#         blendedness_lsst_list.append((Blendedness_lsst))
#         blendedness_euclid_list.append((Blendedness_euclid))
#         blendedness_total_lsst_list.append((Blendedness_total_lsst))
#         blendedness_total_euclid_list.append((Blendedness_total_euclid))
#         scale_radius_list.append(scale_radius)
#         SNR_peak_old_list.append(SNR_peak_old(galaxy_noiseless,galaxy_noisy)[1])
#         SNR_peak_list.append(utils.SNR_peak(galaxy_noiseless,sky_level)[1])
#         SNR_list.append(utils.SNR(galaxy_noiseless,sky_level)[1])
#         i+=1

# np.save('/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_COSMOS_test_v5.npy', galaxies)
# np.save('/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_COSMOS_test_shift_v5.npy', shift_list)
# np.save('/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_COSMOS_test_redshift_v5.npy', redshift_list)
# np.save('/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_COSMOS_test_magnitude_v5.npy', magnitude_list)
# np.save('/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_COSMOS_test_blendedness_lsst_v5.npy', blendedness_lsst_list)
# np.save('/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_COSMOS_test_blendedness_euclid_v5.npy', blendedness_euclid_list)
# np.save('/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_COSMOS_test_blendedness_total_lsst_v5.npy', blendedness_total_lsst_list)
# np.save('/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_COSMOS_test_blendedness_total_euclid_v5.npy', blendedness_total_euclid_list)
# np.save('/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_COSMOS_test_scale_radius_v5.npy', scale_radius_list)
# np.save('/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_COSMOS_test_SNR_peak_old_v5.npy', SNR_peak_old_list)
# np.save('/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_COSMOS_test_SNR_peak_v5.npy', SNR_peak_list)
# np.save('/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_COSMOS_test_SNR_v5.npy', SNR_list)
