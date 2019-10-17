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

from cosmos_generation import Gal_generator_noisy_pix_same

sys.path.insert(0,'../tools_for_VAE/')
import tools_for_VAE
from tools_for_VAE import utils

# The script is used as, eg,
# # python main_single_generation_cosmos.py training 10 1000
# to produce 10 files in the training sample with 1000 images each.
training_or_test = 'training' #str(sys.argv[1])
N_files = 1 #int(sys.argv[2])
N_per_file = 10000#int(sys.argv[3])
#assert training_or_test in ['training', 'validation', 'test']
training_or_test = 'test'

# where to save images and data
save_dir = '/sps/lsst/users/barcelin/data/single/PSF_lsst_O.65/independant/' + training_or_test
# what to call those files
root = 'galaxies_isolated_20190927_'

# Loading the COSMOS catalog
cosmos_cat = galsim.COSMOSCatalog('real_galaxy_catalog_25.2.fits', dir='/sps/lsst/users/barcelin/COSMOS_25.2_training_sample')
# Select galaxies to keep for the test sample
if training_or_test == 'test':
    used_idx = np.arange(5000)
else:
    used_idx = np.arange(5000,cosmos_cat.nobjects)

for icat in trange(N_files):
    # Run params
    root_i = root+str(icat)

    galaxies = []
    df = pd.DataFrame(index=np.arange(N_per_file), columns=['redshift', 'moment_sigma', 'e1', 'e2', 'SNR', 'SNR_peak'])

    res = utils.apply_ntimes(Gal_generator_noisy_pix_same, N_per_file, (cosmos_cat, training_or_test, used_idx, 100))

    for i in trange(N_per_file):
        if training_or_test == 'test':
            gal_noiseless, gal_noisy, data = res[i]
            df.loc[i] = np.array(data)
        else:
            gal_noiseless, gal_noisy = res[i]
        galaxies.append((gal_noiseless,gal_noisy))

    # Save noisy blended images and denoised single central galaxy images
    np.save(os.path.join(save_dir, root_i+'_images.npy'), galaxies)
    # If the created sample is a test sample, also save the shifts and differents data
    if training_or_test == 'test':
        df.to_csv(os.path.join(save_dir, root_i+'_data.csv'), index=False)
