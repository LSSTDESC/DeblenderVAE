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
import random

path, filename = os.path.split('__file__')    
datapath = galsim.meta_data.share_dir
datapath2 = os.path.abspath(os.path.join(path,'/sps/lsst/users/barcelin/EUCLID_Filters/'))
#print('passed here')
# initialize (pseudo-)random number generator

        # read in the Euclid NIR filters
filter_names_euclid_nir = 'HJY'
filter_names_euclid_vis = 'V'

# read in the LSST filters
filter_names_lsst = 'ugrizy'
filters = {}
#print('passed here 2')

filter_names_all = 'HJYVugrizy'

for filter_name in filter_names_euclid_nir:
    filter_filename = os.path.join(datapath2, 'Euclid_NISP0.{0}.dat'.format(filter_name))
    filters[filter_name] = galsim.Bandpass(filter_filename, wave_type='Angstrom')
    filters[filter_name] = filters[filter_name].thin(rel_err=1e-4)

for filter_name in filter_names_euclid_vis:
    filter_filename = os.path.join(datapath2, 'Euclid_VIS.dat'.format(filter_name))
    filters[filter_name] = galsim.Bandpass(filter_filename, wave_type='Angstrom')
    filters[filter_name] = filters[filter_name].thin(rel_err=1e-4)

for filter_name in filter_names_lsst:
    filter_filename = os.path.join(datapath, 'bandpasses/LSST_{0}.dat'.format(filter_name))
    filters[filter_name] = galsim.Bandpass(filter_filename, wave_type='nm')
    filters[filter_name] = filters[filter_name].thin(rel_err=1e-4)

pixel_scale_lsst = 0.2 # arcseconds # LSST Science book
pixel_scale_euclid_nir = 0.3 # arcseconds # Euclid Science book
pixel_scale_euclid_vis = 0.1 # arcseconds # Euclid Science book


#random_seed = 1234567
rng = galsim.BaseDeviate(None)

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

sky_level_pixel_lsst = [int((sky_level_lsst_u* 15 * pixel_scale_lsst**2)),
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

########### PSF #####################
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
def lsst_PSF():
    #Fig 1 : https://arxiv.org/pdf/0805.2366.pdf
    mu = -0.43058681997903414 # np.log(0.65)
    sigma = 0.3404334041976153  # Fixed to have corresponding percentils as in paper
    p_unnormed = lambda x : (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
                    / (x * sigma * np.sqrt(2 * np.pi)))#((1/(2*z0))*((z/z0)**2)*np.exp(-z/z0))
    p_normalization = scipy.integrate.quad(p_unnormed, 0., np.inf)[0]
    p = lambda z : p_unnormed(z) / p_normalization

    from scipy import stats
    class PSF_distribution(stats.rv_continuous):
        def __init__(self):
            super(PSF_distribution, self).__init__()
            self.a = 0.
            self.b = 10.
        def _pdf(self, x):
            return p(x)

    pdf = PSF_distribution()
    return 0.65#pdf.rvs()

#fwhm_lsst = 0.65 # pdf.rvs() ## Fixed at median value : Fig 1 : https://arxiv.org/pdf/0805.2366.pdf

fwhm_euclid_nir = 0.22 # EUCLID PSF is supposed invariant (no atmosphere) despite the optical and wavelengths variations
fwhm_euclid_vis = 0.18 # EUCLID PSF is supposed invariant (no atmosphere) despite the optical and wavelengths variations
beta = 2.5
#PSF_lsst = galsim.Kolmogorov(fwhm=fwhm_lsst)#galsim.Moffat(fwhm=fwhm_lsst, beta=beta)
PSF_euclid_nir = galsim.Moffat(fwhm=fwhm_euclid_nir, beta=beta)
PSF_euclid_vis = galsim.Moffat(fwhm=fwhm_euclid_vis, beta=beta)


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

############# SIZE OF STAMPS ################
# The stamp size of NIR instrument is taken equal to the one of LSST to have a nb of pixels which is 
# an integer and in the same time the max_stamp_size a power of 2 (works better for FFT)
# The physical size in NIR instrument is then larger than in the other stamps but the information needed
# for the VAE and deblender is contained in the stamp.
phys_stamp_size = 6.4 # Arcsecond
lsst_stamp_size = int(phys_stamp_size/pixel_scale_lsst) # Nb of pixels
nir_stamp_size = int(phys_stamp_size/pixel_scale_euclid_nir)+1 # Nb of pixels
vis_stamp_size = int(phys_stamp_size/pixel_scale_euclid_vis) # Nb of pixels

max_stamp_size = np.max((lsst_stamp_size,nir_stamp_size,vis_stamp_size))


# Generation function
def Gal_generator_noisy_pix_same(cosmos_cat):
    count = 0
    try:
        ############## SHAPE OF THE GALAXY ##################
        ud = galsim.UniformDeviate()#cosmos_cat.nobjects -10000-1
        gal = cosmos_cat.makeGalaxy(random.randint(cosmos_cat.nobjects-10000 -1, cosmos_cat.nobjects - 5000 -1), gal_type='parametric', chromatic=True, noise_pad_size = 0)

        gal = gal.rotate(ud() * 360. * galsim.degrees)
        redshift = gal.SED.redshift
        scale_radius = get_scale_radius(gal)
        
        ############ LUMINOSITY ############# 
        # The luminosity is multiplied by the ratio of the noise in the LSST R band and the assumed cosmos noise             
        bdgal_lsst =  (15. * (6.68**2)/((2.4**2)*(1.-0.33**2))) * gal * N_exposures_lsst
        bdgal_euclid_nir =  (1800. * ((1.25)**2 - (0.37)**2)/((2.4**2)*(1.-0.33**2))) * gal * N_exposures_euclid
        bdgal_euclid_vis =  (1800. * ((1.25)**2 - (0.37)**2)/((2.4**2)*(1.-0.33**2))) * gal * N_exposures_euclid         
        
        galaxy_noiseless = np.zeros((10,max_stamp_size,max_stamp_size))
        galaxy_noisy = np.zeros((10,max_stamp_size,max_stamp_size))

        # LSST PSF
        fwhm_lsst = lsst_PSF()
        PSF_lsst = galsim.Kolmogorov(fwhm=fwhm_lsst)

        # i = 0
        # for filter_name, filter_ in filters.items():
        for i, filter_name in enumerate(filter_names_all):
            if (i < 3):
                poissonian_noise_nir = galsim.PoissonNoise(rng, sky_level=sky_level_pixel_nir[i])
                img = galsim.ImageF(max_stamp_size,max_stamp_size, scale=pixel_scale_euclid_nir)  
                bdfinal = galsim.Convolve([bdgal_euclid_nir, PSF_euclid_nir])
                bdfinal.drawImage(filters[filter_name], image=img)
                # Noiseless galaxy
                galaxy_noiseless[i] = img.array.data
                # Noisy galaxy
                img.addNoise(poissonian_noise_nir)
                galaxy_noisy[i] = img.array.data
            else:
                if (i==3):
                    poissonian_noise_vis = galsim.PoissonNoise(rng, sky_level=sky_level_pixel_vis)
                    img = galsim.ImageF(max_stamp_size,max_stamp_size, scale=pixel_scale_euclid_vis)  
                    bdfinal = galsim.Convolve([bdgal_euclid_vis, PSF_euclid_vis])
                    bdfinal.drawImage(filters[filter_name], image=img)
                    # Noiseless galaxy
                    galaxy_noiseless[i] = img.array.data
                    # Noisy galaxy
                    img.addNoise(poissonian_noise_vis)
                    galaxy_noisy[i] = img.array.data
                else:
                    poissonian_noise_lsst = galsim.PoissonNoise(rng, sky_level=sky_level_pixel_lsst[i-4])
                    img = galsim.ImageF(max_stamp_size,max_stamp_size, scale=pixel_scale_lsst)  
                    bdfinal = galsim.Convolve([bdgal_lsst, PSF_lsst])
                    bdfinal.drawImage(filters[filter_name], image=img)
                    # Noiseless galaxy
                    galaxy_noiseless[i] = img.array.data
                    # Noisy galaxy
                    img.addNoise(poissonian_noise_lsst)
                    galaxy_noisy[i]= img.array.data
        
        return galaxy_noiseless, galaxy_noisy, redshift, scale_radius
    except RuntimeError: 
            count +=1
            print("nb of error : "+str(count))
            return Gal_generator_noisy_pix_same(cosmos_cat)
