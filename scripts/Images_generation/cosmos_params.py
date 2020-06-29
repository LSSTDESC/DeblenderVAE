# Import packages

import numpy as np
import sys
import os
import galsim

############# SIZE OF STAMPS ################
# The stamp size of NIR instrument is taken equal to the one of LSST to have a nb of pixels which is 
# an integer and in the same time the max_stamp_size a power of 2 (works better for FFT)
# The physical size in NIR instrument is then larger than in the other stamps but the information needed
# for the VAE and deblender is contained in the stamp.
pixel_scale_lsst = 0.2 # arcseconds # LSST Science book
pixel_scale_euclid_nir = 0.3 # arcseconds # Euclid Science book
pixel_scale_euclid_vis = 0.1 # arcseconds # Euclid Science book
pixel_scale = [pixel_scale_euclid_nir]*3 + [pixel_scale_euclid_vis] + [pixel_scale_lsst]*6

max_stamp_size = 64 #np.max((lsst_stamp_size,nir_stamp_size,vis_stamp_size))

#################### FILTERS ###################
filters = {}
euclid_filters_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data/EUCLID_Filters/')
lsst_filters_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data/share_galsim/bandpasses')

# read in the Euclid NIR filters
filter_names_euclid_nir = 'HJY'
filter_names_euclid_vis = 'V'

for filter_name in filter_names_euclid_nir:
    filter_filename = os.path.join(euclid_filters_dir, 'Euclid_NISP0.{0}.dat'.format(filter_name))
    filters[filter_name] = galsim.Bandpass(filter_filename, wave_type='Angstrom')
    filters[filter_name] = filters[filter_name].thin(rel_err=1e-4)

filter_filename = os.path.join(euclid_filters_dir, 'Euclid_VIS.dat')
filters['V'] = galsim.Bandpass(filter_filename, wave_type='Angstrom')
filters['V'] = filters[filter_name].thin(rel_err=1e-4)

# read in the LSST filters
filter_names_lsst = 'ugrizy'
for filter_name in filter_names_lsst:
    filter_filename = os.path.join(lsst_filters_dir, 'LSST_{0}.dat'.format(filter_name))
    filters[filter_name] = galsim.Bandpass(filter_filename, wave_type='nm')
    filters[filter_name] = filters[filter_name].thin(rel_err=1e-4)

filter_names_all = 'HJYVugrizy'

#################### NOISE ###################
# Poissonian noise according to sky_level
n_years = 1
N_exposures_lsst = [56, 80, 184, 184, 160, 160] #Over the ten years (https://arxiv.org/pdf/0805.2366.pdf)
N_exposures_euclid = 4

sky_level_lsst_u = (2.512 **(26.50-22.95)) * N_exposures_lsst[0] # in e-.s-1.arcsec_2
sky_level_lsst_g = (2.512 **(28.30-22.24)) * N_exposures_lsst[1] # in e-.s-1.arcsec_2
sky_level_lsst_r = (2.512 **(28.13-21.20)) * N_exposures_lsst[2] # in e-.s-1.arcsec_2
sky_level_lsst_i = (2.512 **(27.79-20.47)) * N_exposures_lsst[3] # in e-.s-1.arcsec_2
sky_level_lsst_z = (2.512 **(27.40-19.60)) * N_exposures_lsst[4] # in e-.s-1.arcsec_2
sky_level_lsst_y = (2.512 **(26.58-18.63)) * N_exposures_lsst[5] # in e-.s-1.arcsec_2

sky_level_pixel_lsst = [sky_level_lsst_u* 15 * pixel_scale_lsst**2,
                        sky_level_lsst_g* 15 * pixel_scale_lsst**2,
                        sky_level_lsst_r* 15 * pixel_scale_lsst**2,
                        sky_level_lsst_i* 15 * pixel_scale_lsst**2,
                        sky_level_lsst_z* 15 * pixel_scale_lsst**2,
                        sky_level_lsst_y* 15 * pixel_scale_lsst**2]# in e-/pixel/15s

# average background level for Euclid observations : 22.35 mAB.arcsec-2 in VIS (Consortium book) ##
# For NIR bands, a coefficient is applied : it is calculated by comparing magnitudes AB of one point in the
# sky to the magnitude AB in VIS on this point. The choosen point is (-30;30) in galactic coordinates 
# (EUCLID and LSST overlap on this point).https://irsa.ipac.caltech.edu/applications/BackgroundModel/
coeff_noise_y = (22.57/21.95)
coeff_noise_j = (22.53/21.95)
coeff_noise_h = (21.90/21.95)

sky_level_nir_Y = (2.512 **(24.25-22.35*coeff_noise_y)) * N_exposures_euclid # in e-.s-1.arcsec_2
sky_level_nir_J = (2.512 **(24.29-22.35*coeff_noise_j)) * N_exposures_euclid # in e-.s-1.arcsec_2
sky_level_nir_H = (2.512 **(24.92-22.35*coeff_noise_h)) * N_exposures_euclid # in e-.s-1.arcsec_2
sky_level_vis = (2.512 **(25.58-22.35)) * N_exposures_euclid # in e-.s-1.arcsec_2
sky_level_pixel_nir = [ sky_level_nir_Y * 450. * pixel_scale_euclid_nir**2,
                        sky_level_nir_J * 450. * pixel_scale_euclid_nir**2,
                        sky_level_nir_H * 450. * pixel_scale_euclid_nir**2] # in e-/pixel/1800s
sky_level_pixel_vis = sky_level_vis * 450. * pixel_scale_euclid_vis**2 # in e-/pixel/1800s

sky_level_pixel = sky_level_pixel_nir + [sky_level_pixel_vis] + sky_level_pixel_lsst

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

PSF = [PSF_euclid_nir]*3 + [PSF_euclid_vis] + [PSF_lsst]*6

#################### EXPOSURE AND LUMISOITY ###################
# The luminosity is multiplied by the ratio of the noise in the LSST R band and the assumed cosmos noise             
coeff_exp_euclid =  (450. * ((1.25)**2 - (0.37)**2)/((2.4**2)*(1.-0.33**2)))
coeff_exp_lsst =  (15. * (6.68**2)/((2.4**2)*(1.-0.33**2))) 
coeff_exp = [coeff_exp_euclid*N_exposures_euclid]*4 + [coeff_exp_lsst* N_exposures_lsst[0]]+ [coeff_exp_lsst* N_exposures_lsst[1]]+[coeff_exp_lsst* N_exposures_lsst[2]]+[coeff_exp_lsst* N_exposures_lsst[3]]+[coeff_exp_lsst* N_exposures_lsst[4]]+[coeff_exp_lsst* N_exposures_lsst[5]]











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


