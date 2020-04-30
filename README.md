# Deblender_VAE

Project to do deblending of galaxies (separation of overlapped galaxies on a survey image) using deep learning.
We use two networks:
- a variational autoencoder (Kingma 2014) to denoise isolated galaxy images.
- and another network, which has the same architecture as the VAE, to deblend the galaxies. In this network, only the encoder is trained, since the decoder is fixed: weights are fixed from those of the VAE's decoder.

This folder contains the scripts for the images generation, the VAE and deblender training and the differents plots and tests.

The images are generated with GalSim (https://github.com/GalSim-developers/GalSim, doc: http://galsim-developers.github.io/GalSim/_build/html/index.html) from parametric models fitted to real galaxies from the HST COSMOS catalog (which can be found from here: https://github.com/GalSim-developers/GalSim/wiki/RealGalaxy%20Data).

# Required packages
- tensorflow : 1.13.1 (or tensorflow-gpu)
- tensorflow-probability : 0.6.0
- galsim
