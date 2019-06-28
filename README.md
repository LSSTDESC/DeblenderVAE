# Deblender_VAE

Project to do deblending of galaxies (separation of overlapped galaxies on an image) using deep learning.
We use two networks:
- a variational autoencoder (Kingma 2014) to denoise images of single galaxy.
- then another network is used to deblend the galaxies. This network has a VAE architecture but train only the encoder since the decoder is fixed: it is the trained decoder from the previous VAE.

This folder contains the scripts for the images generation, the VAE, the deblender and the differents plots and tests.
