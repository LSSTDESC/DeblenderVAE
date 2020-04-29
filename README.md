# Deblender_VAE

Project to do deblending of galaxies (separation of overlapped galaxies on a survey image) using deep learning.
We use two networks:
- a variational autoencoder (Kingma 2014) to denoise isolated galaxy images.
- and another network, which has the same architecture as the VAE, to deblend the galaxies. In this network, only the encoder is trained, since the decoder is fixed: weights are fixed from those of the VAE's decoder.

This folder contains the scripts for the images generation, the VAE and deblender training and the differents plots and tests.
