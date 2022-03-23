# Learnable Visual Markers

This repository contains code, trained weights of networks, as well as visualizations related to the course project on Machine learning at Skoltech in the spring of 2022.

Authors: Viacheslav Vasilev, Farukh Yaushev, Kezhik Kyzylool, Vera Soboleva


## Repository structure

**python** - contains all code for launching, training and testing networks both in the form of scripts and in the form of an ipynb notebook, which can be easily transferred to GoogleColab.

**trained_nets** - contains the trained weights of the networks used in the project.

**images** - contains the images of visual markers obtained by training the Synthesizer and Recognizer for different lengths of the input bit string.

**photos** - contains code and visualizations of experiments on recognition by the Recognizer network (can be found in `trained_nets/rec_net.pth`) of markers obtained by photographing. Here the Recognizer network was trained in conjunction with the Synthesizer (`trained_nets/synt_net.pth`) and Rendering network (`trained_nets/rend_net.pth`).

To start the training and testing process, go to `python/` and run `python pipeline.py`.

***Attention***: Running using a superimposing requires a `background` directory with image data in the top directory (here). The archive can be found [here](https://drive.google.com/drive/folders/1nQolB0GQWXROYKacWqg0OzaPPw9V6Uxy).
