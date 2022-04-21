# FSGANPerformanceTesting
This repository contains code for class project for Deep Learning at New York University.

The first part of the code comes from the GitHub repo [YuvalNirkin](https://github.com/YuvalNirkin/fsgan). This repository contains the source code for the video face swapping and face reenactment method described in the paper:
> **FSGAN: Subject Agnostic Face Swapping and Reenactment**  
> *International Conference on Computer Vision (ICCV), Seoul, Korea, 2019*  
> Yuval Nirkin, Yosi Keller, Tal Hassner  
> [Paper](https://arxiv.org/pdf/1908.05932.pdf) &nbsp; [Video](https://www.youtube.com/watch?v=BsITEVX6hkE)
>
> **Abstract:** *We present Face Swapping GAN (FSGAN) for face swapping and reenactment. Unlike previous work, FSGAN is subject agnostic and can be applied to pairs of faces without requiring training on those faces. To this end, we describe a number of technical contributions. We derive a novel recurrent neural network (RNN)–based approach for face reenactment which adjusts for both pose and expression variations and can be applied to a single image or a video sequence. For video sequences, we introduce continuous interpolation of the face views based on reenactment, Delaunay Triangulation, and barycentric coordinates. Occluded face regions are handled by a face completion network. Finally, we use a face blending network for seamless blending of the two faces while preserving target skin color and lighting conditions. This network uses a novel Poisson blending loss which combines Poisson optimization with perceptual loss. We compare our approach to existing state-of-the-art systems and show our results to be both qualitatively and quantitatively superior.*


## Requirements
- High-end NVIDIA GPUs with at least 11GB of DRAM.
- Either Linux or Windows. We recommend Linux for better performance.
- CUDA Toolkit 10.1, CUDNN 7.5, and the latest NVIDIA driver.
- Python 3.6+ and PyTorch 1.4.0+.


## Installation
Please refer to the link above for detailed installation guide. Please note that the author requires to fill out [this form](https://docs.google.com/forms/d/e/1FAIpQLScyyNWoFvyaxxfyaPLnCIAxXgdxLEMwR9Sayjh3JpWseuYlOA/viewform?usp=sf_link) to gain access to the weight files used in this project. 
