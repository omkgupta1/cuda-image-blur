# cuda-image-blur
# CUDA Image Blurring (GPU Accelerated)

## Project Overview
This project implements a basic image blurring filter (Box Blur) using CUDA to utilize GPU acceleration for faster image processing.

## Features:
- Load an image using OpenCV
- Apply Box Blur using CUDA kernel
- Save blurred output image

## How to Run:
1. Install CUDA Toolkit & OpenCV.
2. Compile:
```bash
nvcc -o blur src/image_blur.cu `pkg-config --cflags --libs opencv4`
