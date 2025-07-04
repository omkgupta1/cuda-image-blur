#include <iostream>
#include <opencv2/opencv.hpp>   // For image I/O
#include <cuda_runtime.h>

using namespace std;
using namespace cv;

__global__ void blurKernel(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  
    int y = blockIdx.y * blockDim.y + threadIdx.y;  

    if (x >= width || y >= height) return;

    for (int c = 0; c < channels; ++c) {
        int pixelSum = 0;
        int count = 0;

        // Simple 3x3 box blur
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    pixelSum += input[(ny * width + nx) * channels + c];
                    count++;
                }
            }
        }

        output[(y * width + x) * channels + c] = pixelSum / count;
    }
}
int main() {
    // Load image (in color)
    Mat image = imread("../data/input.jpg", IMREAD_COLOR);
    if (image.empty()) {
        cout << "Error: Image not found!" << endl;
        return -1;
    }

    int width = image.cols;
    int height = image.rows;
    int channels = image.channels();
    size_t imageSize = width * height * channels;

    // Allocate host memory
    unsigned char *h_input = image.data;
    unsigned char *h_output = new unsigned char[imageSize];

    // Allocate device memory
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);

    // Copy input image to device
    cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice);

    // Launch Kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);
    blurKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);

    // Save the output image
    Mat outputImage(height, width, image.type(), h_output);
    imwrite("../data/output.jpg", outputImage);
    cout << "Image blurred and saved as output.jpg!" << endl;

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_output;

    return 0;
}
