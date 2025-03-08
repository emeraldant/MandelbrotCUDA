#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

__device__ void hsv_to_rgb(float h, float s, float v, float& r, float& g, float& b) {
    if (s <= 0.0f) {
        r = g = b = v;
        return;
    }

    h = fmodf(h, 360.0f) / 60.0f;
    int i = static_cast<int>(h);
    float f = h - i;
    float p = v * (1.0f - s);
    float q = v * (1.0f - s * f);
    float t = v * (1.0f - s * (1.0f - f));

    switch (i) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        default: r = v; g = p; b = q; break;
    }
}

__global__ void mandelbrotKernel(uint8_t* pixels, int width, int height,
                                float centerX, float centerY,
                                float scale, int maxIterations)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    int idx = (py * width + px) * 4;

    // Map pixel to complex plane
    float x0 = centerX + (px - width / 2.0f) * (2.0f * scale / width);
    float y0 = centerY + (py - height / 2.0f) * (2.0f * scale / width);

    float x = 0.0f, y = 0.0f;
    int iteration = 0;
    const float bailout = 65536.0f;

    // Main iteration
    while (x * x + y * y <= bailout && iteration < maxIterations) {
        float x_temp = x * x - y * y + x0;
        y = 2.0f * x * y + y0;
        x = x_temp;
        iteration++;
    }

    // Smooth coloring
    float smoothIteration = iteration;
    if (iteration < maxIterations) {
        float log_zn = logf(x * x + y * y) / 2.0f;
        float nu = logf(log_zn / logf(2.0f)) / logf(2.0f);
        smoothIteration = iteration + 1 - nu;
    }

    if (iteration == maxIterations) {
        // Inside set - black
        pixels[idx + 0] = 0;
        pixels[idx + 1] = 0;
        pixels[idx + 2] = 0;
        pixels[idx + 3] = 255;
    } else {
        // Enhanced color mapping
        const float colorFactor = 7.0f;  // Increased for more color variation
        float hue = fmodf(smoothIteration * colorFactor, 360.0f);
        
        // Adjust saturation based on iteration count for more depth
        float saturation = 0.8f + 0.2f * (float)iteration / maxIterations;
        
        // Value (brightness) varies with iteration count
        float value = 0.7f + 0.3f * (float)iteration / maxIterations;
        
        float r, g, b;
        hsv_to_rgb(hue, saturation, value, r, g, b);

        // Write RGBA values
        pixels[idx + 0] = static_cast<uint8_t>(r * 255);
        pixels[idx + 1] = static_cast<uint8_t>(g * 255);
        pixels[idx + 2] = static_cast<uint8_t>(b * 255);
        pixels[idx + 3] = 255;
    }
}

extern "C" void computeMandelbrotCUDA(uint8_t* pixels,
                                   int width, int height,
                                   float centerX, float centerY,
                                   float scale, int maxIterations)
{
    const int threadsPerBlockX = 16;
    const int threadsPerBlockY = 16;
    
    dim3 threadsPerBlock(threadsPerBlockX, threadsPerBlockY);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    uint8_t* d_pixels = nullptr;
    size_t pitch = 0;
    cudaError_t err;
    
    // Allocate device memory with error checking
    err = cudaMallocPitch(&d_pixels, &pitch, width * 4, height);
    if (err != cudaSuccess) {
        printf("CUDA memory allocation error: %s\n", cudaGetErrorString(err));
        return; // Early return on allocation failure
    }
    
    // Launch kernel with L1 cache preference for better performance
    cudaFuncSetCacheConfig(mandelbrotKernel, cudaFuncCachePreferL1);
    
    mandelbrotKernel<<<numBlocks, threadsPerBlock>>>(d_pixels, width, height,
                                                   centerX, centerY,
                                                   scale, maxIterations);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        cudaFree(d_pixels);
        return; // Early return on kernel error
    }
    
    // Copy result back to host using pitched memory
    err = cudaMemcpy2D(pixels, width * 4, d_pixels, pitch,
                width * 4, height, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA memory copy error: %s\n", cudaGetErrorString(err));
    }
    
    // Free device memory
    cudaFree(d_pixels);
    
    // Final synchronization
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA synchronization error: %s\n", cudaGetErrorString(err));
    }
}
