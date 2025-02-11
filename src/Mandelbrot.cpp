#include "Mandelbrot.hpp"
#include "CudaKernels.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>

// CUDA kernel declaration (defined in Mandelbrot.cu)
void computeMandelbrotCUDA(std::uint8_t* d_pixels, int width, int height,
                          float centerX, float centerY, float zoom, int maxIterations);

Mandelbrot::Mandelbrot(int width, int height, float centerX, float centerY, float zoom, int maxIterations)
    : width(width)
    , height(height)
    , centerX(centerX)
    , centerY(centerY)
    , zoom(zoom)
    , maxIterations(maxIterations)
    , pixels(width * height * 4) // 4 bytes per pixel (RGBA)
    , targetCenterX(centerX)
    , targetCenterY(centerY)
    , targetZoom(zoom)
    , animating(false)
    , isDragging(false)
    , dragStartX(0.0f)
    , dragStartY(0.0f)
    , dragStartCenterX(0.0f)
    , dragStartCenterY(0.0f)
    , dragStartFractalX(0.0f)
    , dragStartFractalY(0.0f)
{
    // Initialize CUDA here if needed
}

Mandelbrot::~Mandelbrot() {
    // Cleanup CUDA resources if needed
}

void Mandelbrot::screenToFractalCoords(float screenX, float screenY, float& fractalX, float& fractalY) const {
    float aspectRatio = static_cast<float>(width) / height;
    float fractalWidth = 4.0f / zoom;
    float fractalHeight = fractalWidth / aspectRatio;
    
    // Convert screen coordinates to [-2, 2] range, accounting for aspect ratio
    fractalX = ((screenX / width) - 0.5f) * fractalWidth;
    fractalY = ((screenY / height) - 0.5f) * fractalHeight;
    
    // Offset by current center
    fractalX += centerX;
    fractalY += centerY;
}

void Mandelbrot::setZoomTarget(float x, float y, float zoomFactor) {
    if (isDragging) return; // Don't zoom while dragging
    
    // Calculate the point we want to zoom into in fractal coordinates
    float mouseX, mouseY;
    screenToFractalCoords(x, y, mouseX, mouseY);
    
    // Calculate new zoom level
    targetZoom = zoom * zoomFactor;
    
    // Calculate new center position to keep mouse point fixed
    float newFractalWidth = 4.0f / targetZoom;
    float newFractalHeight = newFractalWidth / (static_cast<float>(width) / height);
    
    targetCenterX = mouseX - (x - width / 2) * (newFractalWidth / width);
    targetCenterY = mouseY - (y - height / 2) * (newFractalHeight / height);
    
    animating = true;
}

void Mandelbrot::startDrag(float x, float y) {
    isDragging = true;

    // Store initial mouse position in screen coordinates
    dragStartX = x;
    dragStartY = y;

    // Store the current center position when drag starts
    dragStartCenterX = centerX;
    dragStartCenterY = centerY;

    // Calculate and store the fractal coordinate at the start of the drag
    screenToFractalCoords(x, y, dragStartFractalX, dragStartFractalY);

    // Stop any ongoing animation
    animating = false;
    targetZoom = zoom;
    targetCenterX = centerX;
    targetCenterY = centerY;
}

void Mandelbrot::updateDrag(float x, float y) {
    if (!isDragging) return;

    // Compute current fractal coordinates using dragStartCenter values
    float aspectRatio = static_cast<float>(width) / height;
    float fractalWidth = 4.0f / zoom;
    float fractalHeight = fractalWidth / aspectRatio;
    
    // Convert current screen position to fractal coordinates using dragStartCenter
    float currentFractalX = ((x / width) - 0.5f) * fractalWidth + dragStartCenterX;
    float currentFractalY = ((y / height) - 0.5f) * fractalHeight + dragStartCenterY;

    // Update center position based on the difference between start and current fractal coordinates
    centerX = dragStartCenterX + (dragStartFractalX - currentFractalX);
    centerY = dragStartCenterY + (dragStartFractalY - currentFractalY);

    // Update target position to match current position
    targetCenterX = centerX;
    targetCenterY = centerY;
}

void Mandelbrot::endDrag() {
    isDragging = false;
}

void Mandelbrot::update(float deltaTime) {
    if (!animating || isDragging) {
        return;
    }

    // Smooth interpolation for zoom
    float zoomDelta = (targetZoom - zoom) * ZOOM_SPEED * deltaTime;
    if (std::abs(zoomDelta) > std::abs(targetZoom - zoom)) {
        zoom = targetZoom;
    } else {
        zoom += zoomDelta;
    }

    // Smooth interpolation for position
    float xDelta = (targetCenterX - centerX) * POSITION_SPEED * deltaTime;
    float yDelta = (targetCenterY - centerY) * POSITION_SPEED * deltaTime;
    
    if (std::abs(xDelta) > std::abs(targetCenterX - centerX)) {
        centerX = targetCenterX;
    } else {
        centerX += xDelta;
    }
    
    if (std::abs(yDelta) > std::abs(targetCenterY - centerY)) {
        centerY = targetCenterY;
    } else {
        centerY += yDelta;
    }

    // Check if we've reached the target
    if (std::abs(zoom - targetZoom) < 0.001f &&
        std::abs(centerX - targetCenterX) < 0.0001f &&
        std::abs(centerY - targetCenterY) < 0.0001f) {
        animating = false;
    }
}

void Mandelbrot::compute() {
    // Call CUDA kernel to compute the Mandelbrot set
    computeMandelbrotCUDA(pixels.data(), width, height, centerX, centerY, zoom, maxIterations);
}
