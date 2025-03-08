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
    , initialCenterX(centerX)
    , initialCenterY(centerY)
    , initialZoom(zoom)
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
    
    // Convert screen coordinates to fractal space directly
    fractalX = centerX + (screenX / width - 0.5f) * fractalWidth;
    fractalY = centerY + (screenY / height - 0.5f) * fractalHeight;
}

void Mandelbrot::setZoomTarget(float x, float y, float zoomFactor) {
    // Calculate new zoom level with safety limits
    float newZoom = zoom * zoomFactor;
    
    // Prevent extreme zoom levels that could cause numerical issues
    static const float MIN_ZOOM = 0.000001f;  // Minimum zoom level
    static const float MAX_ZOOM = 10000000.0f; // Maximum zoom level
    
    if (newZoom < MIN_ZOOM) {
        newZoom = MIN_ZOOM;
    } else if (newZoom > MAX_ZOOM) {
        newZoom = MAX_ZOOM;
    }
    
    targetZoom = newZoom;
    
    // Keep the current center - zooming into screen center
    targetCenterX = centerX;
    targetCenterY = centerY;
    
    // If we're dragging, immediately update zoom while maintaining drag
    if (isDragging) {
        zoom = targetZoom;
        // No need to recalculate drag coordinates since we're using relative movement
    } else {
        animating = true;
    }
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

    // Calculate movement as a percentage of screen dimensions
    float deltaXPercent = (x - dragStartX) / width;
    float deltaYPercent = (y - dragStartY) / height;

    const float BASE_MOVEMENT_SCALE = 4.0f;
    float zoomFactor = std::log2f(zoom + 1.0f) * 0.3f;  // Logarithmic scaling to prevent extreme speeds
    float adjustedScale = BASE_MOVEMENT_SCALE * zoomFactor;
    
    // Calculate movement in fractal coordinates with zoom-aware scale
    float aspectRatio = static_cast<float>(width) / height;
    float moveX = deltaXPercent * adjustedScale / aspectRatio;
    float moveY = deltaYPercent * adjustedScale;
    
    // Apply the scaled movement to the starting position
    centerX = dragStartCenterX - moveX;
    centerY = dragStartCenterY - moveY;

    // Update target position to match current position
    targetCenterX = centerX;
    targetCenterY = centerY;
}

void Mandelbrot::endDrag() {
    isDragging = false;
}

void Mandelbrot::reset() {
    // Stop any ongoing animation or drag
    animating = false;
    isDragging = false;
    
    // Reset to stored initial values
    centerX = initialCenterX;
    centerY = initialCenterY;
    zoom = initialZoom;
    
    // Update targets to match reset values
    targetCenterX = centerX;
    targetCenterY = centerY;
    targetZoom = zoom;
    
    // Force a recomputation
    compute();
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
