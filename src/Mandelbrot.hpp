#ifndef MANDELBROT_HPP
#define MANDELBROT_HPP

#include <vector>
#include <cstdint>

class Mandelbrot {
public:
    Mandelbrot(int width, int height, float centerX, float centerY, float zoom, int maxIterations);
    ~Mandelbrot();

    void update(float deltaTime);
    void compute();
    const std::uint8_t* getPixels() const { return pixels.data(); }
    
    // Methods for zoom and pan control
    void setZoomTarget(float x, float y, float zoomFactor);
    void startDrag(float x, float y);
    void updateDrag(float x, float y);
    void endDrag();
    bool isAnimating() const { return animating; }
    float getCurrentZoom() const { return zoom; }
    std::pair<float, float> getCurrentCenter() const { return {centerX, centerY}; }
    
    // Reset to initial state
    void reset();
    
    // Helper method to convert screen to fractal coordinates
    void screenToFractalCoords(float screenX, float screenY, float& fractalX, float& fractalY) const;

private:
    int width;
    int height;
    float centerX;
    float centerY;
    float zoom;
    int maxIterations;
    std::vector<std::uint8_t> pixels;

    // Members for smooth zooming and dragging
    float targetCenterX;
    float targetCenterY;
    float targetZoom;
    bool animating;
    bool isDragging;
    float dragStartX;
    float dragStartY;
    float dragStartCenterX;
    float dragStartCenterY;
    float dragStartFractalX;
    float dragStartFractalY;
    
    // Initial state values for reset
    const float initialCenterX;
    const float initialCenterY;
    const float initialZoom;
    
    static constexpr float ZOOM_SPEED = 6.0f;  // Increased for smoother zoom
    static constexpr float POSITION_SPEED = 6.0f;  // Matched with zoom speed
    static constexpr float DRAG_SENSITIVITY = 0.25f; // Reduce overall drag movement
};

#endif // MANDELBROT_HPP
