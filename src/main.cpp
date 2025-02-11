#include "Mandelbrot.hpp"
#include "Renderer.hpp"
#include <SFML/System/Clock.hpp>
#include <iostream>

int main() {
    const int windowWidth = 1280;   // Increased resolution
    const int windowHeight = 720;   // 16:9 aspect ratio

    Renderer renderer(windowWidth, windowHeight);
    
    // Start at an interesting location with good detail potential
    Mandelbrot fractal(windowWidth, windowHeight, -0.75f, 0.1f, 1.5f, 1000);

    // Set up zoom callback
    renderer.setZoomCallback([&fractal](float x, float y, float zoomFactor) {
        fractal.setZoomTarget(x, y, zoomFactor);
    });

    // Set up drag callbacks
    renderer.setDragCallbacks(
        [&fractal](float x, float y) { fractal.startDrag(x, y); },
        [&fractal](float x, float y) { fractal.updateDrag(x, y); },
        [&fractal]() { fractal.endDrag(); }
    );

    sf::Clock clock;
    sf::Clock fpsClock;
    int frameCount = 0;
    
    while (renderer.isOpen()) {
        float deltaTime = clock.restart().asSeconds();
        
        // Handle events, quit if needed
        if (!renderer.handleEvents()) {
            break;
        }

        // Update fractal parameters
        fractal.update(deltaTime);
        
        // Compute new frame
        fractal.compute();
        
        // Update display
        renderer.updateTexture(fractal.getPixels());
        
        // FPS calculation
        frameCount++;
        if (fpsClock.getElapsedTime().asSeconds() >= 1.0f) {
            float fps = static_cast<float>(frameCount) / fpsClock.getElapsedTime().asSeconds();
            renderer.updateFPSCounter(fps);
            frameCount = 0;
            fpsClock.restart();
        }

        renderer.clear();
        renderer.draw();
    }

    return 0;
}
