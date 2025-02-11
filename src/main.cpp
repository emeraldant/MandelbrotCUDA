#include "Mandelbrot.hpp"
#include "Renderer.hpp"
#include <SFML/System/Clock.hpp>
#include <iostream>

int main() {
    const int windowWidth = 2560;   
    const int windowHeight = 1440;  

    Renderer renderer(windowWidth, windowHeight);
    
    Mandelbrot fractal(windowWidth, windowHeight, -0.75f, 0.1f, 1.5f, 25000);

    renderer.setZoomCallback([&fractal](float x, float y, float factor) {
        fractal.setZoomTarget(x, y, factor);
    });
    
    renderer.setDragCallbacks(
        [&fractal](float x, float y) { fractal.startDrag(x, y); },
        [&fractal](float x, float y) { fractal.updateDrag(x, y); },
        [&fractal]() { fractal.endDrag(); }
    );
    
    renderer.setResetCallback([&fractal]() {
        fractal.reset();
    });

    sf::Clock clock;
    float lastTime = 0.0f;
    
    while (renderer.isOpen()) {
        float currentTime = clock.getElapsedTime().asSeconds();
        float deltaTime = currentTime - lastTime;
        lastTime = currentTime;
        
        if (!renderer.handleEvents()) {
            break;
        }
        
        fractal.update(deltaTime);
        fractal.compute();
        
        // Update coordinate display
        auto [centerX, centerY] = fractal.getCurrentCenter();
        renderer.updateCoordinates(centerX, centerY, fractal.getCurrentZoom());
        
        renderer.updateTexture(fractal.getPixels());
        renderer.updateFPSCounter(1.0f / deltaTime);
        renderer.clear();
        renderer.draw();
    }

    return 0;
}
