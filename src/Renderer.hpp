#ifndef RENDERER_HPP
#define RENDERER_HPP

#include <SFML/Graphics.hpp>
#include <functional>
#include <memory>
#include <string>

class Renderer {
public:
    Renderer(int width, int height);
    ~Renderer();

    bool isOpen() const;
    bool handleEvents();
    void clear();
    void draw();
    void updateFPSCounter(float fps);
    void updateTexture(const void* pixels);

    // Methods for mouse interaction
    void setZoomCallback(std::function<void(float, float, float)> callback);
    void setDragCallbacks(
        std::function<void(float, float)> startCallback,
        std::function<void(float, float)> updateCallback,
        std::function<void()> endCallback
    );
    sf::Vector2i getMousePosition() const;

private:
    sf::RenderWindow window;
    sf::Texture texture;
    sf::Sprite sprite;
    sf::Font font;
    std::unique_ptr<sf::Text> fpsText;
    
    // Callbacks for mouse events
    std::function<void(float, float, float)> zoomCallback;
    std::function<void(float, float)> dragStartCallback;
    std::function<void(float, float)> dragUpdateCallback;
    std::function<void()> dragEndCallback;
    
    static constexpr float ZOOM_IN_FACTOR = 1.5f;   // More gradual zoom in
    static constexpr float ZOOM_OUT_FACTOR = 0.75f; // More gradual zoom out
};

#endif // RENDERER_HPP
