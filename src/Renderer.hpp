#ifndef RENDERER_HPP
#define RENDERER_HPP

#include <SFML/Graphics.hpp>
#include <functional>
#include <memory>
#include <string>
#include <vector>

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
    void updateCoordinates(float x, float y, float zoom);

    // Methods for mouse interaction
    void setZoomCallback(std::function<void(float, float, float)> callback);
    void setDragCallbacks(
        std::function<void(float, float)> startCallback,
        std::function<void(float, float)> updateCallback,
        std::function<void()> endCallback
    );
    void setResetCallback(std::function<void()> callback);
    sf::Vector2i getMousePosition() const;

    // Minimap update methods
    void updateMinimapViewport(float centerX, float centerY, float scale);

private:
    void loadFont();
    
    sf::RenderWindow window;
    sf::Texture texture;
    sf::Sprite sprite;
    sf::Font font;
    std::unique_ptr<sf::Text> fpsText;
    std::unique_ptr<sf::Text> coordsText;
    
    // Reset button
    sf::RectangleShape resetButton;
    std::unique_ptr<sf::Text> resetButtonText;
    bool isMouseOverButton(float x, float y) const;
    
    // Callbacks for mouse events
    std::function<void(float, float, float)> zoomCallback;
    std::function<void(float, float)> dragStartCallback;
    std::function<void(float, float)> dragUpdateCallback;
    std::function<void()> dragEndCallback;
    std::function<void()> resetCallback;
    
    static constexpr float ZOOM_IN_FACTOR = 1.5f;   // More gradual zoom in
    static constexpr float ZOOM_OUT_FACTOR = 0.75f; // More gradual zoom out

    // Minimap members
    static constexpr int MINIMAP_SIZE = 200;
    sf::Texture minimapTexture;
    sf::Sprite minimapSprite;
    sf::RectangleShape minimapBackground;
    sf::RectangleShape minimapBorder;
    sf::RectangleShape minimapViewport;
    std::vector<std::uint8_t> minimapPixels;
    
    // Minimap methods
    void setupMinimap();
    void updateMinimapTexture();
};

#endif // RENDERER_HPP
