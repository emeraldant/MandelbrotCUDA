#include "Renderer.hpp"
#include <iostream>
#include <filesystem>
#include <optional>
#include <sstream>

Renderer::Renderer(int width, int height)
    : window(sf::VideoMode({static_cast<unsigned int>(width), static_cast<unsigned int>(height)}), "Mandelbrot Set", sf::Style::Default)
    , texture(sf::Vector2u(width, height))  // Create texture with size in constructor
    , sprite(texture)  // Initialize sprite with texture
    , font()
    , fpsText(nullptr)
    , coordsText(nullptr)
    , resetButton()
    , resetButtonText(nullptr)
    , zoomCallback(nullptr)
    , dragStartCallback(nullptr)
    , dragUpdateCallback(nullptr)
    , dragEndCallback(nullptr)
    , resetCallback(nullptr)
{
    // Load font first
    if (!font.openFromFile("./resources/arial.ttf")) {
        // Fallback to system fonts
        if (!font.openFromFile("C:/Windows/Fonts/arial.ttf")) {
            if (!font.openFromFile("C:/Windows/Fonts/segoeui.ttf")) {
                throw std::runtime_error("Failed to load font");
            }
        }
    }
    
    // Create text objects after font is loaded
    fpsText = std::make_unique<sf::Text>(font);
    coordsText = std::make_unique<sf::Text>(font);
    resetButtonText = std::make_unique<sf::Text>(font);
    
    // Setup FPS counter
    fpsText->setCharacterSize(14);
    fpsText->setFillColor(sf::Color::Yellow);
    fpsText->setPosition({10.f, 10.f});
    
    // Setup coordinates display
    coordsText->setCharacterSize(14);
    coordsText->setFillColor(sf::Color::Yellow);
    coordsText->setPosition({10.f, 30.f});
    
    // Setup reset button
    resetButton.setSize({80.f, 30.f});
    resetButton.setPosition({static_cast<float>(width - 90), 10.f});
    resetButton.setFillColor(sf::Color(100, 100, 100));
    
    // Setup reset button text
    resetButtonText->setString("Reset");
    resetButtonText->setCharacterSize(14);
    resetButtonText->setFillColor(sf::Color::White);
    
    // Center the text in the button
    sf::FloatRect textBounds = resetButtonText->getLocalBounds();
    resetButtonText->setPosition({
        static_cast<float>(width - 90 + (80 - textBounds.size.x) / 2),
        10.f + (30.f - textBounds.size.y) / 2
    });
    
    // Center the window on the screen
    sf::VideoMode desktop = sf::VideoMode::getDesktopMode();
    window.setPosition(sf::Vector2i(
        (desktop.size.x  - width) / 2,
        (desktop.size.y - height) / 2
    ));
}

void Renderer::loadFont() {
    // Try to load font from resources directory
    if (!font.openFromFile("./resources/arial.ttf")) {
        // Fallback to system fonts
        if (!font.openFromFile("C:/Windows/Fonts/arial.ttf")) {
            if (!font.openFromFile("C:/Windows/Fonts/segoeui.ttf")) {
                throw std::runtime_error("Failed to load font");
            }
        }
    }
}

Renderer::~Renderer() {
    window.close();
}

bool Renderer::isOpen() const {
    return window.isOpen();
}

void Renderer::setZoomCallback(std::function<void(float, float, float)> callback) {
    zoomCallback = callback;
}

void Renderer::setDragCallbacks(
    std::function<void(float, float)> startCallback,
    std::function<void(float, float)> updateCallback,
    std::function<void()> endCallback
) {
    dragStartCallback = startCallback;
    dragUpdateCallback = updateCallback;
    dragEndCallback = endCallback;
}

sf::Vector2i Renderer::getMousePosition() const {
    return sf::Mouse::getPosition(window);
}

bool Renderer::handleEvents() {
    while (const std::optional<sf::Event> event = window.pollEvent()) {
        if (event->is<sf::Event::Closed>()) {
            window.close();
            return false;
        }
        else if (const auto* mousePress = event->getIf<sf::Event::MouseButtonPressed>()) {
            sf::Vector2f mousePos(mousePress->position);
            
            if (mousePress->button == sf::Mouse::Button::Left) {
                // Improved reset button detection
                if (resetButton.getGlobalBounds().contains(mousePos)) {
                    std::cout << "Reset button clicked!" << std::endl;
                    if (resetCallback) {
                        resetCallback();
                    } else {
                        std::cout << "Reset callback is not set!" << std::endl;
                    }
                    return true; 
                } else if (dragStartCallback) {
                    dragStartCallback(mousePos.x, mousePos.y);
                }
            }
            else if (mousePress->button == sf::Mouse::Button::Right) {
                if (zoomCallback) {
                    zoomCallback(mousePos.x, mousePos.y, ZOOM_OUT_FACTOR);
                }
            }
        }
        else if (const auto* mouseRelease = event->getIf<sf::Event::MouseButtonReleased>()) {
            if (mouseRelease->button == sf::Mouse::Button::Left) {
                sf::Vector2f mousePos(mouseRelease->position);
                if (dragEndCallback && !resetButton.getGlobalBounds().contains(sf::Vector2f(mousePos))) {
                    dragEndCallback();
                }
            }
        }
        else if (const auto* mouseMove = event->getIf<sf::Event::MouseMoved>()) {
            sf::Vector2f mousePos(mouseMove->position);
            
            if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Left)) {
                if (dragUpdateCallback && !resetButton.getGlobalBounds().contains(sf::Vector2f(mousePos))) {
                    dragUpdateCallback(mousePos.x, mousePos.y);
                }
            }
            
            // Update button hover state
            if (resetButton.getGlobalBounds().contains(sf::Vector2f(mousePos))) {
                resetButton.setFillColor(sf::Color(120, 120, 120));
            } else {
                resetButton.setFillColor(sf::Color(100, 100, 100));
            }
        }
        else if (const auto* mouseWheel = event->getIf<sf::Event::MouseWheelScrolled>()) {
            if (mouseWheel->wheel == sf::Mouse::Wheel::Vertical) {
                sf::Vector2f mousePos(mouseWheel->position);
                if (zoomCallback) {
                    float factor = mouseWheel->delta > 0 ? ZOOM_OUT_FACTOR : ZOOM_IN_FACTOR;
                    zoomCallback(mousePos.x, mousePos.y, factor);
                }
            }
        }
    }
    return true;
}

void Renderer::clear() {
    window.clear();
}

void Renderer::draw() {
    window.clear();
    window.draw(sprite);
    window.draw(resetButton);
    window.draw(*resetButtonText);
    window.draw(*fpsText);
    window.draw(*coordsText);
    window.display();
}

void Renderer::updateFPSCounter(float fps) {
    fpsText->setString("FPS: " + std::to_string(static_cast<int>(fps)));
}

void Renderer::updateCoordinates(float x, float y, float zoom) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(6) << "X: " << x << " Y: " << y << " Zoom: " << zoom;
    coordsText->setString(ss.str());
}

bool Renderer::isMouseOverButton(float x, float y) const {
    sf::FloatRect buttonBounds = resetButton.getGlobalBounds();
    return buttonBounds.contains(sf::Vector2f(x, y));
}

void Renderer::setResetCallback(std::function<void()> callback) {
    resetCallback = callback;
}

void Renderer::updateTexture(const void* pixels) {
    texture.update(static_cast<const std::uint8_t*>(pixels));
}
