#include "Renderer.hpp"
#include <iostream>
#include <filesystem>
#include <optional>

Renderer::Renderer(int width, int height)
    : window(sf::VideoMode({static_cast<unsigned int>(width), static_cast<unsigned int>(height)}), "Mandelbrot Set")
    , texture(sf::Vector2u(width, height))
    , sprite(texture) // Initialize sprite with texture
    , font()
    , fpsText(nullptr) // Initialize as nullptr until font is loaded
    , zoomCallback(nullptr)
    , dragStartCallback(nullptr)
    , dragUpdateCallback(nullptr)
    , dragEndCallback(nullptr)
{
    // Load font
    if (!font.openFromFile("C:\\Windows\\Fonts\\arial.ttf")) {
        if (!font.openFromFile("C:\\Windows\\Fonts\\segoeui.ttf")) {
            std::cerr << "Error loading font" << std::endl;
            return;
        }
    }

    // Initialize text properties
    fpsText = std::make_unique<sf::Text>(font);
    fpsText->setCharacterSize(24);
    fpsText->setFillColor(sf::Color::White);
    fpsText->setPosition({10.f, 10.f});
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
        else if (const auto* keyPressed = event->getIf<sf::Event::KeyPressed>()) {
            if (keyPressed->scancode == sf::Keyboard::Scancode::Escape) {
                window.close();
                return false;
            }
        }
        else if (const auto* mouseWheel = event->getIf<sf::Event::MouseWheelScrolled>()) {
            if (zoomCallback) {
                auto mousePos = sf::Mouse::getPosition(window);
                float zoomFactor = (mouseWheel->delta > 0) ? ZOOM_OUT_FACTOR : ZOOM_IN_FACTOR;
                zoomCallback(static_cast<float>(window.getSize().x / 2), 
                           static_cast<float>(window.getSize().y / 2), 
                           zoomFactor);
            }
        }
        else if (const auto* mouseButton = event->getIf<sf::Event::MouseButtonPressed>()) {
            if (mouseButton->button == sf::Mouse::Button::Left && dragStartCallback) {
                auto pos = sf::Mouse::getPosition(window);
                dragStartCallback(static_cast<float>(pos.x),
                                static_cast<float>(pos.y));
            }
        }
        else if (const auto* mouseButton = event->getIf<sf::Event::MouseButtonReleased>()) {
            if (mouseButton->button == sf::Mouse::Button::Left && dragEndCallback) {
                dragEndCallback();
            }
        }
        else if (const auto* mouseMoved = event->getIf<sf::Event::MouseMoved>()) {
            if (dragUpdateCallback && sf::Mouse::isButtonPressed(sf::Mouse::Button::Left)) {
                dragUpdateCallback(static_cast<float>(mouseMoved->position.x),
                                 static_cast<float>(mouseMoved->position.y));
            }
        }
    }
    return true;
}

void Renderer::clear() {
    window.clear(sf::Color::Black);
}

void Renderer::draw() {
    window.draw(sprite);
    if (fpsText) {
        window.draw(*fpsText);
    }
    window.display();
}

void Renderer::updateFPSCounter(float fps) {
    if (fpsText) {
        fpsText->setString("FPS: " + std::to_string(static_cast<int>(fps)));
    }
}

void Renderer::updateTexture(const void* pixels) {
    texture.update(static_cast<const std::uint8_t*>(pixels));
}
