#include <bits/stdc++.h>

#include <SFML/Graphics.hpp>

#include "nn.h"

NeuralNet nn("nn/mnist-96.4%.nn");

const int HEIGHT = 280;
const int WIDTH  = 280;

int main() {
    sf::RenderWindow window(sf::VideoMode(HEIGHT, WIDTH), "MNIST edtior");

    window.setFramerateLimit(30);

    sf::Image pixels;
    pixels.create(WIDTH, HEIGHT, sf::Color::White);

    sf::Texture texture;
    texture.loadFromImage(pixels);

    sf::Sprite canvas(texture);

    sf::CircleShape circle;

    float thickness = 10;

    circle.setFillColor(sf::Color::Transparent);
    circle.setRadius(thickness - 0.1f);
    circle.setOutlineThickness(2);
    circle.setOrigin(thickness - 1, thickness - 1);
    circle.setOutlineColor(sf::Color::Black);

    auto change_thickness = [&](float t) {
        thickness = t;
        circle.setRadius(t);
        circle.setOrigin(t + 1, t + 1);
    };

    while(window.isOpen()) {
        for(sf::Event event; window.pollEvent(event); ) {
            switch(event.type) {
                case sf::Event::Closed:
                    window.close();
                    break;
                default:
                    break;
            }
        }

        static sf::Vector2i prev_pos;
        static bool pressed = false;
        static bool paint = true;

        static sf::Color color = sf::Color::White;

        if(sf::Keyboard::isKeyPressed(sf::Keyboard::C)) {
            for(int x = 0; x < WIDTH; x++) {
                for(int y = 0; y < HEIGHT; y++) {
                    pixels.setPixel(x, y, sf::Color::White);
                }
            }
            
            texture.update(pixels);

        }

        if(sf::Keyboard::isKeyPressed(sf::Keyboard::S)) {
            paint = false;
        }

        if(sf::Keyboard::isKeyPressed(sf::Keyboard::E)) {
            paint = true;
            color = sf::Color::White;
        }

        if(sf::Keyboard::isKeyPressed(sf::Keyboard::R)) {
            paint = true;
            color = sf::Color::Black;
        }

        circle.setPosition(sf::Vector2f(sf::Mouse::getPosition(window)));

        if(paint && sf::Mouse::isButtonPressed(sf::Mouse::Button::Left)) {
        
            sf::Vector2i mouse_pos = sf::Mouse::getPosition(window);
            sf::Vector2i dir = mouse_pos - prev_pos;

            if(!pressed) {
                for(int x = mouse_pos.x - thickness - 1; x <= mouse_pos.x + thickness + 1; x++) {
                    for(int y = mouse_pos.y - thickness - 1; y <= mouse_pos.y + thickness + 1; y++) {
                        if(0 <= x && x < WIDTH && 0 <= y && y <= HEIGHT) {
                            int dx = mouse_pos.x - x;
                            int dy = mouse_pos.y - y;
                            if(dx * dx + dy * dy <= thickness * thickness) {
                                pixels.setPixel(x, y, color);
                            }
                        }
                    }
                }
            } else {
                for(int per = 1; per <= 100; per++) {
                    sf::Vector2i pos = prev_pos + (dir * per) / 100;
                    for(int x = pos.x - thickness; x <= pos.x + thickness; x++) {
                        for(int y = pos.y - thickness; y <= pos.y + thickness; y++) {
                            if(0 <= x && x < WIDTH && 0 <= y && y <= HEIGHT) {
                                int dx = pos.x - x;
                                int dy = pos.y - y;
                                if(dx * dx + dy * dy <= thickness * thickness) {
                                    pixels.setPixel(x, y, color);
                                }
                            }
                        }
                    }
                }
            }

            prev_pos = mouse_pos;
            pressed = true;

            texture.update(pixels);
            // std::cerr << mouse_pos.x << ' ' << mouse_pos.y << '\n';
        } else {
            pressed = false;
        }

        static float input[28 * 28];
        static float output[10];

        for(int x = 0; x < 28; x++) {
            for(int y = 0; y < 28; y++) {
                float sum = 0;
                for(int px = x * 10; px < x * 10 + 10; px++) {
                    for(int py = y * 10; py < y * 10 + 10; py++) {
                        if(pixels.getPixel(py, px) == sf::Color::Black) {
                            sum ++;
                        }
                    }
                }
                sum /= 100;

                input[x * 28 + y] = sum;
            }
        }

        // for(int i = 0; i < 28; i++) {
        //     for(int j = 0; j < 28; j++) {
        //         if(input[i * 28 + j] > 0.5) {
        //             std::cerr << "x";
        //         } else {
        //             std::cerr << " ";
        //         }
        //     }
        //     std::cerr << '\n';
        // }

        nn.forward(input, output);

        int label = 0;
        float value = output[0];
        for(int i = 1; i < 10; i++) {
            if(output[i] > value) {
                label = i;
                value = output[i];
            }
        }

        std::cerr << label << '\n';

        window.setTitle(std::to_string(label));

        window.clear();

        window.draw(canvas);
        window.draw(circle);

        window.display();
    }


}