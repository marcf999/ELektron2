#pragma once

#include "electron.h"
#include "physical_data.h"
#include <SFML/Graphics.hpp>
#include <cmath>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>

class PlotDots {
public:
    static void show(Electron& electron) {
        const auto& cam = electron.stateCamera;
        int n = (int)cam.size();
        if (n < 2) return;

        // Extract q (mass) and r (charge) positions in physical meters
        std::vector<double> qx(n), qz(n), rx(n), rz(n);
        for (int i = 0; i < n; i++) {
            qx[i] = cam[i][QX] * PhysicalData::zitterRadius;
            qz[i] = cam[i][QZ] * PhysicalData::zitterRadius;
            rx[i] = cam[i][RX] * PhysicalData::zitterRadius;
            rz[i] = cam[i][RZ] * PhysicalData::zitterRadius;
        }

        // Auto-range
        double maxVal = 1e-14;
        for (int i = 0; i < n; i++) {
            maxVal = std::max(maxVal, std::abs(qx[i]));
            maxVal = std::max(maxVal, std::abs(qz[i]));
            maxVal = std::max(maxVal, std::abs(rx[i]));
            maxVal = std::max(maxVal, std::abs(rz[i]));
        }
        double range = maxVal * 1.1;

        // Window title
        std::ostringstream titleSS;
        titleSS << "ELektron2 | E=" << std::fixed << std::setprecision(0) << electron.getKineticEnergy()
                << "eV | Angle=" << (int)electron.getAngle() << "\xC2\xB0"
                << " | Spin=" << (PhysicalData::spin >= 0 ? "+" : "") << PhysicalData::spin
                << " | Apex=" << std::scientific << std::setprecision(6) << electron.minimalDistance;

        const int W = 1000, H = 1000;
        sf::RenderWindow window(sf::VideoMode(W, H), titleSS.str());
        window.setFramerateLimit(30);

        // Load a monospace font — try common WSL paths
        sf::Font font;
        bool fontLoaded = font.loadFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
                       || font.loadFromFile("/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf")
                       || font.loadFromFile("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf");

        double zoom = 1.0;
        double panX = 0, panY = 0;
        bool dragging = false;
        int lastMX = 0, lastMY = 0;

        while (window.isOpen()) {
            sf::Event event;
            while (window.pollEvent(event)) {
                if (event.type == sf::Event::Closed)
                    window.close();

                // Scroll to zoom
                if (event.type == sf::Event::MouseWheelScrolled) {
                    double factor = (event.mouseWheelScroll.delta > 0) ? 1.15 : 1.0 / 1.15;
                    zoom *= factor;
                }

                // Drag to pan
                if (event.type == sf::Event::MouseButtonPressed && event.mouseButton.button == sf::Mouse::Left) {
                    dragging = true;
                    lastMX = event.mouseButton.x;
                    lastMY = event.mouseButton.y;
                }
                if (event.type == sf::Event::MouseButtonReleased && event.mouseButton.button == sf::Mouse::Left) {
                    dragging = false;
                }
                if (event.type == sf::Event::MouseMoved && dragging) {
                    panX += event.mouseMove.x - lastMX;
                    panY += event.mouseMove.y - lastMY;
                    lastMX = event.mouseMove.x;
                    lastMY = event.mouseMove.y;
                }
            }

            int w = (int)window.getSize().x;
            int h = (int)window.getSize().y;

            // Coordinate mapping lambdas
            auto toScreenX = [&](double physX) -> float {
                double norm = (physX + range) / (2.0 * range);
                return (float)(panX + w / 2.0 + (norm - 0.5) * w * zoom);
            };
            auto toScreenY = [&](double physY) -> float {
                double norm = (physY + range) / (2.0 * range);
                return (float)(panY + h / 2.0 - (norm - 0.5) * h * zoom);  // flip Y
            };

            window.clear(sf::Color(20, 20, 30));

            // Crosshairs at nucleus
            float cx = toScreenX(0), cy = toScreenY(0);
            sf::Vertex crossH[] = {
                sf::Vertex(sf::Vector2f(0, cy), sf::Color(60, 60, 80)),
                sf::Vertex(sf::Vector2f((float)w, cy), sf::Color(60, 60, 80))
            };
            sf::Vertex crossV[] = {
                sf::Vertex(sf::Vector2f(cx, 0), sf::Color(60, 60, 80)),
                sf::Vertex(sf::Vector2f(cx, (float)h), sf::Color(60, 60, 80))
            };
            window.draw(crossH, 2, sf::Lines);
            window.draw(crossV, 2, sf::Lines);

            // Bohr radius circle
            double bohrM = PhysicalData::bohrRadius;
            float bx1 = toScreenX(-bohrM), bx2 = toScreenX(bohrM);
            float by1 = toScreenY(bohrM), by2 = toScreenY(-bohrM);
            float bDiam = std::abs(bx2 - bx1);
            sf::CircleShape bohrCircle(bDiam / 2.0f);
            bohrCircle.setPosition(std::min(bx1, bx2), std::min(by1, by2));
            bohrCircle.setFillColor(sf::Color::Transparent);
            bohrCircle.setOutlineColor(sf::Color(80, 80, 50, 120));
            bohrCircle.setOutlineThickness(1.0f);
            window.draw(bohrCircle);

            // Nucleus dot
            sf::CircleShape nucleus(4.0f);
            nucleus.setPosition(cx - 4, cy - 4);
            nucleus.setFillColor(sf::Color(255, 255, 100));
            window.draw(nucleus);

            // Trajectory dots — mass center (yellow->red) and charge center (cyan->blue)
            for (int i = 0; i < n; i++) {
                float t = (float)i / (float)(n - 1);
                float sx1 = toScreenX(qx[i]);
                float sy1 = toScreenY(qz[i]);
                float sx2 = toScreenX(rx[i]);
                float sy2 = toScreenY(rz[i]);

                // Cull offscreen
                if (sx1 < -100 || sx1 > w + 100 || sy1 < -100 || sy1 > h + 100) continue;

                // Mass center: yellow -> red
                sf::CircleShape massDot(1.5f);
                massDot.setPosition(sx1 - 1.5f, sy1 - 1.5f);
                massDot.setFillColor(sf::Color(255, (sf::Uint8)(255 * (1.0f - t)), 0, 200));
                window.draw(massDot);

                // Charge center: cyan -> blue
                sf::CircleShape chargeDot(1.5f);
                chargeDot.setPosition(sx2 - 1.5f, sy2 - 1.5f);
                chargeDot.setFillColor(sf::Color(0, (sf::Uint8)(255 * (1.0f - t)), 255, 200));
                window.draw(chargeDot);

                // Connecting line every 5th point
                if (i % 5 == 0) {
                    sf::Vertex line[] = {
                        sf::Vertex(sf::Vector2f(sx1, sy1), sf::Color(100, 100, 100, 80)),
                        sf::Vertex(sf::Vector2f(sx2, sy2), sf::Color(100, 100, 100, 80))
                    };
                    window.draw(line, 2, sf::Lines);
                }
            }

            // HUD text overlay
            if (fontLoaded) {
                // Dark backdrop
                sf::RectangleShape hud(sf::Vector2f((float)(w - 10), 65.0f));
                hud.setPosition(5, 5);
                hud.setFillColor(sf::Color(0, 0, 0, 180));
                window.draw(hud);

                auto fmt = [](double v) {
                    std::ostringstream oss;
                    oss << std::scientific << std::setprecision(6) << v;
                    return oss.str();
                };

                // Params line
                sf::Text paramsTxt;
                paramsTxt.setFont(font);
                paramsTxt.setCharacterSize(11);
                paramsTxt.setFillColor(sf::Color(180, 180, 180));
                std::ostringstream pss;
                pss << "PARAMS | rangeMin: " << PhysicalData::rangeMin
                    << " | rangeMax: " << PhysicalData::rangeMax;
                paramsTxt.setString(pss.str());
                paramsTxt.setPosition(10, 7);
                window.draw(paramsTxt);

                // Exit line
                sf::Text exitTxt;
                exitTxt.setFont(font);
                exitTxt.setCharacterSize(11);
                exitTxt.setFillColor(sf::Color(200, 200, 200));
                std::ostringstream ess;
                ess << "Start: " << fmt(electron.stateHistory.front()[QX])
                    << " | Apex: " << fmt(electron.minimalDistance)
                    << " | Finish: " << fmt(electron.stateHistory.back()[QX])
                    << " | Angle: " << (int)electron.getAngle() << "\xC2\xB0"
                    << " | Energy: " << fmt(electron.getKineticEnergy()) << "eV";
                exitTxt.setString(ess.str());
                exitTxt.setPosition(10, 22);
                window.draw(exitTxt);

                // Constraints line
                sf::Text consTxt;
                consTxt.setFont(font);
                consTxt.setCharacterSize(11);
                consTxt.setFillColor(sf::Color(160, 160, 160));
                std::ostringstream css;
                css << "Steps: " << electron.internalCount
                    << " | Camera pts: " << n
                    << " | minR: " << std::setprecision(6) << electron.minR
                    << " | maxR: " << electron.maxR;
                consTxt.setString(css.str());
                consTxt.setPosition(10, 37);
                window.draw(consTxt);

                // Energy out — bright orange, larger font
                sf::Text energyTxt;
                energyTxt.setFont(font);
                energyTxt.setCharacterSize(18);
                energyTxt.setStyle(sf::Text::Bold);
                energyTxt.setFillColor(sf::Color(255, 140, 0));
                energyTxt.setString("Energy OUT: " + fmt(electron.getKineticEnergy()) + " eV");
                energyTxt.setPosition(10, (float)(h - 65));
                window.draw(energyTxt);

                // Angle out — green
                sf::Text angleTxt;
                angleTxt.setFont(font);
                angleTxt.setCharacterSize(16);
                angleTxt.setStyle(sf::Text::Bold);
                angleTxt.setFillColor(sf::Color(100, 255, 100));
                angleTxt.setString("Angle: " + std::to_string((int)electron.getAngle()) + "\xC2\xB0");
                angleTxt.setPosition(10, (float)(h - 40));
                window.draw(angleTxt);

                // Legend
                sf::RectangleShape legendBg(sf::Vector2f(215, 55));
                legendBg.setPosition((float)(w - 225), (float)(h - 75));
                legendBg.setFillColor(sf::Color(0, 0, 0, 160));
                window.draw(legendBg);

                int lx = w - 220, ly = h - 70;

                sf::CircleShape massLeg(5);
                massLeg.setPosition((float)lx, (float)(ly + 2));
                massLeg.setFillColor(sf::Color(255, 200, 0));
                window.draw(massLeg);
                sf::Text massLabel;
                massLabel.setFont(font);
                massLabel.setCharacterSize(12);
                massLabel.setFillColor(sf::Color(200, 200, 200));
                massLabel.setString("Mass center (q)");
                massLabel.setPosition((float)(lx + 16), (float)(ly));
                window.draw(massLabel);

                sf::CircleShape chargeLeg(5);
                chargeLeg.setPosition((float)lx, (float)(ly + 20));
                chargeLeg.setFillColor(sf::Color(0, 150, 255));
                window.draw(chargeLeg);
                sf::Text chargeLabel;
                chargeLabel.setFont(font);
                chargeLabel.setCharacterSize(12);
                chargeLabel.setFillColor(sf::Color(200, 200, 200));
                chargeLabel.setString("Charge center (r)");
                chargeLabel.setPosition((float)(lx + 16), (float)(ly + 18));
                window.draw(chargeLabel);

                sf::Text bohrLabel;
                bohrLabel.setFont(font);
                bohrLabel.setCharacterSize(12);
                bohrLabel.setFillColor(sf::Color(180, 180, 100));
                bohrLabel.setString("--- Bohr radius");
                bohrLabel.setPosition((float)(lx + 2), (float)(ly + 36));
                window.draw(bohrLabel);

                // Zoom indicator
                sf::Text zoomTxt;
                zoomTxt.setFont(font);
                zoomTxt.setCharacterSize(10);
                zoomTxt.setFillColor(sf::Color(120, 120, 120));
                std::ostringstream zss;
                zss << "Zoom: " << std::fixed << std::setprecision(1) << zoom
                    << "x  (scroll to zoom, drag to pan)";
                zoomTxt.setString(zss.str());
                zoomTxt.setPosition(10, (float)(h - 15));
                window.draw(zoomTxt);
            }

            window.display();
        }
    }
};
