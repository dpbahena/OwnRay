#pragma once

#include "definitions.h"
#include <SDL2/SDL.h>
#include <string>







namespace rayos {
    class Window {
    public:
        Window(std::string name, int w = 800, int h = 600 );
        ~Window();

        Window(const Window &) = delete;
        Window &operator=(const Window &) = delete;

    SDL_Renderer *renderer = nullptr;
    SDL_Texture *colorBufferTexture;

    // helpers
        bool windowIsOpen() { return windowIsRunning; }

    // getters
        Extent2D getExtent() { return {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};}
        SDL_Window* getWindow() {return window;}
    // setters
        void closeWindow() { windowIsRunning = false;}


    private:

    // main functions
        void initWindow();
        void displayMode();

    // helpers

    // variable members
        int width, height, x_winPos, y_winPos;
        


        std::string windowName;
        SDL_Window *window = nullptr;
        
        


        bool windowIsRunning = false;



    };

} // namespace mge
