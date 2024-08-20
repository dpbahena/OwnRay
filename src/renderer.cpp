#include "renderer.hpp"
// #include <iostream>




namespace rayos {

    Renderer::Renderer(Window &window): window(window)
    {   
        extent = window.getExtent();
        // offset.width = extent.width / 2; 
        // offset.height = extent.height / 2;
        windowSize = window.getExtent().width * window.getExtent().height;
        
        createBuffers();
        

        
    }

    Renderer::~Renderer()
    {
       
        
        
    }

    void Renderer::render(uint32_t* colorBuffer)
    {
        SDL_SetRenderDrawColor(window.renderer, 255, 255, 255, 255);
        SDL_RenderClear(window.renderer);

        // copy all content of the colorBuffer to the render
        SDL_UpdateTexture(window.colorBufferTexture, nullptr, colorBuffer, static_cast<int>(window.getExtent().width) * sizeof(uint32_t));

        SDL_RenderCopy(window.renderer, window.colorBufferTexture, nullptr, nullptr);
        // initBuffers();

        SDL_RenderPresent(window.renderer);
    }

   

    void Renderer::createBuffers()
    {
        

        //     // Allocate memory for colorBuffer
        // int allocation =  windowSize *  sizeof(uint32_t);
        // colorBuffer = new uint32_t[allocation];

        // if (colorBuffer == nullptr) {
        //     throw std::runtime_error("Failed to allocate buffer memory");
        // }

        // // Allocate memory for Z-buffer
        // allocation = windowSize * sizeof(float);
        // zBuffer = new float[allocation];

        // if (zBuffer == nullptr) {
        //     throw std::runtime_error("Failed to allocate Z-buffer memory");
        // }
    }

    void Renderer::initBuffers()
    {
        // // black 
        // memset(colorBuffer, 0xFF000000, windowSize * sizeof(uint32_t));

        // // Init zbuffer to max Float
        // std::vector<float> host_zBuffer(windowSize, FLT_MAX);
        // memcpy(zBuffer, host_zBuffer.data(), windowSize * sizeof(float));


    }

    void Renderer::SimpleRenderSystem()
    {
        
        
    }

} // namespace