#pragma once


#include "window.hpp"



namespace rayos {
    class Renderer {
    public:

        
        Renderer(Window &window);
        Renderer();
        ~Renderer();

        Renderer(const Renderer &) = delete;
        Renderer &operator=(const Renderer &) = delete;

        
    // Variables

    
    
    // helpers
        
    // getters
        float getAspectRatio () const { return (float) extent.width / extent.height; }
    // main function

        void SimpleRenderSystem();
        void render(uint32_t* colorBuffer);
        
    private:

    // variable members
    

    Window& window;
    
    
    Extent2D extent;
    Extent2D offset;

    int windowSize{};  // holds width * height size
    // main functions

    
    
    void createBuffers();
    void initBuffers();

        

    // helpers

        


        



    };

} // namespace mge
