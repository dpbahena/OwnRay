#pragma once

#include "window.hpp"
#include "renderer.hpp"
#include "cudaCall.h"
#include "mycam.hpp"




namespace rayos {

    class App {
    public:
        
        static constexpr int WIDTH = 1280;
        static constexpr int HEIGHT = 720;
        App();
        ~App();
        void run();

        App(const App &) = delete;
        App &operator=(const App &) = delete;
    
    private:

    Window window{"by Dario", WIDTH, HEIGHT};
    // Renderer renderer{window};
    CudaCall rayTracer{window};
    MyCam camera{};
    
    Extent2D extent;


    
    
    


    
    void loadObjects();
    void processInput();

    };


    


} // namespace mge
