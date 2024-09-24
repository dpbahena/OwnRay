#pragma once

#include "window.hpp"
#include "renderer.hpp"
#include "cudaCall.h"




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
    CudaCall rayTracer{window};
   
    
    Extent2D extent;
    int samples = 100;
    int depth = 50;
    bool info_flag = true;


    
    
    


    
    void loadObjects();
    void processInput();

    };


    


} // namespace mge
