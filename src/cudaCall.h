#pragma once

#include "definitions.h"
#include "window.hpp"


namespace rayos {


    class CudaCall {
    public:
        CudaCall(Window& window);
        ~CudaCall();

        
        void cudaCall(int width, int height);
        
        
       
        
        
        
    private:

    Window& window;

    
        



    };

    

    
    
    


    
}  // namespace sdl


