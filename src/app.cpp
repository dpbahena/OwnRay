#include "app.hpp"



namespace rayos {

    
    App::App()
    {
        
        
        
    }

    App::~App()
    {
        
    }

    void App::run() {

        // Data data;
        // camera.camera_center = vec3(0.0f, 0.0f, 0.0f);
        // camera.samples_per_pixel = 100;
        // camera.update();

        // data.center     = camera.camera_center;
        // data.delta_u    = camera.pixel_delta_u;
        // data.delta_v    = camera.pixel_delta_v;
        // data.pixel000   = camera.pixel00_loc;
        // data.samples    = camera.samples_per_pixel;
        // data.scale      = camera.sample_scale;



        // uint32_t* colorBuffer = nullptr;
        // cam.image_width = WIDTH;
        // cam.vFOV = 20.0f; // degrees
        // cam.samples_per_pixel = 500;
        // cam.max_depth = 50;
        // cam.defocus_angle = 0.6; // degrees
        // cam.focusDistance = 10.0;
        
        // cam.position    = vec3(13.0, 2.0, 3.0);
        // cam.target      = vec3(0.0, 0.0, 0.0);
        // cam.up          = vec3(0.0, 1.0, 0.0);
        // cam.update();
        
        
        while (window.windowIsOpen()){

            rayTracer.cudaCall(window.getExtent().width, window.getExtent().height, samples, depth);
            
            // data.samples    = camera.samples_per_pixel;
            // data.scale      = camera.sample_scale;
                       
            
            processInput();

            // SDL_Delay(10000);

            // break;
            
        }
        
    }
    void App::loadObjects()
    {
        printf("dario\n");
    }

    void App::processInput()
    {
        SDL_Event event;
        SDL_PollEvent(&event);

        switch (event.type)
        {
            case SDL_QUIT:
                window.closeWindow();
                break;
            case SDL_KEYDOWN:
                switch (event.key.keysym.sym) {
                    case SDLK_ESCAPE:
                        window.closeWindow();
                        break;
                    
                    case SDLK_SPACE:
                        
                        
                        break;
                    case SDLK_PAGEUP:
                       
                        break;
                    case SDLK_PAGEDOWN:
                        
                        break;
                    case SDLK_z:
                        
                        break;
                    
                    case SDLK_x: 
                       
                        break;
                    case SDLK_f:
                       
                        break;
                    case SDLK_i:
                        break;
                    case SDLK_KP_7:
                        samples -= 1;
                        break;
                    case SDLK_KP_8:
                        samples = 50;
                        break;
                    case SDLK_KP_9:
                        samples += 1;
                        break;
                    case SDLK_KP_2:
                        break;
                    case SDLK_KP_4:
                        depth -= 1;
                        break;
                    case SDLK_KP_5:
                        depth = 10;
                        break;
                    case SDLK_KP_6:
                        depth += 1;
                        break;
                    case SDLK_KP_D:
                        break;
                    case SDLK_KP_C:
                        break;
                    case SDLK_1:
                        break;
                    case SDLK_2:
                        break;
                    case SDLK_3:
                        break;
                    case SDLK_4:
                        break;
                    case SDLK_0:
                        break;
                    case SDLK_EQUALS:
                        
                        break;
                    case SDLK_MINUS:
                        
                        break;
                  
                    default:
                        
                        break;
                }
            
            default:
            break; 
        }


    }
} // namespace mge