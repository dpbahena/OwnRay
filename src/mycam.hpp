#pragma once



#include "definitions.h"


namespace rayos{

    class MyCam {
        public:
            MyCam(int width, int heigh) : image_width(width), image_height(heigh){
                aspectRatio = image_width / static_cast<float>(image_height);
            }

            point camera_center             = vec3(0.0f, 0.0f, 0.0f);
            vec3 pixel00_loc;
            vec3 pixel_delta_u;
            vec3 pixel_delta_v;


        void update(){
            
            float vieport_width = viewport_height * aspectRatio;
            
            /* Calculate the vectors across the horizontal and down the vertical viewport edges */
            viewport_u = vec3(vieport_width, 0.0f, 0.0f);
            viewport_v = vec3(0.0f, -viewport_height, 0.0f);

            /* Calculate the horizontal and vertical delta vectors from pixel to pixel */
            pixel_delta_u = viewport_u / static_cast<float>(image_width);
            pixel_delta_v = viewport_v / static_cast<float>(image_height);

            /* Calculate the locations of the upper left pixel */
            auto viewport_upper_left = camera_center - vec3(0.0f, 0.0f, focal_length) - viewport_u / 2.0f - viewport_v / 2.0f;
            pixel00_loc = viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);





            


        }

        private:
            float viewport_height           = 2.0f;
            
            float focal_length              = 1.0f;
            vec3 viewport_u;
            vec3 viewport_v;
            
            
            int image_width;
            int image_height;
            float aspectRatio;

            
            



            
        
    };
}

