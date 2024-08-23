#pragma once



#include "definitions.h"
// #include "my_classes.h"
#include "my_cudahelpers.h"


namespace rayos{

    class MyCam {
        public:
            __device__
            MyCam(int width, int heigh) : image_width(width), image_height(heigh){
                aspectRatio = image_width / static_cast<float>(image_height);

                
            }

            point camera_center             = vec3(0.0f, 0.0f, 0.0f);
            vec3 pixel00_loc;
            vec3 pixel_delta_u;
            vec3 pixel_delta_v;
            int samples_per_pixel           = 100;
            float sample_scale;


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

            sample_scale = 1.0f / static_cast<float>(samples_per_pixel);
        }

        __device__
        ray get_ray(int& i, int& j, curandState_t* states) {
            /* construct a ray originating from the origin and directed at randomly sampled point around the pixel location i, j */
            auto offset = sample_square(states, i, j);
            auto pixel_sample = pixel00_loc + ((i + offset.x) * pixel_delta_u) + ((j + offset.y) * pixel_delta_v);

            auto ray_origin     = camera_center;
            auto ray_direction  = pixel_sample - ray_origin;

            return ray(ray_origin, ray_direction); 

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

