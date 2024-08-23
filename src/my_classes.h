#pragma once

#include "definitions.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace rayos {


    __device__
    vec3 sample_square(curandState_t* states, int& i, int& j);



    class interval {
        public:
            float min, max;
            __device__ 
            interval() : min(FLT_MAX), max(-FLT_MAX) {} // default interval empty
            __device__
            interval (float min, float max) : min(min), max(max){}
            __device__
            float size() const {
                return max - min;
            }
            __device__ 
            bool contains(float x) const {
                return min <= x && x <= max;
            }
            __device__
            bool surrounds(float x) const {
                return min < x && x < max;
            }
            
            static const interval empty, universe;

    };

    


    class ray {
        public:
            
            __device__ ray(){}
            __device__ ray(const point& origin, const vec3& direction) : origin_(origin), direction_(direction) {}

            
            __device__  const point& origin() const { return origin_; }
            __device__  const vec3& direction() const { return direction_;}

            __device__  point at(float t) const {
                return origin_ + t * direction_;
            }

        private:

        point origin_;
        vec3 direction_;
    };


    class hit_record{
        public:
            point p;
            vec3 normal;
            float t;
            bool front_face;

            __device__ 
            void set_face_normal(const ray&r, const vec3& outward_normal){
                /* sets the hit record notmal vector. Assume outward_normal is of unit length */
                front_face = glm::dot(r.direction(), outward_normal) < 0;
                normal = front_face ? outward_normal : -outward_normal;
            }
            
    };

    class hittable {
        public:
            
            __device__ 
            virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const = 0;

    };

    class sphere : public hittable {
        public:
            __device__ 
            sphere (const point& center, float radius) : center(center), radius(radius) {}
            __device__ 
            bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
                vec3 oc = center - r.origin();
                float a = glm::dot(r.direction(), r.direction());
                float h = glm::dot(r.direction(), oc);
                float c = glm::dot(oc, oc) - radius * radius;
                float discriminant = h * h -  a * c;
                if (discriminant < 0 ) {
                    return false;
                }

                float sqrtd = sqrt(discriminant); 

                /* Find the nearest root that lies in the acceptable range */
                float root = (h - sqrtd) / a;
                if (!ray_t.surrounds(root)) {
                    root = (h + sqrtd) / a;
                    if (!ray_t.surrounds(root))
                        return false;
                }

                rec.t = root;
                rec.p = r.at(rec.t);
                vec3 outward_normal = (rec.p - center) / radius;
                rec.set_face_normal(r, outward_normal);

                return true;
            }

        private:
            point center;
            float radius;
    };


    class hittable_list : public hittable {
        public:
        hittable** list;
        int list_size;

        __device__ 
        hittable_list() {}
        __device__ 
        hittable_list(hittable **list, int n) : list(list), list_size(n)  {}
        // __device__
        // void clear() {list_size = 0; }

        __device__ 
        bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
            hit_record temp_rec;
            bool hit_anything = false;
            auto closest_so_far = ray_t.max;

            for (int i = 0; i < list_size; i++) {
                if (list[i]->hit(r, interval(ray_t.min, closest_so_far), temp_rec)){
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
            }

            return hit_anything;
        }





    };


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
            int samples_per_pixel           = 50;
            float sample_scale;
            int depth                       = 5;

        __device__
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
        ray get_ray(int& i, int& j, curandState_t* states) const {
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

