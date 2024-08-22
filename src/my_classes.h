#pragma once

#include "definitions.h"
#include <cuda_runtime.h>

namespace rayos {

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
            bool constains(float x) const {
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



}

