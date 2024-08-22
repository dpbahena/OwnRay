#pragma once

#include "definitions.h"
#include "my_classes.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace rayos {

    __device__  __forceinline__
    vec3 ray_color(const ray& r);
    __device__  __forceinline__
    vec3 ray_color(const ray& r, hittable** world);
    __device__  __forceinline__
    float hit_sphere(const point& center, float radius, const ray& r);
    __device__   __forceinline__
    uint32_t colorToUint32_t(glm::vec3& c);
    __device__
    float random_float(curandState_t* state);
    __device__ 
    float random_float_range(curandState_t* state, float a, float b);
    __device__
    vec3 random_vector(curandState_t* states,  int i, int j);
    __device__
    vec3 random_vector_in_range(curandState_t* states, int i, int j, double min, double max);
     __device__
    vec3 random_in_unit_sphere(curandState_t* states,  int i, int j);
    __device__
    vec3 random_unit_vector(curandState_t* states,  int i, int j);
     __device__
    vec3 random_on_hemisphere(curandState_t* states, int i, int j, const vec3& normal);
    __device__
    vec3 sample_square(curandState_t* states, int& i, int& j);

    const interval interval::empty          = interval(+MAXFLOAT, -MAXFLOAT);
    const interval interval::universe       = interval(-MAXFLOAT, +MAXFLOAT);


    __device__
    float random_float(curandState_t* state) {
        return curand_uniform(state);
    }

    __device__ 
    float random_float_range(curandState_t* state, float a, float b){
        // return a + (b - a) * curand_uniform(state);  // this does not include b  e.g -1 to 1.0  it does not include 1.0
        return a + (b - a) * (curand_uniform(state) - 0.5f) * 2.0f;   // this approach includes the upper limit   -1 to 1.0  it includes 1.0
    }

    __device__
    vec3 random_vector(curandState_t* states,  int i, int j){
        curandState_t x = states[i];
        
        double a = random_float(&x);
        double b = random_float(&x);
        double c = random_float(&x); //a * b;
        states[i] = x; // save value back
        return vec3(a, b, c);

    }

    __device__
    vec3 random_vector_in_range(curandState_t* states, int i, int j,  double min, double max){
        curandState_t x = states[i];
        double a = random_float_range(&x, min, max);
        double b = random_float_range(&x, min, max);
        double c = random_float_range(&x, min, max);
        states[i] = x; // save value back
        return vec3(a, b, c);
    }

    __device__
    vec3 random_unit_vector(curandState_t* states,  int i, int j) {

        auto p = random_in_unit_sphere(states, i, j);
        return glm::normalize(p);
    }

    __device__
    vec3 random_in_unit_sphere(curandState_t* states,  int i, int j) {
        while (true) {
            vec3 p = random_vector_in_range(states, i, j,  -1.0,1.0);
            if (glm::dot(p,p) < 1.0f){
                return p;
            }
        }
    }

    __device__
    vec3 random_on_hemisphere(curandState_t* states, int i, int j, const vec3& normal) {
        vec3 on_unit_sphere = random_unit_vector(states, i, j);
        if (glm::dot(on_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
            return on_unit_sphere;
        else
            return -on_unit_sphere;
    }


    /* Returns the vector to a random point in the  [-0.5, -0.5] - [0.5, 0.5] unit square */
    __device__
    vec3 sample_square(curandState_t* states, int& i, int& j) {
        curandState_t x = states[i];
        curandState_t y = states[j];
        states[i] = x; // save back the value  
        states[j] = y; // save back the value  
        return vec3(random_float(&x) - 0.5f, random_float(&y) - 0.5f, 0);
        // return vec3(random_float(&x) - 0.0, random_float(&y) - 0.0, 0);
         
    }

    /* Returns the vector to a random point in the  [-0.5, -0.5] - [0.5, 0.5] unit square */
    __device__
    vec3 sample_square2(float& x, float& y) {
          
        return vec3(x - 0.5, y - 0.5, 0);
         
    }


    __device__  __forceinline__
    uint32_t colorToUint32_t(glm::vec3& c)
    {
        /* Ensure that the input values within the range [0.0, 1.0] */
        c.x = (c.x < 0.0f) ? 0.0f : ((c.x > 1.0f) ? 1.0f : c.x);  // red
        c.y = (c.y < 0.0f) ? 0.0f : ((c.y > 1.0f) ? 1.0f : c.y);  // green
        c.z = (c.z < 0.0f) ? 0.0f : ((c.z > 1.0f) ? 1.0f : c.z);  // blue

        // Apply a linear to gamma transform for gamma 2
        // c.x = linear_to_gamma(c.x);
        // c.y = linear_to_gamma(c.y);
        // c.z = linear_to_gamma(c.z);

        // convert to integers
        uint32_t ri = static_cast<uint32_t>(c.x * 255.0);
        uint32_t gi = static_cast<uint32_t>(c.y * 255.0);
        uint32_t bi = static_cast<uint32_t>(c.z * 255.0);

        // Combine into a single uint32_t with FF for alpha (opacity)
        uint32_t color = (0x00 << 24) | (ri << 16) | (gi << 8) | bi;

        return color;
    }

    __device__ __forceinline__
    vec3 ray_color(const ray& r){
        float t = hit_sphere(point(0.0f, 0.0f, -1.0f), 0.5f, r);
        if (t > 0.0f){
            vec3 normal = glm::normalize(r.at(t) - vec3(0.0f, 0.0f, -1.0f));
            return 0.5f * vec3(normal.x + 1.0f, normal.y + 1.0f, normal.z + 1.0f);
        }
        vec3 unit_vector = glm::normalize(r.direction());
        float a = 0.5f * (unit_vector.y + 1.0f);
        return (1.0f - a) * vec3(1.0f, 1.0f, 1.0f) + a * vec3(0.5f, 0.7f, 1.0f);
    }

    __device__ __forceinline__
    float hit_sphere(const point& center, float radius, const ray& r) {
        vec3 oc = center - r.origin();
        float a = glm::dot(r.direction(), r.direction());
        float h = glm::dot(r.direction(), oc);
        float c = glm::dot(oc, oc) - radius * radius;
        float discriminant = h * h -  a * c;
        if (discriminant < 0 ) {
            return -1.0f;
        } else {
             return (h - sqrt(discriminant) ) / a;
        }
    }

    __device__  __forceinline__
    vec3 ray_color(const ray& r, hittable** world){
        hit_record rec;
        vec3 color = vec3(1.0f, 1.0f, 1.0f);
        if ((*world)->hit(r, interval(0, MAXFLOAT), rec)){
            return 0.5f * (rec.normal  + color); 
        }
        vec3 unit_vector = glm::normalize(r.direction());
        float a = 0.5f * (unit_vector.y + 1.0f);
        return (1.0f - a) * vec3(1.0f, 1.0f, 1.0f) + a * vec3(0.5f, 0.7f, 1.0f);


    }
    __device__
    ray get_ray(int& i, int& j, vec3& pixel00_loc, vec3& cameraCenter, vec3& delta_u, vec3& delta_v, curandState_t* states) {
        /* construct a ray originating from the origin and directed at randomly sampled point around the pixel location i, j */
        auto offset = sample_square(states, i, j);
        auto pixel_sample = pixel00_loc + ((i + offset.x) * delta_u) + ((j + offset.y) * delta_v);

        auto ray_origin     = cameraCenter;
        auto ray_direction  = pixel_sample - ray_origin;

        return ray(ray_origin, ray_direction); 

    }

    __device__
    ray get_ray2(int& i, int& j, vec3& pixel00_loc, vec3& cameraCenter, vec3& delta_u, vec3& delta_v, float u, float v) {
        /* construct a ray originating from the origin and directed at randomly sampled point around the pixel location i, j */
        // vec3 offset = sample_square2(u, v);
        // auto pixel_sample = pixel00_loc + (i + offset.x) * delta_u + (j + offset.y) * delta_v;
        auto pixel_sample = pixel00_loc + (i+u) *delta_u + (j+v)*delta_v;

        auto ray_origin     = cameraCenter;
        auto ray_direction  = pixel_sample - ray_origin;

        return ray(ray_origin, ray_direction); 
        

    }

}
