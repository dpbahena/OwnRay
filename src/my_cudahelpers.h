#pragma once

#include "definitions.h"
#include "my_classes.h"
#include <cuda_runtime.h>


namespace rayos {

    
    __device__   //__forceinline__
    inline uint32_t colorToUint32_t(glm::vec3& c);
    __device__ 
    inline float random_float_range(curandState_t* state, float a, float b);
    __device__
    inline vec3 random_vector(curandState_t* states,  int& i, int& j);
    __device__
    inline vec3 random_vector_in_range(curandState_t* states, int& i, int& j, float min, float max);
     __device__
    inline vec3 random_in_unit_sphere(curandState_t* states,  int& i, int& j);
     __device__
    inline vec3 random_on_hemisphere(curandState_t* states, int& i, int& j, const vec3& normal);
    



    const interval interval::empty          = interval(+MAXFLOAT, -MAXFLOAT);
    const interval interval::universe       = interval(-MAXFLOAT, +MAXFLOAT);


    __device__
    inline bool near_zero(vec3 v)  {
        // Return true if the vector is close to zero in all dimensions.
        float s = 1e-8;
        return (fabs(v.x) < s) && (fabs(v.y) < s) && (fabs(v.z) < s);
    }

    __device__
    inline vec3 reflect(const vec3& v, const vec3& n) {
        return v - 2.0f * glm::dot(v,n) * n;
    }

    __device__
    inline vec3 refract(const vec3& uv, const vec3& n, float etai_over_etat) {
        auto cos_theta = fmin(glm::dot(-uv, n), 1.0f);
        vec3 r_out_perp =  etai_over_etat * (uv + cos_theta * n);
        vec3 r_out_parallel = -sqrt(fabs(1.0f - glm::dot(r_out_perp, r_out_perp))) * n;
        return (r_out_perp + r_out_parallel);
    }

    __device__
    inline float reflectance(float cosine, float refraction_index){
        /* Use Schlick's approximation for reflectance */
        float r0 = (1.0f - refraction_index) / (1.0f + refraction_index);
        r0 = r0 * r0;
        return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);

    }


    __device__
    inline float random_float(curandState_t* state) {
        return curand_uniform(state);
    }

    __device__ 
    inline float random_float_range(curandState_t* state, float a, float b){
        // return a + (b - a) * curand_uniform(state);  // this does not include b  e.g -1 to 1.0  it does not include 1.0
        return a + (b - a) * (curand_uniform(state) - 0.5f) * 2.0f;   // this approach includes the upper limit   -1 to 1.0  it includes 1.0
    }

   

    __device__
    inline vec3 random_vector(curandState_t* states,  int &i, int &j){
        curandState_t x = states[i];
        // curandState_t y = states[j];
        
        float a = random_float(&x);
        float b = random_float(&x);
        float c = random_float(&x); 
        
        states[i] = x; // save value back
        // states[j] = y; // save value back
        return vec3(a, b, c);

    }

    __device__
    inline vec3 random_vector_in_range(curandState_t* states, int &i, int &j,  float min, float max){
        curandState_t x = states[i];
        // curandState_t y = states[j];
        float a = random_float_range(&x, min, max);
        float b = random_float_range(&x, min, max);
        float c = random_float_range(&x, min, max);
        states[i] = x; // save value back
        // states[j] = y; // save value back
        return vec3(a, b, c);
    }

    __device__
    inline vec3 random_unit_vector(curandState_t* states,  int &i, int &j) {

        auto p = random_in_unit_sphere(states, i, j);
        return glm::normalize(p);
    }

    __device__
    inline vec3 random_in_unit_sphere(curandState_t* states,  int &i, int &j) {
        while (true) {
            vec3 p = random_vector_in_range(states, i, j,  -1.0f,1.0f);
            if (glm::dot(p,p) < 1.0f){
                return p;
            }
        }
    }

    __device__
    inline vec3 random_in_unit_disk(curandState_t* states, int&i, int& j){
        curandState_t x = states[i];
        // curandState_t y = states[j];
        while (true) {
            vec3 p = vec3(random_float_range(&x, -1.0f, 1.0f), random_float_range(&x, -1.0f, 1.0f), 0.0f);
            states[i] = x; // saves value back
            // states[j] = y; // saves value back
            if (glm::dot(p, p) < 1.0f)
                return p;
        }
    }

    __device__
    inline vec3 random_on_hemisphere(curandState_t* states, int &i, int &j, const vec3& normal) {
        vec3 on_unit_sphere = random_unit_vector(states, i, j);
        if (glm::dot(on_unit_sphere, normal) > 0.0f) // In the same hemisphere as the normal
            return on_unit_sphere;
        else
            return -on_unit_sphere;
    }


    /* Returns the vector to a random point in the  [-0.5, -0.5] - [0.5, 0.5] unit square */
    __device__
    inline vec3 sample_square(curandState_t* states, int& i, int& j) {
        curandState_t x = states[i];
        // curandState_t y = states[j];
        vec3 random_vector =  vec3(random_float(&x) - 0.5f, random_float(&x) - 0.5f, 0);
        
        states[i] = x; // save back the value  
        // states[j] = y; // save back the value 
        
        return random_vector;
        
    }

    __device__  //__forceinline__
    inline float linear_to_gamma(float linear_component){
        if (linear_component > 0.0f){
            return sqrt(linear_component);
        }
        return 0;
    }

    __device__ // __forceinline__
    inline uint32_t colorToUint32_t(glm::vec3& c)
    {
       
        

        /* Ensure that the input values within the range [0.0, 1.0] */
        c.x = (c.x < 0.0f) ? 0.0f : ((c.x > 1.0f) ? 1.0f : c.x);  // red
        c.y = (c.y < 0.0f) ? 0.0f : ((c.y > 1.0f) ? 1.0f : c.y);  // green
        c.z = (c.z < 0.0f) ? 0.0f : ((c.z > 1.0f) ? 1.0f : c.z);  // blue   

        // Apply a linear to gamma transform for gamma 2
        c.x = linear_to_gamma(c.x);
        c.y = linear_to_gamma(c.y);
        c.z = linear_to_gamma(c.z);             

        // convert to integers
        uint32_t ri = static_cast<uint32_t>(c.x * 255.0);
        uint32_t gi = static_cast<uint32_t>(c.y * 255.0);
        uint32_t bi = static_cast<uint32_t>(c.z * 255.0);

        // Combine into a single uint32_t with FF for alpha (opacity)
        uint32_t color = (0x00 << 24) | (ri << 16) | (gi << 8) | bi;

        return color;
    }

   

   

    __device__  //__forceinline__
    inline vec3 ray_color(const ray& r, hittable** world, int depth, curandState_t* states, int &i, int &j){
        
        ray current_ray = r;
        vec3 current_attenuation = vec3(1.0f, 1.0f, 1.0f);
        for(int k = 0; k < depth; k++){
            hit_record rec;
            if ((*world)->hit(current_ray, interval(0.001f, FLT_MAX), rec)){
                ray scattered;
                vec3 attenuation;
                if(rec.mat_ptr->scatter(current_ray, rec, attenuation, scattered, states, i, j)){
                    current_attenuation *= attenuation;
                    current_ray = scattered;
                } /* else {
                    return vec3(0.0f, 0.0f, 0.0f);
                } */
                
                
                
            } else {
                vec3 unit_vector = glm::normalize(r.direction());
                float a = 0.5f * (unit_vector.y + 1.0f);
                auto background = (1.0f - a) * vec3(1.0f, 1.0f, 1.0f) + a * vec3(0.5f, 0.7f, 1.0f); //vec3(0.9f, 0.2f, .9f); // pink
                return current_attenuation * background;
                break;
            }
        }

        return current_attenuation;
        // return vec3(0.0f, 0.0f, 0.0f);
        


    }

    
    

}
