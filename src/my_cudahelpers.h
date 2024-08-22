#pragma once

#include "definitions.h"
#include "my_classes.h"
#include <cuda_runtime.h>

namespace rayos {

    __device__  __forceinline__
    vec3 ray_color(const ray& r);
    __device__  __forceinline__
    vec3 ray_color(const ray& r, hittable** world);
    __device__  __forceinline__
    float hit_sphere(const point& center, float radius, const ray& r);
    __device__   __forceinline__
    uint32_t colorToUint32_t(glm::vec3& c);

    const interval interval::empty          = interval(+MAXFLOAT, -MAXFLOAT);
    const interval interval::universe       = interval(-MAXFLOAT, +MAXFLOAT);


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

}
