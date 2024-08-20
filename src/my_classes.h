#pragma once

#include "definitions.h"
#include <cuda_runtime.h>

namespace rayos {
    class ray {
        public:
            
            __device__ ray(){}
            __device__ ray(const point& origin, const vec3& direction) : origin_(origin), direction_(direction) {}

            
            __device__ const point& origin() const { return origin_; }
            __device__ const vec3& direction() const { return direction_;}

            __device__ point at(float t) const {
                return origin_ + t * direction_;
            }

        private:

        point origin_;
        vec3 direction_;
    };


}