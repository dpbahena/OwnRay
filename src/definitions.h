#pragma once


#include <cmath>
#include <vector>





#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <stdint.h>
#include <cmath>



#ifdef NDEBUG
const bool enablePrinting= false;
#else
const bool enablePrinting = true;
#endif



namespace rayos {

    using point = glm::vec3;
    using vec3 = glm::vec3;
    

    typedef struct  {
        uint32_t  width;
        uint32_t height;
    }Extent2D;

    struct Data{
        point center;
        vec3 pixel000;
        vec3 delta_u;
        vec3 delta_v;
    };

    

    // Constants

    const double infinity = std::numeric_limits<double>::infinity();
    const double pi = 3.1415926535897932385;

    


}