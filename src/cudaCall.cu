#include "cudaCall.h"
#include "my_cudahelpers.h"
#include "my_classes.h"
#include "renderer.hpp"

#include <iostream>




namespace rayos {

    // #define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

    // void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    //     if (result) {
    //         std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
    //             file << ":" << line << " '" << func << "' \n";
    //         // Make sure we call CUDA Device Reset before exiting
    //         cudaDeviceReset();
    //         exit(99);
    //     }
    // }

    #define checkCudaErrors(result) { gpuAssert((result), __FILE__, __LINE__); }
    inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
        if (code != cudaSuccess) {
            fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
            if (abort) assert(code == cudaSuccess);
        }
    }



    __global__ void render_kernel(uint32_t* buffer, int width, int height, vec3 cameraCenter, vec3 delta_u, vec3 delta_v, vec3 pixel00){
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        int idx = width * j + i;
        
        if (idx >= (width * height)) return;
        
        auto pixel_center = pixel00 + (static_cast<float>(i) * delta_u) + (static_cast<float>(j) * delta_v);
        auto ray_direction = pixel_center - cameraCenter;
        ray r(cameraCenter, ray_direction);

        vec3 color = ray_color(r);
              
        buffer[idx] = colorToUint32_t(color);
        
    }


    void CudaCall::cudaCall(int width, int height, Data& data)
    {
        
        Renderer renderer{window};
        uint32_t* colorBuffer;
        
        

        checkCudaErrors(cudaMallocManaged(&colorBuffer, width * height * sizeof(uint32_t)));

        clock_t start, stop;
        start = clock();
        int threads = 32;
        dim3 blockSize(threads, threads);
        int blocks_x = (width + blockSize.x - 1) / blockSize.x;
        int blocks_y = (height + blockSize.y - 1) / blockSize.y;
        dim3 gridSize(blocks_x, blocks_y);

        render_kernel<<<gridSize, blockSize>>>(colorBuffer, width, height, data.center, data.delta_u, data.delta_v, data.pixel000);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize() );
        stop = clock();
        double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
        std::cerr << "took " << timer_seconds << " seconds.\n";


        renderer.render(colorBuffer);
        
        cudaFree(colorBuffer);



    }

    CudaCall::~CudaCall()
    {
    }

     rayos::CudaCall::CudaCall(Window& window) : window(window)
    {
    }

    

} // namespace