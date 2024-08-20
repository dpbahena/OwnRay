#include "cudaCall.h"
#include "my_cudahelpers.h"
#include "renderer.hpp"

#include <iostream>




namespace rayos {

    #define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

    void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
        if (result) {
            std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                file << ":" << line << " '" << func << "' \n";
            // Make sure we call CUDA Device Reset before exiting
            cudaDeviceReset();
            exit(99);
        }
    }



    __global__ void render_kernel(uint32_t* buffer, int width, int height){
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        int idx = width * j + i;
        vec3 color = vec3(0.0f, 0.0f, 0.0f);

        if (idx >= (width * height)) return;
        color.x = i / (width - 1.0f);
        color.y = j / (height - 1.0f);
        color.z = 0.0;
        
        buffer[idx] = colorToUint32_t(color);
        
    }


    void CudaCall::cudaCall(int width, int height)
    {
        
        Renderer renderer{window};
        uint32_t* colorBuffer;
        
        

        checkCudaErrors(cudaMallocManaged(&colorBuffer, width * height * sizeof(uint32_t)));
        int threads = 32;
        dim3 blockSize(threads, threads);
        int blocks_x = (width + blockSize.x - 1) / blockSize.x;
        int blocks_y = (height + blockSize.y - 1) / blockSize.y;
        dim3 gridSize(blocks_x, blocks_y);

        render_kernel<<<gridSize, blockSize>>>(colorBuffer, width, height);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize() );

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