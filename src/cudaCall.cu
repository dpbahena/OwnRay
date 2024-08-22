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


    __global__ void init_random(unsigned int seed, curandState_t* states){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        curand_init(seed, idx, 0, &states[idx]);
    }

    __global__ void rand_init(curandState *rand_state) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            curand_init(1984, 0, 0, rand_state);
        }
    }


    __global__ void createWorld(/* sphere** d_sphere, */ hittable** list, hittable** world){
        if (threadIdx.x == 0 && blockIdx.x == 0){

            *(list)     = new sphere(vec3(0.0f, 0.0f, -1.0f), 0.5f);
            *(list+1)   = new sphere(vec3(0.0f, -100.5f, -1.0f), 100.0f);
            *world      = new hittable_list(list, 2);  // the list has 2 spheres
            
            
            
        }
    }

    __global__ void render_init(int max_x, int max_y, unsigned int seed, curandState *rand_state) {
        // int i = threadIdx.x + blockIdx.x * blockDim.x;
        // int j = threadIdx.y + blockIdx.y * blockDim.y;
        // if((i >= max_x) || (j >= max_y)) return;
        // int pixel_index = j*max_x + i;

        // Original: Each thread gets same seed, a different sequence number, no offset
        // curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
        // BUGFIX, see Issue#2: Each thread gets different seed, same sequence for
        // performance improvement of about 2x!
        // curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index]);

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        curand_init(seed, idx, 0, &rand_state[idx]);
    }

    __global__ void render_kernel(uint32_t* buffer, int width, int height, vec3 cameraCenter, vec3 delta_u, vec3 delta_v, vec3 pixel00, int samples, float scale, hittable** world, curandState_t* states){
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        int idx = width * j + i;
        // if (i == 0 && j == 0)
        //     printf("samples: %d\t", samples);
        if (idx >= (width * height)) return;
        vec3 color = vec3(0.0f, 0.0f, 0.0f);
        for (int x = 0; x < samples; x++){
            ray r = get_ray(i, j, pixel00, cameraCenter, delta_u, delta_v, states);
            color += ray_color(r, world);
        }

        // auto pixel_center = pixel00 + (static_cast<float>(i) * delta_u) + (static_cast<float>(j) * delta_v);
        // auto ray_direction = pixel_center - cameraCenter;
        // ray r(cameraCenter, ray_direction);

        // vec3 color = ray_color(r, world);
        color *= scale;
        buffer[idx] = colorToUint32_t(color);
        
    }

    __global__ void render_kernel2(uint32_t* buffer, int width, int height, vec3 cameraCenter, vec3 delta_u, vec3 delta_v, vec3 pixel00, int samples, float scale, hittable** world, curandState* rand_state){
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        int idx = width * j + i;
        curandState local_rand_state = rand_state[idx];
        if (i == 0 && j == 0)
            printf("samples: %d\t", samples);
        if (idx >= (width * height)) return;
        vec3 color = vec3(0.0f, 0.0f, 0.0f);
        for (int x = 0; x < samples; x++){
            float u = float(i + curand_uniform(&local_rand_state)) / float(width);
            float v = float(j + curand_uniform(&local_rand_state)) / float(height);
            ray r = get_ray2(i, j, pixel00, cameraCenter, delta_u, delta_v, u, v);
            color += ray_color(r, world);
        }

        // auto pixel_center = pixel00 + (static_cast<float>(i) * delta_u) + (static_cast<float>(j) * delta_v);
        // auto ray_direction = pixel_center - cameraCenter;
        // ray r(cameraCenter, ray_direction);

        // vec3 color = ray_color(r, world);
        color *= scale;
        buffer[idx] = colorToUint32_t(color);
        
    }

    __global__ void freeWorld(hittable** list, hittable** world){
        // delete buffer;
        delete *(list);
        delete *(list + 1);
        delete *world;
    }


    void CudaCall::cudaCall(int width, int height, Data& data)
    {
        
        Renderer renderer{window};
        uint32_t* colorBuffer;  
        checkCudaErrors(cudaMallocManaged(&colorBuffer, width * height * sizeof(uint32_t)));

        
        // Create world memory

        hittable** d_list;
        checkCudaErrors(cudaMalloc((void**)&d_list, 2 * sizeof(hittable*)) );
        hittable** d_world;
        checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*) ));

       
        createWorld<<<1, 1>>>(d_list, d_world);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize() );

        



        clock_t start, stop;
        start = clock();
        int threads = 32;
        dim3 blockSize(threads, threads);
        int blocks_x = (width + blockSize.x - 1) / blockSize.x;
        int blocks_y = (height + blockSize.y - 1) / blockSize.y;
        dim3 gridSize(blocks_x, blocks_y);

        //generate random seed to be used in rayTracer kernel
        int num_threads = threads * threads * blocks_x * blocks_y;
        curandState_t* d_states;
        checkCudaErrors(cudaMalloc((void**)&d_states, num_threads * sizeof(curandState_t)) );
        init_random<<<gridSize, blockSize>>>(time(0), d_states);

        // curandState* r_state;
        // checkCudaErrors(cudaMalloc((void**)&r_state, num_threads * sizeof(curandState)) );
        // render_init<<<gridSize, blockSize>>>(width, height, time(0), r_state);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize() );


        render_kernel<<<gridSize, blockSize>>>(colorBuffer, width, height, data.center, data.delta_u, data.delta_v, data.pixel000, data.samples, data.scale, d_world, d_states);
        // render_kernel2<<<gridSize, blockSize>>>(colorBuffer, width, height, data.center, data.delta_u, data.delta_v, data.pixel000, data.samples, data.scale, d_world, r_state);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize() );
        stop = clock();
        double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
        std::cerr << "took " << timer_seconds << " seconds.\n";


        renderer.render(colorBuffer);


        freeWorld<<<1, 1>>>(d_list, d_world);
        cudaFree(colorBuffer);
        
        cudaFree(d_list);
        cudaFree(d_world);
        cudaFree(d_states);
        // cudaFree(r_state);

    }

    CudaCall::~CudaCall()
    {
    }

     rayos::CudaCall::CudaCall(Window& window) : window(window)
    {
    }

    

} // namespace