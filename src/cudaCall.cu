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


    __global__ void createWorld(hittable** list, hittable** world, MyCam** camera, int width, int height, int samples, int depth){
        // if (threadIdx.x == 0 && blockIdx.x == 0){
            auto R = cos(M_PI / 4.0f);


            // list[0]     = new sphere(vec3(-R, 0.0f, -1.0f), R, new lambertian(vec3(0.f, 0.f, 1.0f)));
            // list[1]     = new sphere(vec3(R, 0.0f, -1.0f), R, new lambertian(vec3(1.0f, 0.0f, 0.0f)));

            list[0]     = new sphere(vec3(0.0f, 0.0f, -1.2f), 0.5f, new lambertian(vec3(0.1f, 0.2f, 0.5f)));
            list[1]     = new sphere(vec3(0.0f, -100.5f, -1.0f), 100.0f, new lambertian(vec3(0.8f, 0.8f, 0.0f)));
            list[2]     = new sphere(vec3(-1.0f, 0.0f, -1.0f), 0.5f, new dielectric(1.50f));
            list[3]     = new sphere(vec3(-1.0f, 0.0f, -1.0f), 0.4f, new dielectric(1.0f / 1.50f));
            list[4]     = new sphere(vec3(1.0f, 0.0f, -1.0f), 0.5f, new metal(vec3(0.8f, 0.6f, 0.2f), 1.0f));
            *world      = new hittable_list(list, 5);  // the list has 5 spheres
            *camera     = new MyCam(width, height);  
            (*camera)->samples_per_pixel = samples;
            (*camera)->depth = depth;
            // (*camera)->camera_center = vec3(0.0f, 0.0f, 1.0f);
            (*camera)->update();      
            
        // }
    }

    

    __global__ void render_kernel(uint32_t* buffer, int width, int height, MyCam** camera, hittable** world, curandState_t* states){
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        int idx = width * j + i;
       
        if (idx >= (width * height)) return;
        vec3 color = vec3(0.0f, 0.0f, 0.0f);
        for (int x = 0; x < (*camera)->samples_per_pixel; x++){
            ray r = (*camera)->get_ray(i, j, states);
            color += ray_color(r, world, (*camera)->depth, states, i, j);

        }

        // vec3 color = ray_color(r, world);
        color *= (*camera)->sample_scale;
        buffer[idx] = colorToUint32_t(color);
        
    }


    __global__ void freeWorld(hittable** list, hittable** world, MyCam** camera){
      
        for (int i = 0; i < 5; i++){
            delete((sphere *)list[i])->mat_ptr;
            delete list[i];
        }
        delete *world;
        delete *camera;

    }


    void CudaCall::cudaCall(int width, int height, int samples, int depth, bool info_flag)
    {
        
        
        Renderer renderer{window};
        uint32_t* colorBuffer;  
        checkCudaErrors(cudaMallocManaged(&colorBuffer, width * height * sizeof(uint32_t)));

        
        // Create world  (create pointers to different classes)

        hittable** d_list;
        checkCudaErrors(cudaMalloc((void**)&d_list, 5 * sizeof(hittable*)) );
        hittable** d_world;
        checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*) ));
        MyCam** d_camera;
        checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(MyCam*) ));

       
        createWorld<<<1, 1>>>(d_list, d_world, d_camera, width, height, samples, depth);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize() );

        
        


        clock_t start, stop;
        start = clock();
        int threads = 16;
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


        render_kernel<<<gridSize, blockSize>>>(colorBuffer, width, height, d_camera, d_world, d_states);
        // render_kernel2<<<gridSize, blockSize>>>(colorBuffer, width, height, data.center, data.delta_u, data.delta_v, data.pixel000, data.samples, data.scale, d_world, r_state);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize() );
        stop = clock();
        double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
        if (info_flag)
            std::cerr << "took " << timer_seconds << " seconds with " << samples << " samples and depth of " << depth << "\n";


        renderer.render(colorBuffer);


        freeWorld<<<1, 1>>>(d_list, d_world, d_camera);
        cudaFree(colorBuffer);
        
        cudaFree(d_list);
        cudaFree(d_world);
        cudaFree(d_states);
        cudaFree(d_camera);
        // cudaFree(r_state);

    }

    CudaCall::~CudaCall()
    {
    }

     rayos::CudaCall::CudaCall(Window& window) : window(window)
    {
    }

    

} // namespace