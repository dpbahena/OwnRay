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

    
    
    #define RNDi            curand_uniform(&state_x)
    #define RNDj            curand_uniform(&state_y)
    #define RND(a, b) ((a) + (b - (a)) * curand_uniform(&state_x))


    __global__ void createRandomHittableList(hittable** list, int across_x, int across_z, curandState_t* states, int* count){

        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        curandState_t state_x;
        curandState_t state_y;
        

        if (i >= across_x  || j >= across_z ) return;
        if (threadIdx.x == 0 && blockIdx.x == 0 && threadIdx.y == 0 && blockIdx.y == 0){ 
        // Load and initialize the random states for each thread
            state_x = states[i + j * across_x];
            state_y = states[i + j * across_x];
            
        }

        // Calculate positions directly using the thread indices
        float base_m = -across_x / 2.0 + ((float)i / (float)across_x) * across_x; // -11 to 11 range if across_x = 22
        float base_n = -across_x / 2.0 + ((float)j / (float)across_z) * across_x; // -11 to 11 range
    
        

        // Add random offset within a smaller range to avoid alignment
        point center = vec3(base_m + (RNDi - 0.5f) * 0.9f, 0.2f, base_n + (RNDj - 0.5f) * 0.9f);
        material* sphere_material;
        float fuzz;
        if (glm::length(center - point(4.0f, 0.2f, 0.0f)) > 0.9f){
            // atomicAdd(count, 1);
        
            
            float choose_mat = RNDi;
            // printf("count:%d\n", current_count);
            if (choose_mat < 0.8f){
                vec3 albedo = vec3(RNDi, RNDi, RNDi) * vec3(RNDj, RNDj, RNDj);
                sphere_material = new lambertian(albedo);
                list[across_x * j + i] = new sphere(center, 0.2f, sphere_material);
            } else if (choose_mat < 0.95f) {
                vec3 albedo = vec3(0.5f * (1.0f + RNDi), 0.5f * (1.0f + RNDj), 0.5f * (1.0f + RNDi));
                fuzz = 0.5f * RNDj;
                sphere_material = new metal(albedo, fuzz);
                list[across_x * j + i] = new sphere(center, 0.2f, sphere_material);
            } else {
                sphere_material = new dielectric(1.5f);
                list[across_x * j + i] = new sphere(center, 0.2f, sphere_material);
            }
        } else {
            // // Retry loop to ensure the new center is far enough from the fixed point
            // do {
            //     center = vec3(across_x / 2.0f * RND(-1.0f, 1.0f), 0.2f, -across_x / 2.0f * RND(-1.0f, 1.0f));
            // } while (glm::length(center - point(4.0f, 0.2f, 0.0f)) <= 0.9f);
            center = vec3(across_x / 2.0  * RND(-1.0f, 1.0f), 0.2f, -across_x/2.0f * RND(-1.0f, 1.0f));    
            vec3 albedo = vec3(RNDi, RNDi, RNDi) * vec3(RNDj, RNDj, RNDj);
            fuzz = 0.5f * RNDj;
            sphere_material = new metal(albedo, fuzz);
            // sphere_material = new dielectric(1.75);
            list[across_x * j + i] = new sphere(center, 0.2f, sphere_material);
            
            
        }
        // Save back the updated random states to insure continued randomness
            states[i + j * across_x] = state_x;
            states[i + j * across_x] = state_y;
        
        
    }





    

    
    // #define RND(a, b) ((a) + (b - (a)) * curand_uniform(&state_x))
    __global__ void createRandomHittableListIterate(hittable** list, curandState_t* states){
        if (threadIdx.x == 0 && blockIdx.x == 0){
            curandState_t state_x = *states;
            float choose_mat;
            material* sphere_material;
            
            int idx = 0;
            for(int a = -11; a < 11; a++) {
                for (int b = -11; b < 11; b++) {
                    choose_mat = RNDi;
                    point center = vec3(a + 0.9f * RNDi  , 0.2f, b + 0.9f *  RNDi);
                    if (glm::length(center - point(4.0f, 0.2f, 0.0f)) > 0.9f){
                        
                        if (choose_mat < 0.8f){
                            //diffuse
                            vec3 albedo =  vec3(RNDi, RNDi, RNDi) * vec3(RNDi, RNDi, RNDi);
                            sphere_material = new lambertian(albedo);
                            list[idx++] = new sphere(center, 0.2f, sphere_material);
                        } else if (choose_mat < 0.95f) {
                            // metal
                            vec3 albedo =  vec3(0.5f * (1.0f + RNDi), 0.5f * (1.0f + RNDi), 0.5f * (1.0f + RNDi));
                            float fuzz = 0.5f * RNDi;
                            sphere_material = new metal(albedo, fuzz);
                            list[idx++] = new sphere(center, 0.2f, sphere_material);
                        } else {
                            // glass
                            sphere_material = new dielectric(1.5f);
                            list[idx++] = new sphere(center, 0.2f, sphere_material);
                        }
                    } else {
                        float fuzz;
                        center = vec3(11.0f  * RND(-1.0f, 1.0f), 0.2f, -11.0f * RND(-1.0f, 1.0f));    
                        vec3 albedo = vec3(RNDi, RNDi, RNDi) * vec3(RNDi, RNDi, RNDi);
                        fuzz = 0.5f * RNDi;
                        sphere_material = new metal(albedo, fuzz);
                        // sphere_material = new dielectric(1.75);
                        list[idx++] = new sphere(center, 0.2f, sphere_material);
                        
                        
                    }

                }
            }
                
        }

    }


    __global__ void createWorld1(hittable** list, hittable** world, MyCam** camera, int width, int height, int samples, int depth, int randomList, int N){

        lambertian* ground_material = new lambertian(vec3(0.5f, 0.5f, 0.5f));
        metal*      left_material   = new metal(vec3(0.7f, 0.6f, 0.5f), 0.0f);
        dielectric* middle_material = new dielectric(1.5f);
        lambertian* right_material  = new lambertian(vec3(0.4f, 0.2f, 0.1f));

        int diff = N - randomList;
        // int dem = abs(4 - diff);
        // randomList -= dem;  
        
        if (diff == 1) {
            printf("dif: %d\n",diff);
            list[randomList + 0] = new sphere(point(0, -1000.0f, 0.0f), 1000.0f, ground_material);
            delete left_material;
            delete right_material;
            delete middle_material;
            *world      = new hittable_list(list, randomList + 1 );  // the list has N + 4 spheres


        }else if (diff == 2) {
            printf("dif: %d\n",diff);
            list[randomList + 0] = new sphere(point(0, -1000.0f, 0.0f), 1000.0f, ground_material);
            list[randomList + 1] = new sphere(point(0.0f, 1.0f, 0.0f), 1.0f, middle_material);
            delete left_material;
            delete right_material;
            *world      = new hittable_list(list, randomList + 2 );  // the list has N + 4 spheres

        } else if (diff == 3) {
            printf("dif: %d\n",diff);
            list[randomList + 0] = new sphere(point(0, -1000.0f, 0.0f), 1000.0f, ground_material);
            list[randomList + 1] = new sphere(point(0.0f, 1.0f, 0.0f), 1.0f, middle_material);
            list[randomList + 2] = new sphere(point(-4.0f, 1.0f, 0.0f), 1.0f, right_material);
            delete left_material;
            *world      = new hittable_list(list, randomList + 3 );  // the list has N + 4 spheres


        } else  if (diff == 4) {
            printf("dif: %d\n",diff);
            list[randomList + 0] = new sphere(point(0, -1000.0f, 0.0f), 1000.0f, ground_material);
            list[randomList + 1] = new sphere(point(0.0f, 1.0f, 0.0f), 1.0f, middle_material);
            list[randomList + 2] = new sphere(point(-4.0f, 1.0f, 0.0f), 1.0f, right_material);
            list[randomList + 3] = new sphere(point(4.0f, 1.0f, 0.0f), 1.0f, left_material);
            *world      = new hittable_list(list, randomList + 4 );  // the list has N + 4 spheres

        } else if (diff == 0){
            printf("dif: %d\n",diff);
            delete left_material;
            delete right_material;
            delete middle_material;
            delete ground_material;
            *world      = new hittable_list(list, randomList);  // the list has N + 4 spheres

        
        }

        if (abort) assert(diff <= 4);
        
        /* N is the number of random spheres already allocated in the list.
         *  Now we will just continue adding more spheres
         */
        
        // list[randomList + 0] = new sphere(point(0, -1000.0f, 0.0f), 1000.0f, ground_material);
        // list[randomList + 1] = new sphere(point(0.0f, 1.0f, 0.0f), 1.0f, middle_material);
        // list[randomList + 2] = new sphere(point(-4.0f, 1.0f, 0.0f), 1.0f, right_material);
        // list[randomList + 3] = new sphere(point(4.0f, 1.0f, 0.0f), 1.0f, left_material);
        
        
        // *world      = new hittable_list(list, randomList + 2 );  // the list has N + 4 spheres
        printf("camera update\n");
        
        *camera     = new MyCam(width, height);  
        (*camera)->samples_per_pixel = samples;
        (*camera)->depth = depth;
        (*camera)->update();  
        



    }

    __global__ void createWorld(hittable** list, hittable** world, MyCam** camera, int width, int height, int samples, int depth, int N){

        lambertian* ground_material = new lambertian(vec3(0.5f, 0.5f, 0.5f));
        metal*      left_material   = new metal(vec3(0.7f, 0.6f, 0.5f), 0.0f);
        dielectric* middle_material = new dielectric(1.5f);
        lambertian* right_material  = new lambertian(vec3(0.4f, 0.2f, 0.1f));

        // // /* N is the number of random spheres already allocated in the list.
        // //  *  Now we will just continue adding more spheres
        // //  */
        
        list[N + 0] = new sphere(point(0, -1000.0f, 0.0f), 1000.0f, ground_material);
        list[N + 1] = new sphere(point(0.0f, 1.0f, 0.0f), 1.0f, middle_material);
        list[N + 2] = new sphere(point(-4.0f, 1.0f, 0.0f), 1.0f, right_material);
        list[N + 3] = new sphere(point(4.0f, 1.0f, 0.0f), 1.0f, left_material);
        
        
        *world      = new hittable_list(list, N + 4);  // the list has N + 4 spheres
        
        *camera     = new MyCam(width, height);  
        (*camera)->samples_per_pixel = samples;
        (*camera)->depth = depth;
        (*camera)->update();  
        
    }

    __global__ void copy_list (hittable** sourceList, hittable** destList, int N){
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= N ) return;
        // if(((sphere*)sourceList[idx])->mat_ptr)
            destList[idx] = sourceList[idx];
            // *(destList + idx) = *(sourceList + idx);
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


    __global__ void freeWorld(hittable** list, hittable** world, MyCam** camera, int N){
      
        for (int i = 0; i < N; i++){
            delete((sphere *)list[i])->mat_ptr;
            delete list[i];
        }
        delete *world;
        delete *camera;

    }

    __global__ void freeWorld(hittable** list, int N){
      
        for (int i = 0; i < N; i++){
            delete((sphere *)list[i])->mat_ptr;
            delete list[i];
        }
       

    }



    void CudaCall::cudaCall(int width, int height, int samples, int depth, bool info_flag)
    {
        // checkCudaErrors(cudaDeviceSynchronize() );
        Renderer renderer{window};
        uint32_t* colorBuffer;  
        // checkCudaErrors(cudaMallocManaged(&colorBuffer, width * height * sizeof(uint32_t)));
        checkCudaErrors(cudaMalloc((void**)&colorBuffer, width * height * sizeof(uint32_t)));
        

        int across_x = 22;
        int across_z = 22;
        int random_spheres = across_x * across_z;

        int threads = 8;
        dim3 blockSize(threads, threads);
        int blocks_x = (across_x  + blockSize.x - 1) / blockSize.x; 
        int blocks_y = (across_z  + blockSize.y - 1) / blockSize.y;
        dim3 gridSize(blocks_x, blocks_y);

        //generate random seed to be used in rayTracer kernel
        int num_threads = threads * threads * blocks_x * blocks_y;

        curandState_t* d_states_0;
        checkCudaErrors(cudaMalloc((void**)&d_states_0, num_threads * sizeof(curandState_t)) );
        init_random<<<gridSize, blockSize>>>(time(0), d_states_0);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize() );


        hittable** d_list;
        
        int total_spheres = random_spheres + 4;
        checkCudaErrors(cudaMalloc((void**)&d_list, total_spheres* sizeof(hittable*)) );  
        int* d_count;
        checkCudaErrors(cudaMalloc(&d_count, sizeof(int)));
        int zero = 0;
        checkCudaErrors(cudaMemcpy(d_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
        // int total_random_spheres = across_x * across_z;
                           
                     
        // createRandomHittableList<<<gridSize, blockSize>>>(d_list, across_x, across_z, d_states_0, d_count);
        // checkCudaErrors(cudaGetLastError());
        // checkCudaErrors(cudaDeviceSynchronize() );
        // int h_count;
        // checkCudaErrors(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
        // printf("number of random spheres %d\n", h_count);



        createRandomHittableListIterate<<<1, 1>>>(d_list, d_states_0);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize() );





        // create the world
        hittable** d_world;
        checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*) ));
        MyCam** d_camera;
        checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(MyCam*) ));
        createWorld<<<1, 1>>>(d_list, d_world, d_camera, width, height, samples, depth, random_spheres);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize() );


        


        clock_t start, stop;
        start = clock();
        threads = 8;
        dim3 blockSize1(threads, threads);
        blocks_x = (width + blockSize.x - 1) / blockSize.x;
        blocks_y = (height + blockSize.y - 1) / blockSize.y;
        dim3 gridSize1(blocks_x, blocks_y);

        //generate random seed to be used in rayTracer kernel
        num_threads = threads * threads * blocks_x * blocks_y;
        curandState_t* d_states;
        checkCudaErrors(cudaMalloc((void**)&d_states, num_threads * sizeof(curandState_t)) );
        init_random<<<gridSize1, blockSize1>>>(time(0), d_states);
        // checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize() );


        render_kernel<<<gridSize1, blockSize1>>>(colorBuffer, width, height, d_camera, d_world, d_states);
        // checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaDeviceSynchronize() );
        stop = clock();
        double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
        if (info_flag)
            std::cerr << "took " << timer_seconds << " seconds with " << samples << " samples and depth of " << depth << "\n";

        uint32_t* h_image;
        int allocation =  width * height *  sizeof(uint32_t);
        h_image = new uint32_t[allocation];
        checkCudaErrors(cudaMemcpy(h_image, colorBuffer, width * height * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        renderer.render(h_image);

        delete[] h_image;

        // c heckCudaErrors(cudaDeviceSynchronize() );

        
        freeWorld<<<1, 1>>>(d_list, d_world, d_camera, total_spheres);
        // freeWorld<<<1, 1>>>(d_list, h_count );
        
        
        
        
        cudaFree(d_world);
        cudaFree(d_states);
        cudaFree(d_states_0);
        cudaFree(d_camera);
        cudaFree(d_count);
        cudaFree(d_list);
        // cudaFree(d_list_final);
        cudaFree(colorBuffer);

        cudaDeviceReset();
        
        

        

    }

    CudaCall::~CudaCall()
    {
    }

     rayos::CudaCall::CudaCall(Window& window) : window(window)
    {
    }

    

} // namespace