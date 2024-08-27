# OwnRay
combine both my approach and JoeAllan 

# Chapter 2. 

![Screenshot of the project](sampleImages/chap2.png)

# Chapter 4.

![Screenshot of the project](sampleImages/chap4.png)

# Chapter 5.

![Screenshot of the project](sampleImages/chap5.png)

# Chapter 6.
## Introduction of abstract class and virtual functions

![Screenshot of the project](sampleImages/chap6.png)

# Chapter 7. Antialiasing
## Introduction of  curandState_t* states in Cuda
![Screenshot of the project](sampleImages/chap7.png)

# Chapter 8. Antialiasing
## Camera class as pointer
![Screenshot of the project](sampleImages/chap7.png)

# Chapter 9. Recursion vs Iteration
## Introduction to diffuse material 
![Screenshot of the project](sampleImages/chap9.png)
### Notes:
I could use both i and j  but using only i is faster by 3x:
vec3 random_vector(curandState_t* states,  int &i, int &j){
        curandState_t x = states[i];
        // curandState_t y = states[j];
        
        float a = random_float(&x);
        float b = random_float(&x);
        float c = random_float(&x); 
        
        states[i] = x; // save value back
        // states[j] = y; // save value back
        return vec3(a, b, c);
}

# Chapter 10. Abstract class of  Material
## Lambertian and Metal/fuzz
![Screenshot of the project](sampleImages/chap10_samples50_depth10.png)
![Screenshot of the project](sampleImages/chap10_samples500_depth100.png)

# Chapter 11. Camera movement 
![Screenshot of the project](sampleImages/chap11.png)
