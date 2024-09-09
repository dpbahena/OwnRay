#pragma once

#include "definitions.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/sort.h>

namespace rayos {

    __device__
    inline vec3 random_unit_vector(curandState_t* states,  int& i, int& j);
    __device__
    inline vec3 sample_square(curandState_t* states, int& i, int& j);
    __device__ inline bool near_zero(vec3 v);
    __device__
    inline vec3 reflect(const vec3& v, const vec3& n);
    __device__ inline vec3 refract(const vec3& uv, const vec3& n, float etai_over_etat);
    __device__ inline float random_float(curandState_t* state);
    __device__ inline float reflectance(float cosine, float refraction_index);
    __device__ inline vec3 random_in_unit_disk(curandState_t* states, int&i, int& j);
    template<typename T>
    __device__ void swap(T& a, T& b);



    class interval {
        public:
            float min, max;
            __device__ 
            interval() : min(FLT_MAX), max(-FLT_MAX) {} // default interval empty
            __device__
            interval (float min, float max) : min(min), max(max){}
            __device__
            interval (const interval& a, const interval& b) {
                /*  Create the interval tightly enclosing the two input intervals */
                min = a.min <= b.min ? a.min : b.min;
                max = a.max >= b.max ? a.max : b.max;

            }
            __device__
            float size() const {
                return max - min;
            }
            __device__ 
            bool contains(float x) const {
                return min <= x && x <= max;
            }
            __device__
            bool surrounds(float x) const {
                return min < x && x < max;
            }

            __device__
            interval expand(float delta) const {
                auto padding = delta / 2.0;
                return interval(min - padding, max + padding);
            }

            
            static const interval empty, universe;

    };

    


    class ray {
        public:
            
            __device__ ray(){}
            __device__ ray(const point& origin, const vec3& direction, float time) : origin_(origin), direction_(direction), tm(time) {}
            __device__ ray(const point& origin, const vec3& direction) : ray(origin, direction, 0.0f) {}

            
            __device__  const point& origin() const { return origin_; }
            __device__  const vec3& direction() const { return direction_;}

            __device__ float time() const { return tm; }

            __device__  point at(float t) const {
                return origin_ + t * direction_;
            }

        private:

        point origin_;
        vec3 direction_;
        float tm;
    };

    class material;  // placeholder

    class hit_record{
        public:
            point p;
            vec3 normal;
            float t;
            float u;
            float v;
            bool front_face;
            material* mat_ptr;

            __device__ 
            void set_face_normal(const ray&r, const vec3& outward_normal){
                /* sets the hit record notmal vector. Assume outward_normal is of unit length */
                front_face = glm::dot(r.direction(), outward_normal) < 0.0f;
                normal = front_face ? outward_normal : -outward_normal;
            }
            
    };

    

    // class aabb {
    //     public:
    //         __device__
    //         aabb() {} // default AABB is empty, since intervals are empty by default

    //         __device__
    //         aabb(const interval& x, const interval& y, const interval& z) : x(x), y(y), z(z) {}

    //         __device__
    //         aabb(const point& a, const point& b) {
    //             /* Treat the 2 points a and b as extrema for the bounding box so we don't require a particular min/max coordinate order */
    //             x = a.x <= b.x ? interval(a.x, b.x) : interval(b.x, a.x);
    //             y = a.y <= b.y ? interval(a.y, b.y) : interval(b.y, a.y);
    //             z = a.z <= b.z ? interval(a.z, b.z) : interval(b.z, a.z); 
    //         }
    //         __device__
    //         aabb(const aabb& box0, const aabb& box1) {
    //             x = interval(box0.x, box1.x);
    //             y = interval(box0.y, box1.y);
    //             z = interval(box0.z, box1.z);
    //         }

    //         __device__
    //         const interval& axis_interval(int n) const {

    //             if (n == 1) return y;
    //             if (n == 2) return z;
    //             return x;

    //         }
    //         __device__
    //         bool hit(const ray& r, interval ray_t) const {

    //             const point& ray_orig = r.origin();
    //             const vec3& ray_dir = r.direction();

    //             for (int axis = 0; axis < 3; axis++) {
    //                 const interval& ax = axis_interval(axis);
    //                 const float adinv = 1.0f / ray_dir[axis];

    //                 auto t0 = (ax.min - ray_orig[axis]) * adinv;
    //                 auto t1 = (ax.max - ray_orig[axis]) * adinv;

    //                 if (t0 < t1) {
    //                     if (t0 > ray_t.min) ray_t.min = t0;
    //                     if (t1 < ray_t.max) ray_t.max = t1;
    //                 } else {
    //                     if (t1 > ray_t.min) ray_t.min = t1;
    //                     if (t0 < ray_t.max) ray_t.max = t0;
    //                 }

    //                 if (ray_t.max <= ray_t.min) {
    //                     return false;
    //                 }
    //             }

    //             return true;
    //         }

            
    //         interval x, y, z;


    //     private:

    // };

    class AABB {
    public:
        point minimum, maximum;

        __device__ AABB() {}
        __device__ AABB(const point& a, const point& b) { minimum = a; maximum = b; }

        __device__ bool hit(const ray& r, interval ray_t) const {
            for (int a = 0; a < 3; a++) {
                auto invD = 1.0f / r.direction()[a];
                auto t0 = (minimum[a] - r.origin()[a]) * invD;
                auto t1 = (maximum[a] - r.origin()[a]) * invD;
                if (invD < 0.0f) swap(t0, t1);
                ray_t.min = t0 > ray_t.min ? t0 : ray_t.min;
                ray_t.max = t1 < ray_t.max ? t1 : ray_t.max;
                if (ray_t.max <= ray_t.min) return false;
            }
            return true;
        }

        __device__ static AABB surrounding_box(const AABB& box0, const AABB& box1) {
            point small(fmin(box0.minimum.x, box1.minimum.x),
                        fmin(box0.minimum.y, box1.minimum.y),
                        fmin(box0.minimum.z, box1.minimum.z));
            point big(fmax(box0.maximum.x, box1.maximum.x),
                    fmax(box0.maximum.y, box1.maximum.y),
                    fmax(box0.maximum.z, box1.maximum.z));
            return AABB(small, big);
        }
    };


    // class hittable {
    //     public:
            
    //         __device__ 
    //         virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const = 0;
    //         __device__
    //         virtual aabb bounding_box() const = 0;

    // };

    class hittable {
        public:
            __device__ 
            virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const = 0;

            __device__
            virtual bool bounding_box(float time0, float time1, AABB& output_box) const = 0;
    };

    

    class sphere : public hittable {
    public:
        point center_start, center_end;  // Start and end points of the sphere movement
        float time_start, time_end;      // Time range for the sphere's movement
        float radius;
        material* mat_ptr;

        // Constructor for static sphere
        __device__
        sphere(const point& static_center, float r, material* mat)
            : center_start(static_center), center_end(static_center),
              time_start(0), time_end(0), radius(r), mat_ptr(mat) {}

        // Constructor for moving sphere
        __device__
        sphere(const point& center1, const point& center2, float t0, float t1, float r, material* mat)
            : center_start(center1), center_end(center2),
              time_start(t0), time_end(t1), radius(r), mat_ptr(mat) {}

        // Method to get the sphere's center at a given time
        __device__
        point center(float time) const {
            // Linear interpolation between center_start and center_end based on the time
            return center_start + ((time - time_start) / (time_end - time_start)) * (center_end - center_start);
        }

        // Hit method to detect ray-sphere intersections
        __device__
        bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
            point sphere_center = center(r.time());
            vec3 oc = r.origin() - sphere_center;
            float a = glm::dot(r.direction(), r.direction());
            float half_b = glm::dot(oc, r.direction());
            float c = glm::dot(oc, oc) - radius * radius;
            float discriminant = half_b * half_b - a * c;

            if (discriminant < 0) return false;

            float sqrtd = sqrt(discriminant);

            // Find the nearest root that lies in the acceptable range
            float root = (-half_b - sqrtd) / a;
            if (!ray_t.surrounds(root)) {
                root = (-half_b + sqrtd) / a;
                if (!ray_t.surrounds(root))
                    return false;
            }

            rec.t = root;
            rec.p = r.at(rec.t);
            vec3 outward_normal = (rec.p - sphere_center) / radius;
            rec.set_face_normal(r, outward_normal);
            rec.mat_ptr = mat_ptr;
            return true;
        }

        // Bounding box method to return the AABB for the sphere
        __device__
        bool bounding_box(float time0, float time1, AABB& output_box) const override {
            // Calculate bounding boxes for both the start and end times of the sphere's movement
            AABB box_start(
                center(time0) - vec3(radius, radius, radius),
                center(time0) + vec3(radius, radius, radius)
            );
            AABB box_end(
                center(time1) - vec3(radius, radius, radius),
                center(time1) + vec3(radius, radius, radius)
            );

            output_box = AABB::surrounding_box(box_start, box_end);
            return true;
        }
    };


    // class sphere : public hittable {
    // public:
        
    //     point center_start, center_end;  // Start and end points of the sphere movement
    //     float time_start, time_end;
    //     // point center;
    //     float radius;
    //     material* mat_ptr;

    //     // Constructor for stationary sphere
    //     __device__ 
    //     sphere(const point& static_center, float r, material* mat) 
    //         : center(static_center), radius(r), mat_ptr(mat) {}

    //     // // Constructor for moving sphere
    //     // __device__ 
    //     // sphere(const point& center1, const point& center2, float time1, float time2, float r, material* mat) 
    //     //     : center(center1, center2), radius(r), mat_ptr(mat) {}

    //     // Constructor for moving sphere
    //     __device__ 
    //     sphere(const point& center1, const point& center2, float time1, float time2, float r, material* mat)
    //         : center_start(center1), center_end(center2), time_start(time1), time_end(time2), radius(r), mat_ptr(mat) {}

    //     __device__ 
    //     point center(float time) const {
    //         return center_start + ((time - time_start) / (time_end - time_start)) * (center_end - center_start);
    //     }

    //     // Implement hit function as before
    //     __device__ 
    //     bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
    //         vec3 oc = r.origin() - center;
    //         float a = glm::dot(r.direction(), r.direction());
    //         float half_b = glm::dot(oc, r.direction());
    //         float c = glm::dot(oc, oc) - radius * radius;
    //         float discriminant = half_b * half_b - a * c;

    //         if (discriminant < 0) return false;

    //         float sqrtd = sqrt(discriminant);

    //         // Find the nearest root that lies in the acceptable range
    //         float root = (-half_b - sqrtd) / a;
    //         if (!ray_t.surrounds(root)) {
    //             root = (-half_b + sqrtd) / a;
    //             if (!ray_t.surrounds(root))
    //                 return false;
    //         }

    //         rec.t = root;
    //         rec.p = r.at(rec.t);
    //         vec3 outward_normal = (rec.p - center) / radius;
    //         rec.set_face_normal(r, outward_normal);
    //         rec.mat_ptr = mat_ptr;
    //         return true;
    //     }

    //     // Implement bounding_box function
    //     __device__
    //     bool bounding_box(float time0, float time1, aabb& output_box) const override {
    //         output_box = aabb(
    //             center - vec3(radius, radius, radius),
    //             center + vec3(radius, radius, radius)
    //         );
    //         return true;
    //     }
    // };



    // class hittable_list : public hittable {
    //     public:
    //     hittable** list;
    //     int list_size;
    //     aabb bbox;

    //     __device__ 
    //     hittable_list() {}
    //     __device__ 
    //     hittable_list(hittable **list, int n) : list(list), list_size(n)  {
    //         for(int i = 0; i < list_size; i++){
    //             bbox = aabb(bbox, list[i]->bounding_box());
    //         }
    //     }
    //     // __device__
    //     // void clear() {list_size = 0; }

    //     __device__ 
    //     bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
    //         hit_record temp_rec;
    //         bool hit_anything = false;
    //         auto closest_so_far = ray_t.max;

    //         for (int i = 0; i < list_size; i++) {
    //             if (list[i]->hit(r, interval(ray_t.min, closest_so_far), temp_rec)){
    //                 hit_anything = true;
    //                 closest_so_far = temp_rec.t;
    //                 rec = temp_rec;
    //             }
    //         }

    //         return hit_anything;
    //     }
    //     __device__
    //     aabb bounding_box() const override { return bbox; }
    // };


    class hittable_list : public hittable {
    public:
        hittable** list;
        int list_size;

        __device__ hittable_list() {}
        __device__ hittable_list(hittable** l, int n) : list(l), list_size(n) {}

        __device__ 
        bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
            hit_record temp_rec;
            bool hit_anything = false;
            auto closest_so_far = ray_t.max;

            for (int i = 0; i < list_size; i++) {
                if (list[i]->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
            }

            return hit_anything;
        }

        // Implement bounding_box function
        __device__
        bool bounding_box(float time0, float time1, AABB& output_box) const override {
            if (list_size == 0) return false;

            AABB temp_box;
            bool first_box = true;

            for (int i = 0; i < list_size; i++) {
                if (!list[i]->bounding_box(time0, time1, temp_box)) return false;
                output_box = first_box ? temp_box : AABB::surrounding_box(output_box, temp_box);
                first_box = false;
            }

            return true;
        }
    };


    // class bvh_node : public hittable {
    //     public:
    //         // bvh_node(hittable** objects) : bvh_node(objects, 0, )
    //         __device__
    //         bvh_node(hittable** objects, size_t start, size_t end, curandState_t* states){
    //             curandState_t  x = states[0];
    //             // int axis = (int)random_float_range(&x, 0.0, 2.0f);
    //             int axis = (int)(3 * random_float(&x)); // 0, 1, or 2

    //             auto comparator = (axis == 0) ? box_x_compare : (axis == 1) ? box_y_compare : box_z_compare;

    //             size_t object_span = end - start;
    //             if (object_span == 1) {
    //                 left = right = objects[start];
    //             } else if (object_span == 2) {
    //                 left = objects[start];
    //                 right = objects[start + 1];
    //             } else {
    //                 thrust::sort(thrust::device, objects + start, objects + end, comparator);
    //                 auto mid = start + object_span / 2;

    //                 left = new bvh_node(objects, start, mid, states);
    //                 right = new bvh_node(objects, mid, end, states);
    //             }
    //             bbox = aabb(left->bounding_box(), right->bounding_box());

    //         }
    //         __device__
    //         bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
    //             if (!bbox.hit(r, ray_t)) return false;
    //             bool hit_left = left->hit(r, ray_t, rec);
    //             bool hit_right = right->hit(r, interval(ray_t.min, hit_left ? rec.t : ray_t.max), rec);

    //             return hit_left || hit_right;
    //         }
    //         __device__
    //         aabb bounding_box() const override { return bbox; }
            
    //     private:
    //         hittable* left;
    //         hittable* right;
    //         aabb bbox;

    //         __device__
    //         static bool box_compare( hittable* a, hittable* b, int axis_index) {
    //             auto a_axis_interval = a->bounding_box().axis_interval(axis_index);
    //             auto b_axis_interval = b->bounding_box().axis_interval(axis_index);
    //             return a_axis_interval.min < b_axis_interval.min;

    //         }
    //         __device__
    //         static bool box_x_compare (hittable* a, hittable* b){
    //             return box_compare(a, b, 0);
    //         }

    //         __device__
    //         static bool box_y_compare (hittable* a, hittable* b){
    //             return box_compare(a, b, 1);
    //         }

    //         __device__
    //         static bool box_z_compare (hittable* a, hittable* b){
    //             return box_compare(a, b, 2);
    //         }

    // };
    // class bvh_node : public hittable {
    // public:
    //     hittable* left;
    //     hittable* right;
    //     AABB box;

    //     __device__ bvh_node() {}

    //     __device__ bvh_node(hittable** list, int start, int end, curandState_t* state) {
    //         int axis = int(3 * random_float(state));
    //         auto comparator = (axis == 0) ? box_x_compare : (axis == 1) ? box_y_compare : box_z_compare;

    //         int object_span = end - start;

    //         if (object_span == 1) {
    //             left = right = list[start];
    //         } else if (object_span == 2) {
    //             if (comparator(list[start], list[start + 1])) {
    //                 left = list[start];
    //                 right = list[start + 1];
    //             } else {
    //                 left = list[start + 1];
    //                 right = list[start];
    //             }
    //         } else {
    //             thrust::sort(thrust::device, list + start, list + end, comparator);
    //             auto mid = start + object_span / 2;
    //             left = new bvh_node(list, start, mid, state);
    //             right = new bvh_node(list, mid, end, state);
    //         }

    //         AABB box_left, box_right;

    //         if (!left->bounding_box(0, 0, box_left) || !right->bounding_box(0, 0, box_right))
    //             printf("No bounding box in bvh_node constructor.\n");

    //         box = AABB::surrounding_box(box_left, box_right);
    //     }

    //     __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
    //         if (!box.hit(r, ray_t))
    //             return false;

    //         bool hit_left = left->hit(r, ray_t, rec);
    //         bool hit_right = right->hit(r, interval(ray_t.min, hit_left ? rec.t : ray_t.max), rec);

    //         return hit_left || hit_right;
    //     }

    //     __device__ bool bounding_box(float time0, float time1, AABB& output_box) const {
    //         output_box = box;
    //         return true;
    //     }

    //     __device__ static bool box_compare(const hittable* a, const hittable* b, int axis) {
    //         AABB box_a, box_b;
    //         if (!a->bounding_box(0, 0, box_a) || !b->bounding_box(0, 0, box_b))
    //             printf("No bounding box in bvh_node constructor.\n");
    //         return box_a.minimum[axis] < box_b.minimum[axis];
    //     }

    //     __device__ static bool box_x_compare(const hittable* a, const hittable* b) { return box_compare(a, b, 0); }
    //     __device__ static bool box_y_compare(const hittable* a, const hittable* b) { return box_compare(a, b, 1); }
    //     __device__ static bool box_z_compare(const hittable* a, const hittable* b) { return box_compare(a, b, 2); }
    // };

    class bvh_node : public hittable {
public:
    hittable* left;
    hittable* right;
    AABB box;

    __device__ bvh_node() {}

    __device__ bvh_node(hittable** list, int start, int end, curandState_t* state) {
        int axis = int(3 * random_float(state));  // Randomly choose an axis
        auto comparator = (axis == 0) ? box_x_compare : (axis == 1) ? box_y_compare : box_z_compare;

        int object_span = end - start;

        if (object_span == 1) {
            left = right = list[start];
        } else if (object_span == 2) {
            if (comparator(list[start], list[start + 1])) {
                left = list[start];
                right = list[start + 1];
            } else {
                left = list[start + 1];
                right = list[start];
            }
        } else {
            thrust::sort(thrust::device, list + start, list + end, comparator);
            auto mid = start + object_span / 2;
            left = new bvh_node(list, start, mid, state);
            right = new bvh_node(list, mid, end, state);
        }

        AABB box_left, box_right;

        if (!left->bounding_box(0, 0, box_left) || !right->bounding_box(0, 0, box_right)) {
            printf("No bounding box in bvh_node constructor.\n");
        }

        box = AABB::surrounding_box(box_left, box_right);
    }

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        if (!box.hit(r, ray_t))
            return false;

        bool hit_left = left->hit(r, ray_t, rec);
        bool hit_right = right->hit(r, interval(ray_t.min, hit_left ? rec.t : ray_t.max), rec);

        return hit_left || hit_right;
    }

    __device__ bool bounding_box(float time0, float time1, AABB& output_box) const {
        output_box = box;
        return true;
    }

    __device__ static bool box_compare(const hittable* a, const hittable* b, int axis) {
        AABB box_a, box_b;
        if (!a->bounding_box(0, 0, box_a) || !b->bounding_box(0, 0, box_b))
            printf("No bounding box in bvh_node constructor.\n");
        return box_a.minimum[axis] < box_b.minimum[axis];
    }

    __device__ static bool box_x_compare(const hittable* a, const hittable* b) { return box_compare(a, b, 0); }
    __device__ static bool box_y_compare(const hittable* a, const hittable* b) { return box_compare(a, b, 1); }
    __device__ static bool box_z_compare(const hittable* a, const hittable* b) { return box_compare(a, b, 2); }
};



    class MyCam {
        public:
            __device__
            MyCam(int width, int heigh) : image_width(width), image_height(heigh){
                aspectRatio = image_width / static_cast<float>(image_height);

                
            }

            point camera_center             = vec3(0.0f, 0.0f, 0.0f);
            vec3 pixel00_loc;
            vec3 pixel_delta_u;
            vec3 pixel_delta_v;
            int samples_per_pixel           = 50;
            float sample_scale;
            int depth                       = 5;
            float vFOV                       = 20.0f; // in degrees
            point lookfrom = point(13.0f,2.0f,3.0f);   // Point camera is looking from
            point lookat   = point(0.0f,0.0f,0.0f);  // Point camera is looking at
            vec3   vup      = vec3(0.0f,1.0f,0.0f);     // Camera-relative "up" direction

            float defocus_angle = 0.6f;  // Variation angle of rays through each pixel
            float focus_dist = 10.0;    // Distance from camera lookfrom point to plane of perfect focus


        __device__
        void update(){

            
            camera_center = lookfrom;
            sample_scale = 1.0f / static_cast<float>(samples_per_pixel);
            // Determine viewport dimensions

            // focal_length = glm::length(lookfrom - lookat);


            float h = tan(glm::radians(vFOV) / 2.0);
            viewport_height = 2.0f * h * focus_dist;
            float vieport_width = viewport_height * aspectRatio;

            // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
            w = glm::normalize(lookfrom - lookat);
            u = glm::normalize(cross(vup, w));
            v = glm::cross(w, u);
            
            /* Calculate the vectors across the horizontal and down the vertical viewport edges */
            viewport_u = vieport_width * u;
            viewport_v = -viewport_height * v;

            /* Calculate the horizontal and vertical delta vectors from pixel to pixel */
            pixel_delta_u = viewport_u / static_cast<float>(image_width);
            pixel_delta_v = viewport_v / static_cast<float>(image_height);

            /* Calculate the locations of the upper left pixel */
            auto viewport_upper_left = camera_center - (focus_dist * w) - viewport_u / 2.0f - viewport_v / 2.0f;
            pixel00_loc = viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);

            // Calculate the camera defocus disk basis vectors.
            auto defocus_radius = focus_dist * tan(glm::radians(defocus_angle / 2.0f));
            defocus_disk_u = u * defocus_radius;
            defocus_disk_v = v * defocus_radius;
            
        }

        __device__
        ray get_ray(int& i, int& j, curandState_t* states) const {
            /* Construct a camera ray originating from the defocus disk and directed at a randomly sampled point around the pixel location i, j. */
            
            auto offset = sample_square(states, i, j);
            auto pixel_sample = pixel00_loc + ((i + offset.x) * pixel_delta_u) + ((j + offset.y) * pixel_delta_v);

            auto ray_origin     = (defocus_angle <= 0) ? camera_center : defocus_disk_sample(states, i, j);
            auto ray_direction  = pixel_sample - ray_origin;
            curandState_t x = states[i];
            auto ray_time = random_float(&x);
            states[i] = x; // save value back


            return ray(ray_origin, ray_direction, ray_time); 

        }


        private:
            float viewport_height           = 2.0f;
            
            float focal_length              = 1.0f;
            vec3 viewport_u;
            vec3 viewport_v;
            
            
            int image_width;
            int image_height;
            float aspectRatio;
            vec3   u, v, w;              // Camera frame basis vectors
            vec3   defocus_disk_u;       // Defocus disk horizontal radius
            vec3   defocus_disk_v;       // Defocus disk vertical radius

            __device__
            point defocus_disk_sample(curandState_t* states, int &i, int& j) const {
                /* returns a random point in the camera defocus disk */
                vec3 p = random_in_unit_disk(states, i, j);
                return camera_center + (p.x * defocus_disk_u) + (p.y * defocus_disk_v);

            }




    };
    

    class material {
        public:
        __device__
        virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState_t* states, int i, int j) const {return false;} //const = 0;
    };

    class lambertian : public material {
        public:
            __device__
            lambertian(const vec3& albedo) : albedo(albedo) {}
            __device__
            bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState_t* states, int i, int j) const override {
                vec3 scatter_direction = rec.normal + random_unit_vector(states, i, j);
                /* Catch if it is degenerate scatter_direction vector (avoids infinites and NaNs) */
                if (near_zero(scatter_direction))
                    scatter_direction = rec.normal;

                scattered = ray(rec.p, scatter_direction, r_in.time());
                attenuation = albedo;
                return true;
            }
        private:
            vec3 albedo;
    };


    class metal : public material {
        public:
            __device__
            metal(const vec3& albedo, float fuzz) : albedo(albedo), fuzz(fuzz < 1? fuzz : 1) {}
            __device__
            bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState_t* states, int i, int j) const override {
                vec3 reflected = reflect(r_in.direction(), rec.normal);
                reflected = glm::normalize(reflected) + (fuzz * random_unit_vector(states, i, j));
                scattered = ray(rec.p, reflected, r_in.time());
                attenuation = albedo;
                return (glm::dot(scattered.direction(), rec.normal) > 0);
            }
        private:
            vec3 albedo;
            float fuzz;
    };


    class dielectric : public material {
        public:
            __device__
            dielectric(float refraction_index) : refraction_index(refraction_index) {}
            __device__
            bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState_t* states, int i, int j) const override {
                attenuation = vec3(1.0f, 1.0f, 1.0f);
                float ri = rec.front_face ? (1.0f / refraction_index ) : refraction_index;
                vec3 unit_direction = glm::normalize(r_in.direction());
                float cos_theta = fmin(glm::dot(-unit_direction, rec.normal), 1.0f);
                float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
                bool cannot_refract = ri * sin_theta > 1.0f;
                vec3 direction;
                curandState_t x = states[i];
                if (cannot_refract || reflectance(cos_theta, ri) > random_float(&x)) {
                    direction = reflect(unit_direction, rec.normal);
                } else {
                    direction = refract(unit_direction, rec.normal, ri);
                }
                states[i] = x;  //saves back the value
                scattered = ray(rec.p, direction, r_in.time());
                return true;
            }
        private:
            /* Refractive index is vacuum or air, or the ratio of the material's ri over the ri of the enclosing media */
            float refraction_index;
            
    };

    template<typename T>
    __device__ void swap(T& a, T& b) {
        const T temp = a;
        a = b;
        b = temp;
    }





}

