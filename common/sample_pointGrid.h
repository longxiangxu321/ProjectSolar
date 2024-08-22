#ifndef PG_H
#define PG_H

#include <sstream>
#include <vector>
#include <fstream>
#include "Model.h"

using json = nlohmann::json;

namespace osc {

    struct Triangle_info {
        vec3f direction;
        uint32_t surf_gmlid;
        int surface_type;
        float surface_albedo;
    };



    struct GridPoint {
    vec3f position;
    Triangle_info triangle_info;
    // std::vector<vec3> hemisphere_samples;

    GridPoint() = default;
    GridPoint(const vec3f ori, const Triangle_info info): position(ori), triangle_info(info) {}
    };






    void calculate_mass_center(const vec3f &A, const vec3f &B, const vec3f &C, const int splits,std::vector<GridPoint> &grid_n, const int &current_depth,
                            const int &max_depth, const Triangle_info triangle_info);



    std::vector<GridPoint> create_point_grid(const Model& citymodel, float sampling_density);

    float inline calculate_dist(vec3f a, vec3f b) {
        return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2) + pow(a.z - b.z, 2));
    }

    void save_point_grid(const std::vector<GridPoint> &grid_n, const vec3f &translation, const std::string &filename);


    std::vector<GridPoint> clean_point_grid(const std::vector<GridPoint> &grid_n);


    
    
}   

#endif