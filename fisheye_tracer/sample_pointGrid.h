#ifndef PG_H
#define PG_H

#include <sstream>
#include <vector>
#include <fstream>
// #include "include/Triangle.h"
// #include "include/vec3.h"
// #include "include/grid_point.h"
#include "json.hpp"
#include "Model.h"

using json = nlohmann::json;

namespace osc {
    struct GridPoint {
    vec3f position;
    vec3f normal;
    uint32_t surf_gmlid;
    // std::vector<vec3> hemisphere_samples;

    GridPoint() = default;
    GridPoint(const vec3f ori, const vec3f dir): position(ori), normal(dir) {}
    };


    void calculate_mass_center(const vec3f &A, const vec3f &B, const vec3f &C, const vec3f &direction, const int splits,
                            std::stringstream &output_stream, std::vector<GridPoint> &grid_n, const int &current_depth,
                            const int &max_depth, const uint32_t surf_gmlid);

    std::vector<GridPoint> create_point_grid(const Model& citymodel);
    
}

#endif