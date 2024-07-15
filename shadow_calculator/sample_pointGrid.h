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

    GridPoint copy() {
        GridPoint gp;
        gp.position = position;
        gp.normal = normal;
        gp.surf_gmlid = surf_gmlid;
        return gp;
    }

    };




    void calculate_mass_center(const vec3f &A, const vec3f &B, const vec3f &C, const vec3f &direction, const int splits,
                            std::stringstream &output_stream, std::vector<GridPoint> &grid_n, const int &current_depth,
                            const int &max_depth, const uint32_t surf_gmlid);

    struct KDTree
    {
        GridPoint* kdtree;
        int* node_depth;
        int grid_size;
    };

    struct KNNResult
    {
        int K;
        uint32_t* indices;
        float* distances;
    };

    void buildKDTree(std::vector<GridPoint> &grid, GridPoint* kdtree, int* depth_list, uint32_t start, uint32_t end, uint32_t current_node, int depth);

    KNNResult findKNN(const KDTree &kdTree, const int K, float radius, const GridPoint &target);

    std::vector<GridPoint> create_point_grid(const Model& citymodel);

    float inline calculate_dist(vec3f a, vec3f b) {
        return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2) + pow(a.z - b.z, 2));
    }


    
    
}   

#endif