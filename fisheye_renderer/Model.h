#pragma once

#include "gdt/math/AffineSpace.h"
#include <vector>
#include <sstream>
#include <fstream> 
#include <tuple>

namespace osc {
    using namespace gdt;

    struct TriangleMesh {
        std::vector<vec3f> vertex;
        std::vector<vec3f> normal;
        std::vector<vec3i> index;
        std::vector<int> triangleID;

        std::vector<int> materialID;
        vec3f diffuse;

        int buildingID;

    };

    struct Model {
        ~Model()
        {
            for (auto mesh: meshes) delete mesh;
        }

        std::vector<TriangleMesh *> meshes;
        vec3f original_center;
        box3f bounds;

        void transformModel();
    };

    Model *loadOBJ(const std::string &objFile);

    

}