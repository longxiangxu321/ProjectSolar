#pragma once

#include "gdt/math/AffineSpace.h"
#include <vector>
#include <sstream>
#include <fstream> 
#include <tuple>
#include "3rdParty/json.hpp"
#include <unordered_set>
using json = nlohmann::json;

namespace osc {
    using namespace gdt;

    struct TriangleMesh {
        std::vector<vec3f> vertex;
        std::vector<vec3f> normal;
        std::vector<vec3i> index;
        // std::vector<int> triangleID;

        std::vector<int> globalID;
        std::vector<int> surfaceType;
        std::vector<float> albedo;
        vec3f diffuse;

        // int buildingID;

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

    std::vector<vec3f> get_coordinates(const json& j, bool translate);

    Model *loadOBJ(const std::string &objFile);

    Model *loadCityJSON(const std::string &jsonFile);

    Model *loadWeatherStation(const std::string &jsonFile);

    Model *loadTUDelft(const std::string &jsonFile);
    

}