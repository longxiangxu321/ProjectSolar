#pragma once

#include "gdt/math/AffineSpace.h"
#include <vector>
#include <sstream>
#include <fstream> 
#include <tuple>
#include "3rdParty/json.hpp"
#include <unordered_set>
#include <filesystem>
using json = nlohmann::json;

namespace solarcity {
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

    void obtain_model_statistics_and_save_area_info(const Model* model, const std::string filename);
    // void obtain_model_statistics_and_save_area_info(const Model* model);

    std::vector<vec3f> get_coordinates(const json& j, bool translate);

    Model *loadOBJ(const std::string &objFile);

    Model *loadCityJSON(const std::string &jsonFile);

    Model *loadWeatherStation(const std::string &jsonFile);

    Model *loadTUDelft(const std::string &jsonFile, const std::string &objFile);

    

    // void loadTUDOBJ(const std::string &objFile);
    

}