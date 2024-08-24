#include "Model.h"
#include <map>
#include <array>

namespace osc {

    std::map<std::string, int> createSurfaceTypeMap() {
    std::map<std::string, int> surface_type_map;
    surface_type_map["WallSurface"] = 0;
    surface_type_map["RoofSurface"] = 1;
    surface_type_map["GroundSurface"] = 2;
    surface_type_map["TIN"] = 3;
    surface_type_map["Tree"] = 4;
    return surface_type_map;
    }

    std::map<std::string, int> surface_type_map = createSurfaceTypeMap();
    std::map<std::string, float> surface_albedo_map = {{"WallSurface", 0.2}, 
                                    {"RoofSurface", 0.1}, {"GroundSurface", 0.2}, 
                                    {"TIN", 0.2}, {"Tree", 0.2}};

    std::vector<vec3f> transformMesh(const std::array<float, 16>& matrix, const std::array<float, 3>& translate) {
        const std::vector<vec3f> mesh = {
            vec3f(-0.025f, -0.001f, 0.41f),
            vec3f(-0.025f, -0.001f, 0.0f),
            vec3f(0.025f, 0.001f, 0.0f),
            vec3f(0.025f, 0.001f, 0.41f),
            vec3f(0.0f, 0.0f, 0.4f),
            vec3f(0.001f, -0.025f, 0.41f),
            vec3f(0.001f, -0.025f, 0.0f),
            vec3f(-0.001f, 0.025f, 0.0f),
            vec3f(-0.001f, 0.025f, 0.41f),
            vec3f(-0.006f, 0.212f, 0.488f),
            vec3f(-0.008f, 0.299f, 0.7f),
            vec3f(-0.006f, 0.212f, 0.912f),
            vec3f(0.0f, 0.0f, 1.0f),
            vec3f(0.006f, -0.212f, 0.912f),
            vec3f(0.008f, -0.299f, 0.7f),
            vec3f(0.006f, -0.212f, 0.488f),
            vec3f(0.212f, 0.006f, 0.488f),
            vec3f(0.299f, 0.008f, 0.7f),
            vec3f(0.212f, 0.006f, 0.912f),
            vec3f(-0.212f, -0.006f, 0.912f),
            vec3f(-0.299f, -0.008f, 0.7f),
            vec3f(-0.212f, -0.006f, 0.488f)
        };

        std::vector<vec3f> transformedcoords;
        transformedcoords.reserve(mesh.size());

        for (const auto& vertex : mesh) {
            float x = vertex.x * matrix[0] + vertex.y * matrix[4] + vertex.z * matrix[8] + matrix[12] + translate[0];
            float y = vertex.x * matrix[1] + vertex.y * matrix[5] + vertex.z * matrix[9] + matrix[13] + translate[1];
            float z = vertex.x * matrix[2] + vertex.y * matrix[6] + vertex.z * matrix[10] + matrix[14] + translate[2];
            transformedcoords.emplace_back(x, y, z);
        }

        return transformedcoords;
    }


    // std::tuple<int, int> global local
    int addOrGetVertexIndex(std::map<int, int>& vertexMap, int globalIndex) {
      auto it = vertexMap.find(globalIndex);
      if (it != vertexMap.end()) {
          // Global index already exists, return the corresponding local index
          return it->second;
      } else {
          // Global index does not exist, insert it
          int localIndex = vertexMap.size();
          vertexMap[globalIndex] = localIndex;
          return localIndex;
      }
    }   

    void Model::transformModel() {
        for (auto mesh : meshes) {
            for (int i = 0; i < mesh->vertex.size(); i++) {
                vec3f translated_vertex = vec3f(mesh->vertex[i].x - bounds.lower.x, 
                                        mesh->vertex[i].y - bounds.lower.y, mesh->vertex[i].z - bounds.lower.z);
                mesh->vertex[i] = translated_vertex;
            }
        }

        bounds = box3f(bounds.lower - bounds.lower, bounds.upper - bounds.lower);
        
    }


    Model *loadOBJ(const std::string &objFile)
    {

    Model *model = new Model;

    std::ifstream input_stream;
    input_stream.open(objFile);
    std::vector<vec3f> points;


    bool first_object_read = false;
    int building_index = 0;
    int total_triangles = 0;
    TriangleMesh *mesh = nullptr;
    int matID = 0;
    std::map<int, int> vertexMap;

    if (input_stream.is_open()) {
        std::cout << "Reading " << objFile << std::endl;
        std::string line;
        while (std::getline(input_stream, line)) {
            if (line[0] == 'v') {
                double x,y,z;
                std::stringstream ss(line);
                std::string temp;
                ss >> temp >> x >> y >> z;
                double sx = x;
                double sy = y;
                points.push_back(vec3f(sx, sy, z));
                model->bounds.extend(vec3f(sx, sy, z));
            }
            if (line[0] == 'o') {//every o means a new object
                if (first_object_read) {
                    for (const auto& entry : vertexMap) {
                      mesh->vertex.push_back(points[entry.first]);
                    }
                    model->meshes.push_back(mesh);
                    mesh = new TriangleMesh;
                }
                else {
                    first_object_read = true;
                    mesh = new TriangleMesh;
                }
                vertexMap.clear();
                std::stringstream ss(line);
                std::string gmlid;
                std::string temp;
                ss >> temp >> gmlid;
                // mesh->diffuse = vec3f(0.8f, 0.f, 0.f);
                mesh->diffuse = gdt::randomColor(building_index);
                // mesh->buildingID = building_index;
                building_index++;
            }
            if (line[0] == 'u') {
                std::stringstream ss(line);
                std::string temp;
                ss >> temp >> matID;
            }
            if (line[0] == 'f') {//shell has different Faces
                unsigned long v0, v1, v2;
                std::stringstream ss(line);
                std::string temp;
                ss >> temp >> v0 >> v1 >> v2;

                int global_v0 = v0 -1;
                int global_v1 = v1 -1;
                int global_v2 = v2 -1;

                int local_v0 = addOrGetVertexIndex(vertexMap, global_v0);
                int local_v1 = addOrGetVertexIndex(vertexMap, global_v1);
                int local_v2 = addOrGetVertexIndex(vertexMap, global_v2);
                mesh->index.push_back(vec3i(local_v0, local_v1, local_v2));

                vec3f x0 = points[global_v0];
                vec3f x1 = points[global_v1];
                vec3f x2 = points[global_v2];

                vec3f normal = normalize(cross(x1 - x0, x2 - x0));
                mesh->normal.push_back(normal);
                mesh->globalID.push_back(total_triangles);
                mesh->surfaceType.push_back(matID);
                total_triangles++;
            }
            else {
                continue;
            }
        }
        if (mesh != nullptr) {
                std::cout<<"find last one"<<std::endl;
                for (const auto& entry : vertexMap) {
                      mesh->vertex.push_back(points[entry.first]);
                    }

                model->meshes.push_back(mesh);
            }

    }
    std::cout<<"bouding box low "<<model->bounds.lower.x<<" "<<model->bounds.lower.y<<" "<<model->bounds.lower.z<<std::endl;
    std::cout<<"bouding box high "<<model->bounds.upper.x<<" "<<model->bounds.upper.y<<" "<<model->bounds.upper.z<<std::endl;
    std::cout<<"Total building num " << building_index<<std::endl;
    std::cout<<"Total triangle num " << total_triangles<<std::endl;
    std::cout<<"Total vertex num " << points.size()<<std::endl;
    model->original_center = model->bounds.center();
    return model;

    }


    
    std::vector<vec3f> get_coordinates(const json& j, bool translate) {
        std::vector<vec3f> lspts;
        std::vector<std::vector<int>> lvertices = j["vertices"];
        if (translate) {
            for (auto& vi : lvertices) {
                double x = (vi[0] * j["transform"]["scale"][0].get<double>()) + j["transform"]["translate"][0].get<double>();
                double y = (vi[1] * j["transform"]["scale"][1].get<double>()) + j["transform"]["translate"][1].get<double>();
                double z = (vi[2] * j["transform"]["scale"][2].get<double>()) + j["transform"]["translate"][2].get<double>();
                lspts.push_back(vec3f(x, y, z));
            }
        } else {
            //-- do not translate, useful to keep the values low for downstream processing of data
            for (auto& vi : lvertices) {
                double x = (vi[0] * j["transform"]["scale"][0].get<double>());
                double y = (vi[1] * j["transform"]["scale"][1].get<double>());
                double z = (vi[2] * j["transform"]["scale"][2].get<double>());
                lspts.push_back(vec3f(x, y, z));
            }
        }
        return lspts;
    }

    Model *loadCityJSON(const std::string &jsonFile)
    {

        Model *model = new Model;
        std::unordered_set<std::string> targetTypes = {"WallSurface", "RoofSurface"};

        // std::map<int, int> vertexMap;

        std::fstream input(jsonFile);
        json j;
        input >> j;
        input.close();
        std::vector<vec3f> lspts = get_coordinates(j, true);
        for (auto const &lspt:lspts) {
            model->bounds.extend(lspt);
        }

        int building_index = 0;
        int total_triangles = 0;

        TriangleMesh *mesh = nullptr;
        std::map<int, int> vertexMap;

        for (auto &co: j["CityObjects"].items()) {
            if (co.value()["type"] == "BuildingPart") {
                vertexMap.clear();
                mesh = new TriangleMesh;

                for (auto &g: co.value()["geometry"]) {
                    for (int i = 0; i< g["boundaries"][0].size(); i++) {
                        std::vector<std::vector<int>> triangle = g["boundaries"][0][i];
                        if (triangle[0].size() != 3 || 
                            triangle[0][0]==triangle[0][1] || 
                            triangle[0][0]==triangle[0][2] || 
                            triangle[0][1]==triangle[0][2]) {
                            int gmlid_int = g["semantics"]["surfaces"][i]["global_idx"];

                            std::cout<<"a triangle is not valid, identifier: "<<gmlid_int<<std::endl;
                            continue;
                        }  
                        else{
                            // std::string surf_type = g["semantics"]["surfaces"][i]["type"];
                            // if (targetTypes.find(surf_type) != targetTypes.end()) {
                                int gmlid_int = g["semantics"]["surfaces"][i]["global_idx"];
                                int global_v0 = triangle[0][0];
                                int global_v1 = triangle[0][1];
                                int global_v2 = triangle[0][2];

                                int original_size = vertexMap.size();

                                int local_v0 = addOrGetVertexIndex(vertexMap, global_v0);

                                if (original_size != vertexMap.size()) {
                                    mesh->vertex.push_back(lspts[global_v0]);
                                }

                                original_size = vertexMap.size();
                                int local_v1 = addOrGetVertexIndex(vertexMap, global_v1);
                                if (original_size != vertexMap.size()) {
                                    mesh->vertex.push_back(lspts[global_v1]);
                                }

                                original_size = vertexMap.size();
                                int local_v2 = addOrGetVertexIndex(vertexMap, global_v2);
                                if (original_size != vertexMap.size()) {
                                    mesh->vertex.push_back(lspts[global_v2]);
                                }
                                // std::cout<<"local_v0: "<<local_v0<<" local_v1: "<<local_v1<<" local_v2: "<<local_v2<<std::endl;

                                mesh->index.push_back(vec3i(local_v0, local_v1, local_v2));
                                vec3f vx = lspts[global_v0];
                                vec3f vy = lspts[global_v1];
                                vec3f vz = lspts[global_v2];
                                vec3f normal = normalize(cross(vy - vx, vz - vx));
                                mesh->normal.push_back(normal);
                                mesh->globalID.push_back(gmlid_int);
                                std::string type = g["semantics"]["surfaces"][i]["type"];
                                int surface_type_id = surface_type_map[type];
                                float surface_albedo = surface_albedo_map[type];
                                mesh->albedo.push_back(surface_albedo);
                                mesh->surfaceType.push_back(surface_type_id);
                                
                                total_triangles++;
                        }
  
                    }
                }
                
                
                mesh->diffuse = gdt::randomColor(building_index);
                building_index++;
                
                assert(mesh->vertex.size() == vertexMap.size());
                assert(mesh->index.size() == mesh->normal.size());
                assert(mesh->index.size() == mesh->globalID.size());

                
                if (mesh->vertex.size() > 0 && mesh->index.size() > 0){
                    model->meshes.push_back(mesh);
                }   else {
                    std::cout<<"Empty mesh, deleting"<< building_index <<std::endl;
                    delete mesh;
                }
            }

        }



        std::cout<<"bouding box low "<<model->bounds.lower.x<<" "<<model->bounds.lower.y<<" "<<model->bounds.lower.z<<std::endl;
        std::cout<<"bouding box high "<<model->bounds.upper.x<<" "<<model->bounds.upper.y<<" "<<model->bounds.upper.z<<std::endl;
        std::cout<<"Total building num " << building_index<<std::endl;
        std::cout<<"Total triangle num " << total_triangles<<std::endl;
        std::cout<<"Total vertex num " << lspts.size()<<std::endl;
        model->original_center = model->bounds.center();
        return model;


    }
    
    Model *loadWeatherStation(const std::string &jsonFile)
    {

        Model *model = new Model;
        std::unordered_set<std::string> targetTypes = {"WallSurface", "RoofSurface"};

        // std::map<int, int> vertexMap;

        std::fstream input(jsonFile);
        json j;
        input >> j;
        input.close();
        std::vector<vec3f> lspts = get_coordinates(j, true);
        for (auto const &lspt:lspts) {
            model->bounds.extend(lspt);
        }

        int building_index = 0;
        int total_triangles = 0;
        int TIN_index = 0;
        int Tree_index = 0;

        TriangleMesh *mesh = nullptr;
        std::map<int, int> vertexMap;

        for (auto &co: j["CityObjects"].items()) {
            if (co.value()["type"] == "BuildingPart" || co.value()["type"] == "Building") {
                vertexMap.clear();
                mesh = new TriangleMesh;

                for (auto &g: co.value()["geometry"]) {
                    if (g["lod"] != "2") {
                            continue;
                    }
                    else {
                    for (int i = 0; i< g["boundaries"].size(); i++) {
                        std::vector<std::vector<int>> triangle = g["boundaries"][i];
                        if (triangle[0].size() != 3 || 
                            triangle[0][0]==triangle[0][1] || 
                            triangle[0][0]==triangle[0][2] || 
                            triangle[0][1]==triangle[0][2])
                        {
                            int gmlid_int = g["semantics"]["surfaces"][i]["global_idx"];

                            std::cout<<"a triangle is not valid, identifier: "<<gmlid_int<<std::endl;
                            continue;
                        }  
                        else {
                            // std::string surf_type = g["semantics"]["surfaces"][i]["type"];
                            // if (targetTypes.find(surf_type) != targetTypes.end()) {
                                int gmlid_int = g["semantics"]["surfaces"][i]["global_idx"];
                                int global_v0 = triangle[0][0];
                                int global_v1 = triangle[0][1];
                                int global_v2 = triangle[0][2];

                                int original_size = vertexMap.size();

                                int local_v0 = addOrGetVertexIndex(vertexMap, global_v0);

                                if (original_size != vertexMap.size()) {
                                    mesh->vertex.push_back(lspts[global_v0]);
                                }

                                original_size = vertexMap.size();
                                int local_v1 = addOrGetVertexIndex(vertexMap, global_v1);
                                if (original_size != vertexMap.size()) {
                                    mesh->vertex.push_back(lspts[global_v1]);
                                }

                                original_size = vertexMap.size();
                                int local_v2 = addOrGetVertexIndex(vertexMap, global_v2);
                                if (original_size != vertexMap.size()) {
                                    mesh->vertex.push_back(lspts[global_v2]);
                                }
                                // std::cout<<"local_v0: "<<local_v0<<" local_v1: "<<local_v1<<" local_v2: "<<local_v2<<std::endl;

                                mesh->index.push_back(vec3i(local_v0, local_v1, local_v2));
                                vec3f vx = lspts[global_v0];
                                vec3f vy = lspts[global_v1];
                                vec3f vz = lspts[global_v2];
                                vec3f normal = normalize(cross(vy - vx, vz - vx));
                                mesh->normal.push_back(normal);
                                mesh->globalID.push_back(gmlid_int);
                                std::string type = g["semantics"]["surfaces"][i]["type"];
                                int surface_type_id = surface_type_map[type];
                                float surface_albedo = surface_albedo_map[type];
                                mesh->surfaceType.push_back(surface_type_id);
                                mesh->albedo.push_back(surface_albedo);
                                total_triangles++;
                        }
  
                        }
                    }
                }
                
                
                
                mesh->diffuse = gdt::randomColor(building_index);

                
                assert(mesh->vertex.size() == vertexMap.size());
                assert(mesh->index.size() == mesh->normal.size());
                assert(mesh->index.size() == mesh->globalID.size());

                
                if (mesh->vertex.size() > 0 && mesh->index.size() > 0){
                    model->meshes.push_back(mesh);
                    building_index++;
                }   else {
                    std::cout<<"Empty mesh, deleting building"<< building_index <<std::endl;
                    delete mesh;
                }
            }

            else if (co.value()["type"] == "TINRelief") {
                vertexMap.clear();
                mesh = new TriangleMesh;
                for (auto &g: co.value()["geometry"]) {
                    if (g["lod"] != "1") {
                            continue;
                    }
                    else {
                    for (int i = 0; i< g["boundaries"].size(); i++) {
                        std::vector<std::vector<int>> triangle = g["boundaries"][i];
                        if (triangle[0].size() != 3 || 
                            triangle[0][0]==triangle[0][1] || 
                            triangle[0][0]==triangle[0][2] || 
                            triangle[0][1]==triangle[0][2])
                        {
                            int gmlid_int = g["semantics"]["surfaces"][i]["global_idx"];

                            std::cout<<"a triangle is not valid, identifier: "<<gmlid_int<<std::endl;
                            continue;
                        }  
                        else {
                                int gmlid_int = g["semantics"]["surfaces"][i]["global_idx"];
                                int global_v0 = triangle[0][0];
                                int global_v1 = triangle[0][1];
                                int global_v2 = triangle[0][2];

                                int original_size = vertexMap.size();

                                int local_v0 = addOrGetVertexIndex(vertexMap, global_v0);

                                if (original_size != vertexMap.size()) {
                                    mesh->vertex.push_back(lspts[global_v0]);
                                }

                                original_size = vertexMap.size();
                                int local_v1 = addOrGetVertexIndex(vertexMap, global_v1);
                                if (original_size != vertexMap.size()) {
                                    mesh->vertex.push_back(lspts[global_v1]);
                                }

                                original_size = vertexMap.size();
                                int local_v2 = addOrGetVertexIndex(vertexMap, global_v2);
                                if (original_size != vertexMap.size()) {
                                    mesh->vertex.push_back(lspts[global_v2]);
                                }
                                // std::cout<<"local_v0: "<<local_v0<<" local_v1: "<<local_v1<<" local_v2: "<<local_v2<<std::endl;

                                mesh->index.push_back(vec3i(local_v0, local_v1, local_v2));
                                vec3f vx = lspts[global_v0];
                                vec3f vy = lspts[global_v1];
                                vec3f vz = lspts[global_v2];
                                vec3f normal = normalize(cross(vy - vx, vz - vx));
                                mesh->normal.push_back(normal);
                                mesh->globalID.push_back(gmlid_int);
                                std::string type = "TIN";
                                int surface_type_id = surface_type_map[type];
                                float surface_albedo = surface_albedo_map[type];
                                mesh->surfaceType.push_back(surface_type_id);
                                mesh->albedo.push_back(surface_albedo);
                                total_triangles++;
                        }                
                    }
                    }
                    
                }
                
                mesh->diffuse = gdt::randomColor(TIN_index);
 
                
                assert(mesh->vertex.size() == vertexMap.size());
                assert(mesh->index.size() == mesh->normal.size());
                assert(mesh->index.size() == mesh->globalID.size());

                
                if (mesh->vertex.size() > 0 && mesh->index.size() > 0){
                    model->meshes.push_back(mesh);
                    TIN_index++;
                }   else {
                    std::cout<<"Empty mesh, deleting TIN terrain"<< TIN_index <<std::endl;
                    delete mesh;
                }
                
            }

            else if (co.value()["type"] == "SolitaryVegetationObject") {
                vertexMap.clear();
                mesh = new TriangleMesh;


                std::array<float, 16> transformation_matrix;
                for (int i = 0; i < 16; ++i) {
                    transformation_matrix[i] = co.value()["geometry"][0]["transformationMatrix"][i].get<float>();
                }

                std::array<float, 3> translate_arr;
                for (int i = 0; i < 3; ++i) {
                    translate_arr[i] = co.value()["geographicalExtent"][i].get<float>();
                }

                std::vector<vec3f> transformedcoords = transformMesh(transformation_matrix, translate_arr);



                for (int i = 0; i < transformedcoords.size(); i++) {
                    mesh->vertex.push_back(transformedcoords[i]);
                }

                const std::vector<vec3i> faces = {
                    vec3i(4, 1, 2),
                    vec3i(1, 4, 0),
                    vec3i(3, 4, 2),
                    vec3i(4, 3, 0),
                    vec3i(4, 6, 7),
                    vec3i(6, 4, 5),
                    vec3i(8, 4, 7),
                    vec3i(4, 8, 5),
                    vec3i(5, 13, 14),
                    vec3i(13, 5, 12),
                    vec3i(15, 5, 14),
                    vec3i(5, 8, 12),
                    vec3i(8, 11, 12),
                    vec3i(11, 8, 10),
                    vec3i(8, 9, 10),
                    vec3i(8, 5, 4),
                    vec3i(0, 19, 20),
                    vec3i(19, 0, 12),
                    vec3i(21, 0, 20),
                    vec3i(0, 3, 12),
                    vec3i(3, 18, 12),
                    vec3i(18, 3, 17),
                    vec3i(3, 16, 17),
                    vec3i(3, 0, 4)
                };

                for (int i = 0; i < faces.size(); i++) {
                    mesh->index.push_back(faces[i]);
                    vec3f normal = normalize(cross(transformedcoords[faces[i].y] - transformedcoords[faces[i].x], transformedcoords[faces[i].z] - transformedcoords[faces[i].x]));
                    mesh->normal.push_back(normal);
                    int globalID = co.value()["attributes"]["global_idx"];
                    mesh->globalID.push_back(globalID);
                    std::string type = "Tree";
                    int surface_type_id = surface_type_map[type];
                    float surface_albedo = surface_albedo_map[type];
                    mesh->surfaceType.push_back(surface_type_id);
                    mesh->albedo.push_back(surface_albedo);
                    total_triangles++;
                }




                
                mesh->diffuse = gdt::randomColor(Tree_index);
 
                
                // assert(mesh->vertex.size() == vertexMap.size());
                // assert(mesh->index.size() == mesh->normal.size());
                // assert(mesh->index.size() == mesh->globalID.size());

                
                if (mesh->vertex.size() > 0 && mesh->index.size() > 0){
                    model->meshes.push_back(mesh);
                    Tree_index++;
                }   else {
                    std::cout<<"Empty mesh, deleting Tree"<< Tree_index <<std::endl;
                    delete mesh;
                }
                
            }
        }

        



        std::cout<<"bouding box low "<<model->bounds.lower.x<<" "<<model->bounds.lower.y<<" "<<model->bounds.lower.z<<std::endl;
        std::cout<<"bouding box high "<<model->bounds.upper.x<<" "<<model->bounds.upper.y<<" "<<model->bounds.upper.z<<std::endl;
        std::cout<<"Total building num " << building_index<<std::endl;
        std::cout<<"Total TIN num " << TIN_index<<std::endl;
        std::cout<<"Total Tree num " << Tree_index<<std::endl;
        std::cout<<"Total triangle num " << total_triangles<<std::endl;
        std::cout<<"Total vertex num " << lspts.size()<<std::endl;
        model->original_center = model->bounds.center();
        return model;


    }

    }

