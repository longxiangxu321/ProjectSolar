#include "Model.h"
#include <map>

namespace osc {

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
                mesh->vertex[i] = vec3f(mesh->vertex[i].x - bounds.center().x, 
                                        mesh->vertex[i].y - bounds.center().y, mesh->vertex[i].z - bounds.center().z);
            }
        }
        bounds = box3f(bounds.lower - bounds.center(), bounds.upper - bounds.center());
        
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
                mesh->materialID.push_back(matID);
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
                                mesh->materialID.push_back(gmlid_int);
                                total_triangles++;
                        }
  
                    }
                }
                
                
                mesh->diffuse = gdt::randomColor(building_index);
                building_index++;
                
                assert(mesh->vertex.size() == vertexMap.size());
                assert(mesh->index.size() == mesh->normal.size());
                assert(mesh->index.size() == mesh->materialID.size());

                
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



}
