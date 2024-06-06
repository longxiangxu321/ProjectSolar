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
        bounds = box3f(bounds.upper - bounds.center(), bounds.lower - bounds.center());
        
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
                mesh->buildingID = building_index;
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

}
