#include "Model.h"
#include <map>
#include <array>

namespace osc {

    inline vec3f compute_normal(vec3f vx, vec3f vy, vec3f vz) {
        vec3d dvx = vec3d(vx.x, vx.y, vx.z);
        vec3d dvy = vec3d(vy.x, vy.y, vy.z);
        vec3d dvz = vec3d(vz.x, vz.y, vz.z);

        vec3d edge1 = dvy - dvx;
        vec3d edge2 = dvz - dvx;

        double length1 = length(edge1);
        double length2 = length(edge2);

        if (length1 != 0) {
            edge1 = edge1 / length1;  // 归一化 edge1
        }

        if (length2 != 0) {
            edge2 = edge2 / length2;  // 归一化 edge2
        }

        vec3f scaled_edge_1 = vec3f(edge1.x, edge1.y, edge1.z);
        vec3f scaled_edge_2 = vec3f(edge2.x, edge2.y, edge2.z);

        vec3f normal = normalize(cross(scaled_edge_1, scaled_edge_2));

        return normal;
    }

    inline double compute_area(vec3f vx, vec3f vy, vec3f vz) {
        vec3f cross_product = cross(vz-vx, vy-vx);
        double area = length(cross_product);
        return area;
    }

    struct vec4_type {
        int x;
        int y;
        int z;
        std::string type;
        vec4_type(int x, int y, int z, std::string type) : x(x), y(y), z(z), type(type) {}
    };

    std::map<std::string, int> createSurfaceTypeMap() {
    std::map<std::string, int> surface_type_map;
    surface_type_map["WallSurface"] = 0;
    surface_type_map["RoofSurface"] = 1;
    surface_type_map["GroundSurface"] = 2;
    surface_type_map["TIN"] = 3;
    surface_type_map["Tree"] = 4;
    surface_type_map["Window"] = 5;
    return surface_type_map;
    }

    std::map<std::string, int> surface_type_map = createSurfaceTypeMap();
    std::map<std::string, float> surface_albedo_map = {{"WallSurface", 0.4}, 
                                    {"RoofSurface", 0.1}, {"GroundSurface", 0.2}, 
                                    {"TIN", 0.05}, {"Tree", 0.3}, {"Window", 0.3}};

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

    void obtain_model_statistics_and_save_area_info(const Model* model, const std::string filename) {
        int building_face_num = 0;
        int tin_face_num = 0;
        int tree_face_num = 0;
        int total_face = 0;

        std::filesystem::path filepath = filename;


        double total_area = 0;

        std::vector<double> all_areas;

        for (auto const all_mesh:model->meshes) {
            for (int i = 0; i <all_mesh->surfaceType.size();i++){
                int temp_type = all_mesh->surfaceType[i];
                if (temp_type==3) {tin_face_num++;}
                else if (temp_type==4) {tree_face_num++;}
                else {building_face_num++;}

                vec3f v0 = all_mesh->vertex[all_mesh->index[i][0]];
                vec3f v1 = all_mesh->vertex[all_mesh->index[i][1]];
                vec3f v2 = all_mesh->vertex[all_mesh->index[i][2]];

                double area = compute_area(v0,v1,v2);
                total_area+=area;
                total_face++;
                all_areas.push_back(area);
            }
        }

        double avg_area = total_area/total_face;
        std::cout<<"average triangle area "<<  avg_area<<std::endl;

        std::ofstream areaFile(filepath, std::ios::binary | std::ios::out);

        double *areas = all_areas.data();

        if (!areaFile) {
            std::cerr << "cannot create shadow_map" << std::endl;
        }

        areaFile.write(reinterpret_cast<const char*>(areas), total_face* sizeof(double));



        std::cout<<"Building face number "<<building_face_num<<std::endl;
        std::cout<<"TIN face number "<<tin_face_num<<std::endl;
        std::cout<<"Tree face number "<<tree_face_num<<std::endl;
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


    // void loadTUDOBJ(const std::string &objFile)
    // {
    //         Model *model = new Model;
    //         std::ifstream input_stream;
    //         input_stream.open(objFile);
    //         std::vector<vec3f> points;
    //         std::vector<vec3i> wall_surfaces;
    //         std::vector<vec3i> roof_surfaces;
    //         std::vector<vec3i> window_surfaces;
    //         std::vector<vec3i> vegetation_surfaces;
    //         std::vector<vec3i>* current_surfaces = nullptr;

    //         if (input_stream.is_open()) {
    //             std::cout << "Reading " << objFile << std::endl;
    //             std::string line;
    //             while (std::getline(input_stream, line)) {
    //                 if (line[0] == 'v' && line[1] == ' ') {
    //                     double x,y,z;
    //                     std::stringstream ss(line);
    //                     std::string temp;
    //                     ss >> temp >> x >> y >> z;
    //                     double sx = x;
    //                     double sy = y;
    //                     points.push_back(vec3f(sx, sy, z));
    //                     model->bounds.extend(vec3f(sx, sy, z));
    //                 }
    //                 if (line[0] == 'u') {
    //                     std::stringstream ss(line);
    //                     std::string temp, material;
    //                     ss >> temp >> material;
    //                     std::cout << "material: " << material << std::endl;

    //                     if (material == "material_0") {
    //                         current_surfaces = &wall_surfaces;
    //                     } else if (material == "material_1") {
    //                         current_surfaces = &roof_surfaces;
    //                     } else if (material == "material_2") {
    //                         current_surfaces = &window_surfaces;
    //                     } else if (material == "material_3") {
    //                         current_surfaces = &vegetation_surfaces;
    //                     }
    //                 }
    //                 if (line[0] == 'f') {
    //                     unsigned long v0, v1, v2;
    //                     std::stringstream ss(line);
    //                     std::string temp;
    //                     ss >> temp >> v0 >> v1 >> v2;

    //                     int global_v0 = v0 -1;
    //                     int global_v1 = v1 -1;
    //                     int global_v2 = v2 -1;
    //                     if (current_surfaces) {
    //                         current_surfaces->push_back(vec3i(global_v0, global_v1, global_v2));
    //                     }
    //                 }
    //                 else {
    //                     continue;
    //                 }
    //             }
    //         }


    // }

    
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
                        std::string surf_type = g["semantics"]["surfaces"][i]["type"];
                        if (triangle[0].size() != 3 || 
                            triangle[0][0]==triangle[0][1] || 
                            triangle[0][0]==triangle[0][2] || 
                            triangle[0][1]==triangle[0][2]) {
                            int gmlid_int = g["semantics"]["surfaces"][i]["global_idx"];

                            std::cout<<"a triangle is not valid, identifier: "<<gmlid_int<<std::endl;
                            continue;
                        }
                        else if (targetTypes.find(surf_type) == targetTypes.end()) {
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
        int Tree_index = 0;
        int TIN_index = 0;

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
                        std::string surf_type = g["semantics"]["surfaces"][i]["type"];
                        if (triangle[0].size() != 3 || 
                            triangle[0][0]==triangle[0][1] || 
                            triangle[0][0]==triangle[0][2] || 
                            triangle[0][1]==triangle[0][2])
                        {
                            int gmlid_int = g["semantics"]["surfaces"][i]["global_idx"];

                            std::cout<<"a triangle is not valid, identifier: "<<gmlid_int<<std::endl;
                            continue;
                        }
                        else if (targetTypes.find(surf_type) == targetTypes.end()) {
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
                
                
                
                // mesh->diffuse = gdt::randomColor(building_index);
                mesh->diffuse = vec3f(0.5f, 0.5f, 0.5f);

                
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
                
                // mesh->diffuse = gdt::randomColor(TIN_index);
                mesh->diffuse = vec3f(0.6f, 0.8f, 0.2f);
 
                
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




                
                // mesh->diffuse = gdt::randomColor(Tree_index);
                mesh->diffuse = vec3f(0.0f, 0.5f, 0.0f);
 
                
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

    Model *loadTUDelft(const std::string &jsonFile, const std::string &objFile)
    {

        Model *model = new Model;
        std::unordered_set<std::string> targetTypes = {"WallSurface", "RoofSurface"};

        // std::map<int, int> vertexMap;
        for (auto const &material:surface_albedo_map) {
            std::cout<<"material: "<<material.first<<" albedo: "<<material.second<<std::endl;
        }

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

        int TIN_num_faces = 0;

        TriangleMesh *mesh = nullptr;
        std::map<int, int> vertexMap;

        int max_global_idx = 0;

        for (auto &co: j["CityObjects"].items()) {
            if (co.value()["type"] == "BuildingPart" || co.value()["type"] == "Building") {
                vertexMap.clear();
                if (co.key() == "NL.IMBAG.Pand.0503100000025162" || co.key() == "NL.IMBAG.Pand.0503100000031391") {
                    std::cout<<"Skipping building: "<<co.key()<<std::endl;
                   continue;
                }
                mesh = new TriangleMesh;
                for (auto &g: co.value()["geometry"]) {
                    for (int i = 0; i< g["boundaries"].size(); i++) {
                        std::vector<std::vector<int>> triangle = g["boundaries"][i];
                        std::string surf_type = g["semantics"]["surfaces"][i]["type"];
                        if (triangle[0].size() != 3 || 
                            triangle[0][0]==triangle[0][1] || 
                            triangle[0][0]==triangle[0][2] || 
                            triangle[0][1]==triangle[0][2])
                        {
                            int gmlid_int = g["semantics"]["surfaces"][i]["global_idx"];

                            std::cout<<"a triangle is not valid, identifier: "<<gmlid_int<<std::endl;
                            continue;
                        }
                        else if (targetTypes.find(surf_type) == targetTypes.end()) {
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
                                max_global_idx = std::max(max_global_idx, gmlid_int);
                                std::string type = g["semantics"]["surfaces"][i]["type"];
                                int surface_type_id = surface_type_map[type];
                                float surface_albedo = surface_albedo_map[type];
                                mesh->surfaceType.push_back(surface_type_id);
                                mesh->albedo.push_back(surface_albedo);
                                total_triangles++;
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
                    for (int i = 0; i< g["boundaries"].size(); i++) {
                        std::vector<std::vector<int>> triangle = g["boundaries"][i];
                        if (triangle[0].size() != 3 || 
                            triangle[0][0]==triangle[0][1] || 
                            triangle[0][0]==triangle[0][2] || 
                            triangle[0][1]==triangle[0][2])
                            {
                                int gmlid_int = g["semantics"]["surfaces"][i]["global_idx"];

                                // std::cout<<"a TIN is not valid, identifier: "<<gmlid_int<<std::endl;
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
                                max_global_idx = std::max(max_global_idx, gmlid_int);
                                std::string type = "TIN";
                                int surface_type_id = surface_type_map[type];
                                float surface_albedo = surface_albedo_map[type];
                                mesh->surfaceType.push_back(surface_type_id);
                                mesh->albedo.push_back(surface_albedo);
                                total_triangles++;
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
                    max_global_idx = std::max(max_global_idx, globalID);
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

        
        std::ifstream input_stream;
        input_stream.open(objFile);
        std::vector<vec3f> points;

        std::vector<vec4_type> surfaces;
        std::string currentSurfaceType;

        if (input_stream.is_open()) {
            std::cout << "Reading " << objFile << std::endl;
            std::string line;
            while (std::getline(input_stream, line)) {
                if (line[0] == 'v' && line[1] == ' ') {
                    double x,y,z;
                    std::stringstream ss(line);
                    std::string temp;
                    ss >> temp >> x >> y >> z;
                    double sx = x;
                    double sy = y;
                    points.push_back(vec3f(sx, sy, z));
                    // model->bounds.extend(vec3f(sx, sy, z));
                }
                if (line[0] == 'u') {
                    std::stringstream ss(line);
                    std::string temp, material;
                    ss >> temp >> material;

                    if (material == "material_0") {
                        currentSurfaceType = "WallSurface";
                    } else if (material == "material_1") {
                        currentSurfaceType = "RoofSurface";
                    } else if (material == "material_2") {
                        currentSurfaceType = "Window";
                    } else if (material == "material_3") {
                        currentSurfaceType = "Tree";
                        Tree_index++;
                    } else {
                        currentSurfaceType = "Unknown";  // 未知的 material
                    }
                }
                if (line[0] == 'f') {
                    unsigned long v0, v1, v2;
                    char slash;

                    std::stringstream ss(line);
                    std::string temp;
                    ss >> temp >> v0 >> v1 >> v2;

                    int global_v0 = v0 - 1;
                    int global_v1 = v1 - 1;
                    int global_v2 = v2 - 1;
                    if (currentSurfaceType != "Unknown") {
                        surfaces.emplace_back(global_v0, global_v1, global_v2, currentSurfaceType);
                    }
                }
                else {
                    continue;
                }
            }
        }

        mesh = new TriangleMesh;
        std::cout<<"obj surface number: "<<surfaces.size()<<std::endl;
        std::cout <<"obj vertex number: "<<points.size()<<std::endl;
        for (auto const &point:points) {
            mesh->vertex.push_back(vec3f(point.x, point.y, point.z));
        }

        for (auto const &surface:surfaces) {
            int global_v0 = surface.x;
            int global_v1 = surface.y;
            int global_v2 = surface.z;
            std::string surface_type = surface.type;

            if (global_v0 == global_v1 || global_v0 == global_v2 || global_v1 == global_v2) {
                // std::cout<<"a triangle is not valid, identifier: "<<global_v0<<std::endl;
                continue;
            }

            if (global_v0 == 37 && global_v1 == 35 && global_v2 == 34)
            {
                mesh->globalID.push_back(55241912);
                std::cout<<"found s1"<<std::endl;
            }
            else if (global_v0 == 2489 && global_v1 == 2494 && global_v2 == 2493)
            {
                std::cout<<"found s3"<<std::endl;
                mesh->globalID.push_back(19125524);
            }
            else {
                mesh->globalID.push_back(++max_global_idx);
            }

            mesh->index.push_back(vec3i(global_v0, global_v1, global_v2));
            vec3f normal = compute_normal(points[global_v0], points[global_v1], points[global_v2]);
            mesh->normal.push_back(normal);
            

            int surface_type_id = surface_type_map[surface_type];
            float surface_albedo = surface_albedo_map[surface_type];
            mesh->surfaceType.push_back(surface_type_id);
            mesh->albedo.push_back(surface_albedo);
            total_triangles++;
        }
        mesh->diffuse = gdt::randomColor(total_triangles);

        
        assert(mesh->vertex.size() == points.size());
        assert(mesh->index.size() == mesh->normal.size());
        assert(mesh->index.size() == mesh->globalID.size());
        if (mesh->vertex.size() > 0 && mesh->index.size() > 0){
            model->meshes.push_back(mesh);
            std::cout<<"obj mesh added"<<std::endl;
                    // Tree_index++;
            }   else {
                // std::cout<<"Empty mesh, deleting Tree"<< Tree_index <<std::endl;
                delete mesh;
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

