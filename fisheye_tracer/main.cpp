// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "SampleRenderer.h"

// our helper library for window handling
#include "glfWindow/GLFWindow.h"
#include <GL/gl.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "3rdParty/stb_image_write.h"

#include <chrono>
#include <cstdlib>
#include <ctime>
#include "sample_pointGrid.h"
#include "read_solar.h"
/*! \namespace osc - Optix Siggraph Course */
namespace osc {

  
  /*! main entry point to this example - initially optix, print hello
    world, then exit */
  extern "C" int main(int ac, char **av)
  {
    try {

    try {
        std::filesystem::current_path(PROJECT_ROOT); // Set the current working directory
        std::cout << "Current working directory changed to: " << std::filesystem::current_path() << std::endl;
      } catch(const std::filesystem::filesystem_error& e) {
            std::cerr << "Error when changing current directory: " << e.what() << std::endl;}

    json CFG;
    if (std::filesystem::exists("./config.json")) {
        std::cout << "Config file exists." << std::endl;
        CFG = readConfig("config.json");
    }
    else {
          throw std::runtime_error("Config file does not exist.");}

    std::filesystem::path root_folder = CFG["study_area"]["data_root"];
    std::filesystem::path target_tiles = root_folder / "citymodel";
    std::filesystem::path solar_position_path = root_folder / CFG["output_folder_name"] 
                                        / "intermediate" /"sun_pos.csv";
    
    std::filesystem::path point_grid_path = root_folder / CFG["output_folder_name"] 
                                        / "intermediate" /"point_grid.dat";

    std::filesystem::path indexFile_path = root_folder / CFG["output_folder_name"] 
                                         /"index_map.dat";
    std::filesystem::path cosFile_path = root_folder / CFG["output_folder_name"]
                                          /"cosine_map.dat";
    std::filesystem::path horizonfactorFile_path = root_folder / CFG["output_folder_name"]
                                          /"horizon_factor_map.dat";
    std::filesystem::path skyviewfactorFile_path = root_folder / CFG["output_folder_name"]
                                          /"sky_view_factor_map.dat";
    std::filesystem::path config_backup_path = root_folder / CFG["output_folder_name"]
                                            /"config.json";
    
    const float point_grid_sampling_density = CFG["area_per_point"];
    const int batch_size = 2500;
    const int azimuth_resolution = CFG["azimuth_resolution"];
    const int elevation_resolution = CFG["elevation_resolution"];
    const vec2i hemisphere_resolution = vec2i(azimuth_resolution,elevation_resolution);
    const int voxel_resolution = CFG["voxel_resolution"];

    std::cout<<"Reading city model"<<std::endl;

    Model* model = nullptr;
    int entry_count = 0;

    // std::filesystem::directory_iterator dir_iter(target_tiles);

    // if (dir_iter == std::filesystem::directory_iterator()) {
    //     std::cerr <<GDT_TERMINAL_RED<< "The directory: "<< target_tiles
    //     <<" is empty or does not exist." <<GDT_TERMINAL_DEFAULT <<std::endl;
    //     return 1;
    // } else {
    //     // 读取第一个 entry
    //     const auto& first_entry = *dir_iter;
    //     if (first_entry.is_regular_file()) {
    //         model = loadTUDelft(first_entry.path().string());
    //         std::cout<<GDT_TERMINAL_GREEN<<"Model loaded"<<GDT_TERMINAL_DEFAULT<<std::endl;

            
    //         if (model == nullptr) {
    //             std::cerr<<GDT_TERMINAL_RED << "Failed to load: " <<
    //             first_entry.path().string() << GDT_TERMINAL_DEFAULT<<std::endl;
    //         }
    //     } else {
    //         std::cerr << first_entry.path().string() << " is not a regular file." << std::endl;
    //         return 1;
    //     }
    //     entry_count++;

    //     ++dir_iter;
    //     if (dir_iter != std::filesystem::directory_iterator()) {
    //         std::cerr<< GDT_TERMINAL_RED<< "There is more than one file in this folder, keep only one." 
    //         <<GDT_TERMINAL_DEFAULT<< std::endl;
    //         return 1;
    //     }
    // }
      std::vector<std::filesystem::path> obj_files;
      std::filesystem::path json_file;
      int json_count = 0;

      for (const auto& entry : std::filesystem::directory_iterator(target_tiles)) {
          if (entry.is_regular_file()) {
              auto path = entry.path();
              if (path.extension() == ".json") {
                  json_file = path;
                  json_count++;
              } else if (path.extension() == ".obj") {
                  obj_files.push_back(path);
              }
          }
      }

      if (json_count == 0) {
          std::cerr << GDT_TERMINAL_RED << "No cityjson file found in the directory." << GDT_TERMINAL_DEFAULT << std::endl;
          return 1;
      } else if (json_count > 1) {
          std::cerr << GDT_TERMINAL_RED << "There is more than cityjson file in this folder, keep only one." << GDT_TERMINAL_DEFAULT << std::endl;
          return 1;
      }

      model = loadTUDelft(json_file.string(), obj_files[0].string());
      // model = loadWeatherStation(json_file.string());
      std::cout << GDT_TERMINAL_GREEN << "Model loaded from " << json_file.string() << " and "<< obj_files[0].string()<< GDT_TERMINAL_DEFAULT << std::endl;

      if (model == nullptr) {
          std::cerr << GDT_TERMINAL_RED << "Failed to load: " << json_file.string() << GDT_TERMINAL_DEFAULT << std::endl;
          return 1;
      }


      model->transformModel();
      SampleRenderer *renderer = new SampleRenderer(model);
      // std::cout<<renderer->bbox.lower<<std::endl;
      // std::cout<<renderer->bbox.upper<<std::endl;
      // std::cout<<renderer->launchParams.bbox_min<<std::endl;
      // std::cout<<renderer->launchParams.bbox_max<<std::endl;

      std::vector<GridPoint> raw_gridpoints= create_point_grid(*model, point_grid_sampling_density);
      std::vector<GridPoint> gridpoints = clean_point_grid(raw_gridpoints);
      std::cout<<"Grid points created: "<< gridpoints.size()<<std::endl;

      vec3f translation = model->bounds.center() - model->original_center;
      save_point_grid(gridpoints, translation, point_grid_path.string());
      std::cout<<"Grid points saved"<<std::endl;

      // const vec2i fbSize(vec2i(360/azimuth_resolution,90/elevation_resolution));
      vec2i fbSize = vec2i(360/azimuth_resolution, 90/elevation_resolution);
      std::cout<<GDT_TERMINAL_YELLOW<<"Frame buffer size: "<<fbSize.x<<" "<<fbSize.y<<GDT_TERMINAL_DEFAULT<<std::endl;

      auto start = std::chrono::high_resolution_clock::now();


      // srand(static_cast<unsigned>(time(0)));

      // std::ofstream indexFile("../index_map.dat", std::ios::binary | std::ios::out);
      // std::ofstream azimuthFile("../azimuth_map.dat", std::ios::binary | std::ios::out);
      // std::ofstream elevationFile("../elevation_map.dat", std::ios::binary | std::ios::out);
      // std::ofstream horizonfactorFile("../horizon_factor_map.dat", std::ios::binary | std::ios::out);
      // std::ofstream skyviewfactorFile("../sky_view_factor_map.dat", std::ios::binary | std::ios::out);
      std::ofstream indexFile(indexFile_path, std::ios::binary | std::ios::out);
      std::ofstream cosFile(cosFile_path, std::ios::binary | std::ios::out);
      std::ofstream horizonfactorFile(horizonfactorFile_path, std::ios::binary | std::ios::out);
      std::ofstream skyviewfactorFile(skyviewfactorFile_path, std::ios::binary | std::ios::out);


      if (!indexFile || !cosFile) {
        std::cerr << "cannot create semantic_map files" << std::endl;
        return 1;
      }


      uint32_t num_samplepoints = gridpoints.size();
      
      const uint32_t last_batch_size = num_samplepoints % batch_size;
      uint32_t num_batches = num_samplepoints / batch_size;
      
      
      // vec2i spliting = vec2i(360, 90);

      int point_id = 0;
      int choosen_id = 19402;

      std::cout<<"Rendering with batch size "<<batch_size<<std::endl;
      std::cout <<"Total number of batches: "<<num_batches<<std::endl;
      int batch_offset = 0;
      for (uint32_t i = 0; i < num_batches; i++) {
        if (i % 10 == 0) {
          std::cout<<GDT_TERMINAL_GREEN<<"Rendering batch "<<i<<" to "<< i+10 <<std::endl;
        }
        
        Camera cameras[batch_size];

        std::vector<GridPoint> batch_points(gridpoints.begin() + i * batch_size, gridpoints.begin() + (i + 1) * batch_size);
        for (uint32_t j = 0; j < batch_points.size(); j++) {
          GridPoint point = batch_points[j];
          vec3f position = point.position;
          vec3f direction = point.triangle_info.direction;
          vec3f up = vec3f(0.f, 1.f, 0.f);
          Camera cam = {position, direction, up};
          cameras[j] = cam;
          if (point_id == choosen_id) {
            std::cout<<position.x -translation.x <<" "<<position.y -translation.y <<" "<<position.z - translation.z<<std::endl;
            std::cout<<direction.x<<" "<<direction.y<<" "<<direction.z<<std::endl;
          }
          point_id++;
        }
        renderer->setCameraGroup(cameras, batch_size, hemisphere_resolution, voxel_resolution);
        renderer->render();
        std::vector<uint32_t> pixels(batch_size*fbSize.x*fbSize.y);
        std::vector<half> cosine_factors(batch_size*fbSize.x*fbSize.y*6);
        std::vector<float> horizon_factor(batch_size);
        std::vector<int> sky_view_factor(batch_size);


        renderer->downloadPixels(pixels.data());
        renderer->downloadIncidentFactors(cosine_factors.data());
        renderer->downloadHorizonFactors(horizon_factor.data());
        renderer->downloadSVF(sky_view_factor.data());

        indexFile.write(reinterpret_cast<const char*>(pixels.data()), pixels.size() * sizeof(uint32_t));
        cosFile.write(reinterpret_cast<const char*>(cosine_factors.data()), cosine_factors.size() * sizeof(half));
        horizonfactorFile.write(reinterpret_cast<const char*>(horizon_factor.data()), horizon_factor.size() * sizeof(float));
        skyviewfactorFile.write(reinterpret_cast<const char*>(sky_view_factor.data()), sky_view_factor.size() * sizeof(int));

        batch_offset += batch_size;
      }
      
      if (last_batch_size > 0) {
      std::cout<<GDT_TERMINAL_GREEN<<"Rendering last batch with size: "<<last_batch_size<<GDT_TERMINAL_DEFAULT<<std::endl;
      
      Camera* last_cameras = new Camera[last_batch_size];
      std::vector<GridPoint> last_batch_points(gridpoints.begin() + num_batches * batch_size, gridpoints.end());
      for (uint32_t j = 0; j < last_batch_points.size(); j++) {
        GridPoint point = last_batch_points[j];
        vec3f position = point.position;
        vec3f direction = point.triangle_info.direction;
        vec3f up = vec3f(0.f, 1.f, 0.f);
        Camera cam = {position, direction, up};
        last_cameras[j] = cam;
        if (point_id == choosen_id) {
            std::cout << position.x + translation.x << " " << position.y + translation.y << " " << position.z + translation
                .z << std::endl;
        }
        point_id++;
      }
      renderer->setCameraGroup(last_cameras, last_batch_size, hemisphere_resolution, voxel_resolution);
      renderer->render();

      std::vector<uint32_t> pixels(last_batch_size*fbSize.x*fbSize.y);
      std::vector<half> cosine_factors(last_batch_size*fbSize.x*fbSize.y*6);
      std::vector<float> horizon_factor(last_batch_size);
      std::vector<int> sky_view_factor(last_batch_size);


      renderer->downloadPixels(pixels.data());
      renderer->downloadIncidentFactors(cosine_factors.data());
      renderer->downloadHorizonFactors(horizon_factor.data());
      renderer->downloadSVF(sky_view_factor.data());

      indexFile.write(reinterpret_cast<const char*>(pixels.data()), pixels.size() * sizeof(uint32_t));
      cosFile.write(reinterpret_cast<const char*>(cosine_factors.data()), cosine_factors.size() * sizeof(half));
      horizonfactorFile.write(reinterpret_cast<const char*>(horizon_factor.data()), horizon_factor.size() * sizeof(float));
      skyviewfactorFile.write(reinterpret_cast<const char*>(sky_view_factor.data()), sky_view_factor.size() * sizeof(int));
      delete[] last_cameras;
      }  



      indexFile.close();
      cosFile.close();
      horizonfactorFile.close();
      skyviewfactorFile.close();

      std::cout<<"Finito"<<std::endl;

      

      voxel_dim voxel_dimension = renderer->print_dimension();
      CFG["result"]["voxel_dim_x"] = voxel_dimension.num_x;
      CFG["result"]["voxel_dim_y"] = voxel_dimension.num_y;
      CFG["result"]["voxel_dim_z"] = voxel_dimension.num_z;
      CFG["result"]["bbox_min"] = {voxel_dimension.bbox_min.x - translation.x, voxel_dimension.bbox_min.y - translation.y, voxel_dimension.bbox_min.z - translation.z};
      CFG["result"]["bbox_max"] = {voxel_dimension.bbox_max.x - translation.x, voxel_dimension.bbox_max.y - translation.y, voxel_dimension.bbox_max.z - translation.z};
      auto end = std::chrono::high_resolution_clock::now();

      std::chrono::duration<double, std::milli> elapsed = end - start; // 计算经过的毫秒数
      float compute_time = static_cast<float>(elapsed.count()) / 1000;
      std::cout << "Total rendering time: " << compute_time << " second" << std::endl;
      CFG["result"]["viewshed_rendering_time"] = compute_time;

      std::ofstream out_json("config.json");
      std::ofstream out_backup_json(config_backup_path);
      out_json << std::setw(4) << CFG << std::endl;
      out_backup_json << std::setw(4) << CFG << std::endl;

      





      // std::cout << GDT_TERMINAL_GREEN
      //           << std::endl
      //           << "Image rendered, and saved to " << fileName << " ... done." << std::endl
      //           << GDT_TERMINAL_DEFAULT
      //           << std::endl;

      
    } catch (std::runtime_error& e) {
      std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what()
                << GDT_TERMINAL_DEFAULT << std::endl;
	  std::cout << "Did you forget to copy sponza.obj and sponza.mtl into your optix7course/models directory?" << std::endl;
	  exit(1);
    }
    return 0;
  }
  
} // ::osc
