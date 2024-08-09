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

/*! \namespace osc - Optix Siggraph Course */
namespace osc {

  
  /*! main entry point to this example - initially optix, print hello
    world, then exit */
  extern "C" int main(int ac, char **av)
  {
    try {
      Model *model = loadCityJSON(
#ifdef _WIN32
      // on windows, visual studio creates _two_ levels of build dir
      // (x86/Release)
      "../../models/Delft.city.json"
#else
      // on linux, common practice is to have ONE level of build dir
      // (say, <project>/build/)...
      "../models/sponza.obj"
#endif
                             );

      model->transformModel();
      SampleRenderer *renderer = new SampleRenderer(model);
      // std::cout<<renderer->bbox.lower<<std::endl;
      // std::cout<<renderer->bbox.upper<<std::endl;
      // std::cout<<renderer->launchParams.bbox_min<<std::endl;
      // std::cout<<renderer->launchParams.bbox_max<<std::endl;

      std::vector<GridPoint> raw_gridpoints= create_point_grid(*model);
      std::vector<GridPoint> gridpoints = clean_point_grid(raw_gridpoints);
      std::cout<<"Grid points created: "<< gridpoints.size()<<std::endl;

      vec3f translation = model->bounds.center() - model->original_center;
      save_point_grid(gridpoints, translation, "../grid_points.dat");
      std::cout<<"Grid points saved"<<std::endl;

      const vec2i fbSize(vec2i(360,90));

      auto start = std::chrono::high_resolution_clock::now();


      // srand(static_cast<unsigned>(time(0)));

      std::ofstream indexFile("../index_map.dat", std::ios::binary | std::ios::out);
      std::ofstream azimuthFile("../azimuth_map.dat", std::ios::binary | std::ios::out);
      std::ofstream elevationFile("../elevation_map.dat", std::ios::binary | std::ios::out);
      std::ofstream horizonfactorFile("../horizon_factor_map.dat", std::ios::binary | std::ios::out);
      std::ofstream skyviewfactorFile("../sky_view_factor_map.dat", std::ios::binary | std::ios::out);

      if (!indexFile || !azimuthFile || !elevationFile) {
        std::cerr << "cannot create semantic_map files" << std::endl;
        return 1;
      }


      uint32_t num_samplepoints = gridpoints.size();
      const uint32_t batch_size = 2500;
      const uint32_t last_batch_size = num_samplepoints % batch_size;
      uint32_t num_batches = num_samplepoints / batch_size;
      vec2i hemisphere_resolution = vec2i(1,1);
      
      // vec2i spliting = vec2i(360, 90);

      int point_id = 0;
      int choosen_id = 19402;

      std::cout<<"Rendering with batch size "<<batch_size<<std::endl;
      std::cout <<"Total number of batches: "<<num_batches<<std::endl;
      int batch_offset = 0;
      for (uint32_t i = 0; i < num_batches - 1; i++) {
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
        renderer->setCameraGroup(cameras, batch_size, hemisphere_resolution);
        renderer->render();
        std::vector<uint32_t> pixels(batch_size*fbSize.x*fbSize.y);
        std::vector<half> incident_azimuth(batch_size*fbSize.x*fbSize.y);
        std::vector<half> incident_elevation(batch_size*fbSize.x*fbSize.y);
        std::vector<float> horizon_factor(batch_size);
        std::vector<int> sky_view_factor(batch_size);


        renderer->downloadPixels(pixels.data());
        renderer->downloadIncidentAngles(incident_azimuth.data(), incident_elevation.data());
        renderer->downloadHorizonFactors(horizon_factor.data());
        renderer->downloadSVF(sky_view_factor.data());

        indexFile.write(reinterpret_cast<const char*>(pixels.data()), pixels.size() * sizeof(uint32_t));
        azimuthFile.write(reinterpret_cast<const char*>(incident_azimuth.data()), incident_azimuth.size() * sizeof(half));
        elevationFile.write(reinterpret_cast<const char*>(incident_elevation.data()), incident_elevation.size() * sizeof(half));
        horizonfactorFile.write(reinterpret_cast<const char*>(horizon_factor.data()), horizon_factor.size() * sizeof(float));
        skyviewfactorFile.write(reinterpret_cast<const char*>(sky_view_factor.data()), sky_view_factor.size() * sizeof(int));

        batch_offset += batch_size;
      }
      

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
      renderer->setCameraGroup(last_cameras, last_batch_size, hemisphere_resolution, batch_offset);
      renderer->render();

      renderer->print_dimension();

      std::vector<uint32_t> pixels(batch_size*fbSize.x*fbSize.y);
      std::vector<half> incident_azimuth(batch_size*fbSize.x*fbSize.y);
      std::vector<half> incident_elevation(batch_size*fbSize.x*fbSize.y);
      std::vector<float> horizon_factor(batch_size);
      std::vector<int> sky_view_factor(batch_size);


      renderer->downloadPixels(pixels.data());
      renderer->downloadIncidentAngles(incident_azimuth.data(), incident_elevation.data());
      renderer->downloadHorizonFactors(horizon_factor.data());
      renderer->downloadSVF(sky_view_factor.data());

      indexFile.write(reinterpret_cast<const char*>(pixels.data()), pixels.size() * sizeof(uint32_t));
      azimuthFile.write(reinterpret_cast<const char*>(incident_azimuth.data()), incident_azimuth.size() * sizeof(half));
      elevationFile.write(reinterpret_cast<const char*>(incident_elevation.data()), incident_elevation.size() * sizeof(half));
      horizonfactorFile.write(reinterpret_cast<const char*>(horizon_factor.data()), horizon_factor.size() * sizeof(float));
      skyviewfactorFile.write(reinterpret_cast<const char*>(sky_view_factor.data()), sky_view_factor.size() * sizeof(int));

      indexFile.close();
      azimuthFile.close();
      elevationFile.close();
      horizonfactorFile.close();
      skyviewfactorFile.close();

      std::cout<<"Finito"<<std::endl;

      delete[] last_cameras;
      


      auto end = std::chrono::high_resolution_clock::now();

      std::chrono::duration<double, std::milli> elapsed = end - start; // 计算经过的毫秒数

      std::cout << "Total rendering time: " << elapsed.count() << " ms" << std::endl;


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
