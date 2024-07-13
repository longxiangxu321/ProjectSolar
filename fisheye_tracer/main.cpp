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
      Model *model = loadOBJ(
#ifdef _WIN32
      // on windows, visual studio creates _two_ levels of build dir
      // (x86/Release)
      "../../models/sponza.obj"
#else
      // on linux, common practice is to have ONE level of build dir
      // (say, <project>/build/)...
      "../models/sponza.obj"
#endif
                             );

      model->transformModel();
      SampleRenderer *renderer = new SampleRenderer(model);

      std::vector<GridPoint> gridpoints= create_point_grid(*model);
      std::cout<<"Grid points created: "<< gridpoints.size()<<std::endl;

      const vec2i fbSize(vec2i(360,90));

      auto start = std::chrono::high_resolution_clock::now();


      // srand(static_cast<unsigned>(time(0)));

      std::ofstream outFile("../semantic_map.dat", std::ios::binary | std::ios::out);
      if (!outFile) {
        std::cerr << "无法打开文件进行写操作" << std::endl;
        return 1;
      }


      uint32_t num_samplepoints = gridpoints.size();
      const uint32_t batch_size = 2500;
      const uint32_t last_batch_size = num_samplepoints % batch_size;
      uint32_t num_batches = num_samplepoints / batch_size;
      vec2i spliting = vec2i(360, 90);

      std::cout<<"Rendering with batch size "<<batch_size<<std::endl;
      std::cout <<"Total number of batches: "<<num_batches<<std::endl;
      for (uint32_t i = 0; i < num_batches - 1; i++) {
        if (i % 10 == 0) {
          std::cout<<"Rendering batch "<<i<<"to "<< i+10 <<std::endl;
        }
        
        Camera cameras[batch_size];

        std::vector<GridPoint> batch_points(gridpoints.begin() + i * batch_size, gridpoints.begin() + (i + 1) * batch_size);
        for (uint32_t j = 0; j < batch_points.size(); j++) {
          GridPoint point = batch_points[j];
          vec3f position = point.position;
          vec3f direction = point.normal;
          vec3f up = vec3f(0.f, 1.f, 0.f);
          Camera cam = {position, position + direction, up};
          cameras[j] = cam;
        }
        renderer->setCameraGroup(cameras, batch_size, spliting);
        renderer->render();
        std::vector<uint32_t> pixels(batch_size*fbSize.x*fbSize.y);
        std::vector<float> pixels_float(batch_size*fbSize.x*fbSize.y*3);
        renderer->downloadPixels(pixels.data());
        renderer->downloadIncidentAngles(pixels_float.data());

        outFile.write(reinterpret_cast<const char*>(pixels.data()), pixels.size() * sizeof(uint32_t));
        outFile.write(reinterpret_cast<const char*>(pixels_float.data()), pixels_float.size() * sizeof(float));

      }
      outFile.close();

      std::cout<<"Rendering last batch of "<<last_batch_size<<std::endl;
      
      Camera* last_cameras = new Camera[last_batch_size];
      std::vector<GridPoint> last_batch_points(gridpoints.begin() + num_batches * batch_size, gridpoints.end());
      for (uint32_t j = 0; j < last_batch_points.size(); j++) {
        GridPoint point = last_batch_points[j];
        vec3f position = point.position;
        vec3f direction = point.normal;
        vec3f up = vec3f(0.f, 1.f, 0.f);
        Camera cam = {position, position + direction, up};
        last_cameras[j] = cam;
      }
      renderer->setCameraGroup(last_cameras, last_batch_size, spliting);
      renderer->render();
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
