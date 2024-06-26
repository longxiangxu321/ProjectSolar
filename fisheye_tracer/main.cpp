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

      std::vector<GridPoint> gridpoints= create_point_grid(*model);
      std::cout<<"Grid points created: "<< gridpoints.size()<<std::endl;
    
      SampleRenderer *renderer = new SampleRenderer(model);
      const vec2i fbSize(vec2i(360,90));

      auto start = std::chrono::high_resolution_clock::now();


      srand(static_cast<unsigned>(time(0)));

      const int numCameras = 100;
      Camera cameras[numCameras];

    for (int i = 0; i < numCameras; ++i) {
        vec3f randomPosition(
            -1000 + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (2000))),
            -1000 + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (2000))),
            0 + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (100)))
        );

        vec3f direction = vec3f(0.f, -400.f, 0.f); // 目标位置
        vec3f up = vec3f(0.f, 1.f, 0.f); // 上方向
        vec3f horizontal, vertical, tangent, bitangent; // 根据需要计算这些向量
        Camera cam= {randomPosition, direction, up};
        cameras[i] = cam;
    }
    vec2i spliting = vec2i(360, 90);
    renderer->setCameraGroup(cameras, numCameras, spliting);
    renderer->render();
    std::vector<uint32_t> pixels(numCameras*spliting.x*spliting.y);
    renderer->downloadPixels(pixels.data());

      // std::vector<vec3f> lookfroms = {vec3f(-10.07f, 20.681f, 20), vec3f(0, 0, 20), vec3f(300, 100, 10), vec3f(200, -50, 15)};


      // for (int i = 0; i < lookfroms.size(); i++) {
        // Camera camera = { /*from*/lookfroms[i],
        //                   /* at */model->bounds.center()-vec3f(0,400,0),
        //                   /* up */vec3f(0.f,1.f,0.f) };
      //   renderer->resize(fbSize);
      //   renderer->setCamera(camera);
      //   renderer->render();
      //   std::vector<uint32_t> pixels(fbSize.x*fbSize.y);
      //   renderer->downloadPixels(pixels.data());
      // }

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
