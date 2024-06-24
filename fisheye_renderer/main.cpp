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


/*! \namespace osc - Optix Siggraph Course */
namespace osc {

  struct SampleWindow : public GLFCameraWindow
  {
    SampleWindow(const std::string &title,
                 const Model *model,
                 const Camera &camera,
                 const float worldScale)
      : GLFCameraWindow(title,camera.from,camera.at,camera.up,worldScale),
        sample(model)
    {
      sample.setCamera(camera);
    }
    
    virtual void render() override
    {
      if (cameraFrame.modified) {
        sample.setCamera(Camera{ cameraFrame.get_from(),
                                 cameraFrame.get_at(),
                                 cameraFrame.get_up() });
        cameraFrame.modified = false;
      }
      sample.render();
    }
    
    virtual void draw() override
    {
      sample.downloadPixels(pixels.data());
      if (fbTexture == 0)
        glGenTextures(1, &fbTexture);
      
      glBindTexture(GL_TEXTURE_2D, fbTexture);
      GLenum texFormat = GL_RGBA;
      GLenum texelType = GL_UNSIGNED_BYTE;
      glTexImage2D(GL_TEXTURE_2D, 0, texFormat, fbSize.x, fbSize.y, 0, GL_RGBA,
                   texelType, pixels.data());

      glDisable(GL_LIGHTING);
      glColor3f(1, 1, 1);

      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

      glEnable(GL_TEXTURE_2D);
      glBindTexture(GL_TEXTURE_2D, fbTexture);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      
      glDisable(GL_DEPTH_TEST);

      glViewport(0, 0, fbSize.x, fbSize.y);

      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      glOrtho(0.f, (float)fbSize.x, 0.f, (float)fbSize.y, -1.f, 1.f);

      glBegin(GL_QUADS);
      {
        glTexCoord2f(0.f, 0.f);
        glVertex3f(0.f, 0.f, 0.f);
      
        glTexCoord2f(0.f, 1.f);
        glVertex3f(0.f, (float)fbSize.y, 0.f);
      
        glTexCoord2f(1.f, 1.f);
        glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);
      
        glTexCoord2f(1.f, 0.f);
        glVertex3f((float)fbSize.x, 0.f, 0.f);
      }
      glEnd();
    }
    
    virtual void resize(const vec2i &newSize) 
    {
      fbSize = newSize;
      sample.resize(newSize);
      pixels.resize(newSize.x*newSize.y);
    }

    vec2i                 fbSize;
    GLuint                fbTexture {0};
    SampleRenderer        sample;
    std::vector<uint32_t> pixels;
  };
  
  
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
      // Camera camera = { /*from*/vec3f(-10.07f, 20.681f, -2.7304f),
      //                   /* at */model->bounds.center()-vec3f(0,400,0),
      //                   /* up */vec3f(0.f,1.f,0.f) };
      model->transformModel();

      // vec3f lookfrom = model->bounds.center() + vec3f(0, 0, 500);
      // vec3f lookat = model->bounds.center();

      // Camera camera = { /*from*/ lookfrom,
      //             /* at */ lookat,
      //             /* up */ vec3f(0.f, 1.f, 0.f) };
      // // something approximating the scale of the world, so the
      // // camera knows how much to move for any given user interaction:
      // SampleRenderer *renderer = new SampleRenderer(model);

      // const vec2i fbSize(vec2i(360,90));
      // renderer->resize(fbSize);
      // renderer->setCamera(camera);
      // renderer->render();

      // std::vector<uint32_t> pixels(fbSize.x*fbSize.y);
      // renderer->downloadPixels(pixels.data());

      // const std::string fileName = "rendered_fisheye.png";
      // stbi_write_png(fileName.c_str(),fbSize.x,fbSize.y,4,
      //                pixels.data(),fbSize.x*sizeof(uint32_t));

      SampleRenderer *renderer = new SampleRenderer(model);
      const vec2i fbSize(vec2i(360,90));

      auto start = std::chrono::high_resolution_clock::now();
      std::vector<vec3f> lookfroms = {vec3f(-10.07f, 20.681f, 20), vec3f(0, 0, 20), vec3f(300, 100, 10), vec3f(200, -50, 15)};
      for (int i = 0; i < lookfroms.size(); i++) {
        Camera camera = { /*from*/lookfroms[i],
                          /* at */model->bounds.center()-vec3f(0,400,0),
                          /* up */vec3f(0.f,1.f,0.f) };
        renderer->resize(fbSize);
        renderer->setCamera(camera);
        renderer->render();
        std::vector<uint32_t> pixels(fbSize.x*fbSize.y);
        renderer->downloadPixels(pixels.data());
        // const std::string fileName = "rendered_fisheye_" + std::to_string(i) + ".png";
        // stbi_write_png(fileName.c_str(),fbSize.x,fbSize.y,4,
        //                pixels.data(),fbSize.x*sizeof(uint32_t));
        // std::cout << GDT_TERMINAL_GREEN
        //     << std::endl
        //     << "Image rendered, and saved to " << fileName << " ... done." << std::endl
        //     << GDT_TERMINAL_DEFAULT
        //     << std::endl;
      }

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
