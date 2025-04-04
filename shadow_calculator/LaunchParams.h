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

#pragma once

#include "gdt/math/vec.h"
#include "sample_pointGrid.h"
#include "optix7.h"

namespace solarcity {
  using namespace gdt;

  struct TriangleMeshSBTData {
    vec3f  color;
    vec3f *vertex;
    vec3i *index;
  };

  // struct Camera_info{
  //     vec3f position;
  //     vec3f direction;

  //     vec3f tangent;
  //     vec3f bitangent;
  // };
  
  struct LaunchParams
  {
    bool* shadowBuffer;

    vec3f* positions;
    vec3f* directions;

    uint32_t n_cameras;
    uint32_t n_directions;

    vec3f bbox_min;
    vec3f bbox_max;


    OptixTraversableHandle traversable;

    // void free_temp() {
    //   free(positions);
    //   free(directions);
    //   free(tangents);
    //   free(bitangents);
    //   free(colorBuffer);
    // }
  };

} // ::osc
