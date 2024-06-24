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
#include "optix7.h"

namespace osc {
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
    uint32_t *colorBuffer;
    


    int n_elevation;
    int n_azimuth;

    vec3f* positions;
    vec3f* directions;
    vec3f* tangents;
    vec3f* bitangents;
    int n_cameras;

    OptixTraversableHandle traversable;
  };

} // ::osc
