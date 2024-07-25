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

#include <optix_device.h>

#include "LaunchParams.h"

#define UINT32_MAX_VALUE 4294967295U

using namespace osc;

namespace osc {
  
  /*! launch parameters in constant memory, filled in by optix upon
      optixLaunch (this gets filled in from the buffer we pass to
      optixLaunch) */
  extern "C" __constant__ LaunchParams optixLaunchParams;

  // for this simple example, we have a single ray type
  enum { SURFACE_RAY_TYPE=0, RAY_TYPE_COUNT };
  
  struct PRD {
    float azimuth;
    float elevation;
    uint32_t voxel_id;
};


  static __forceinline__ __device__
  void *unpackPointer( uint32_t i0, uint32_t i1 )
  {
    const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>( uptr ); 
    return ptr;
  }

  static __forceinline__ __device__
  void  packPointer( void* ptr, uint32_t& i0, uint32_t& i1 )
  {
    const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
  }

  template<typename T>
  static __forceinline__ __device__ T *getPRD()
  { 
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>( unpackPointer( u0, u1 ) );
  }

  static __forceinline__ __device__
  double degrees_to_radians(float degrees) {
    return degrees * M_PI / 180.0;
  }
  
  //------------------------------------------------------------------------------
  // closest hit and anyhit programs for radiance-type rays.
  //
  // Note eventually we will have to create one pair of those for each
  // ray type and each geometry type we want to render; but this
  // simple example doesn't use any actual geometries yet, so we only
  // create a single, dummy, set of them (we do have to have at least
  // one group of them to set up the SBT)
  //------------------------------------------------------------------------------
  
  extern "C" __global__ void __closesthit__radiance()
  {
    const TriangleMeshSBTData &sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

    // compute normal:
    const int   primID = optixGetPrimitiveIndex();
    const vec3i index  = sbtData.index[primID];
    const vec3f &A     = sbtData.vertex[index.x];
    const vec3f &B     = sbtData.vertex[index.y];
    const vec3f &C     = sbtData.vertex[index.z];
    // const vec3f Ng     = normalize(cross(B-A,C-A));

    // const vec3f rayDir = optixGetWorldRayDirection();

    // const half cosDN  = fabsf(dot(normalize(rayDir),Ng));

    float2 hit_barycentric = optixGetTriangleBarycentrics();
    vec3f hit_point = A * (1.0f - hit_barycentric.x - hit_barycentric.y) +
                      B * hit_barycentric.x +
                      C * hit_barycentric.y;

    vec3f rayOrigin = optixGetWorldRayOrigin();
    
    int voxel_x = (hit_point.x - optixLaunchParams.bbox_min.x)/optixLaunchParams.resolution;
    int voxel_y = (hit_point.y - optixLaunchParams.bbox_min.y)/optixLaunchParams.resolution;
    int voxel_z = (hit_point.z - optixLaunchParams.bbox_min.z)/optixLaunchParams.resolution;

    vec3f voxel_center = vec3f(voxel_x * optixLaunchParams.resolution + optixLaunchParams.resolution/2,
                               voxel_y * optixLaunchParams.resolution + optixLaunchParams.resolution/2,
                               voxel_z * optixLaunchParams.resolution + optixLaunchParams.resolution/2) + optixLaunchParams.bbox_min;
    
    vec3f voxel_ray_dir = normalize(rayOrigin - voxel_center);
    
    half azimuth = __float2half(atan2(voxel_ray_dir.y, voxel_ray_dir.x)*180.0f / M_PI); // from -180 to 180
    half elevation = __float2half(atan2(voxel_ray_dir.z, sqrt(voxel_ray_dir.x * voxel_ray_dir.x + voxel_ray_dir.y * voxel_ray_dir.y)*180.0f / M_PI)); // from -90 to 90
    
    uint32_t voxel_id = voxel_x + voxel_y * optixLaunchParams.voxel_dim.x + voxel_z * optixLaunchParams.voxel_dim.x * optixLaunchParams.voxel_dim.y;
    

    PRD &prd = *(PRD*)getPRD<PRD>();
    
    prd.azimuth = azimuth;
    prd.elevation = elevation;
    prd.voxel_id = voxel_id;
  }
  
  extern "C" __global__ void __anyhit__radiance()
  { /*! for this simple example, this will remain empty */ }


  
  //------------------------------------------------------------------------------
  // miss program that gets called for any ray that did not have a
  // valid intersection
  //
  // as with the anyhit/closest hit programs, in this example we only
  // need to have _some_ dummy function to set up a valid SBT
  // ------------------------------------------------------------------------------
  
  extern "C" __global__ void __miss__radiance()
  {
    // vec3f &prd = *(vec3f*)getPRD<vec3f>();
    // // set to constant white as background color
    // prd = vec3f(1.f);
    PRD &prd = *(PRD*)getPRD<PRD>();
    
    prd.azimuth = 181.0f;
    prd.elevation = 91.0f;
    prd.voxel_id = UINT32_MAX_VALUE;

  }

  //------------------------------------------------------------------------------
  // ray gen program - the actual rendering happens in here
  //------------------------------------------------------------------------------
  extern "C" __global__ void __raygen__renderFrame()
  {
    const int cam_id = optixGetLaunchIndex().x; // index for camera position
    const int ray_id = optixGetLaunchIndex().y; // index for ray direction




    // 将屏幕空间坐标转换为球坐标
    int ix = ray_id % optixLaunchParams.n_azimuth;
    int iy = ray_id / optixLaunchParams.n_azimuth;
    // int ix = ray_id % 360;
    // int iy = ray_id / 360;


    float theta = degrees_to_radians(float(ix) * optixLaunchParams.hemisphere_resolution.x);
    float phi = degrees_to_radians(float(iy) * optixLaunchParams.hemisphere_resolution.y) * 0.5f;
    // float theta = screen.x * 2.0f * M_PI;  // 从 0 到 2*PI
    // float phi = screen.y * 0.5f * M_PI;           // 从 0 到 PI

    // 使用球坐标计算射线方向
    float x = sin(phi) * cos(theta);
    float y = sin(phi) * sin(theta);
    float z = cos(phi);

    vec3f camera_position = optixLaunchParams.positions[cam_id];
    vec3f camera_direction = optixLaunchParams.directions[cam_id];
    vec3f tangent = optixLaunchParams.tangents[cam_id];
    vec3f bitangent = optixLaunchParams.bitangents[cam_id];

    // 计算并归一化射线方向
    vec3f rayDir = normalize(x * tangent + y * bitangent + z * camera_direction);

    // vec3f pixelColorPRD = vec3f(0.f);
    PRD pixelColorPRD = {0.f,0.f, 0};

    // the values we store the PRD pointer in:
    uint32_t u0, u1;
    packPointer( &pixelColorPRD, u0, u1 );

    optixTrace(optixLaunchParams.traversable,
               camera_position,
               rayDir,
               0.f,    // tmin
               1e20f,  // tmax
               0.0f,   // rayTime
               OptixVisibilityMask( 255 ),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
               SURFACE_RAY_TYPE,             // SBT offset
               RAY_TYPE_COUNT,               // SBT stride
               SURFACE_RAY_TYPE,             // missSBTIndex 
               u0, u1 );

    // const int r = int(255.99f*pixelColorPRD.x);
    // const int g = int(255.99f*pixelColorPRD.y);
    // const int b = int(255.99f*pixelColorPRD.z);
    const float azimuth = pixelColorPRD.azimuth;
    const float elevation = pixelColorPRD.elevation;
    const uint32_t voxel_id = pixelColorPRD.voxel_id;

    const int r = 1;
    const int g = 1;
    const int b = 1;

    // convert to 32-bit rgba value (we explicitly set alpha to 0xff
    // to make stb_image_write happy ...
    const uint32_t rgba = 0xff000000
      | (r<<0) | (g<<8) | (b<<16);

    // and write to frame buffer ...
    // const uint32_t fbIndex = ix+iy*optixLaunchParams.frame.size.x;
    // optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
    const uint32_t fbIndex = ray_id + cam_id * optixLaunchParams.n_azimuth * optixLaunchParams.n_elevation;
    optixLaunchParams.colorBuffer[fbIndex] = voxel_id;
    optixLaunchParams.incident_azimuthBuffer[fbIndex] = azimuth;
    optixLaunchParams.incident_elevationBuffer[fbIndex] = elevation;
    // optixLaunchParams.incident_angleBuffer[fbIndex] = cosDN;
  }
  
} // ::osc
