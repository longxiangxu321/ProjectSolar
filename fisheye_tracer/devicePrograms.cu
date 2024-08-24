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
    uint32_t voxel_id;
    float up;
    float down;
    float left;
    float right;
    float front;
    float back;
    int mask;
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
    
    // half azimuth = __float2half(atan2(voxel_ray_dir.y, voxel_ray_dir.x)); // from -180 to 180
    // half elevation = __float2half(atan2(voxel_ray_dir.z, sqrt(voxel_ray_dir.x * voxel_ray_dir.x + voxel_ray_dir.y * voxel_ray_dir.y))); // from -90 to 90
    float cos_up = dot(voxel_ray_dir, vec3f(0.0f, 0.0f, 1.0f));
    float cos_down = dot(voxel_ray_dir, vec3f(0.0f, 0.0f, -1.0f));
    float cos_left = dot(voxel_ray_dir, vec3f(-1.0f, 0.0f, 0.0f));
    float cos_right = dot(voxel_ray_dir, vec3f(1.0f, 0.0f, 0.0f));
    float cos_front = dot(voxel_ray_dir, vec3f(0.0f, 1.0f, 0.0f));
    float cos_back = dot(voxel_ray_dir, vec3f(0.0f, -1.0f, 0.0f));

    uint32_t voxel_id = voxel_x + voxel_y * optixLaunchParams.voxel_dim.x + voxel_z * optixLaunchParams.voxel_dim.x * optixLaunchParams.voxel_dim.y;
    

    PRD &prd = *(PRD*)getPRD<PRD>();
    
    prd.up = cos_up;
    prd.down = cos_down;
    prd.left = cos_left;
    prd.right = cos_right;
    prd.front = cos_front;
    prd.back = cos_back;

    prd.voxel_id = voxel_id;
    prd.mask = 0;
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
    
    prd.up = 0.0f;
    prd.down = 0.0f;
    prd.left = 0.0f;
    prd.right = 0.0f;
    prd.front = 0.0f;
    prd.back = 0.0f;
    prd.voxel_id = UINT32_MAX_VALUE;
    prd.mask = 1;

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

    float theta = degrees_to_radians(float(ix) * optixLaunchParams.hemisphere_resolution.x);
    float phi = degrees_to_radians(float(iy) * optixLaunchParams.hemisphere_resolution.y);

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

    float angle_with_horizon  = acos(dot(rayDir, vec3f(0.0f, 0.0f, 1.0f)));


    // vec3f pixelColorPRD = vec3f(0.f);
    PRD pixelColorPRD = {0.f,0.f, 0, 0};

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


    const uint32_t voxel_id = pixelColorPRD.voxel_id;
    float cos_phi = cos(M_PI/2 - phi);
    half up = __float2half(pixelColorPRD.up * cos_phi);
    half down = __float2half(pixelColorPRD.down * cos_phi);
    half left = __float2half(pixelColorPRD.left * cos_phi);
    half right = __float2half(pixelColorPRD.right * cos_phi);
    half front = __float2half(pixelColorPRD.front * cos_phi);
    half back = __float2half(pixelColorPRD.back * cos_phi);

    // and write to frame buffer ...
    // const uint32_t fbIndex = ix+iy*optixLaunchParams.frame.size.x;
    // optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
    // int offset = optixLaunchParams.batch_offset * optixLaunchParams.n_azimuth * optixLaunchParams.n_elevation;
    const uint32_t fbIndex = ray_id + cam_id * optixLaunchParams.n_azimuth * optixLaunchParams.n_elevation;
    // float importance = pow((1 - fabs(cos(angle_with_horizon))), optixLaunchParams.horizon_gamma);
    // float pixel_horizon_intensity = (pixelColorPRD.mask *importance);
    

    // optixLaunchParams.horizon_factorBuffer[cam_id] += 1.0f;
    // optixLaunchParams.horizon_importanceBuffer[cam_id] += 1.0f;

    // atomicAdd(&optixLaunchParams.horizon_factorBuffer[cam_id], importance* pixelColorPRD.mask);
    // atomicAdd(&optixLaunchParams.horizon_importanceBuffer[cam_id], importance);
    atomicAdd(&optixLaunchParams.sky_view_factorBuffer[cam_id], pixelColorPRD.mask);

    optixLaunchParams.colorBuffer[fbIndex] = voxel_id;

    const uint32_t cosindex = ray_id * 6 + cam_id * optixLaunchParams.n_azimuth * optixLaunchParams.n_elevation * 6;
    optixLaunchParams.incident_factorBuffer[cosindex] = up;
    optixLaunchParams.incident_factorBuffer[cosindex + 1] = down;
    optixLaunchParams.incident_factorBuffer[cosindex + 2] = left;
    optixLaunchParams.incident_factorBuffer[cosindex + 3] = right;
    optixLaunchParams.incident_factorBuffer[cosindex + 4] = front;
    optixLaunchParams.incident_factorBuffer[cosindex + 5] = back;
    // optixLaunchParams.incident_angleBuffer[fbIndex] = cosDN;
  }
  
} // ::osc
