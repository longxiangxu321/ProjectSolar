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
// this include may only appear in a single source file:
#include <optix_function_table_definition.h>
#include<cmath>

/*! \namespace osc - Optix Siggraph Course */
namespace osc {

  extern "C" char embedded_ptx_code[];

  /*! SBT record for a raygen program */
  struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) RaygenRecord
  {
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void *data;
  };

  /*! SBT record for a miss program */
  struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) MissRecord
  {
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void *data;
  };

  /*! SBT record for a hitgroup program */
  struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) HitgroupRecord
  {
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    TriangleMeshSBTData data;
  };


  /*! constructor - performs all setup, including initializing
    optix, creates module, pipeline, programs, SBT, etc. */
  SampleRenderer::SampleRenderer(const Model *model)
    : model(model)
  {
    bbox = model->bounds;
    

    initOptix();
      
    std::cout << "#osc: creating optix context ..." << std::endl;
    createContext();
      
    std::cout << "#osc: setting up module ..." << std::endl;
    createModule();

    std::cout << "#osc: creating raygen programs ..." << std::endl;
    createRaygenPrograms();
    std::cout << "#osc: creating miss programs ..." << std::endl;
    createMissPrograms();
    std::cout << "#osc: creating hitgroup programs ..." << std::endl;
    createHitgroupPrograms();

    launchParams.traversable = buildAccel();
    
    std::cout << "#osc: setting up optix pipeline ..." << std::endl;
    createPipeline();

    std::cout << "#osc: building SBT ..." << std::endl;
    buildSBT();

    // launchParamsBuffer.alloc(sizeof(launchParams));
    std::cout << "#osc: context, module, pipeline, etc, all set up ..." << std::endl;

    std::cout << GDT_TERMINAL_GREEN;
    std::cout << "#osc: Optix 7 Sample fully set up" << std::endl;
    std::cout << GDT_TERMINAL_DEFAULT;
  }

  OptixTraversableHandle SampleRenderer::buildAccel()
  {
    PING;
    PRINT(model->meshes.size());
    
    vertexBuffer.resize(model->meshes.size());
    indexBuffer.resize(model->meshes.size());
    
    OptixTraversableHandle asHandle { 0 };
    
    // ==================================================================
    // triangle inputs
    // ==================================================================
    std::vector<OptixBuildInput> triangleInput(model->meshes.size());
    std::vector<CUdeviceptr> d_vertices(model->meshes.size());
    std::vector<CUdeviceptr> d_indices(model->meshes.size());
    std::vector<uint32_t> triangleInputFlags(model->meshes.size());

    for (int meshID=0;meshID<model->meshes.size();meshID++) {
      // upload the model to the device: the builder
      TriangleMesh &mesh = *model->meshes[meshID];
      vertexBuffer[meshID].alloc_and_upload(mesh.vertex);
      indexBuffer[meshID].alloc_and_upload(mesh.index);

      triangleInput[meshID] = {};
      triangleInput[meshID].type
        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

      // create local variables, because we need a *pointer* to the
      // device pointers
      d_vertices[meshID] = vertexBuffer[meshID].d_pointer();
      d_indices[meshID]  = indexBuffer[meshID].d_pointer();
      
      triangleInput[meshID].triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
      triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(vec3f);
      triangleInput[meshID].triangleArray.numVertices         = (int)mesh.vertex.size();
      triangleInput[meshID].triangleArray.vertexBuffers       = &d_vertices[meshID];
    
      triangleInput[meshID].triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
      triangleInput[meshID].triangleArray.indexStrideInBytes  = sizeof(vec3i);
      triangleInput[meshID].triangleArray.numIndexTriplets    = (int)mesh.index.size();
      triangleInput[meshID].triangleArray.indexBuffer         = d_indices[meshID];
    
      triangleInputFlags[meshID] = 0 ;
    
      // in this example we have one SBT entry, and no per-primitive
      // materials:
      triangleInput[meshID].triangleArray.flags               = &triangleInputFlags[meshID];
      triangleInput[meshID].triangleArray.numSbtRecords               = 1;
      triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer        = 0; 
      triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes   = 0; 
      triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0; 
    }
    // ==================================================================
    // BLAS setup
    // ==================================================================
    
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags             = OPTIX_BUILD_FLAG_NONE
      | OPTIX_BUILD_FLAG_ALLOW_COMPACTION
      ;
    accelOptions.motionOptions.numKeys  = 1;
    accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;
    
    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage
                (optixContext,
                 &accelOptions,
                 triangleInput.data(),
                 (int)model->meshes.size(),  // num_build_inputs
                 &blasBufferSizes
                 ));
    
    // ==================================================================
    // prepare compaction
    // ==================================================================
    
    CUDABuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));
    
    OptixAccelEmitDesc emitDesc;
    emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.d_pointer();
    
    // ==================================================================
    // execute build (main stage)
    // ==================================================================
    
    CUDABuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);
    
    CUDABuffer outputBuffer;
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);
      
    OPTIX_CHECK(optixAccelBuild(optixContext,
                                /* stream */0,
                                &accelOptions,
                                triangleInput.data(),
                                (int)model->meshes.size(),
                                tempBuffer.d_pointer(),
                                tempBuffer.sizeInBytes,
                                
                                outputBuffer.d_pointer(),
                                outputBuffer.sizeInBytes,
                                
                                &asHandle,
                                
                                &emitDesc,1
                                ));
    CUDA_SYNC_CHECK();
    
    // ==================================================================
    // perform compaction
    // ==================================================================
    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize,1);
    
    asBuffer.alloc(compactedSize);
    OPTIX_CHECK(optixAccelCompact(optixContext,
                                  /*stream:*/0,
                                  asHandle,
                                  asBuffer.d_pointer(),
                                  asBuffer.sizeInBytes,
                                  &asHandle));
    CUDA_SYNC_CHECK();
    
    // ==================================================================
    // aaaaaand .... clean up
    // ==================================================================
    outputBuffer.free(); // << the UNcompacted, temporary output buffer
    tempBuffer.free();
    compactedSizeBuffer.free();
    
    return asHandle;
  }
  
  /*! helper function that initializes optix and checks for errors */
  void SampleRenderer::initOptix()
  {
    std::cout << "#osc: initializing optix..." << std::endl;
      
    // -------------------------------------------------------
    // check for available optix7 capable devices
    // -------------------------------------------------------
    cudaFree(0);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
      throw std::runtime_error("#osc: no CUDA capable devices found!");
    std::cout << "#osc: found " << numDevices << " CUDA devices" << std::endl;

    // -------------------------------------------------------
    // initialize optix
    // -------------------------------------------------------
    OPTIX_CHECK( optixInit() );
    std::cout << GDT_TERMINAL_GREEN
              << "#osc: successfully initialized optix... yay!"
              << GDT_TERMINAL_DEFAULT << std::endl;
  }

  static void context_log_cb(unsigned int level,
                             const char *tag,
                             const char *message,
                             void *)
  {
    fprintf( stderr, "[%2d][%12s]: %s\n", (int)level, tag, message );
  }

  /*! creates and configures a optix device context (in this simple
    example, only for the primary GPU device) */
  void SampleRenderer::createContext()
  {
    // for this sample, do everything on one device
    const int deviceID = 0;
    CUDA_CHECK(SetDevice(deviceID));
    CUDA_CHECK(StreamCreate(&stream));
      
    cudaGetDeviceProperties(&deviceProps, deviceID);
    std::cout << "#osc: running on device: " << deviceProps.name << std::endl;
      
    CUresult  cuRes = cuCtxGetCurrent(&cudaContext);
    if( cuRes != CUDA_SUCCESS ) 
      fprintf( stderr, "Error querying current context: error code %d\n", cuRes );
      
    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
    OPTIX_CHECK(optixDeviceContextSetLogCallback
                (optixContext,context_log_cb,nullptr,4));
  }



  /*! creates the module that contains all the programs we are going
    to use. in this simple example, we use a single module from a
    single .cu file, using a single embedded ptx string */
  void SampleRenderer::createModule()
  {
    moduleCompileOptions.maxRegisterCount  = 50;
    moduleCompileOptions.optLevel          = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.usesMotionBlur     = false;
    pipelineCompileOptions.numPayloadValues   = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags     = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";
      
    pipelineLinkOptions.maxTraceDepth          = 2;
      
    const std::string ptxCode = embedded_ptx_code;
      
    char log[2048];
    size_t sizeof_log = sizeof( log );
#if OPTIX_VERSION >= 70700
    OPTIX_CHECK(optixModuleCreate(optixContext,
                                         &moduleCompileOptions,
                                         &pipelineCompileOptions,
                                         ptxCode.c_str(),
                                         ptxCode.size(),
                                         log,&sizeof_log,
                                         &module
                                         ));
#else
    OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
                                         &moduleCompileOptions,
                                         &pipelineCompileOptions,
                                         ptxCode.c_str(),
                                         ptxCode.size(),
                                         log,      // Log string
                                         &sizeof_log,// Log string sizse
                                         &module
                                         ));
#endif
    if (sizeof_log > 1) PRINT(log);
  }
    


  /*! does all setup for the raygen program(s) we are going to use */
  void SampleRenderer::createRaygenPrograms()
  {
    // we do a single ray gen program in this example:
    raygenPGs.resize(1);
      
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module            = module;           
    pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,&sizeof_log,
                                        &raygenPGs[0]
                                        ));
    if (sizeof_log > 1) PRINT(log);
  }
    
  /*! does all setup for the miss program(s) we are going to use */
  void SampleRenderer::createMissPrograms()
  {
    // we do a single ray gen program in this example:
    missPGs.resize(1);
      
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module            = module;           
    pgDesc.miss.entryFunctionName = "__miss__radiance";

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,&sizeof_log,
                                        &missPGs[0]
                                        ));
    if (sizeof_log > 1) PRINT(log);
  }
    
  /*! does all setup for the hitgroup program(s) we are going to use */
  void SampleRenderer::createHitgroupPrograms()
  {
    // for this simple example, we set up a single hit group
    hitgroupPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH            = module;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    pgDesc.hitgroup.moduleAH            = module;
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,&sizeof_log,
                                        &hitgroupPGs[0]
                                        ));
    if (sizeof_log > 1) PRINT(log);
  }
    

  /*! assembles the full pipeline of all programs */
  void SampleRenderer::createPipeline()
  {
    std::vector<OptixProgramGroup> programGroups;
    for (auto pg : raygenPGs)
      programGroups.push_back(pg);
    for (auto pg : missPGs)
      programGroups.push_back(pg);
    for (auto pg : hitgroupPGs)
      programGroups.push_back(pg);
      
    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixPipelineCreate(optixContext,
                                    &pipelineCompileOptions,
                                    &pipelineLinkOptions,
                                    programGroups.data(),
                                    (int)programGroups.size(),
                                    log,&sizeof_log,
                                    &pipeline
                                    ));
    if (sizeof_log > 1) PRINT(log);

    OPTIX_CHECK(optixPipelineSetStackSize
                (/* [in] The pipeline to configure the stack size for */
                 pipeline, 
                 /* [in] The direct stack size requirement for direct
                    callables invoked from IS or AH. */
                 2*1024,
                 /* [in] The direct stack size requirement for direct
                    callables invoked from RG, MS, or CH.  */                 
                 2*1024,
                 /* [in] The continuation stack requirement. */
                 2*1024,
                 /* [in] The maximum depth of a traversable graph
                    passed to trace. */
                 1));
    if (sizeof_log > 1) PRINT(log);
  }


  /*! constructs the shader binding table */
  void SampleRenderer::buildSBT()
  {
    // ------------------------------------------------------------------
    // build raygen records
    // ------------------------------------------------------------------
    std::vector<RaygenRecord> raygenRecords;
    for (int i=0;i<raygenPGs.size();i++) {
      RaygenRecord rec;
      OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i],&rec));
      rec.data = nullptr; /* for now ... */
      raygenRecords.push_back(rec);
    }
    raygenRecordsBuffer.alloc_and_upload(raygenRecords);
    sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

    // ------------------------------------------------------------------
    // build miss records
    // ------------------------------------------------------------------
    std::vector<MissRecord> missRecords;
    for (int i=0;i<missPGs.size();i++) {
      MissRecord rec;
      OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i],&rec));
      rec.data = nullptr; /* for now ... */
      missRecords.push_back(rec);
    }
    missRecordsBuffer.alloc_and_upload(missRecords);
    sbt.missRecordBase          = missRecordsBuffer.d_pointer();
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount         = (int)missRecords.size();

    // ------------------------------------------------------------------
    // build hitgroup records
    // ------------------------------------------------------------------
    int numObjects = (int)model->meshes.size();
    std::vector<HitgroupRecord> hitgroupRecords;
    for (int meshID=0;meshID<numObjects;meshID++) {
      HitgroupRecord rec;
      // all meshes use the same code, so all same hit group
      OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[0],&rec));
      rec.data.color  = model->meshes[meshID]->diffuse;
      rec.data.vertex = (vec3f*)vertexBuffer[meshID].d_pointer();
      rec.data.index  = (vec3i*)indexBuffer[meshID].d_pointer();
      hitgroupRecords.push_back(rec);
    }
    hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
    sbt.hitgroupRecordBase          = hitgroupRecordsBuffer.d_pointer();
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt.hitgroupRecordCount         = (int)hitgroupRecords.size();
  }



  /*! render one frame */
  void SampleRenderer::render()
  {
    // sanity check: make sure we launch only after first resize is
    // already done:
    // if (launchParams.frame.size.x == 0) return;
    launchParamsBuffer.free();

    CUDABuffer launchParamsBuffer;
    launchParamsBuffer.alloc(sizeof(launchParams));


    launchParamsBuffer.upload(&launchParams,1);

      
    int fmsize = launchParams.n_azimuth * launchParams.n_elevation;
    
    OPTIX_CHECK(optixLaunch(
                            pipeline,stream,
                            /*! parameters and SBT */
                            launchParamsBuffer.d_pointer(),
                            launchParamsBuffer.sizeInBytes,
                            &sbt,
                            /*! dimensions of the launch: */
                            launchParams.n_cameras,
                            fmsize,
                            1
                            ));

    // sync - make sure the frame is rendered before we download and
    // display (obviously, for a high-performance application you
    // want to use streams and double-buffering, but for this simple
    // example, this will have to do)
    CUDA_SYNC_CHECK();
  }

  /*! set camera to render with */
  void SampleRenderer::setCameraGroup(const Camera* cameras, const int numCameras, 
  const vec2i &hemisphere_resolution, const float resolution,
                      const float horizon_angle_threshold , const int horizon_gamma)
  {

    vec2i spliting = vec2i(360/hemisphere_resolution.x, 90/hemisphere_resolution.y);
    
    vec3f* h_positions = new vec3f[numCameras];
    vec3f* h_directions = new vec3f[numCameras];
    vec3f* h_tangents = new vec3f[numCameras];
    vec3f* h_bitangents = new vec3f[numCameras];

    // float* h_horizon_masks = new float[numCameras];
    // std::fill(h_horizon_masks, h_horizon_masks + numCameras, 0.0f);
    

    for (int i = 0; i < numCameras; i++) {
      vec3f origin = cameras[i].from;
      vec3f direction = cameras[i].at;



      vec3f tangent = normalize(cross(direction, vec3f(0.f, 1.f, 0.f)));
      if (length(tangent) < 1e-6 || isnan(tangent.x) || isnan(tangent.y) || isnan(tangent.z)) {
          tangent = normalize(cross(direction, vec3f(0.f, 0.f, 1.f)));
      }
      

      vec3f bitangent = normalize(cross(direction, tangent));

      // Camera_info cam = {origin, direction, tangent, bitangent};
      // launchParams.cameras[i] = cam;
      h_positions[i] = origin;
      h_directions[i] = direction;
      h_tangents[i] = tangent;
      h_bitangents[i] = bitangent;
    }

    vec3f* d_positions;
    vec3f* d_directions;
    vec3f* d_tangents;
    vec3f* d_bitangents;

    // float* d_horizon_masks;

    
    cudaMalloc(&d_positions, numCameras * sizeof(vec3f));
    cudaMalloc(&d_directions, numCameras * sizeof(vec3f));
    cudaMalloc(&d_tangents, numCameras * sizeof(vec3f));
    cudaMalloc(&d_bitangents, numCameras * sizeof(vec3f));

    // cudaMalloc(&d_horizon_masks, numCameras * sizeof(float));


    // Copy data from host to device
    cudaMemcpy(d_positions, h_positions, numCameras * sizeof(vec3f), cudaMemcpyHostToDevice);
    cudaMemcpy(d_directions, h_directions, numCameras * sizeof(vec3f), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tangents, h_tangents, numCameras * sizeof(vec3f), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bitangents, h_bitangents, numCameras * sizeof(vec3f), cudaMemcpyHostToDevice);

    // cudaMemcpy(d_horizon_masks, h_horizon_masks, numCameras * sizeof(float), cudaMemcpyHostToDevice);


    //launchParams.batch_offset = batch_offset;

    // Update the device pointers in launchParams
    launchParams.positions = d_positions;
    launchParams.directions = d_directions;
    launchParams.tangents = d_tangents;
    launchParams.bitangents = d_bitangents;

    // launchParams.horizon_masks = d_horizon_masks;

    // launchParams.kdTree = d_kdtree;




    delete[] h_positions;
    delete[] h_directions;
    delete[] h_tangents;
    delete[] h_bitangents;


    launchParams.n_cameras = numCameras;
    launchParams.n_azimuth = spliting.x;
    launchParams.n_elevation = spliting.y;
    launchParams.bbox_min = bbox.lower;
    launchParams.bbox_max = bbox.upper;

    // std::cout<<bbox.lower<<std::endl;

    // std::cout<<"bbox_min: "<<launchParams.bbox_min<<std::endl;
    // std::cout<<"bbox_max: "<<launchParams.bbox_max<<std::endl;

    launchParams.resolution = resolution;
    // launchParams.horizon_threshold = horizon_angle_threshold;
    launchParams.horizon_gamma = horizon_gamma;
    launchParams.hemisphere_resolution = hemisphere_resolution;
    launchParams.voxel_dim = make_int3(ceil((bbox.upper.x - bbox.lower.x) / resolution),
                                             ceil((bbox.upper.y - bbox.lower.y) / resolution),
                                             ceil((bbox.upper.z - bbox.lower.z) / resolution));

    colorBuffer.resize(numCameras * spliting.x * spliting.y * sizeof(uint32_t));
    incident_azimuthBuffer.resize(numCameras * spliting.x * spliting.y * sizeof(half));
    incident_elevationBuffer.resize(numCameras * spliting.x * spliting.y * sizeof(half));

    horizon_factorBuffer.resize(numCameras * sizeof(float));
    horizon_importanceBuffer.resize(numCameras * sizeof(float));
    sky_view_factorBuffer.resize(numCameras * sizeof(int));
    cudaMemset(reinterpret_cast<void*>(horizon_factorBuffer.d_pointer()), 0.0f, numCameras * sizeof(float));
    cudaMemset(reinterpret_cast<void*>(horizon_importanceBuffer.d_pointer()), 0.0f, numCameras * sizeof(float));
    cudaMemset(reinterpret_cast<void*>(sky_view_factorBuffer.d_pointer()), 0.0f, numCameras * sizeof(int));

    launchParams.horizon_factorBuffer = (float*)horizon_factorBuffer.d_pointer();
    launchParams.horizon_importanceBuffer = (float*)horizon_importanceBuffer.d_pointer();
    launchParams.sky_view_factorBuffer = (int*)sky_view_factorBuffer.d_pointer();

    launchParams.colorBuffer = (uint32_t*)colorBuffer.d_pointer();
    launchParams.incident_azimuthBuffer = (half*)incident_azimuthBuffer.d_pointer();
    launchParams.incident_elevationBuffer = (half*)incident_elevationBuffer.d_pointer();
  }
  

  /*! download the rendered color buffer */
  void SampleRenderer::downloadPixels(uint32_t h_pixels[])
  {
    colorBuffer.download(h_pixels,
                         launchParams.n_cameras*launchParams.n_azimuth*launchParams.n_elevation);
  }

  void SampleRenderer::downloadIncidentAngles(half h_incident_azimuths[], half h_incident_elevations[])
  {
    incident_azimuthBuffer.download(h_incident_azimuths,
                                  launchParams.n_cameras*launchParams.n_azimuth*launchParams.n_elevation);
    incident_elevationBuffer.download(h_incident_elevations,
                                  launchParams.n_cameras*launchParams.n_azimuth*launchParams.n_elevation);
  }

  void SampleRenderer::downloadHorizonFactors(float h_horizon_factors[])
  {
    float* horizon_importance = new float[launchParams.n_cameras];
    float* horizon_factors = new float[launchParams.n_cameras];

    horizon_importanceBuffer.download(horizon_importance, launchParams.n_cameras);
    horizon_factorBuffer.download(horizon_factors, launchParams.n_cameras);

    for (int i = 0; i < launchParams.n_cameras; i++) {
      // std::cout<<"test original"<<horizon_importance[i]<< " " <<horizon_factors[i] << std::endl;
      h_horizon_factors[i] = horizon_factors[i] / horizon_importance[i];

    }
    
    delete[] horizon_importance; // 释放 horizon_importance 数组的内存
    delete[] horizon_factors;    // 释放 horizon_factors 数组的内存
    
  }

  void SampleRenderer::downloadSVF(int h_svf[]){
    sky_view_factorBuffer.download(h_svf, launchParams.n_cameras);
  }


  voxel_dim SampleRenderer::print_dimension()
  {
    std::cout<<"bbox_min: "<<launchParams.bbox_min<<std::endl;
    std::cout<<"bbox_max: "<<launchParams.bbox_max<<std::endl;

    std::cout<<"voxel resolution: "<<launchParams.resolution<<std::endl;
    std::cout<<"voxel_dim x: "<<launchParams.voxel_dim.x<<std::endl;
    std::cout<<"voxel_dim y: "<<launchParams.voxel_dim.y<<std::endl;
    std::cout<<"voxel_dim z: "<<launchParams.voxel_dim.z<<std::endl;

    std::cout<<"hemisphere resolution x: "<<launchParams.hemisphere_resolution.x<<std::endl;
    std::cout<<"hemisphere resolution y: "<<launchParams.hemisphere_resolution.y<<std::endl;

    voxel_dim dim = {launchParams.voxel_dim.x, launchParams.voxel_dim.y, launchParams.voxel_dim.z, launchParams.bbox_min, launchParams.bbox_max};
    return dim;
  }
  
} // ::osc
