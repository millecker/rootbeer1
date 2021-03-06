#include "CUDARuntime.h"
#include "../../../../at/illecker/HostDeviceInterface.h"
#include "../../../../at/illecker/HostMonitor.h"
#include <cuda.h>

#define CHECK_STATUS(env,msg,status,device) \
if (CUDA_SUCCESS != status) {\
  throw_cuda_errror_exception(env, msg, status, device);\
  return;\
}

/**
* Throws a runtimeexception called CudaMemoryException
* allocd - number of bytes tried to allocate
* id - variable the memory assignment was for
*/
void throw_cuda_errror_exception(JNIEnv *env, const char *message, int error,
  CUdevice device) {

  char msg[1024];
  jclass exp;
  jfieldID fid;
  int a = 0;
  int b = 0;
  char name[1024];

  exp = env->FindClass("org/trifort/rootbeer/runtime/CudaErrorException");

  // we truncate the message to 900 characters to stop any buffer overflow
  switch(error){
    case CUDA_ERROR_OUT_OF_MEMORY:
      sprintf(msg, "CUDA_ERROR_OUT_OF_MEMORY: %.900s",message);
      break;
    case CUDA_ERROR_NO_BINARY_FOR_GPU:
      cuDeviceGetName(name,1024,device);
      cuDeviceComputeCapability(&a, &b, device);
      sprintf(msg, "No binary for gpu. %s Selected %s (%d.%d). 2.0 compatibility required.", message, name, a, b);
      break;
    default:
      sprintf(msg, "ERROR STATUS:%i : %.900s", error, message);
  }

  fid = env->GetFieldID(exp, "cudaError_enum", "I");
  env->SetLongField(exp, fid, (jint)error);
  env->ThrowNew(exp, msg);
  return;
}

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT void JNICALL Java_org_trifort_rootbeer_runtime_CUDAContext_cudaRun
  (JNIEnv *env, jobject this_ref, jint device_index, jbyteArray cubin_file,
   jint cubin_length, jint block_shape_x, jint grid_shape_x, jint num_threads,
   jobject object_mem, jobject handles_mem, jobject exceptions_mem,
   jobject class_mem, jint using_kernel_templates, jint using_exceptions,
   jobject hama_peer)
{
  CUresult status;
  CUdevice device;
  CUcontext context;
  CUmodule module;
  CUfunction function;
  void * fatcubin;
  int offset;
  int info_space_size;

  CUdeviceptr gpu_info_space;
  CUdeviceptr gpu_object_mem;
  CUdeviceptr gpu_handles_mem;
  CUdeviceptr gpu_exceptions_mem;
  CUdeviceptr gpu_class_mem;
  CUdeviceptr gpu_heap_end;
  CUdeviceptr gpu_buffer_size;
  CUdeviceptr gpu_blocksync_barrier_array_in;
  CUdeviceptr gpu_blocksync_barrier_array_out;
  CUdeviceptr gpu_host_device_interface;
  
  void * cpu_object_mem;
  void * cpu_handles_mem;
  void * cpu_exceptions_mem;
  void * cpu_class_mem;
  jlong cpu_object_mem_size;
  jlong cpu_handles_mem_size;
  jlong cpu_exceptions_mem_size;
  jlong cpu_class_mem_size;
  jlong cpu_heap_end;
  HostDeviceInterface *cpu_host_device_interface = NULL;

  jclass cuda_memory_class;
  jmethodID get_address_method;
  jmethodID get_size_method;
  jmethodID get_heap_end_method;

  jlong * info_space;
  
  jclass hama_peer_class;
  jfieldID host_monitor_field;
  HostMonitor *host_monitor = NULL;
  
  //----------------------------------------------------------------------------
  //init device and function
  //----------------------------------------------------------------------------
  status = cuDeviceGet(&device, device_index);
  CHECK_STATUS(env, "Error in cuDeviceGet", status, device)

  status = cuCtxCreate(&context, CU_CTX_MAP_HOST, device);
  CHECK_STATUS(env,"Error in cuCtxCreate", status, device)

  fatcubin = malloc(cubin_length);
  env->GetByteArrayRegion(cubin_file, 0, cubin_length, (jbyte *)fatcubin);

  status = cuModuleLoadFatBinary(&module, fatcubin);
  CHECK_STATUS(env, "Error in cuModuleLoad", status, device)
  free(fatcubin);

  // HamaPeer - Modify function name
  status = cuModuleGetFunction(&function, module, "_Z5entryPcS_PiS0_PxS0_S0_S0_S0_P19HostDeviceInterfaceii");
  CHECK_STATUS(env, "Error in cuModuleGetFunction", status, device)

  //----------------------------------------------------------------------------
  //get handles from java
  //----------------------------------------------------------------------------
  cuda_memory_class = env->FindClass("org/trifort/rootbeer/runtime/FixedMemory");
  get_address_method = env->GetMethodID(cuda_memory_class, "getAddress", "()J");
  get_size_method = env->GetMethodID(cuda_memory_class, "getSize", "()J");
  get_heap_end_method = env->GetMethodID(cuda_memory_class, "getHeapEndPtr", "()J");

  cpu_object_mem = (void *) env->CallLongMethod(object_mem, get_address_method);
  cpu_object_mem_size = env->CallLongMethod(object_mem, get_size_method);
  cpu_heap_end = env->CallLongMethod(object_mem, get_heap_end_method);
  cpu_heap_end >>= 4;

  cpu_handles_mem = (void *) env->CallLongMethod(handles_mem, get_address_method);
  cpu_handles_mem_size = env->CallLongMethod(handles_mem, get_size_method);

  cpu_exceptions_mem = (void *) env->CallLongMethod(exceptions_mem, get_address_method);
  cpu_exceptions_mem_size = env->CallLongMethod(exceptions_mem, get_size_method);

  cpu_class_mem = (void *) env->CallLongMethod(class_mem, get_address_method);
  cpu_class_mem_size = env->CallLongMethod(class_mem, get_size_method);

  info_space_size = 1024;
  info_space = (jlong *) malloc(info_space_size);
  info_space[1] = env->CallLongMethod(object_mem, get_heap_end_method);

  //----------------------------------------------------------------------------
  //allocate mem
  //----------------------------------------------------------------------------
  status = cuMemAlloc(&gpu_info_space, info_space_size);
  CHECK_STATUS(env, "Error in cuMemAlloc: gpu_info_mem", status, device)

  status = cuMemAlloc(&gpu_object_mem, cpu_object_mem_size);
  CHECK_STATUS(env, "Error in cuMemAlloc: gpu_object_mem", status, device)

  status = cuMemAlloc(&gpu_handles_mem, cpu_handles_mem_size);
  CHECK_STATUS(env, "Error in cuMemAlloc: gpu_handles_mem", status, device)

  if(using_exceptions){
    status = cuMemAlloc(&gpu_exceptions_mem, cpu_exceptions_mem_size);
    CHECK_STATUS(env, "Error in cuMemAlloc: gpu_exceptions_mem", status, device)
  }

  status = cuMemAlloc(&gpu_class_mem, cpu_class_mem_size);
  CHECK_STATUS(env, "Error in cuMemAlloc: gpu_class_mem", status, device)

  status = cuMemAlloc(&gpu_heap_end, 8);
  CHECK_STATUS(env, "Error in cuMemAlloc: gpu_heap_end", status, device)

  status = cuMemAlloc(&gpu_buffer_size, 8);
  CHECK_STATUS(env, "Error in cuMemAlloc: gpu_buffer_size", status, device)

  // Allocate gpu_blocksync_barrier_array_in for Inter-Block Lock-Free Synchronization
  status = cuMemAlloc(&gpu_blocksync_barrier_array_in, grid_shape_x * sizeof(jint));
  CHECK_STATUS(env, "Error in cuMemAlloc: gpu_blocksync_barrier_array_in", status, device)

  // Initialize gpu_blocksync_barrier_array_in
  status = cuMemsetD32(gpu_blocksync_barrier_array_in, 0, grid_shape_x);
  CHECK_STATUS(env, "Error in cuMemsetD32: gpu_blocksync_barrier_array_in", status, device)

  // Allocate gpu_blocksync_barrier_array_out for Inter-Block Lock-Free Synchronization
  status = cuMemAlloc(&gpu_blocksync_barrier_array_out, grid_shape_x * sizeof(jint));
  CHECK_STATUS(env, "Error in cuMemAlloc: gpu_blocksync_barrier_array_out", status, device)

  // Initialize gpu_blocksync_barrier_array_out
  status = cuMemsetD32(gpu_blocksync_barrier_array_out, 0, grid_shape_x);
  CHECK_STATUS(env, "Error in cuMemsetD32: gpu_blocksync_barrier_array_out", status, device)

  // HamaPeer - allocate memory
  if (hama_peer != NULL) {
    // Get HostMonitor
    hama_peer_class = env->GetObjectClass(hama_peer);
    host_monitor_field = env->GetFieldID(hama_peer_class, "m_hostMonitor", "J");
    host_monitor = (HostMonitor*) env->GetLongField(hama_peer, host_monitor_field);
    if (host_monitor->isDebugging()) {
      printf("CUDAContext_cudaRun - host_monitor.ptr: %p\n", host_monitor);
    }
    
    // Allocate HostDeviceInterface Pinned Memory
    status = cuMemHostAlloc((void**)&cpu_host_device_interface, sizeof(HostDeviceInterface),
                          CU_MEMHOSTALLOC_WRITECOMBINED | CU_MEMHOSTALLOC_DEVICEMAP);
    CHECK_STATUS(env, "Error in cuMemHostAlloc: cpu_host_device_interface", status, device)
    if (host_monitor->isDebugging()) {
      printf("CUDAContext_cudaRun - allocate cpu_host_device_interface sizeof: %lld bytes\n", 
             (long long) sizeof(HostDeviceInterface));
      printf("CUDAContext_cudaRun - cpu_host_device_interface.ptr: %p\n", cpu_host_device_interface);
    }
    
    // Initialize cpu_hostDeviceInterface
    cpu_host_device_interface->init();
    
    // Set cpu_host_device_interface in HostMonitor
    host_monitor->updateHostDeviceInterface(cpu_host_device_interface);
    
    // Get device pointer of HostDeviceInterface object
    status = cuMemHostGetDevicePointer(&gpu_host_device_interface, cpu_host_device_interface, 0);
    CHECK_STATUS(env, "Error in cuMemHostGetDevicePointer: gpu_host_device_interface", status, device)
    if (host_monitor->isDebugging()) {
      printf("CUDAContext_cudaRun - gpu_host_device_interface: %p\n", (void*)gpu_host_device_interface);
    }
  }
  
  //----------------------------------------------------------------------------
  //set function parameters
  //----------------------------------------------------------------------------
  // HamaPeer - Align argument count
  status = cuParamSetSize(function, (10 * sizeof(CUdeviceptr)) + (2 * sizeof(int)));
  CHECK_STATUS(env, "Error in cuParamSetSize", status, device)

  offset = 0;
  status = cuParamSetv(function, offset, (void *) &gpu_info_space, sizeof(CUdeviceptr));
  CHECK_STATUS(env, "Error in cuParamSetv gpu_info_space", status, device)
  offset += sizeof(CUdeviceptr);

  status = cuParamSetv(function, offset, (void *) &gpu_object_mem, sizeof(CUdeviceptr));
  CHECK_STATUS(env, "Error in cuParamSetv: gpu_object_mem", status, device)
  offset += sizeof(CUdeviceptr);

  status = cuParamSetv(function, offset, (void *) &gpu_handles_mem, sizeof(CUdeviceptr));
  CHECK_STATUS(env, "Error in cuParamSetv: gpu_handles_mem %", status, device)
  offset += sizeof(CUdeviceptr);

  status = cuParamSetv(function, offset, (void *) &gpu_heap_end, sizeof(CUdeviceptr));
  CHECK_STATUS(env, "Error in cuParamSetv: gpu_heap_end", status, device)
  offset += sizeof(CUdeviceptr);

  status = cuParamSetv(function, offset, (void *) &gpu_buffer_size, sizeof(CUdeviceptr));
  CHECK_STATUS(env, "Error in cuParamSetv: gpu_buffer_size", status, device)
  offset += sizeof(CUdeviceptr);

  status = cuParamSetv(function, offset, (void *) &gpu_exceptions_mem, sizeof(CUdeviceptr));
  CHECK_STATUS(env, "Error in cuParamSetv: gpu_exceptions_mem", status, device)
  offset += sizeof(CUdeviceptr);

  status = cuParamSetv(function, offset, (void *) &gpu_class_mem, sizeof(CUdeviceptr));
  CHECK_STATUS(env, "Error in cuParamSetv: gpu_class_mem", status, device)
  offset += sizeof(CUdeviceptr);
  
  // Pass gpu_blocksync_barrier_array_in to kernel function (Inter-Block Lock-Free Synchronization)
  status = cuParamSetv(function, offset, (void *) &gpu_blocksync_barrier_array_in, sizeof(CUdeviceptr));
  CHECK_STATUS(env, "Error in cuParamSetv: gpu_blocksync_barrier_array_in", status, device)
  offset += sizeof(CUdeviceptr);
    
  // Pass gpu_blocksync_barrier_array_out to kernel function (Inter-Block Lock-Free Synchronization)
  status = cuParamSetv(function, offset, (void *) &gpu_blocksync_barrier_array_out, sizeof(CUdeviceptr));
  CHECK_STATUS(env, "Error in cuParamSetv: gpu_blocksync_barrier_array_out", status, device)
  offset += sizeof(CUdeviceptr);

  // Pass PinnedMemory gpu_host_device_interface to kernel function
  if (hama_peer != NULL) {
    status = cuParamSetv(function, offset, (void *) &gpu_host_device_interface, sizeof(CUdeviceptr));
    CHECK_STATUS(env, "Error in cuParamSetv: gpu_host_device_interface", status, device)
  }
  // Submitting a NULL parameter is not supported! -> CUDA_ERROR_INVALID_VALUE
  // gpu_host_device_interface should be NULL if parameter is not submitted
  //
  // else {
  //  status = cuParamSetv(function, offset, (void *) NULL, sizeof(CUdeviceptr));
  //}
  offset += sizeof(CUdeviceptr); // also increase parameter offset if hama_peer == NULL

  status = cuParamSeti(function, offset, num_threads);
  CHECK_STATUS(env, "Error in cuParamSeti: num_threads", status, device)
  offset += sizeof(int);

  status = cuParamSeti(function, offset, using_kernel_templates);
  CHECK_STATUS(env, "Error in cuParamSeti: using_kernel_templates", status, device)
  offset += sizeof(int);

  //----------------------------------------------------------------------------
  //copy data
  //----------------------------------------------------------------------------
  status = cuMemcpyHtoD(gpu_info_space, info_space, info_space_size);
  CHECK_STATUS(env, "Error in cuMemcpyHtoD: info_space", status, device)

  status = cuMemcpyHtoD(gpu_object_mem, cpu_object_mem, cpu_object_mem_size);
  CHECK_STATUS(env, "Error in cuMemcpyHtoD: gpu_object_mem", status, device)

  status = cuMemcpyHtoD(gpu_handles_mem, cpu_handles_mem, cpu_handles_mem_size);
  CHECK_STATUS(env, "Error in cuMemcpyHtoD: gpu_handles_mem", status, device)

  status = cuMemcpyHtoD(gpu_class_mem, cpu_class_mem, cpu_class_mem_size);
  CHECK_STATUS(env, "Error in cuMemcpyHtoD: gpu_class_mem", status, device)

  status = cuMemcpyHtoD(gpu_heap_end, &cpu_heap_end, sizeof(jlong));
  CHECK_STATUS(env, "Error in cuMemcpyHtoD: gpu_heap_end", status, device)

  status = cuMemcpyHtoD(gpu_buffer_size, &cpu_object_mem_size, sizeof(jlong));
  CHECK_STATUS(env, "Error in cuMemcpyHtoD: gpu_buffer_size", status, device)

  if(using_exceptions){
    status = cuMemcpyHtoD(gpu_exceptions_mem, cpu_exceptions_mem, cpu_exceptions_mem_size);
    CHECK_STATUS(env, "Error in cuMemcpyDtoH: gpu_exceptions_mem", status, device)
  }

  //----------------------------------------------------------------------------
  // HamaPeer - start HostMonitor
  //----------------------------------------------------------------------------
  if (host_monitor != NULL) {
    if (host_monitor->isDebugging()) {
      printf("CUDAContext_cudaRun - startMonitoring...\n");
    }
    
    host_monitor->startMonitoring();
    
    if (host_monitor->isDebugging()) {
      printf("CUDAContext_cudaRun - startMonitoring finished!\n");
    }
  }
  
  //----------------------------------------------------------------------------
  //launch
  //----------------------------------------------------------------------------
  status = cuFuncSetBlockShape(function, block_shape_x, 1, 1);
  CHECK_STATUS(env, "Error in cuFuncSetBlockShape", status, device);

  status = cuLaunchGrid(function, grid_shape_x, 1);
  CHECK_STATUS(env, "Error in cuLaunchGrid", status, device)

  status = cuCtxSynchronize();
  CHECK_STATUS(env, "Error in cuCtxSynchronize", status, device)

  //----------------------------------------------------------------------------
  // HamaPeer - stop HostMonitor
  //----------------------------------------------------------------------------
  if (host_monitor != NULL) {
    if (host_monitor->isDebugging()) {
      printf("CUDAContext_cudaRun - stopMonitoring...\n");
    }
    
    host_monitor->stopMonitoring();
    
    if (host_monitor->isDebugging()) {
      printf("CUDAContext_cudaRun - stopMonitoring finished!\n");
    }
  }
  
  //----------------------------------------------------------------------------
  //copy data back
  //----------------------------------------------------------------------------
  status = cuMemcpyDtoH(info_space, gpu_info_space, info_space_size);
  CHECK_STATUS(env, "Error in cuMemcpyDtoH: gpu_info_space", status, device)

  cpu_heap_end = info_space[1];
  cpu_heap_end <<= 4;

  status = cuMemcpyDtoH(cpu_object_mem, gpu_object_mem, cpu_heap_end);
  CHECK_STATUS(env, "Error in cuMemcpyDtoH: gpu_object_mem", status, device)

  if(using_exceptions){
    status = cuMemcpyDtoH(cpu_exceptions_mem, gpu_exceptions_mem, cpu_exceptions_mem_size);
    CHECK_STATUS(env, "Error in cuMemcpyDtoH: gpu_exceptions_mem", status, device)
  }

  //----------------------------------------------------------------------------
  //free resources
  //----------------------------------------------------------------------------
  free(host_monitor);
  free(info_space);

  cuMemFree(gpu_info_space);
  cuMemFree(gpu_object_mem);
  cuMemFree(gpu_handles_mem);
  cuMemFree(gpu_exceptions_mem);
  cuMemFree(gpu_class_mem);
  cuMemFree(gpu_heap_end);
  cuMemFree(gpu_buffer_size);
  cuMemFree(gpu_blocksync_barrier_array_in);
  cuMemFree(gpu_blocksync_barrier_array_out);
  cuMemFree(gpu_host_device_interface);
  
  cuCtxDestroy(context);
}

#ifdef __cplusplus
}
#endif

