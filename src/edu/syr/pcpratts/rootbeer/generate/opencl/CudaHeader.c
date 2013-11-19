#ifndef NAN
#include <math_constants.h>
#define NAN CUDART_NAN
#endif

#ifndef INFINITY
#include <math_constants.h>
#define INFINITY CUDART_INF
#endif

#include <stdio.h>

__shared__ size_t m_Local[3];
__shared__ char m_shared[%%shared_mem_size%%];

__device__
int getThreadId(){
  return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__
int getThreadIdxx(){
  return threadIdx.x;
}

__device__
int getBlockIdxx(){
  return blockIdx.x;
}

__device__
int getBlockDimx(){
  return blockDim.x;
}

__device__
int getGridDimx(){
  return blockDim.x;
}

__device__
void edu_syr_pcpratts_syncthreads(){
  __syncthreads();
}

__device__
void edu_syr_pcpratts_threadfence(){
  __threadfence();
}

__device__
void edu_syr_pcpratts_threadfence_block(){
  __threadfence_block();
}

__device__ clock_t global_now;

/*HAMA_PIPES_CODE_START*/

/* before HostDeviceInterface
nvcc generated.cu --ptxas-options=-v

ptxas info    : 8 bytes gmem, 4 bytes cmem[14]
ptxas info    : Compiling entry function '_Z5entryPcS_PiPxS1_S0_S0_i' for 'sm_10'
ptxas info    : Used 5 registers, 104 bytes smem, 20 bytes cmem[1]
*/

/* after HostDeviceInterface
nvcc generated.cu --ptxas-options=-v -arch sm_20
*/

#include <string>

using std::string;

class HostDeviceInterface {
public:
  // Only one thread is able to use the
  // HostDeviceInterface
  volatile int lock_thread_id; 

  // HostMonitor has_task
  volatile bool has_task;

  // HostMonitor is done (end of communication)
  volatile bool done;

  // Request for HostMonitor
  enum MESSAGE_TYPE {
	UNDEFINED, GET_VALUE, DONE
  };
  volatile MESSAGE_TYPE command;
  volatile int param1;

  // Response of HostMonitor
  volatile bool is_result_available;
  volatile int result_int;
  volatile string result_string;

  __device__ __host__ HostDeviceInterface() {
    lock_thread_id = -1;
    has_task = false;
    done = false;

    command = UNDEFINED;

    is_result_available = false;
    result_int = 0;
  }

  __device__ __host__ ~HostDeviceInterface() {}
};

__device__ HostDeviceInterface *d_host_device_interface;

/*HAMA_PIPES_CODE_END*/
