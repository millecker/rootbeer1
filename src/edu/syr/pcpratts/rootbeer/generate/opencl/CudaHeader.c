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

after HostDeviceInterface

nvcc generated.cu --ptxas-options=-v

ptxas info    : 72 bytes gmem, 36 bytes cmem[14]
ptxas info    : Compiling entry function '_Z5entryPcS_PiPxS1_S0_S0_iS0_' for 'sm_10'
ptxas info    : Used 5 registers, 112 bytes smem, 20 bytes cmem[1]


nvcc generated.cu --ptxas-options=-v -arch sm_20

ptxas info    : 72 bytes gmem, 72 bytes cmem[14]
ptxas info    : Compiling entry function '_Z5entryPcS_PiPxS1_S0_S0_iS0_' for 'sm_20'
ptxas info    : Function properties for _Z5entryPcS_PiPxS1_S0_S0_iS0_
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 12 registers, 24 bytes smem, 104 bytes cmem[0]

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
    START_MESSAGE, SET_BSPJOB_CONF, SET_INPUT_TYPES,
    RUN_SETUP, RUN_BSP, RUN_CLEANUP,
    READ_KEYVALUE, WRITE_KEYVALUE,
    GET_MSG, GET_MSG_COUNT,
    SEND_MSG, SYNC,
    GET_ALL_PEERNAME, GET_PEERNAME,
    GET_PEER_INDEX, GET_PEER_COUNT, GET_SUPERSTEP_COUNT,
    REOPEN_INPUT, CLEAR,
    CLOSE, ABORT,
    DONE, TASK_DONE,
    REGISTER_COUNTER, INCREMENT_COUNTER,
    SEQFILE_OPEN, SEQFILE_READNEXT,
    SEQFILE_APPEND, SEQFILE_CLOSE,
    PARTITION_REQUEST, PARTITION_RESPONSE,
    LOG, END_OF_DATA,
    UNDEFINED
  };
  volatile MESSAGE_TYPE command;
  volatile int param1;

  // Response of HostMonitor
  volatile bool is_result_available;
  volatile int result_int;
  volatile string result_string;

  HostDeviceInterface() {
    init();
  }

  void init() {
    lock_thread_id = -1;
    has_task = false;
    done = false;
    command = UNDEFINED;
    is_result_available = false;
    result_int = 0;
  }

  ~HostDeviceInterface() {}
};

__device__ HostDeviceInterface *host_device_interface;

/*HAMA_PIPES_CODE_END*/
