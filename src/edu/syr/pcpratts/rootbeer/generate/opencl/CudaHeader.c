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
  return gridDim.x;
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

__device__
void at_illecker_threadfence_system(){
  __threadfence_system();
}

// Inter-Block Lock-Based Synchronization
__device__ int global_mutex = 0;
__device__
void at_illecker_syncblocks(int goal_value){
  // threadId in a block
  // threadIdx.x * blockDim.y + threadIdx.y
  int tid_in_block = threadIdx.x;
  
  // only thread 0 is used for synchronization
  if (tid_in_block == 0) {
    atomicAdd(&global_mutex, 1);

    // only when all blocks add 1 to global_mutex
    // global_mutex will equal to goal_value
    int count = 0;
    while (count < 100) {
      __threadfence();
      if (global_mutex == goal_value) {
        break;
      }
      count++;
      if (count > 50) {
        count = 0;
      }
    }
  }
  __syncthreads();
}

/*HAMA_PIPES_HEADER_CODE_IGNORE_IN_TWEAKS_START*/

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

#define STR_SIZE 1024

using std::string;

class HostDeviceInterface {
public:
  volatile bool is_debugging; 

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

  // Command parameter
  volatile bool use_int_val1; // in int_val1
  volatile bool use_int_val2; // in int_val2
  volatile bool use_int_val3; // in int_val3
  volatile bool use_long_val1; // in long_val1
  volatile bool use_long_val2; // in long_val2
  volatile bool use_float_val1; // in float_val1
  volatile bool use_float_val2; // in float_val2
  volatile bool use_double_val1; // in double_val1
  volatile bool use_double_val2; // in double_val2
  volatile bool use_str_val1; // in str_val1
  volatile bool use_str_val2; // in str_val2
  volatile bool use_str_val3; // in str_val3

  // Transfer variables (used in sendCommand and getResult)
  volatile int int_val1;
  volatile int int_val2;
  volatile int int_val3;
  volatile long long_val1;
  volatile long long_val2;
  volatile float float_val1;
  volatile float float_val2;
  volatile double double_val1;
  volatile double double_val2;
  volatile char str_val1[STR_SIZE];
  volatile char str_val2[STR_SIZE];
  volatile char str_val3[255];

  enum TYPE {
    INT, LONG, FLOAT, DOUBLE, STRING, STRING_ARRAY,
    KEY_VALUE_PAIR, NULL_TYPE, NOT_AVAILABLE
  };
  volatile TYPE return_type;
  volatile TYPE key_type;
  volatile TYPE value_type;

  volatile bool end_of_data;

  // Response of HostMonitor
  volatile bool is_result_available;

  HostDeviceInterface() {
    init();
  }

  void init() {
    is_debugging = false;
    lock_thread_id = -1;
    has_task = false;
    done = false;
    command = UNDEFINED;
    use_int_val1 = false;
    use_int_val2 = false;
    use_int_val3 = false;
    use_long_val1 = false;
    use_long_val2 = false;
    use_float_val1 = false;
    use_float_val2 = false;
    use_double_val1 = false;
    use_double_val2 = false;
    use_str_val1 = false;
    use_str_val2 = false;
    use_str_val3 = false;
    int_val1 = 0;
    int_val2 = 0;
    int_val3 = 0;
    long_val1 = 0;
    long_val2 = 0;
    float_val1 = 0;
    float_val2 = 0;
    double_val1 = 0;
    double_val2 = 0;
    key_type = NOT_AVAILABLE;
    value_type = NOT_AVAILABLE;
    end_of_data = true;
    is_result_available = false;
  }

  ~HostDeviceInterface() {}
};

__device__ HostDeviceInterface *host_device_interface;

/*HAMA_PIPES_HEADER_CODE_IGNORE_IN_TWEAKS_END*/
