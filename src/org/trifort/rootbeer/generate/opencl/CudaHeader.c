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
void org_trifort_syncthreads(){
  __syncthreads();
}

__device__
void org_trifort_threadfence(){
  __threadfence();
}

__device__
void org_trifort_threadfence_block(){
  __threadfence_block();
}

__device__
void org_trifort_threadfence_system(){
  __threadfence_system();
}

__device__ clock_t global_now;

__device__ __forceinline__
unsigned int get_smid(void) {
  unsigned int r;
  asm("mov.u32 %0, %%smid;" : "=r"(r));
  return r;
}

__device__ __forceinline__
unsigned int get_nsmid() {
  unsigned int r;
  asm("mov.u32 %0, %%nsmid;" : "=r"(r));
  return r;
}

__device__ __forceinline__
unsigned int get_laneid() {
  unsigned int r;
  asm("mov.u32 %0, %%laneid ;" : "=r"(r));
  return r;
}

__device__ __forceinline__
unsigned int get_warpid() {
  unsigned int r;
  asm("mov.u32 %0, %%warpid ;" : "=r"(r))
  return r;
}

// Inter-Block Simple Synchronization based on
// http://eprints.cs.vt.edu/archive/00001087/01/TR_GPU_synchronization.pdf
/*
__device__ int syncblocks_global_mutex = 0;
__device__
void at_illecker_syncblocks(int goal_value) {
  int thread_idxx = threadIdx.x; // * blockDim.y + threadIdx.y
  int grid_size = blockDim.x;
  int count = 0;

  // goal_value is a multiple of grid_size
  goal_value = goal_value * grid_size;

  // Each block increments the global mutex
  if (thread_idxx == 0) {
    atomicAdd(&syncblocks_global_mutex, 1);

    // Busy wait until all blocks have incremented the mutex to goal_value
    while (count < 100) {
      if (syncblocks_global_mutex == goal_value) {
        break;
      }
      __threadfence(); // required
      count++;
      if (count > 50) {
        count = 0;
      }
    }
  }

  __syncthreads();
}
*/

// Inter-Block Synchronization based on
// http://eprints.cs.vt.edu/archive/00001087/01/TR_GPU_synchronization.pdf
__device__ int *syncblocks_barrier_array_in;
__device__ int *syncblocks_barrier_array_out;
__device__
void at_illecker_syncblocks(int goal_value) {
  int thread_idxx = threadIdx.x; // * blockDim.y + threadIdx.y
  int block_idxx = blockIdx.x; // * gridDim.y + blockIdx.y
  int block_size = gridDim.x; // * gridDim.y
  int grid_size = blockDim.x;
  int count = 0;

  // TODO
  // Known Issues:
  //  - grid_size <= amount of multiprocessors (nsmid) (non preemptive scheduling)
  //  - grid_size might not be valid if using Rootbeer.run(List<Kernel>) 
  //    because blockIdx.x != num_blocks
  
  // Inter-Block Sync only when grid_size > 1
  if (grid_size > 1) {
    
    // Inter-Block Lock-Free Sync algorithm when block_size >= grid_size
    if (block_size >= grid_size) {

      // Each block sets goal_value in array_in
      if (thread_idxx == 0) {
        syncblocks_barrier_array_in[block_idxx] = goal_value;
        // printf("%d\t(%d,%d)\n", block_idxx, get_smid(), get_nsmid());
        //__threadfence(); // maybe needless
      }

      // Only block 0 is used for synchronization
      if (block_idxx == 0) {
      
        if (thread_idxx < grid_size) {
          // busy wait
          count = 0;
          while (count < 100) {
            if (syncblocks_barrier_array_in[thread_idxx] == goal_value) {
              break;
            }
            __threadfence(); // required
            count++;
            if (count > 50) {
              count = 0;
            }
          }
        }
    
        __syncthreads();

        if (thread_idxx < grid_size) {
          syncblocks_barrier_array_out[thread_idxx] = goal_value;
          // printf("%d\n", thread_idxx);
          //__threadfence(); // maybe needless
        }
      }

      // Each block waits for goal_value in array_out
      if (thread_idxx == 0) {
        // busy wait
        count = 0;
        while (count < 100) {
          if (syncblocks_barrier_array_out[block_idxx] == goal_value) {
            break;
          }
          __threadfence(); // required

          count++;
          if (count > 50) {
            count = 0;
          }
        }
      }

    }
  
  }

  __syncthreads();
}

/*HAMA_PIPES_HEADER_CODE_IGNORE_IN_TWEAKS_START*/

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
  volatile long long long_val1;
  volatile long long long_val2;
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
    end_of_data = false;
    is_result_available = false;
  }

  ~HostDeviceInterface() {}
};

__device__ HostDeviceInterface *host_device_interface;

/*HAMA_PIPES_HEADER_CODE_IGNORE_IN_TWEAKS_END*/
