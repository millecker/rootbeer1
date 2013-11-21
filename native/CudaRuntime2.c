#include "edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2.h"

#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_STATUS(env,msg,status) \
if (CUDA_SUCCESS != status) {\
  throw_cuda_errror_exception(env, msg, status);\
  return;\
}

#define CHECK_STATUS_RTN(env,msg,status,rtn) \
if (CUDA_SUCCESS != status) {\
  throw_cuda_errror_exception(env, msg, status);\
  return rtn;\
}

static CUdevice cuDevice;
static CUmodule cuModule;
static CUfunction cuFunction;
static CUcontext cuContext;

static void * toSpace;
static void * textureMemory;
static void * handlesMemory;
static void * exceptionsMemory;

static CUdeviceptr gcInfoSpace;
static CUdeviceptr gpuToSpace;
static CUdeviceptr gpuTexture;
static CUdeviceptr gpuHandlesMemory;
static CUdeviceptr gpuExceptionsMemory;
static CUdeviceptr gpuClassMemory;
static CUdeviceptr gpuHeapEndPtr;
static CUdeviceptr gpuBufferSize;
static CUtexref    cache;

static jclass thisRefClass;

static jlong heapEndPtr;
static jlong bufferSize;
static jlong classMemSize;
static jlong numBlocks;
static int maxGridDim;
static int numMultiProcessors;

static int textureMemSize;
static size_t gc_space_size;

/*****************************************************************************/
/*************************** HAMA_PIPES_CODE_START ***************************/
/*****************************************************************************/

#include <cuda_runtime.h>
#include <errno.h>
#include <netinet/in.h>
#include <pthread.h>
#include <rpc/types.h>
#include <rpc/xdr.h>
#include <signal.h>
#include <string>
#include <sys/socket.h>

#define stringify( name ) # name

using std::string;

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline cudaError_t checkCuda(cudaError_t result, string message) {
  if (result != cudaSuccess) {
    fprintf(stderr, "%s - CUDA Runtime Error: %s\n", message.c_str(), 
            cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

/*****************************************************************************/
// HostDeviceInterface
/*****************************************************************************/
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
	UNDEFINED, GET_NUM_MESSAGES, DONE
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

/* Only needed for debugging output */
const char* messageTypeNames[] = { stringify(UNDEFINED), stringify(GET_NUM_MESSAGES),
  stringify(DONE) };

/*****************************************************************************/
// Hadoop Utils
/*****************************************************************************/
/**
 * Check to make sure that the condition is true, and throw an exception
 * if it is not. The exception will contain the message and a description
 * of the source location.
 */

#define HADOOP_ASSERT(CONDITION, MESSAGE) \
{ \
if (!(CONDITION)) { \
throw Error((MESSAGE), __FILE__, __LINE__, \
__PRETTY_FUNCTION__); \
} \
}

string toString(int32_t x) {
  char str[100];
  sprintf(str, "%d", x);
  return str;
}

class Error {
private:
  string error;
public:
  Error(const string& msg): error(msg) {
  }
  
  Error(const string& msg,
        const string& file, int line,
        const string& function) {
    error = msg + " at " + file + ":" + toString(line) +
    " in " + function;
  }
  
  const string& getMessage() const {
    return error;
  }
};

int toInt(const string& val) {
  int result;
  char trash;
  int num = sscanf(val.c_str(), "%d%c", &result, &trash);
  HADOOP_ASSERT(num == 1,
                "Problem converting " + val + " to integer.");
  return result;
}

float toFloat(const string& val) {
  float result;
  char trash;
  int num = sscanf(val.c_str(), "%f%c", &result, &trash);
  HADOOP_ASSERT(num == 1,
                "Problem converting " + val + " to float.");
  return result;
}

double toDouble(const string& val) {
  const char* begin = val.c_str();
  char* end;
  double result = strtod(begin, &end);
  size_t s = end - begin;
  if(s < val.size()) {
    throw Error("Problem converting "+val+" to double. (result:"
                +toString(result)+")");
  }
  return result;
}

bool toBool(const string& val) {
  if (val == "true") {
    return true;
  } else if (val == "false") {
    return false;
  } else {
    HADOOP_ASSERT(false,
                  "Problem converting " + val + " to boolean.");
  }
}

/*****************************************************************************/
// FileInStream
/*****************************************************************************/
class FileInStream {
private:
  /**
   * The file to write to.
   */
  FILE *mFile;
  /**
   * Does is this class responsible for closing the FILE*?
   */
  bool isOwned;
  
public:
  FileInStream() {
    mFile = NULL;
    isOwned = false;
  }
  ~FileInStream() {
    if (mFile != NULL) {
      close();
    }
  }
  
  bool open(const std::string& name) {
    mFile = fopen(name.c_str(), "rb");
    isOwned = true;
    return (mFile != NULL);
  }
  
  bool open(FILE* file) {
    mFile = file;
    isOwned = false;
    return (mFile != NULL);
  }
  
  void read(void *buf, size_t len) {
    size_t result = fread(buf, len, 1, mFile);
    if (result == 0) {
      if (feof(mFile)) {
        HADOOP_ASSERT(false, "end of file");
      } else {
        HADOOP_ASSERT(false, string("read error on file: ") + strerror(errno));
      }
    }
  }
  
  bool skip(size_t nbytes)
  {
    return (0==fseek(mFile, nbytes, SEEK_CUR));
  }
  
  bool close()
  {
    int ret = 0;
    if (mFile != NULL && isOwned) {
      ret = fclose(mFile);
    }
    mFile = NULL;
    return (ret==0);
  }
  
};

/*****************************************************************************/
// FileOutStream
/*****************************************************************************/
class FileOutStream {
private:
  FILE *mFile;
  bool isOwned;
  
public:
  
  FileOutStream() {
    mFile = NULL;
    isOwned = false;
  }
  
  ~FileOutStream() {
    if (mFile != NULL) {
      close();
    }
  }
  
  bool open(const std::string& name, bool overwrite)
  {
    if (!overwrite) {
      mFile = fopen(name.c_str(), "rb");
      if (mFile != NULL) {
        fclose(mFile);
        return false;
      }
    }
    mFile = fopen(name.c_str(), "wb");
    isOwned = true;
    return (mFile != NULL);
  }
  
  bool open(FILE* file)
  {
    mFile = file;
    isOwned = false;
    return (mFile != NULL);
  }
  
  void write(const void* buf, size_t len)
  {
    size_t result = fwrite(buf, len, 1, mFile);
    HADOOP_ASSERT(result == 1,
                  string("write error to file: ") + strerror(errno));
  }
  
  bool advance(size_t nbytes)
  {
    return (0==fseek(mFile, nbytes, SEEK_CUR));
  }
  
  bool close()
  {
    int ret = 0;
    if (mFile != NULL && isOwned) {
      ret = fclose(mFile);
    }
    mFile = NULL;
    return (ret == 0);
  }
  
  void flush()
  {
    fflush(mFile);
  }
  
};

/*****************************************************************************/
// Serialization and Deserialization
/*****************************************************************************/
void serializeLong(int64_t t, FileOutStream& stream)
{
  if (t >= -112 && t <= 127) {
    int8_t b = t;
    stream.write(&b, 1);
    return;
  }
  
  int8_t len = -112;
  if (t < 0) {
    t ^= -1ll; // reset the sign bit
    len = -120;
  }
  
  uint64_t tmp = t;
  while (tmp != 0) {
    tmp = tmp >> 8;
    len--;
  }
  
  stream.write(&len, 1);
  len = (len < -120) ? -(len + 120) : -(len + 112);
  
  for (uint32_t idx = len; idx != 0; idx--) {
    uint32_t shiftbits = (idx - 1) * 8;
    uint64_t mask = 0xFFll << shiftbits;
    uint8_t b = (t & mask) >> shiftbits;
    stream.write(&b, 1);
  }
}

int64_t deserializeLong(FileInStream& stream)
{
  int8_t b;
  stream.read(&b, 1);
  if (b >= -112) {
    return b;
  }
  bool negative;
  int len;
  if (b < -120) {
    negative = true;
    len = -120 - b;
  } else {
    negative = false;
    len = -112 - b;
  }
  uint8_t barr[len];
  stream.read(barr, len);
  int64_t t = 0;
  for (int idx = 0; idx < len; idx++) {
    t = t << 8;
    t |= (barr[idx] & 0xFF);
  }
  if (negative) {
    t ^= -1ll;
  }
  return t;
}

void serializeInt(int32_t t, FileOutStream& stream) {
  serializeLong(t, stream);
}

int32_t deserializeInt(FileInStream& stream) {
  return deserializeLong(stream);
}


void serializeFloat(float t, FileOutStream& stream)
{
  char buf[sizeof(float)];
  XDR xdrs;
  xdrmem_create(&xdrs, buf, sizeof(float), XDR_ENCODE);
  xdr_float(&xdrs, &t);
  stream.write(buf, sizeof(float));
}

float deserializeFloat(FileInStream& stream)
{
  float t;
  char buf[sizeof(float)];
  stream.read(buf, sizeof(float));
  XDR xdrs;
  xdrmem_create(&xdrs, buf, sizeof(float), XDR_DECODE);
  xdr_float(&xdrs, &t);
  return t;
}

void serializeDouble(double t, FileOutStream& stream)
{
  char buf[sizeof(double)];
  XDR xdrs;
  xdrmem_create(&xdrs, buf, sizeof(double), XDR_ENCODE);
  xdr_double(&xdrs, &t);
  stream.write(buf, sizeof(double));
}

double deserializeDouble(FileInStream& stream)
{
  double t;
  char buf[sizeof(double)];
  stream.read(buf, sizeof(double));
  XDR xdrs;
  xdrmem_create(&xdrs, buf, sizeof(double), XDR_DECODE);
  xdr_double(&xdrs, &t);
  return t;
}

void serializeString(const string& t, FileOutStream& stream)
{
  serializeInt(t.length(), stream);
  if (t.length() > 0) {
    stream.write(t.data(), t.length());
  }
}

string deserializeString(FileInStream& stream)
{
  string t;
  int32_t len = deserializeInt(stream);
  if (len > 0) {
    // resize the string to the right length
    t.resize(len);
    // read into the string in 64k chunks
    const int bufSize = 65536;
    int offset = 0;
    char buf[bufSize];
    while (len > 0) {
      int chunkLength = len > bufSize ? bufSize : len;
      stream.read(buf, chunkLength);
      t.replace(offset, chunkLength, buf, chunkLength);
      offset += chunkLength;
      len -= chunkLength;
    }
  } else {
    t.clear();
  }
  return t;
}

/*****************************************************************************/
// SocketClient Implementation
/*****************************************************************************/
class SocketClient {
private:
  int sock;
  FILE* in_stream;
  FILE* out_stream;
  FileInStream* inStream;
  FileOutStream* outStream;

public:
  volatile int32_t resultInt;
  volatile bool isNewResultInt;
  volatile int64_t resultLong;
  volatile bool isNewResultLong;
  volatile string resultString;
  volatile bool isNewResultString;
  //vector<string> resultVector;
  //bool isNewResultVector;
  //bool isNewKeyValuePair;
  //string currentKey;
  //string currentValue;
  
  SocketClient() {
    sock = -1;
    in_stream = NULL;
    out_stream = NULL;
    isNewResultInt = false;
    isNewResultString = false;
    //isNewResultVector = false;
    //isNewKeyValuePair = false;
  }
  
  ~SocketClient() {
    if (in_stream != NULL) {
      fflush(in_stream);
    }
    if (out_stream != NULL) {
      fflush(out_stream);
    }
    fflush(stdout);
    if (sock != -1) {
      int result = shutdown(sock, SHUT_RDWR);
      //if (result != 0) {
      //	fprintf(stderr, "SocketClient: problem shutting down socket\n");
      //}
      result = shutdown(sock, 2);
      if (result != 0) {
        fprintf(stderr, "SocketClient: problem closing socket\n");
      }
    }
  }
  
  void connectSocket(int port) {
    printf("SocketClient started\n");
    
    if (port <= 0) {
      printf("SocketClient: invalid port number!\n");
      return; /* Failed */
    }
    
    sock = socket(PF_INET, SOCK_STREAM, 0);
    if (sock == -1) {
      fprintf(stderr, "SocketClient: problem creating socket: %s\n",
              strerror(errno));
    }
    
    sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    
    int res = connect(sock, (sockaddr*) &addr, sizeof(addr));
    if (res != 0) {
      fprintf(stderr, "SocketClient: problem connecting command socket: %s\n",
              strerror(errno));
    }
    
    in_stream = fdopen(sock, "r");
    out_stream = fdopen(sock, "w");
    
    inStream = new FileInStream();
    inStream->open(in_stream);
    outStream = new FileOutStream();
    outStream->open(out_stream);
    
    printf("SocketClient is connected to port %d ...\n", port);
  }
  
  void sendCMD(int32_t cmd) volatile {
    serializeInt(cmd, *outStream);
    outStream->flush();
    printf("SocketClient sent CMD %s\n", messageTypeNames[cmd]);
  }
  
  void sendCMD(int32_t cmd, int32_t value) volatile {
    serializeInt(cmd, *outStream);
    serializeInt(value, *outStream);
    outStream->flush();
    printf("SocketClient sent CMD: %s with Value: %d\n", messageTypeNames[cmd],
           value);
  }
  
  void sendCMD(int32_t cmd, const string& value) volatile {
    serializeInt(cmd, *outStream);
    serializeString(value, *outStream);
    outStream->flush();
    printf("SocketClient sent CMD: %s with Value: %s\n", messageTypeNames[cmd],
           value.c_str());
  }
  
  void sendCMD(int32_t cmd, const string values[], int size) volatile {
    serializeInt(cmd, *outStream);
    for (int i = 0; i < size; i++) {
      serializeString(values[i], *outStream);
      printf("SocketClient sent CMD: %s with Param%d: %s\n",
             messageTypeNames[cmd], i + 1, values[i].c_str());
    }
    outStream->flush();
  }
  
  void sendCMD(int32_t cmd, int32_t value, const string values[],
               int size) volatile {
    serializeInt(cmd, *outStream);
    serializeInt(value, *outStream);
    for (int i = 0; i < size; i++) {
      serializeString(values[i], *outStream);
      printf("SocketClient sent CMD: %s with Param%d: %s\n",
             messageTypeNames[cmd], i + 1, values[i].c_str());
    }
    outStream->flush();
  }
  
  void nextEvent() volatile {
    int32_t cmd = deserializeInt(*inStream);
    
    switch (cmd) {
        
      case HostDeviceInterface::GET_NUM_MESSAGES: {
        resultInt = deserializeInt(*inStream);
        printf("SocketClient - GET_NUM_MESSAGES IN=%d\n", resultInt);
        isNewResultInt = true;
        break;
      }
        
      default:
        fprintf(stderr, "SocketClient - Unknown binary command: %d\n", cmd);
        break;
        
    }
  }

};

/*****************************************************************************/
// HostMonitor
/*****************************************************************************/
class HostMonitor {
private:
  pthread_t monitor_thread;
  pthread_mutex_t mutex_process_command;
  SocketClient *socket_client;

public:
  volatile bool is_monitoring;
  volatile HostDeviceInterface *host_device_interface;

  HostMonitor(int port) {
    is_monitoring = false;
    host_device_interface = NULL;
    pthread_mutex_init(&mutex_process_command, NULL);
    socket_client = new SocketClient();

    // connect SocketClient
    socket_client->connectSocket(port);

    printf("HostMonitor init finished...\n");
  }

  ~HostMonitor() {
    pthread_mutex_destroy(&mutex_process_command);
  }

  void setHostDeviceInterface(HostDeviceInterface *h_d_interface) {
    printf("HostMonitor setHostDeviceInterface...\n");
    host_device_interface = h_d_interface;
    reset();
  }

  void reset() volatile {
    host_device_interface->command = HostDeviceInterface::UNDEFINED;
    host_device_interface->has_task = false;
    printf("HostMonitor reset lock_thread_id: %d, has_task: %s, result_available: %s\n",
           host_device_interface->lock_thread_id, 
           (host_device_interface->has_task) ? "true" : "false",
           (host_device_interface->is_result_available) ? "true" : "false");
  }

  void startMonitoring() {
    if ( (host_device_interface != NULL) && (!is_monitoring) ) {
      printf("HostMonitor startMonitoring thread\n");
      pthread_create(&monitor_thread, NULL, &HostMonitor::thread, this);

      // wait for monitoring
      //while (!is_monitoring) {
      //  printf("HostMonitor.startMonitoring is_monitoring: %s\n",
      //    (is_monitoring) ? "true" : "false");
      //}
      printf("HostMonitor.startMonitoring started thread! is_monitoring: %s\n",
            (is_monitoring) ? "true" : "false");
    }
  }

  void stopMonitoring() {
    if ( (host_device_interface != NULL) && (is_monitoring) ) {
      printf("HostMonitor stopMonitoring thread\n");

      host_device_interface->done = true;

      // wait for monitoring to end
      while (is_monitoring) {
        printf("HostMonitor.stopMonitoring is_monitoring: %s\n",
          (is_monitoring) ? "true" : "false");
      }

      printf("HostMonitor.stopMonitoring stopped! done: %s\n",
            (host_device_interface->done) ? "true" : "false");
    }
  }

  static void *thread(void *context) {
    volatile HostMonitor *_this = ((HostMonitor *) context);
    printf("HostMonitor thread started... done: %s\n",
            (_this->host_device_interface->done) ? "true" : "false");

    while (!_this->host_device_interface->done) {
      _this->is_monitoring = true;

      //printf("HostMonitor thread is_monitoring: %s\n",
      //      (_this->is_monitoring) ? "true" : "false");

      //printf("HostMonitor thread running... has_task: %s lock_thread_id: %d command: %d\n",
      //      (_this->host_device_interface->has_task) ? "true" : "false",
      //       _this->host_device_interface->lock_thread_id,
      //       _this->host_device_interface->command);

      if ((_this->host_device_interface->has_task) && 
          (_this->host_device_interface->lock_thread_id >= 0) && 
          (_this->host_device_interface->command != HostDeviceInterface::UNDEFINED)) {

        pthread_mutex_t *lock = (pthread_mutex_t *) &_this->mutex_process_command;
	pthread_mutex_lock(lock);
	
        printf("HostMonitor thread: %p, LOCKED(mutex_process_command)\n", pthread_self());

	_this->processCommand();
        
        _this->reset();

	pthread_mutex_unlock(lock);
	printf("HostMonitor thread: %p, UNLOCKED(mutex_process_command)\n", pthread_self());
      }
    }
    _this->is_monitoring = false;
    return NULL;
  }

  void processCommand() volatile {

    printf("HostMonitor processCommand: %d, lock_thread_id: %d, result_available: %s\n",
           host_device_interface->command, 
           host_device_interface->lock_thread_id, 
           (host_device_interface->is_result_available) ? "true" : "false");

    switch (host_device_interface->command) {
      
      case HostDeviceInterface::GET_NUM_MESSAGES: {
        socket_client->sendCMD(HostDeviceInterface::GET_NUM_MESSAGES);
        
        while (!socket_client->isNewResultInt) {
          socket_client->nextEvent();
        }
        
        socket_client->isNewResultInt = false;
        host_device_interface->result_int = socket_client->resultInt;

        host_device_interface->is_result_available = true;

        printf("HostMonitor got result: %d result_available: %s\n",
               host_device_interface->result_int, 
               (host_device_interface->is_result_available) ? "true" : "false");

	// block until result was consumed
	while (host_device_interface->is_result_available) {
          printf("HostMonitor wait for consuming result! result_int: %d, result_available: %s\n",
                 host_device_interface->result_int, 
                 (host_device_interface->is_result_available) ? "true" : "false");
	}
	
	printf("HostMonitor consumed result: %d\n", host_device_interface->result_int);

        break;
      }

      case HostDeviceInterface::DONE: {
        socket_client->sendCMD(HostDeviceInterface::DONE);
        host_device_interface->is_result_available = true;
        // block until result was consumed
        while (host_device_interface->is_result_available) {}

        break;
      }
    }
  }

};

// Global HostDeviceInterface
HostDeviceInterface *h_host_device_interface = NULL;
HostDeviceInterface *d_host_device_interface = NULL;
// Global HostMonitor
HostMonitor *host_monitor = NULL;

/*****************************************************************************/
/**************************** HAMA_PIPES_CODE_END ****************************/
/*****************************************************************************/

/**
* Throws a runtimeexception called CudaMemoryException
* allocd - number of bytes tried to allocate
* id - variable the memory assignment was for
*/
void throw_cuda_errror_exception(JNIEnv *env, const char *message, int error) {
  char msg[1024];
  jclass exp;
  jfieldID fid;
  int a = 0;
  int b = 0;
  char name[1024];

  if(error == CUDA_SUCCESS){
    return;
  }

  exp = env->FindClass("edu/syr/pcpratts/rootbeer/runtime2/cuda/CudaErrorException");

  // we truncate the message to 900 characters to stop any buffer overflow
  switch(error){
    case CUDA_ERROR_OUT_OF_MEMORY:
      sprintf(msg, "CUDA_ERROR_OUT_OF_MEMORY: %.900s",message);
      break;
    case CUDA_ERROR_NO_BINARY_FOR_GPU:
      cuDeviceGetName(name,1024,cuDevice);
      cuDeviceComputeCapability(&a, &b, cuDevice);
      sprintf(msg, "No binary for gpu. Selected %s (%d.%d). 2.0 compatibility required.", name, a, b);
      break;
    default:
      sprintf(msg, "ERROR STATUS:%i : %.900s", error, message);
  }

  fid = env->GetFieldID(exp, "cudaError_enum", "I");
  env->SetLongField(exp,fid, (jint)error);

  env->ThrowNew(exp,msg);
  
  return;
}

void setLongField(JNIEnv *env, jobject obj, const char * name, jlong value){

  jfieldID fid = env->GetFieldID(thisRefClass, name, "J");
  env->SetLongField(obj, fid, value);
  
  return;
}

void getBestDevice(JNIEnv *env){
  int num_devices;
  int status;
  int i;
  CUdevice temp_device;
  int curr_multiprocessors;
  int max_multiprocessors = -1;
  int max_i = -1;
  
  status = cuDeviceGetCount(&num_devices);
  CHECK_STATUS(env,"error in cuDeviceGetCount",status)
          
  if(num_devices == 0)
      throw_cuda_errror_exception(env,"0 Cuda Devices were found",0);
  
  for(i = 0; i < num_devices; ++i){
    status = cuDeviceGet(&temp_device, i);
    CHECK_STATUS(env,"error in cuDeviceGet",status)
            
    status = cuDeviceGetAttribute(&curr_multiprocessors, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, temp_device);    
    CHECK_STATUS(env,"error in cuDeviceGetAttribute",status)
            
    if(curr_multiprocessors > max_multiprocessors)
    {
      max_multiprocessors = curr_multiprocessors;
      max_i = i;
    }
  }

  status = cuDeviceGet(&cuDevice, max_i); 
  CHECK_STATUS(env,"error in cuDeviceGet",status)
          
  status = cuDeviceGetAttribute(&maxGridDim, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, cuDevice);    
  CHECK_STATUS(env,"error in cuDeviceGetAttribute",status)
          
  numMultiProcessors = max_multiprocessors;

  return;
}

void savePointers(JNIEnv * env, jobject this_ref){
  thisRefClass = env->GetObjectClass(this_ref);
  setLongField(env, this_ref, "m_ToSpaceAddr", (jlong) toSpace);
  setLongField(env, this_ref, "m_GpuToSpaceAddr", (jlong) gpuToSpace);
  setLongField(env, this_ref, "m_TextureAddr", (jlong) textureMemory);
  setLongField(env, this_ref, "m_GpuTextureAddr", (jlong) gpuTexture);
  setLongField(env, this_ref, "m_HandlesAddr", (jlong) handlesMemory);
  setLongField(env, this_ref, "m_GpuHandlesAddr", (jlong) gpuHandlesMemory);
  setLongField(env, this_ref, "m_ExceptionsHandlesAddr", (jlong) exceptionsMemory);
  setLongField(env, this_ref, "m_GpuExceptionsHandlesAddr", (jlong) gpuExceptionsMemory);
  setLongField(env, this_ref, "m_ToSpaceSize", (jlong) bufferSize);
  setLongField(env, this_ref, "m_MaxGridDim", (jlong) maxGridDim);
  setLongField(env, this_ref, "m_NumMultiProcessors", (jlong) numMultiProcessors);
  setLongField(env, this_ref, "m_NumBlocks", (jlong) numBlocks);
  
  return;
}

void initDevice(JNIEnv * env, jobject this_ref, jint max_blocks_per_proc, jint max_threads_per_block, jlong free_space)
{          
  int status;
  int deviceCount = 0;
  size_t f_mem;
  size_t t_mem;
  size_t to_space_size;
  textureMemSize = 1;

  status = cuDeviceGetCount(&deviceCount);
  CHECK_STATUS(env,"error in cuDeviceGetCount",status)

  getBestDevice(env);
  
  status = cuCtxCreate(&cuContext, CU_CTX_MAP_HOST, cuDevice);  
  CHECK_STATUS(env,"error in cuCtxCreate",status)
  
  status = cuMemGetInfo(&f_mem, &t_mem);
  CHECK_STATUS(env,"error in cuMemGetInfo",status)
          
  to_space_size = f_mem;
  
  numBlocks = numMultiProcessors * max_threads_per_block * max_blocks_per_proc;
  
  //space for 100 types in the scene
  classMemSize = sizeof(jint)*100;
  
  gc_space_size = 1024;
  to_space_size -= (numBlocks * sizeof(jlong));
  to_space_size -= (numBlocks * sizeof(jlong));
  to_space_size -= gc_space_size;
  to_space_size -= free_space;
  to_space_size -= classMemSize;
  //leave 10MB for module
  to_space_size -= 10L*1024L*1024L;

  //to_space_size -= textureMemSize;
  bufferSize = to_space_size;

  status = cuMemHostAlloc(&toSpace, to_space_size, 0);  
  CHECK_STATUS(env,"toSpace memory allocation failed",status)
    
  status = cuMemAlloc(&gpuToSpace, to_space_size);
  CHECK_STATUS(env,"gpuToSpace memory allocation failed",status)
    
  status = cuMemAlloc(&gpuClassMemory, classMemSize);
  CHECK_STATUS(env,"gpuClassMemory memory allocation failed",status)
  
/*
  status = cuMemHostAlloc(&textureMemory, textureMemSize, 0);  
  if (CUDA_SUCCESS != status) 
  {
    printf("error in cuMemHostAlloc textureMemory %d\n", status);
  }

  status = cuMemAlloc(&gpuTexture, textureMemSize);
  if (CUDA_SUCCESS != status) 
  {
    printf("error in cuMemAlloc gpuTexture %d\n", status);
  }
*/

  status = cuMemHostAlloc(&handlesMemory, numBlocks * sizeof(jlong), CU_MEMHOSTALLOC_WRITECOMBINED); 
  CHECK_STATUS(env,"handlesMemory memory allocation failed",status)

  status = cuMemAlloc(&gpuHandlesMemory, numBlocks * sizeof(jlong)); 
  CHECK_STATUS(env,"gpuHandlesMemory memory allocation failed",status)

  status = cuMemHostAlloc(&exceptionsMemory, numBlocks * sizeof(jlong), 0); 
  CHECK_STATUS(env,"exceptionsMemory memory allocation failed",status)

  status = cuMemAlloc(&gpuExceptionsMemory, numBlocks * sizeof(jlong)); 
  CHECK_STATUS(env,"gpuExceptionsMemory memory allocation failed",status)

  status = cuMemAlloc(&gcInfoSpace, gc_space_size);  
  CHECK_STATUS(env,"gcInfoSpace memory allocation failed",status)

  status = cuMemAlloc(&gpuHeapEndPtr, 8);
  CHECK_STATUS(env,"gpuHeapEndPtr memory allocation failed",status)

  status = cuMemAlloc(&gpuBufferSize, 8);
  CHECK_STATUS(env,"gpuBufferSize memory allocation failed",status)

  savePointers(env, this_ref);

  if (host_monitor != NULL) {
    printf("initDevice - init hostdevice_interface\n");
    
    // allocate host_device_interface as pinned memory
    checkCuda(cudaHostAlloc((void**) &h_host_device_interface, 
              sizeof(HostDeviceInterface),
              cudaHostAllocWriteCombined | cudaHostAllocMapped),
              "error in cudaHostAlloc(h_host_device_interface)");

    checkCuda(cudaHostGetDevicePointer(&d_host_device_interface, 
              h_host_device_interface, 0),
              "error in cudaHostGetDevicePointer(d_host_device_interface)");

    host_monitor->setHostDeviceInterface(h_host_device_interface);
  }

  printf("initDevice finished!\n");

  return;
}

/*
 * Class:     edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2
 * Method:    reinit
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2_reinit
  (JNIEnv * env, jobject this_ref, jint max_blocks_per_proc, jint max_threads_per_block, jlong free_space)
{
  cuMemFreeHost(toSpace);
  cuMemFree(gpuToSpace);
  cuMemFree(gpuClassMemory);
  cuMemFreeHost(handlesMemory);
  cuMemFree(gpuHandlesMemory);
  cuMemFreeHost(exceptionsMemory);
  cuMemFree(gpuExceptionsMemory);
  cuMemFree(gcInfoSpace);
  cuMemFree(gpuHeapEndPtr);
  cuMemFree(gpuBufferSize);
  cuCtxDestroy(cuContext);
  initDevice(env, this_ref, max_blocks_per_proc, max_threads_per_block, free_space);
  
  return;
}

size_t initContext(JNIEnv * env, jint max_blocks_per_proc, jint max_threads_per_block)
{
  size_t to_space_size;
  int status;
  int deviceCount = 0;
  size_t f_mem;
  size_t t_mem;
  
  status = cuDeviceGetCount(&deviceCount);
  CHECK_STATUS_RTN(env,"error in cuDeviceGetCount",status, 0);

  getBestDevice(env);

  status = cuCtxCreate(&cuContext, CU_CTX_MAP_HOST, cuDevice);
  CHECK_STATUS_RTN(env,"error in cuCtxCreate",status, 0)
  
  status = cuMemGetInfo (&f_mem, &t_mem);
  CHECK_STATUS_RTN(env,"error in cuMemGetInfo",status, 0)
  
  to_space_size = f_mem;

  //space for 100 types in the scene
  classMemSize = sizeof(jint)*100;
  
  numBlocks = numMultiProcessors * max_threads_per_block * max_blocks_per_proc;
  
  gc_space_size = 1024;
  to_space_size -= (numBlocks * sizeof(jlong));
  to_space_size -= (numBlocks * sizeof(jlong));
  to_space_size -= gc_space_size;
  to_space_size -= classMemSize;
  //leave 10MB for module
  to_space_size -= 10L*1024L*1024L;
  
  return to_space_size;
}

/*
 * Class:     edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2
 * Method:    findReserveMem
 * Signature: ()I
 */
JNIEXPORT jlong JNICALL Java_edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2_findReserveMem
  (JNIEnv * env, jobject this_ref, jint max_blocks_per_proc, jint max_threads_per_block)
{
  size_t to_space_size;
  size_t temp_size;
  int status;
  int deviceCount = 0;
  jlong prev_i;
  jlong i;
  size_t f_mem;
  size_t t_mem;

  status = cuInit(0);
  CHECK_STATUS_RTN(env,"error in cuInit",status, 0)

  printf("automatically determining CUDA reserve space...\n");
  
  to_space_size = initContext(env, max_blocks_per_proc, max_threads_per_block);
  numBlocks = numMultiProcessors * max_threads_per_block * max_blocks_per_proc;
  
  for(i = 1024L*1024L; i < to_space_size; i += 100L*1024L*1024L){
    temp_size = to_space_size - i;
  
    printf("attempting allocation with temp_size: %lu to_space_size: %lu i: %ld\n", temp_size, to_space_size, i);
 
    status = cuMemHostAlloc(&toSpace, temp_size, 0);  
    if(status != CUDA_SUCCESS){
      cuCtxDestroy(cuContext);
      initContext(env, max_blocks_per_proc, max_threads_per_block);
      continue;
    }
    
    status = cuMemAlloc(&gpuToSpace, temp_size);
    if(status != CUDA_SUCCESS){
      cuCtxDestroy(cuContext);
      initContext(env, max_blocks_per_proc, max_threads_per_block);
      continue;
    } 

    status = cuMemAlloc(&gpuClassMemory, classMemSize);
    if(status != CUDA_SUCCESS){
      cuCtxDestroy(cuContext);
      initContext(env, max_blocks_per_proc, max_threads_per_block);
      continue;
    } 

    status = cuMemHostAlloc(&handlesMemory, numBlocks * sizeof(jlong), 0); 
    if(status != CUDA_SUCCESS){
      cuCtxDestroy(cuContext);
      initContext(env, max_blocks_per_proc, max_threads_per_block);
      continue;
    } 

    status = cuMemAlloc(&gpuHandlesMemory, numBlocks * sizeof(jlong)); 
    if(status != CUDA_SUCCESS){
      cuCtxDestroy(cuContext);
      initContext(env, max_blocks_per_proc, max_threads_per_block);
      continue;
    } 

    status = cuMemHostAlloc(&exceptionsMemory, numBlocks * sizeof(jlong), 0); 
    if(status != CUDA_SUCCESS){
      cuCtxDestroy(cuContext);
      initContext(env, max_blocks_per_proc, max_threads_per_block);
      continue;
    } 

    status = cuMemAlloc(&gpuExceptionsMemory, numBlocks * sizeof(jlong)); 
    if(status != CUDA_SUCCESS){
      cuCtxDestroy(cuContext);
      initContext(env, max_blocks_per_proc, max_threads_per_block);
      continue;
    } 

    status = cuMemAlloc(&gcInfoSpace, gc_space_size);  
    if(status != CUDA_SUCCESS){
      cuCtxDestroy(cuContext);
      initContext(env, max_blocks_per_proc, max_threads_per_block);
      continue;
    } 

    status = cuMemAlloc(&gpuHeapEndPtr, 8);
    if(status != CUDA_SUCCESS){
      cuCtxDestroy(cuContext);
      initContext(env, max_blocks_per_proc, max_threads_per_block);
      continue;
    } 

    status = cuMemAlloc(&gpuBufferSize, 8);
    if(status != CUDA_SUCCESS){
      cuCtxDestroy(cuContext);
      initContext(env, max_blocks_per_proc, max_threads_per_block);
      continue;
    } 


    bufferSize = temp_size;
    savePointers(env, this_ref);

    return i;
  }
  throw_cuda_errror_exception(env, "unable to find enough space using CUDA", 0); 
  return 0;
}

/*
 * Class:     edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2
 * Method:    printDeviceInfo
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2_printDeviceInfo
  (JNIEnv *env, jclass cls)
{
    int i, a=0, b=0, status;
    int num_devices = 0;
    char str[1024];
    size_t free_mem, total_mem;
 
    status = cuInit(0);
    CHECK_STATUS(env,"error in cuInit",status)
    
    cuDeviceGetCount(&num_devices);
    printf("%d cuda gpus found\n", num_devices);
 
    for (i = 0; i < num_devices; ++i)
    {
        CUdevice dev;
        status = cuDeviceGet(&dev, i);
        CHECK_STATUS(env,"error in cuDeviceGet",status)

        status = cuCtxCreate(&cuContext, CU_CTX_MAP_HOST, dev);
        CHECK_STATUS(env,"error in cuCtxCreate",status)
                
        printf("\nGPU:%d\n", i);
        
        if(cuDeviceComputeCapability(&a, &b, dev) == CUDA_SUCCESS)
            printf("Version:                       %i.%i\n", a, b);
        
        if(cuDeviceGetName(str,1024,dev) == CUDA_SUCCESS)
            printf("Name:                          %s\n", str);
        
        if(cuMemGetInfo(&free_mem, &total_mem) == CUDA_SUCCESS){
          #if (defined linux || defined __APPLE_CC__)
            printf("Total global memory:           %zu/%zu (Free/Total) MBytes\n", free_mem/1024/1024, total_mem/1024/1024);
          #else
            printf("Total global memory:           %Iu/%Iu (Free/Total) MBytes\n", free_mem/1024/1024, total_mem/1024/1024);
          #endif
        }
        
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,dev) == CUDA_SUCCESS)
            printf("Total registers per block:     %i\n", a);
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_WARP_SIZE,dev) == CUDA_SUCCESS)
            printf("Warp size:                     %i\n", a);
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_PITCH,dev) == CUDA_SUCCESS)
            printf("Maximum memory pitch:          %i\n", a);
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,dev) == CUDA_SUCCESS)
            printf("Maximum threads per block:     %i\n", a);        
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,dev) == CUDA_SUCCESS)
            printf("Total shared memory per block  %.2f KB\n", a/1024.0);
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_CLOCK_RATE,dev) == CUDA_SUCCESS)
            printf("Clock rate:                    %.2f MHz\n",  a/1000000.0);
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,dev) == CUDA_SUCCESS)
            printf("Memory Clock rate:             %.2f\n",  a/1000000.0);
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,dev) == CUDA_SUCCESS)
            printf("Total constant memory:         %.2f MB\n",  a/1024.0/1024.0);
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_INTEGRATED,dev) == CUDA_SUCCESS)
            printf("Integrated:                    %i\n",  a);
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,dev) == CUDA_SUCCESS)
            printf("Max threads per multiprocessor:%i\n",  a);    
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,dev) == CUDA_SUCCESS)
            printf("Number of multiprocessors:     %i\n",  a);    
      
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,dev) == CUDA_SUCCESS)
            printf("Maximum dimension x of block:  %i\n", a);
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,dev) == CUDA_SUCCESS)
            printf("Maximum dimension y of block:  %i\n", a);
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,dev) == CUDA_SUCCESS)
            printf("Maximum dimension z of block:  %i\n", a);
        
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,dev) == CUDA_SUCCESS)
            printf("Maximum dimension x of grid:   %i\n", a);
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,dev) == CUDA_SUCCESS)
            printf("Maximum dimension y of grid:   %i\n", a);
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,dev) == CUDA_SUCCESS)
            printf("Maximum dimension z of grid:   %i\n", a);
			
        cuCtxDestroy(cuContext);
    } 
	
	return;
}

/*
 * Class:     edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2
 * Method:    setup
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2_setup
  (JNIEnv *env, jobject this_ref, jint max_blocks_per_proc, jint max_threads_per_block, jlong free_space)
{
  int status;
  
  status = cuInit(0);
  CHECK_STATUS(env,"error in cuInit",status)

  initDevice(env, this_ref, max_blocks_per_proc, max_threads_per_block, free_space);
  
  return;
}

void * readCubinFile(const char * filename){

  int i;
  jlong size;
  char * ret;
  int block_size;
  int num_blocks;
  int leftover;
  char * dest;
  
  FILE * file = fopen(filename, "r");
  fseek(file, 0, SEEK_END);
  size = ftell(file);
  fseek(file, 0, SEEK_SET);

  ret = (char *) malloc(size);
  block_size = 4096;
  num_blocks = (int) (size / block_size);
  leftover = (int) (size % block_size);

  dest = ret;
  for(i = 0; i < num_blocks; ++i){
    fread(dest, 1, block_size, file);
    dest += block_size;
  }
  if(leftover != 0){
    fread(dest, 1, leftover, file);
  }

  fclose(file);
  return (void *) ret;
}

void * readCubinFileFromBuffers(JNIEnv *env, jobject buffers, jint size, jint total_size){
  int i, j;
  int dest_offset = 0;
  int len;
  signed char * data;
  char * ret = (char *) malloc(total_size);

  jclass cls = env->GetObjectClass(buffers);
  jmethodID mid = env->GetMethodID(cls, "get", "(I)Ljava/lang/Object;");
  for(i = 0; i < size; ++i){
    jobject buffer = env->CallObjectMethod(buffers, mid, i);
    jbyteArray * arr = (jbyteArray*) &buffer;
    len = env->GetArrayLength(*arr);
    data = env->GetByteArrayElements(*arr, NULL);
    memcpy((void *) (ret + dest_offset), (void *) data, len);
    dest_offset += len;
    env->ReleaseByteArrayElements(*arr, data, 0);
  }

  return (void *) ret;
}

/*
 * Class:     edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2
 * Method:    writeClassTypeRef
 * Signature: ([I)V
 */
JNIEXPORT void JNICALL Java_edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2_writeClassTypeRef
  (JNIEnv *env, jobject this_ref, jintArray jarray)
{
  int i;
  jint * native_array = env->GetIntArrayElements(jarray, 0);
  cuMemcpyHtoD(gpuClassMemory, native_array, classMemSize);
  env->ReleaseIntArrayElements(jarray, native_array, 0);
  
  return;
}

/*
 * Class:     edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2
 * Method:    loadFunction
 * Signature: (JLjava/lang/Object;III)V
 */
JNIEXPORT void JNICALL Java_edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2_loadFunction
  (JNIEnv *env, jobject this_obj, jlong heap_end_ptr, jobject buffers, jint size, jint total_size, jint num_blocks){

  void * fatcubin;
  int offset;
  CUresult status;
  char * native_filename;
  heapEndPtr = heap_end_ptr;
  
  cuCtxPushCurrent(cuContext);
  fatcubin = readCubinFileFromBuffers(env, buffers, size, total_size);
  status = cuModuleLoadFatBinary(&cuModule, fatcubin);
  CHECK_STATUS(env, "error in cuModuleLoad", status);
  free(fatcubin);

  status = cuModuleGetFunction(&cuFunction, cuModule, "_Z5entryPcS_PiPxS1_S0_S0_iP19HostDeviceInterface"); 
  CHECK_STATUS(env,"error in cuModuleGetFunction",status)

  status = cuFuncSetCacheConfig(cuFunction, CU_FUNC_CACHE_PREFER_L1);
  CHECK_STATUS(env,"error in cuFuncSetCacheConfig",status)

  status = cuParamSetSize(cuFunction, (7 * sizeof(CUdeviceptr) + sizeof(int))); 
  CHECK_STATUS(env,"error in cuParamSetSize",status)

  offset = 0;
  status = cuParamSetv(cuFunction, offset, (void *) &gcInfoSpace, sizeof(CUdeviceptr)); 
  CHECK_STATUS(env,"error in cuParamSetv gcInfoSpace",status)
  offset += sizeof(CUdeviceptr);

  status = cuParamSetv(cuFunction, offset, (void *) &gpuToSpace, sizeof(CUdeviceptr)); 
  CHECK_STATUS(env,"error in cuParamSetv gpuToSpace",status)
  offset += sizeof(CUdeviceptr);

  status = cuParamSetv(cuFunction, offset, (void *) &gpuHandlesMemory, sizeof(CUdeviceptr)); 
  CHECK_STATUS(env,"error in cuParamSetv gpuHandlesMemory %",status)
  offset += sizeof(CUdeviceptr);

  status = cuParamSetv(cuFunction, offset, (void *) &gpuHeapEndPtr, sizeof(CUdeviceptr)); 
  CHECK_STATUS(env,"error in cuParamSetv gpuHeapEndPtr",status)
  offset += sizeof(CUdeviceptr);

  status = cuParamSetv(cuFunction, offset, (void *) &gpuBufferSize, sizeof(CUdeviceptr));
  CHECK_STATUS(env,"error in cuParamSetv gpuBufferSize",status)
  offset += sizeof(CUdeviceptr); 

  status = cuParamSetv(cuFunction, offset, (void *) &gpuExceptionsMemory, sizeof(CUdeviceptr)); 
  CHECK_STATUS(env,"error in cuParamSetv gpuExceptionsMemory",status)
  offset += sizeof(CUdeviceptr);

  status = cuParamSetv(cuFunction, offset, (void *) &gpuClassMemory, sizeof(CUdeviceptr)); 
  CHECK_STATUS(env,"error in cuParamSetv gpuClassMemory",status)
  offset += sizeof(CUdeviceptr);

  status = cuParamSeti(cuFunction, offset, num_blocks); 
  CHECK_STATUS(env,"error in cuParamSetv num_blocks",status)
  offset += sizeof(int);

  status = cuParamSetv(cuFunction, offset, (void *) &d_host_device_interface, sizeof(HostDeviceInterface)); 
  CHECK_STATUS(env,"error in cuParamSetv d_host_device_interface",status)
  offset += sizeof(int);

  cuCtxPopCurrent(&cuContext);
  
  return;
}

/*
 * Class:     edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2
 * Method:    runBlocks
 * Signature: (I)V
 */
JNIEXPORT jint JNICALL Java_edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2_runBlocks
  (JNIEnv *env, jobject this_obj, jint num_blocks, jint block_shape, jint grid_shape){

  CUresult status;
  jlong * infoSpace = (jlong *) malloc(gc_space_size);
  infoSpace[1] = heapEndPtr;
  cuCtxPushCurrent(cuContext);
  cuMemcpyHtoD(gcInfoSpace, infoSpace, gc_space_size);
  cuMemcpyHtoD(gpuToSpace, toSpace, heapEndPtr);
  //cuMemcpyHtoD(gpuTexture, textureMemory, textureMemSize);
  cuMemcpyHtoD(gpuHandlesMemory, handlesMemory, num_blocks * sizeof(jlong));
  cuMemcpyHtoD(gpuHeapEndPtr, &heapEndPtr, sizeof(jlong));
  cuMemcpyHtoD(gpuBufferSize, &bufferSize, sizeof(jlong));
  
/*
  status = cuModuleGetTexRef(&cache, cuModule, "m_Cache");  
  if (CUDA_SUCCESS != status) 
  {
    printf("error in cuModuleGetTexRef %d\n", status);
  }

  status = cuTexRefSetAddress(0, cache, gpuTexture, textureMemSize);
  if (CUDA_SUCCESS != status) 
  {
    printf("error in cuTextRefSetAddress %d\n", status);
  }
*/

  if (host_monitor != NULL) {
    printf("runBlocks - startMonitoring...\n");
    host_monitor->startMonitoring();
  }

  status = cuFuncSetBlockShape(cuFunction, block_shape, 1, 1);
  if(status != CUDA_SUCCESS){
    free(infoSpace);
    cuCtxPopCurrent(&cuContext);
  }
  CHECK_STATUS_RTN(env,"error in cuFuncSetBlockShape",status, (jint)status);

  status = cuLaunchGrid(cuFunction, grid_shape, 1);
  if(status != CUDA_SUCCESS){
    free(infoSpace);
    cuCtxPopCurrent(&cuContext);
  }
  CHECK_STATUS_RTN(env,"error in cuLaunchGrid",status, (jint)status)

  status = cuCtxSynchronize();  
  if(status != CUDA_SUCCESS){
    free(infoSpace);
    cuCtxPopCurrent(&cuContext);
  }
  CHECK_STATUS_RTN(env,"error in cuCtxSynchronize",status, (jint)status)
  
  if (host_monitor != NULL) {
    printf("runBlocks - stopMonitoring...\n");
    host_monitor->stopMonitoring();
  }

  cuMemcpyDtoH(infoSpace, gcInfoSpace, gc_space_size);
  heapEndPtr = infoSpace[1];
  cuMemcpyDtoH(toSpace, gpuToSpace, heapEndPtr);
  cuMemcpyDtoH(exceptionsMemory, gpuExceptionsMemory, num_blocks * sizeof(jlong));
  free(infoSpace);
  cuCtxPopCurrent(&cuContext);

  return 0;
}

/*
 * Class:     edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2
 * Method:    unload
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2_unload
  (JNIEnv *env, jobject this_obj){

  cuModuleUnload(cuModule);
  cuFunction = (CUfunction) 0;  
 
  return;
}

/*
 * Class:     edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2
 * Method:    connect
 * Signature: (I)Z
 */
JNIEXPORT jboolean JNICALL Java_edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2_connect
  (JNIEnv *env, jobject this_ref, jint port) {

  printf("CudaRuntime2.connect started...\n");

  // init HostMonitor
  host_monitor = new HostMonitor(port);

  printf("CudaRuntime2.connect is_monitoring: %s\n",
        (host_monitor->is_monitoring) ? "true" : "false");

  return true;
}
