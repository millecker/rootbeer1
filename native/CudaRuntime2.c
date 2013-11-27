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

#include <errno.h>
#include <netinet/in.h>
#include <pthread.h>
#include <rpc/types.h>
#include <rpc/xdr.h>
#include <signal.h>
#include <sstream> /* ostringstream */
#include <string>
#include <sys/socket.h>
#include <typeinfo> /* typeid */

#define stringify( name ) # name
#define STR_SIZE 1024

using std::string;

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
  volatile bool use_str_val1; // in str_val1
  volatile bool use_str_val2; // in str_val2

  // Transfer variables (used in sendCommand and getResult)
  volatile int int_val1;
  volatile long long_val1;
  volatile char str_val1[STR_SIZE];
  volatile char str_val2[STR_SIZE];

  // Response of HostMonitor
  volatile bool is_result_available;

  HostDeviceInterface() {
    init();
  }

  void init() {
    lock_thread_id = -1;
    has_task = false;
    done = false;
    command = UNDEFINED;
    use_int_val1 = false;
    use_str_val1 = false;
    use_str_val2 = false;
    int_val1 = 0;
    long_val1 = 0;
    is_result_available = false;
  }

  ~HostDeviceInterface() {}
};

/* Only needed for debugging output */
const char* messageTypeNames[] = {
  stringify( START_MESSAGE ), stringify( SET_BSPJOB_CONF ), stringify( SET_INPUT_TYPES ),
  stringify( RUN_SETUP ), stringify( RUN_BSP ), stringify( RUN_CLEANUP ),
  stringify( READ_KEYVALUE ), stringify( WRITE_KEYVALUE ),
  stringify( GET_MSG ), stringify( GET_MSG_COUNT ),
  stringify( SEND_MSG ), stringify( SYNC ),
  stringify( GET_ALL_PEERNAME ), stringify( GET_PEERNAME ),
  stringify( GET_PEER_INDEX ), stringify( GET_PEER_COUNT ), stringify( GET_SUPERSTEP_COUNT ),
  stringify( REOPEN_INPUT ), stringify( CLEAR ),
  stringify( CLOSE ), stringify( ABORT ),
  stringify( DONE ), stringify( TASK_DONE ),
  stringify( REGISTER_COUNTER ), stringify( INCREMENT_COUNTER ),
  stringify( SEQFILE_OPEN ), stringify( SEQFILE_READNEXT ),
  stringify( SEQFILE_APPEND ), stringify( SEQFILE_CLOSE ),
  stringify( PARTITION_REQUEST ), stringify( PARTITION_RESPONSE ),
  stringify( LOG ), stringify( END_OF_DATA ),
  stringify( UNDEFINED )
};

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

/**
 * Generic toString
 */
template <class T>
string toString(const T& t) {
  std::ostringstream oss;
  oss << t;
  return oss.str();
}

/**
 * Generic toString template specializations
 */
template <> string toString<string>(const string& t) {
  return t;
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
    error = msg + " at " + file + ":" + toString<int32_t>(line) +
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
/**
 * Generic serialization
 */
template<class T>
void serialize(T t, FileOutStream& stream) {
  serializeString(toString<T>(t), stream);
}

/**
 * Generic serialization template specializations
 */
template <> void serialize<int64_t>(int64_t t, FileOutStream& stream) {
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

template <> void serialize<int32_t>(int32_t t, FileOutStream& stream) {
  serialize<int64_t>(t, stream);
}

template <> void serialize<float>(float t, FileOutStream& stream) {
  char buf[sizeof(float)];
  XDR xdrs;
  xdrmem_create(&xdrs, buf, sizeof(float), XDR_ENCODE);
  xdr_float(&xdrs, &t);
  stream.write(buf, sizeof(float));
}

template <> void serialize<double>(double t, FileOutStream& stream) {
  char buf[sizeof(double)];
  XDR xdrs;
  xdrmem_create(&xdrs, buf, sizeof(double), XDR_ENCODE);
  xdr_double(&xdrs, &t);
  stream.write(buf, sizeof(double));
}

template <> void serialize<string>(string t, FileOutStream& stream) {
  serialize<int64_t>(t.length(), stream);
  if (t.length() > 0) {
    stream.write(t.data(), t.length());
  }
}
  
/**
 * Generic deserialization
 */
template<class T>
T deserialize(FileInStream& stream) {
  string str = "Not able to deserialize type: ";
  throw Error(str.append(typeid(T).name()));
}
  
/**
 * Generic deserialization template specializations
 */
template <> int64_t deserialize<int64_t>(FileInStream& stream) {
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

template <> int32_t deserialize<int32_t>(FileInStream& stream) {
  return deserialize<int64_t>(stream);
}

template <> float deserialize<float>(FileInStream& stream) {
  float t;
  char buf[sizeof(float)];
  stream.read(buf, sizeof(float));
  XDR xdrs;
  xdrmem_create(&xdrs, buf, sizeof(float), XDR_DECODE);
  xdr_float(&xdrs, &t);
  return t;
}

template <> double deserialize<double>(FileInStream& stream) {
  double t;
  char buf[sizeof(double)];
  stream.read(buf, sizeof(double));
  XDR xdrs;
  xdrmem_create(&xdrs, buf, sizeof(double), XDR_DECODE);
  xdr_double(&xdrs, &t);
  return t;
}

template <> string deserialize<string>(FileInStream& stream) {
  string t;
  int32_t len = deserialize<int32_t>(stream);
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
  int sock_;
  FILE* in_stream_;
  FILE* out_stream_;
  FileInStream* file_in_stream_;
  FileOutStream* file_out_stream_;

public: 
  SocketClient() {
    sock_ = -1;
    in_stream_ = NULL;
    out_stream_ = NULL;
    file_in_stream_ = NULL;
    file_out_stream_ = NULL;
  }
  
  ~SocketClient() {
    if (in_stream_ != NULL) {
      fflush(in_stream_);
    }
    if (out_stream_ != NULL) {
      fflush(out_stream_);
    }
    fflush(stdout);

    if (sock_ != -1) {
      int result = shutdown(sock_, SHUT_RDWR);
      if (result != 0) {
        fprintf(stderr, "SocketClient: problem shutting down socket\n");
      }
    }

    delete in_stream_;
    delete out_stream_;
    delete file_in_stream_;
    delete file_out_stream_;
  }
  
  void connectSocket(int port) {
    printf("SocketClient started\n");
    
    if (port <= 0) {
      fprintf(stderr, "SocketClient: invalid port number!\n");
      return; /* Failed */
    }
    
    sock_ = socket(PF_INET, SOCK_STREAM, 0);
    if (sock_ == -1) {
      fprintf(stderr, "SocketClient: problem creating socket: %s\n",
              strerror(errno));
    }
    
    sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    
    int res = connect(sock_, (sockaddr*) &addr, sizeof(addr));
    if (res != 0) {
      fprintf(stderr, "SocketClient: problem connecting command socket: %s\n",
              strerror(errno));
    }
    
    FILE* in_stream = fdopen(sock_, "r");
    FILE* out_stream = fdopen(sock_, "w");
    
    file_in_stream_ = new FileInStream();
    file_in_stream_->open(in_stream);
    file_out_stream_ = new FileOutStream();
    file_out_stream_->open(out_stream);
    
    printf("SocketClient is connected to port %d ...\n", port);
  }
  
  void sendCMD(int32_t cmd) volatile {
    serialize<int32_t>(cmd, *file_out_stream_);
    file_out_stream_->flush();
    printf("SocketClient sent CMD %s\n", messageTypeNames[cmd]);
  }
  
  template<class T>
  void sendCMD(int32_t cmd, T value) volatile {
    serialize<int32_t>(cmd, *file_out_stream_);
    serialize<T>(value, *file_out_stream_);
    file_out_stream_->flush();
    printf("SocketClient sent CMD: %s with Value: '%s'\n", messageTypeNames[cmd],
           toString<T>(value).c_str());
  }

  template<class T1, class T2>
  void sendCMD(int32_t cmd, T1 value1, T2 value2) volatile {
    serialize<int32_t>(cmd, *file_out_stream_);
    serialize<T1>(value1, *file_out_stream_);
    serialize<T2>(value2, *file_out_stream_);
    file_out_stream_->flush();
    printf("SocketClient sent CMD: %s with Value1: '%s' and Value2: '%s'\n", messageTypeNames[cmd],
           toString<T1>(value1).c_str(), toString<T2>(value2).c_str());
  }

/*
// TODO
  void sendCMD(int32_t cmd, const string values[], int size) volatile {
    serialize<int32_t>(cmd, *file_out_stream_);
    for (int i = 0; i < size; i++) {
      serialize<string>(values[i], *file_out_stream_);
      printf("SocketClient sent CMD: %s with Param%d: %s\n",
             messageTypeNames[cmd], i + 1, values[i].c_str());
    }
    file_out_stream_->flush();
  }
  
  void sendCMD(int32_t cmd, int32_t value, const string values[],
               int size) volatile {
    serialize<int32_t>(cmd, *file_out_stream_);
    serialize<int32_t>(value, *file_out_stream_);
    for (int i = 0; i < size; i++) {
      serialize<string>(values[i], *file_out_stream_);
      printf("SocketClient sent CMD: %s with Param%d: %s\n",
             messageTypeNames[cmd], i + 1, values[i].c_str());
    }
    file_out_stream_->flush();
  }
*/
  /**
   * Wait for next event, which should be a response for
   * a previously sent command (expected_response_cmd)
   * and return the generic result
   */
  template<class T>
  T getResult(int32_t expected_response_cmd) volatile {
    
    T result = T();

    // read response command
    int32_t cmd = deserialize<int32_t>(*file_in_stream_);
    
    // check if response is expected
    if (expected_response_cmd == cmd) {

      switch (cmd) {

        case HostDeviceInterface::GET_MSG: {
          T msg;
          msg = deserialize<T>(*file_in_stream_);
          return msg;
        }
        case HostDeviceInterface::GET_MSG_COUNT: {
          T msg_count;
          msg_count = deserialize<T>(*file_in_stream_);
          return msg_count;
        }
        case HostDeviceInterface::GET_PEERNAME: {
          T peername;
          peername = deserialize<T>(*file_in_stream_);
          return peername;
        }
        case HostDeviceInterface::GET_PEER_INDEX: {
          T peer_index = deserialize<T>(*file_in_stream_);
          return peer_index;
        }
        case HostDeviceInterface::GET_PEER_COUNT: {
          T peer_count = deserialize<T>(*file_in_stream_);
          return peer_count;
        }
        case HostDeviceInterface::GET_SUPERSTEP_COUNT: {
          T superstep_count = deserialize<T>(*file_in_stream_);
          return superstep_count;
        }
        
        case HostDeviceInterface::SEQFILE_OPEN: {
          T file_id = deserialize<T>(*file_in_stream_);
          return file_id;
        }
        case HostDeviceInterface::SEQFILE_APPEND: {
          result = deserialize<T>(*file_in_stream_);
          return result;
        }
        case HostDeviceInterface::SEQFILE_CLOSE: {
          result = deserialize<T>(*file_in_stream_);
          return result;
        }
      }
      // Not expected response
    } else {
      
      /*
       case CLOSE: {
       if(logging)fprintf(stderr,"HamaPipes::BinaryProtocol::nextEvent - got CLOSE\n");
       handler_->close();
       break;
       }
       case ABORT: {
       if(logging)fprintf(stderr,"HamaPipes::BinaryProtocol::nextEvent - got ABORT\n");
       handler_->abort();
       break;
       }
       */
      fprintf(stderr, "SocketClient - Unknown binary command: %d\n", cmd);
      //HADOOP_ASSERT(false, "Unknown binary command " + toString(cmd));
    }
    return result;
  }
};

/*****************************************************************************/
// HostMonitor
/*****************************************************************************/
class HostMonitor {
private:
  pthread_t monitor_thread_;
  pthread_mutex_t mutex_process_command_;
  SocketClient *socket_client_;

public:
  volatile bool is_monitoring;
  volatile HostDeviceInterface *host_device_interface;

  HostMonitor(HostDeviceInterface *h_d_interface, int port) {
    host_device_interface = h_d_interface;
    is_monitoring = false;
    pthread_mutex_init(&mutex_process_command_, NULL);
    socket_client_ = new SocketClient();

    // connect SocketClient
    socket_client_->connectSocket(port);

    reset();
    printf("HostMonitor init finished...\n");
  }

  ~HostMonitor() {
    pthread_mutex_destroy(&mutex_process_command_);
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
      printf("HostMonitor.startMonitoring...\n");
      pthread_create(&monitor_thread_, NULL, &HostMonitor::thread, this);

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
      printf("HostMonitor.stopMonitoring...\n");

      host_device_interface->done = true;

      // wait for monitoring to end
      //while (is_monitoring) {
      //  printf("HostMonitor.stopMonitoring is_monitoring: %s\n",
      //    (is_monitoring) ? "true" : "false");
      //}
      printf("HostMonitor.stopMonitoring stopped! done: %s\n",
            (host_device_interface->done) ? "true" : "false");
    }
  }

  static void *thread(void *context) {
    volatile HostMonitor *_this = ((HostMonitor *) context);
    printf("HostMonitorThread started... done: %s\n",
            (_this->host_device_interface->done) ? "true" : "false");
    fflush(stdout);

    while (!_this->host_device_interface->done) {
      _this->is_monitoring = true;

      //printf("HostMonitorThread is_monitoring: %s\n",
      //      (_this->is_monitoring) ? "true" : "false");
      //fflush(stdout);

      //printf("HostMonitor thread running... has_task: %s lock_thread_id: %d command: %d\n",
      //      (_this->host_device_interface->has_task) ? "true" : "false",
      //       _this->host_device_interface->lock_thread_id,
      //       _this->host_device_interface->command);

      if ((_this->host_device_interface->has_task) && 
          (_this->host_device_interface->lock_thread_id >= 0) && 
          (_this->host_device_interface->command != HostDeviceInterface::UNDEFINED)) {

        pthread_mutex_t *lock = (pthread_mutex_t *) &_this->mutex_process_command_;
        pthread_mutex_lock(lock);
	
        printf("HostMonitor thread: %p, LOCKED(mutex_process_command)\n", pthread_self());

        _this->processCommand();
        
        _this->reset();

        pthread_mutex_unlock(lock);
        printf("HostMonitor thread: %p, UNLOCKED(mutex_process_command)\n", pthread_self());
        fflush(stdout);
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
      
      case HostDeviceInterface::SEND_MSG: {
        socket_client_->sendCMD(HostDeviceInterface::SEND_MSG, 
                                string(const_cast<char *>(host_device_interface->str_val1)), 
                                string(const_cast<char *>(host_device_interface->str_val2)));

        host_device_interface->str_val1[0] = '\0';
        host_device_interface->use_str_val1 = false;

        host_device_interface->str_val2[0] = '\0';
        host_device_interface->use_str_val2 = false;

        host_device_interface->is_result_available = true;
        // block until result was consumed
	while (host_device_interface->is_result_available) {}
        break;
      }

      case HostDeviceInterface::GET_MSG: {
        // TODO
        break;
      }

      case HostDeviceInterface::GET_MSG_COUNT: {
        socket_client_->sendCMD(HostDeviceInterface::GET_MSG_COUNT);

        host_device_interface->int_val1 = socket_client_->getResult<int32_t>(HostDeviceInterface::GET_MSG_COUNT);
        host_device_interface->is_result_available = true;

        printf("HostMonitor got result: %d result_available: %s\n",
               host_device_interface->int_val1,
               (host_device_interface->is_result_available) ? "true" : "false");

        // block until result was consumed
	while (host_device_interface->is_result_available) {}
        printf("HostMonitor consumed result: %d\n", host_device_interface->int_val1);
        break;
      }

      case HostDeviceInterface::SYNC: {
        socket_client_->sendCMD(HostDeviceInterface::SYNC);
        printf("HostMonitor sent SYNC\n");

        host_device_interface->int_val1 = 0;
        host_device_interface->is_result_available = true;
        // block until result was consumed
	while (host_device_interface->is_result_available) {}
        break;
      }

      case HostDeviceInterface::GET_SUPERSTEP_COUNT: {
        socket_client_->sendCMD(HostDeviceInterface::GET_SUPERSTEP_COUNT);

        host_device_interface->long_val1 = socket_client_->getResult<int64_t>(HostDeviceInterface::GET_SUPERSTEP_COUNT);
        host_device_interface->is_result_available = true;

        printf("HostMonitor got result: %ld result_available: %s\n",
               host_device_interface->long_val1,
               (host_device_interface->is_result_available) ? "true" : "false");

        // block until result was consumed
	while (host_device_interface->is_result_available) {}
        printf("HostMonitor consumed result: %ld\n", host_device_interface->long_val1);
        break;
      }

      case HostDeviceInterface::GET_PEERNAME: {
        socket_client_->sendCMD(HostDeviceInterface::GET_PEERNAME, host_device_interface->int_val1);
        
        string result = socket_client_->getResult<string>(HostDeviceInterface::GET_PEERNAME);

        strcpy(const_cast<char *>(host_device_interface->str_val1), result.c_str());
        host_device_interface->is_result_available = true;

        printf("HostMonitor got result: %s result_available: %s\n",
               host_device_interface->str_val1,
               (host_device_interface->is_result_available) ? "true" : "false");

        // block until result was consumed
        while (host_device_interface->is_result_available) {}
        printf("HostMonitor consumed result: %s\n", host_device_interface->str_val1);
        break;
      }

      case HostDeviceInterface::GET_PEER_INDEX: {
        socket_client_->sendCMD(HostDeviceInterface::GET_PEER_INDEX);

        host_device_interface->int_val1 = socket_client_->getResult<int32_t>(HostDeviceInterface::GET_PEER_INDEX);
        host_device_interface->is_result_available = true;

        printf("HostMonitor got result: %d result_available: %s\n",
               host_device_interface->int_val1,
               (host_device_interface->is_result_available) ? "true" : "false");

        // block until result was consumed
	while (host_device_interface->is_result_available) {}
        printf("HostMonitor consumed result: %d\n", host_device_interface->int_val1);
        break;
      }

      case HostDeviceInterface::GET_PEER_COUNT: {
        socket_client_->sendCMD(HostDeviceInterface::GET_PEER_COUNT);
        
        host_device_interface->int_val1 = socket_client_->getResult<int32_t>(HostDeviceInterface::GET_PEER_COUNT);
        host_device_interface->is_result_available = true;

        printf("HostMonitor got result: %d result_available: %s\n",
               host_device_interface->int_val1,
               (host_device_interface->is_result_available) ? "true" : "false");

        // block until result was consumed
	while (host_device_interface->is_result_available) {}
        printf("HostMonitor consumed result: %d\n", host_device_interface->int_val1);
        break;
      }

      case HostDeviceInterface::CLEAR: {
        socket_client_->sendCMD(HostDeviceInterface::CLEAR);
        printf("HostMonitor sent CLEAR\n");

        host_device_interface->int_val1 = 0;
        host_device_interface->is_result_available = true;
        // block until result was consumed
	while (host_device_interface->is_result_available) {}
        break;
      }

      case HostDeviceInterface::REOPEN_INPUT: {
        socket_client_->sendCMD(HostDeviceInterface::CLEAR);
        printf("HostMonitor sent CLEAR\n");

        host_device_interface->int_val1 = 0;
        host_device_interface->is_result_available = true;
        // block until result was consumed
	while (host_device_interface->is_result_available) {}
        break;
      }

      case HostDeviceInterface::SEQFILE_OPEN: {
        // TODO
        break;
      }
      case HostDeviceInterface::SEQFILE_APPEND: {
        // TODO
        break;
      }
      case HostDeviceInterface::SEQFILE_CLOSE: {
        // TODO
        break;
      }
    }
  }

};

// Global HostMonitor
HostMonitor *host_monitor = NULL;

// Global HostDeviceInterface
HostDeviceInterface *h_host_device_interface = NULL;
CUdeviceptr d_host_device_interface;


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

  printf("initDevice - allocate host_device_interface pinned memory.\n");
    
  // allocate host_device_interface as pinned memory
  status = cuMemHostAlloc((void**)&h_host_device_interface, sizeof(HostDeviceInterface),
                          CU_MEMHOSTALLOC_WRITECOMBINED | CU_MEMHOSTALLOC_DEVICEMAP);
  CHECK_STATUS(env,"h_host_device_interface memory allocation failed",status)

  h_host_device_interface->init();

  status = cuMemHostGetDevicePointer(&d_host_device_interface, h_host_device_interface, 0);
  CHECK_STATUS(env,"d_host_device_interface memory allocation failed",status)
  
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

  printf("loadFunction...\n");
  
  cuCtxPushCurrent(cuContext);
  fatcubin = readCubinFileFromBuffers(env, buffers, size, total_size);
  status = cuModuleLoadFatBinary(&cuModule, fatcubin);
  CHECK_STATUS(env, "error in cuModuleLoad", status);
  free(fatcubin);

  status = cuModuleGetFunction(&cuFunction, cuModule, "_Z5entryPcS_PiPxS1_S0_S0_P19HostDeviceInterfacei"); 
  CHECK_STATUS(env,"error in cuModuleGetFunction",status)

  status = cuFuncSetCacheConfig(cuFunction, CU_FUNC_CACHE_PREFER_L1);
  CHECK_STATUS(env,"error in cuFuncSetCacheConfig",status)

  status = cuParamSetSize(cuFunction, (8 * sizeof(CUdeviceptr) + sizeof(int))); 
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

  printf("loadFunction - lock_thread_id: %d\n",h_host_device_interface->lock_thread_id);
  printf("loadFunction - h_host_device_interface: %p\n",h_host_device_interface);
  printf("loadFunction - d_host_device_interface: %p\n",d_host_device_interface);
  fflush(stdout);

  status = cuParamSetv(cuFunction, offset, (void *) &d_host_device_interface, sizeof(CUdeviceptr)); 
  CHECK_STATUS(env,"error in cuParamSetv d_host_device_interface",status)
  offset += sizeof(CUdeviceptr);

  status = cuParamSeti(cuFunction, offset, num_blocks); 
  CHECK_STATUS(env,"error in cuParamSetv num_blocks",status)
  offset += sizeof(int);

  cuCtxPopCurrent(&cuContext);
  
  printf("loadFunction finished!\n");
  fflush(stdout);

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
    printf("runBlocks - startMonitoring finished!\n");
    fflush(stdout);
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
    printf("runBlocks - stopMonitoring finished!\n");
    fflush(stdout);
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
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2_connect
  (JNIEnv *env, jobject this_ref, jint port) {

  // init HostMonitor for Pinned Memory
  host_monitor = new HostMonitor(h_host_device_interface, port);
}
