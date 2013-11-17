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
/**********************************************************************************/
// HamaPipes and Pinnend Memory Communication
/**********************************************************************************/

/* before KernelWrapper
nvcc generated.cu --ptxas-options=-v

ptxas info    : 8 bytes gmem, 4 bytes cmem[14]
ptxas info    : Compiling entry function '_Z5entryPcS_PiPxS1_S0_S0_i' for 'sm_10'
ptxas info    : Used 5 registers, 104 bytes smem, 20 bytes cmem[1]
*/

/* after KernelWrapper
nvcc generated.cu --ptxas-options=-v -arch sm_20

ptxas info    : 16 bytes gmem, 16 bytes cmem[14]
ptxas info    : Compiling entry function '_Z5entryPcS_PiPxS1_S0_S0_i' for 'sm_20'
ptxas info    : Function properties for _Z5entryPcS_PiPxS1_S0_S0_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, 24 bytes smem, 92 bytes cmem[0]
*/

#include <assert.h>
#include <errno.h>
#include <netinet/in.h>
#include <pthread.h>
#include <rpc/types.h>
#include <rpc/xdr.h>
#include <signal.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <sys/socket.h>

#define stringify( name ) # name

using std::string;

string toString(int32_t x) {
  char str[100];
  sprintf(str, "%d", x);
  return str;
}

class Error {
private:
  string error;
public:
  Error(const std::string& msg): error(msg) {
  }
  
  Error(const std::string& msg,
        const std::string& file, int line,
        const std::string& function) {
    error = msg + " at " + file + ":" + toString(line) +
    " in " + function;
  }
  
  const std::string& getMessage() const {
    return error;
  }
};

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

/**********************************************************************************/
// FileInStream
/**********************************************************************************/

/**
 * A class to read a file as a stream.
 */
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

/**********************************************************************************/
// FileOutStream
/**********************************************************************************/

/**
 * A class to write a stream to a file.
 */
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


/**********************************************************************************/
// Serialization and Deserialization
/**********************************************************************************/

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

/**********************************************************************************/
// MESSAGE_TYPE
/**********************************************************************************/

enum MESSAGE_TYPE {
	UNDEFINED, GET_NUM_MESSAGES, DONE
};

/* Only needed for debugging output */
const char* messageTypeNames[] = { stringify(UNDEFINED), stringify(GET_NUM_MESSAGES),
  stringify(DONE) };

/**********************************************************************************/
// SocketClient Implementation
/**********************************************************************************/
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
        
      case GET_NUM_MESSAGES: {
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

/**********************************************************************************/
// KernelWrapper Implementation
/**********************************************************************************/
class KernelWrapper {
private:
  SocketClient socket_client;
	pthread_t t_monitor;
	pthread_mutex_t mutex_process_command;
  
	volatile bool result_available;
	volatile int result_int;
  
public:
	volatile MESSAGE_TYPE command;
	volatile bool has_task;
	volatile int lock_thread_id;
	volatile bool done;
	volatile bool is_monitoring;
  
	KernelWrapper(int port) {
		init(port);
	}
	~KernelWrapper() {
		pthread_mutex_destroy(&mutex_process_command);
	}
  
	void init(int port) {
    // connect SocketClient
    socket_client.connectSocket(port);
    
		is_monitoring = false;
		done = false;
		has_task = false;
    
		result_available = false;
		result_int = 0;
    
		lock_thread_id = -1;
    
		pthread_mutex_init(&mutex_process_command, NULL);
    
		reset();
	}
  
	void reset() volatile {
		command = UNDEFINED;
		has_task = false;
	}
  
	void start_monitoring() {
		pthread_create(&t_monitor, NULL, &KernelWrapper::thread, this);
	}
  
	static void *thread(void *context) {
		volatile KernelWrapper *_this = ((KernelWrapper *) context);
    
		while (!_this->done) {
			_this->is_monitoring = true;
      
			if ((_this->has_task) && (_this->lock_thread_id >= 0)
          && (_this->command != UNDEFINED)) {
        
				pthread_mutex_t *lock =
        (pthread_mutex_t *) &_this->mutex_process_command;
        
				pthread_mutex_lock(lock);
        
				_this->processCommand();
				_this->reset();
				
        pthread_mutex_unlock(lock);
      }
		}
		return NULL;
	}
  
	void processCommand() volatile {
    
		switch (command) {
        
      case GET_NUM_MESSAGES: {
        socket_client.sendCMD(GET_NUM_MESSAGES);
        
        while (!socket_client.isNewResultInt) {
          socket_client.nextEvent();
        }
        socket_client.isNewResultInt = false;
        
        result_int = socket_client.resultInt;
        result_available = true;
        
        // block until result was consumed
        while (result_available) {
        }
        
        break;
      }
      case DONE: {
        socket_client.sendCMD(DONE);
        result_available = true;
        
        // block until result was consumed
        while (result_available) {
        }
        
        break;
      }
        
		}
	}
  
	// Device Method
	// lock_thread_id was already set
	__device__ int getNumCurrentMessages() {
    
		// wait for possible old task to end
		while (has_task) {
		}
    
		command = GET_NUM_MESSAGES;
		has_task = true;
		__threadfence_system();
		//__threadfence();
    
		// wait for socket communication to end
		while (!result_available) {
			__threadfence_system();
		}
    
		result_available = false;
		__threadfence_system();
		//__threadfence();
    
		return result_int;
	}
  
	__device__ __host__ void sendDone() {
    
		command = DONE;
		has_task = true;
		//__threadfence_system();
    
		// wait for socket communication to end
		while (!result_available) {
		}
    
		done = true;
	}
};

// Pinned Memory connection global vars
KernelWrapper *h_kernelWrapper;
KernelWrapper *d_kernelWrapper;
/*HAMA_PIPES_CODE_END*/
