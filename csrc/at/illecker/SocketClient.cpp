/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "SocketClient.h"
#include "HostDeviceInterface.h"

#include <errno.h>
#include <netinet/in.h>
//#include <sys/socket.h>

SocketClient::SocketClient(bool is_debugging) {
  is_debugging = is_debugging;
  sock_ = -1;
  in_stream_ = NULL;
  out_stream_ = NULL;
  file_in_stream_ = NULL;
  file_out_stream_ = NULL;
}

SocketClient::~SocketClient() {
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
      printf("SocketClient: problem shutting down socket\n");
    }
  }
  
  delete in_stream_;
  delete out_stream_;
  delete file_in_stream_;
  delete file_out_stream_;
}

void SocketClient::connectSocket(int port) {
  if (is_debugging) {
    printf("SocketClient started\n");
  }
  
  if (port <= 0) {
    printf("SocketClient: invalid port number!\n");
    return;
  }
  
  sock_ = socket(PF_INET, SOCK_STREAM, 0);
  if (sock_ == -1) {
    printf("SocketClient: problem creating socket: %s\n",
           strerror(errno));
  }
  
  sockaddr_in addr;
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  
  int res = connect(sock_, (sockaddr*) &addr, sizeof(addr));
  if (res != 0) {
    printf("SocketClient: problem connecting command socket: %s\n",
           strerror(errno));
  }
  
  FILE* in_stream = fdopen(sock_, "r");
  FILE* out_stream = fdopen(sock_, "w");
  
  file_in_stream_ = new FileInStream();
  file_in_stream_->open(in_stream);
  file_out_stream_ = new FileOutStream();
  file_out_stream_->open(out_stream);
  
  if (is_debugging) {
    printf("SocketClient is connected to port %d ...\n", port);
  }
}

bool SocketClient::sendCMD(int32_t cmd, bool verify_response) volatile {
  serialize<int32_t>(cmd, *file_out_stream_);
  file_out_stream_->flush();
  if (is_debugging) {
    printf("SocketClient sent CMD %s\n", messageTypeNames[cmd]);
  }
  if (verify_response) {
    int32_t response = deserialize<int32_t>(*file_in_stream_);
    if (response != cmd) {
      return false;
    }
  }
  return true;
}

template<class T>
bool SocketClient::sendCMD(int32_t cmd, bool verify_response, T value) volatile {
  serialize<int32_t>(cmd, *file_out_stream_);
  serialize<T>(value, *file_out_stream_);
  file_out_stream_->flush();
  if (is_debugging) {
    printf("SocketClient sent CMD: %s with Value: '%s'\n", messageTypeNames[cmd],
           toString<T>(value).c_str());
  }
  if (verify_response) {
    int32_t response = deserialize<int32_t>(*file_in_stream_);
    if (response != cmd) {
      return false;
    }
  }
  return true;
}

template<class T1, class T2>
bool SocketClient::sendCMD(int32_t cmd, bool verify_response, T1 value1, T2 value2) volatile {
  serialize<int32_t>(cmd, *file_out_stream_);
  serialize<T1>(value1, *file_out_stream_);
  serialize<T2>(value2, *file_out_stream_);
  file_out_stream_->flush();
  if (is_debugging) {
    printf("SocketClient sent CMD: %s with Value1: '%s' and Value2: '%s'\n", messageTypeNames[cmd],
           toString<T1>(value1).c_str(), toString<T2>(value2).c_str());
  }
  if (verify_response) {
    int32_t response = deserialize<int32_t>(*file_in_stream_);
    if (response != cmd) {
      return false;
    }
  }
  return true;
}

template<class T1, class T2, class T3>
bool SocketClient::sendCMD(int32_t cmd, bool verify_response, T1 value1, T2 value2, T3 value3) volatile {
  serialize<int32_t>(cmd, *file_out_stream_);
  serialize<T1>(value1, *file_out_stream_);
  serialize<T2>(value2, *file_out_stream_);
  serialize<T3>(value3, *file_out_stream_);
  file_out_stream_->flush();
  if (is_debugging) {
    printf("SocketClient sent CMD: %s with Value1: '%s', Value2: '%s' and Value3: '%s'\n", messageTypeNames[cmd],
           toString<T1>(value1).c_str(), toString<T2>(value2).c_str(), toString<T3>(value3).c_str());
  }
  if (verify_response) {
    int32_t response = deserialize<int32_t>(*file_in_stream_);
    if (response != cmd) {
      return false;
    }
  }
  return true;
}

void SocketClient::sendCMD(int32_t cmd, const string values[], int size) volatile {
  serialize<int32_t>(cmd, *file_out_stream_);
  for (int i = 0; i < size; i++) {
    serialize<string>(values[i], *file_out_stream_);
    if (is_debugging) {
      printf("SocketClient sent CMD: %s with Param%d: %s\n",
             messageTypeNames[cmd], i + 1, values[i].c_str());
    }
  }
  file_out_stream_->flush();
}

/**
 * Wait for next event, which should be a response for
 * a previously sent command (expected_response_cmd)
 * and return the generic result
 */
template<class T>
T SocketClient::getResult(int32_t expected_response_cmd) volatile {
  
  T result = T();
  
  // read response command
  int32_t cmd = deserialize<int32_t>(*file_in_stream_);
  
  // check if response is expected
  if (expected_response_cmd == cmd) {
    
    result = deserialize<T>(*file_in_stream_);
    return result;
    
  } else if ( (expected_response_cmd == HostDeviceInterface::GET_MSG) &&
             (cmd == HostDeviceInterface::END_OF_DATA)) {
    
    return result; // default constructor (empty string, int 0)
    
  } else { // Not expected response
    
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
    printf("SocketClient.getResult - Unknown binary command: %d\n", cmd);
    HADOOP_ASSERT(false, "Unknown binary command " + toString(cmd));
  }
  return result;
}

/**
 * Wait for next event, which should be a response for
 * a previously sent command (expected_response_cmd)
 * and return the generic vector result list
 */
template<class T>
vector<T> SocketClient::getVectorResult(int32_t expected_response_cmd) {
  
  vector<T> results;
  
  // read response command
  int32_t cmd = deserialize<int32_t>(*file_in_stream_);
  
  // check if response is expected
  if (expected_response_cmd == cmd) {
    
    switch (cmd) {
      case HostDeviceInterface::GET_ALL_PEERNAME: {
        vector<T> peernames;
        T peername;
        int32_t peername_count = deserialize<int32_t>(*file_in_stream_);
        if (is_debugging) {
          printf("SocketClient.getVectorResult peername_count: %d\n",
                 peername_count);
        }
        
        for (int i=0; i<peername_count; i++)  {
          peername = deserialize<T>(*file_in_stream_);
          peernames.push_back(peername);
          if (is_debugging) {
            printf("SocketClient.getVectorResult peername: '%s'\n",
                   toString<T>(peername).c_str());
          }
        }
        return peernames;
      }
    }
  } else {
    printf("SocketClient.getVectorResult(expected_cmd = %d) - Unknown binary command: %d\n",
           expected_response_cmd, cmd);
    HADOOP_ASSERT(false, "Unknown binary command " + toString(cmd));
  }
  return results;
}

/**
 * Wait for next event, which should be a response for
 * a previously sent command (expected_response_cmd)
 * and return the generic KeyValuePair or an empty one
 * if no data is available
 */
template <class K, class V>
KeyValuePair<K,V> SocketClient::getKeyValueResult(int32_t expected_response_cmd,
                                    bool use_key, bool use_value) {
  
  KeyValuePair<K,V> key_value_pair;
  
  // read response command
  int32_t cmd = deserialize<int32_t>(*file_in_stream_);
  
  // check if response is expected or END_OF_DATA
  if ((expected_response_cmd == cmd) || (cmd == HostDeviceInterface::END_OF_DATA) ) {
    
    switch (cmd) {
      
      case HostDeviceInterface::READ_KEYVALUE:
      case HostDeviceInterface::SEQFILE_READNEXT: {
        K key;
        if (use_key) {
          key = deserialize<K>(*file_in_stream_);
        }
        V value;
        if (use_value) {
          value = deserialize<V>(*file_in_stream_);
        }
        key_value_pair = pair<K,V>(key, value);
        break;
      }
      case HostDeviceInterface::END_OF_DATA: {
        key_value_pair = KeyValuePair<K,V>(true);
        if (is_debugging) {
          printf("SocketClient.getKeyValueResult - got END_OF_DATA\n");
        }
      }
    }
  } else {
    key_value_pair = KeyValuePair<K,V>(true);
    printf("SocketClient.getKeyValueResult(expected_cmd = %d) - Unknown binary command: %d\n",
           expected_response_cmd, cmd);
    HADOOP_ASSERT(false, "Unknown binary command " + toString(cmd));
  }
  return key_value_pair;
}
