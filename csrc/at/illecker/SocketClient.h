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

#ifndef SOCKET_CLIENT_H
#define SOCKET_CLIENT_H

#include "HadoopUtils.h"
#include "HostDeviceInterface.h"

#include <errno.h>
#include <netinet/in.h>
#include <vector>

using std::pair;
using std::vector;

using namespace HadoopUtils;

/**
 * Generic KeyValuePair including is_empty
 */
template <typename K, typename V>
struct KeyValuePair : pair<K, V> {
  typedef pair<K, V> base_t;
  bool is_empty;
  
  KeyValuePair() : is_empty(false) {}
  explicit KeyValuePair(bool x) : is_empty(x) {}
  KeyValuePair(const K& k, const V& v) : base_t(k, v), is_empty(false) {}
  
  template <class X, class Y>
  KeyValuePair(const pair<X,Y> &p) : base_t(p), is_empty(false) {}
  
  template <class X, class Y>
  KeyValuePair(const KeyValuePair<X,Y> &p) : base_t(p), is_empty(p.is_empty) {}
};

/**
 * Override Generic KeyValuePair << operator
 */
template <typename OS, typename K, typename V>
OS &operator<<(OS &os, const KeyValuePair<K, V>& p) {
  os << "<KeyValuePair: ";
  if (!p.is_empty) {
    os << p.first << ", " << p.second;
  } else {
    os << "empty";
  }
  os << ">";
  return os;
}

class SocketClient {
private:
  int sock_;
  FILE* in_stream_;
  FILE* out_stream_;
  FileInStream* file_in_stream_;
  FileOutStream* file_out_stream_;
  bool is_debugging;

public:
  SocketClient(bool is_debugging);
  ~SocketClient();

  void connectSocket(int port);
  
  bool sendCMD(int32_t cmd, bool verify_response) volatile;
  void sendCMD(int32_t cmd, const string values[], int size) volatile;
  
  template<class T>
  bool sendCMD(int32_t cmd, bool verify_response, T value) volatile {
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
  bool sendCMD(int32_t cmd, bool verify_response, T1 value1, T2 value2) volatile {
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
  bool sendCMD(int32_t cmd, bool verify_response, T1 value1, T2 value2, T3 value3) volatile {
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
  vector<T> getVectorResult(int32_t expected_response_cmd) {
    
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
  KeyValuePair<K,V> getKeyValueResult(int32_t expected_response_cmd,
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

};

#endif

