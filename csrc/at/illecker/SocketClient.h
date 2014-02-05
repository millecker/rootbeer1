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

#include <vector>

using std::pair;
using std::vector;

using namespace HadoopUtils;

#define stringify( name ) # name

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
  bool sendCMD(int32_t cmd, bool verify_response, T value) volatile;
  
  template<class T1, class T2>
  bool sendCMD(int32_t cmd, bool verify_response, T1 value1, T2 value2) volatile;
  
  template<class T1, class T2, class T3>
  bool sendCMD(int32_t cmd, bool verify_response, T1 value1, T2 value2, T3 value3) volatile;
  
  template<class T>
  T getResult(int32_t expected_response_cmd) volatile;
  
  template<class T>
  vector<T> getVectorResult(int32_t expected_response_cmd);
  
  template <class K, class V>
  KeyValuePair<K,V> getKeyValueResult(int32_t expected_response_cmd,
                                      bool use_key, bool use_value);
};

#endif
