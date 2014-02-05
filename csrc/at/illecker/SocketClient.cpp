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

