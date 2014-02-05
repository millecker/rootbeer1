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

#ifndef HOST_MONITOR_H
#define HOST_MONITOR_H

#include "HostDeviceInterface.h"
#include "SocketClient.h"

#include <pthread.h>

class HostMonitor {
private:
  pthread_t monitor_thread_;
  pthread_mutex_t mutex_process_command_;
  SocketClient *socket_client_;
  
  static void* thread(void *context);
  void processCommand() volatile;

public:
  volatile bool is_monitoring;
  volatile HostDeviceInterface *host_device_interface;

  HostMonitor(HostDeviceInterface *h_d_interface, int port, bool is_debugging);
  ~HostMonitor();

  void reset() volatile;
  void startMonitoring();
  void stopMonitoring();
};

#endif
