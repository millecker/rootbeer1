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

#include "HostMonitor.h"

HostMonitor::HostMonitor(HostDeviceInterface *h_d_interface, int port, bool is_debugging) {
  host_device_interface = h_d_interface;
  host_device_interface->is_debugging = is_debugging;
  
  is_monitoring = false;
  pthread_mutex_init(&mutex_process_command_, NULL);
  socket_client_ = new SocketClient(is_debugging);
  
  // connect SocketClient
  socket_client_->connectSocket(port);
  
  reset();
  
  if (host_device_interface->is_debugging) {
    printf("HostMonitor init finished...\n");
  }
}

HostMonitor::~HostMonitor() {
  pthread_mutex_destroy(&mutex_process_command_);
}

void HostMonitor::reset() volatile {
  host_device_interface->command = HostDeviceInterface::UNDEFINED;
  host_device_interface->has_task = false;
  if (host_device_interface->is_debugging) {
    printf("HostMonitor reset lock_thread_id: %d, has_task: %s, result_available: %s\n",
           host_device_interface->lock_thread_id,
           (host_device_interface->has_task) ? "true" : "false",
           (host_device_interface->is_result_available) ? "true" : "false");
  }
}

void HostMonitor::startMonitoring() {
  if ( (host_device_interface != NULL) && (!is_monitoring) ) {
    if (host_device_interface->is_debugging) {
      printf("HostMonitor.startMonitoring...\n");
    }
    pthread_create(&monitor_thread_, NULL, &HostMonitor::thread, this);
    
    // wait for monitoring
    //while (!is_monitoring) {
    //  printf("HostMonitor.startMonitoring is_monitoring: %s\n",
    //    (is_monitoring) ? "true" : "false");
    //}
    
    if (host_device_interface->is_debugging) {
      printf("HostMonitor.startMonitoring started thread! is_monitoring: %s\n",
             (is_monitoring) ? "true" : "false");
    }
  }
}

void HostMonitor::stopMonitoring() {
  if ( (host_device_interface != NULL) && (is_monitoring) ) {
    if (host_device_interface->is_debugging) {
      printf("HostMonitor.stopMonitoring...\n");
    }
    
    host_device_interface->done = true;
    
    // wait for monitoring to end
    //while (is_monitoring) {
    //  printf("HostMonitor.stopMonitoring is_monitoring: %s\n",
    //    (is_monitoring) ? "true" : "false");
    //}
    
    if (host_device_interface->is_debugging) {
      printf("HostMonitor.stopMonitoring stopped! done: %s\n",
             (host_device_interface->done) ? "true" : "false");
    }
  }
}

void* HostMonitor::thread(void *context) {
  volatile HostMonitor *_this = ((HostMonitor *) context);
  
  if (_this->host_device_interface->is_debugging) {
    printf("HostMonitorThread started... done: %s\n",
           (_this->host_device_interface->done) ? "true" : "false");
    fflush(stdout);
  }
  
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
      
      if (_this->host_device_interface->is_debugging) {
        printf("HostMonitor thread: %p, LOCKED(mutex_process_command)\n", pthread_self());
      }
      
      _this->processCommand();
      
      _this->reset();
      
      pthread_mutex_unlock(lock);
      
      if (_this->host_device_interface->is_debugging) {
        printf("HostMonitor thread: %p, UNLOCKED(mutex_process_command)\n", pthread_self());
        fflush(stdout);
      }
    }
  }
  _this->is_monitoring = false;
  return NULL;
}

void HostMonitor::processCommand() volatile {
  
  if (host_device_interface->is_debugging) {
    printf("HostMonitor processCommand: %s, lock_thread_id: %d, result_available: %s\n",
           messageTypeNames[host_device_interface->command],
           host_device_interface->lock_thread_id,
           (host_device_interface->is_result_available) ? "true" : "false");
  }
  
  switch (host_device_interface->command) {
    
    /***********************************************************************/
    case HostDeviceInterface::SEND_MSG: {
      
      bool response = false;
      if (host_device_interface->use_str_val2) {
        response = socket_client_->sendCMD(HostDeviceInterface::SEND_MSG, true,
                                           string(const_cast<char *>(host_device_interface->str_val1)),
                                           string(const_cast<char *>(host_device_interface->str_val2)));
        
      } else if (host_device_interface->use_int_val1) {
        response = socket_client_->sendCMD(HostDeviceInterface::SEND_MSG, true,
                                           string(const_cast<char *>(host_device_interface->str_val1)),
                                           host_device_interface->int_val1);
        
      } else if (host_device_interface->use_long_val1) {
        response = socket_client_->sendCMD(HostDeviceInterface::SEND_MSG, true,
                                           string(const_cast<char *>(host_device_interface->str_val1)),
                                           host_device_interface->long_val1);
        
      } else if (host_device_interface->use_float_val1) {
        response = socket_client_->sendCMD(HostDeviceInterface::SEND_MSG, true,
                                           string(const_cast<char *>(host_device_interface->str_val1)),
                                           host_device_interface->float_val1);
        
      } else if (host_device_interface->use_double_val1) {
        response = socket_client_->sendCMD(HostDeviceInterface::SEND_MSG, true,
                                           string(const_cast<char *>(host_device_interface->str_val1)),
                                           host_device_interface->double_val1);
      }
      
      if (response == false) {
        // TODO throw CudaException?
        printf("HostDeviceInterface::SEND_MSG got wrong response command!\n");
      } else if (host_device_interface->is_debugging) {
        printf("HostDeviceInterface::SEND_MSG got response: 'true' \n");
      }
      
      // Set result available for GPU Kernel
      host_device_interface->is_result_available = true;
      if (host_device_interface->is_debugging) {
        printf("HostDeviceInterface::SEND_MSG is_result_available: '%s' \n",
               (host_device_interface->is_result_available) ? "true" : "false");
      }
      
      // block until result was consumed
      while (host_device_interface->is_result_available) {}
      if (host_device_interface->is_debugging) {
        printf("HostDeviceInterface::SEND_MSG result was consumed!\n");
      }
      break;
    }
    
    /***********************************************************************/
    case HostDeviceInterface::GET_MSG: {
      socket_client_->sendCMD(HostDeviceInterface::GET_MSG, false);
      
      // Check return type
      if (host_device_interface->return_type == HostDeviceInterface::INT) {
        host_device_interface->int_val1 = socket_client_->getResult<int32_t>(HostDeviceInterface::GET_MSG);
        if (host_device_interface->is_debugging) {
          printf("HostMonitor got result: '%d' \n", host_device_interface->int_val1);
        }
        
      } else if (host_device_interface->return_type == HostDeviceInterface::LONG) {
        host_device_interface->long_val1 = socket_client_->getResult<int64_t>(HostDeviceInterface::GET_MSG);
        if (host_device_interface->is_debugging) {
          printf("HostMonitor got result: '%lld' \n", host_device_interface->long_val1);
        }
        
      } else if (host_device_interface->return_type == HostDeviceInterface::FLOAT) {
        host_device_interface->float_val1 = socket_client_->getResult<float>(HostDeviceInterface::GET_MSG);
        if (host_device_interface->is_debugging) {
          printf("HostMonitor got result: '%f' \n", host_device_interface->float_val1);
        }
        
      } else if (host_device_interface->return_type == HostDeviceInterface::DOUBLE) {
        host_device_interface->double_val1 = socket_client_->getResult<double>(HostDeviceInterface::GET_MSG);
        if (host_device_interface->is_debugging) {
          printf("HostMonitor got result: '%f' \n", host_device_interface->double_val1);
        }
        
      } else if (host_device_interface->return_type == HostDeviceInterface::STRING) {
        string result = socket_client_->getResult<string>(HostDeviceInterface::GET_MSG);
        strcpy(const_cast<char *>(host_device_interface->str_val1), result.c_str());
        if (host_device_interface->is_debugging) {
          printf("HostMonitor got result: '%s' \n", host_device_interface->str_val1);
        }
      }
      
      // Set result available for GPU Kernel
      host_device_interface->is_result_available = true;
      
      // block until result was consumed
      while (host_device_interface->is_result_available) {}
      
      if (host_device_interface->is_debugging) {
        printf("HostMonitor result was consumed\n");
      }
      break;
    }
    
    /***********************************************************************/
    case HostDeviceInterface::GET_MSG_COUNT: {
      socket_client_->sendCMD(HostDeviceInterface::GET_MSG_COUNT, false);
      
      host_device_interface->int_val1 = socket_client_->getResult<int32_t>(HostDeviceInterface::GET_MSG_COUNT);
      // Set result available for GPU Kernel
      host_device_interface->is_result_available = true;
      
      if (host_device_interface->is_debugging) {
        printf("HostMonitor got result: %d result_available: %s\n",
               host_device_interface->int_val1,
               (host_device_interface->is_result_available) ? "true" : "false");
      }
      
      // block until result was consumed
      while (host_device_interface->is_result_available) {}
      
      if (host_device_interface->is_debugging) {
        printf("HostMonitor result was consumed\n");
      }
      break;
    }
    
    /***********************************************************************/
    case HostDeviceInterface::SYNC: {
      bool response = socket_client_->sendCMD(HostDeviceInterface::SYNC, true);
      
      if (host_device_interface->is_debugging) {
        printf("HostMonitor sent SYNC\n");
      }
      if (response == false) {
        // TODO throw CudaException?
        printf("HostDeviceInterface::SYNC got wrong response command!\n");
      }
      
      host_device_interface->int_val1 = 0;
      // Set result available for GPU Kernel
      host_device_interface->is_result_available = true;
      
      // block until result was consumed
      while (host_device_interface->is_result_available) {}
      break;
    }
    
    /***********************************************************************/
    case HostDeviceInterface::GET_SUPERSTEP_COUNT: {
      socket_client_->sendCMD(HostDeviceInterface::GET_SUPERSTEP_COUNT, false);
      
      host_device_interface->long_val1 = socket_client_->getResult<int64_t>(HostDeviceInterface::GET_SUPERSTEP_COUNT);
      // Set result available for GPU Kernel
      host_device_interface->is_result_available = true;
      
      if (host_device_interface->is_debugging) {
        printf("HostMonitor got result: %lld result_available: %s\n",
               host_device_interface->long_val1,
               (host_device_interface->is_result_available) ? "true" : "false");
      }
      
      // block until result was consumed
      while (host_device_interface->is_result_available) {}
      
      if (host_device_interface->is_debugging) {
        printf("HostMonitor result was consumed\n");
      }
      break;
    }
    
    /***********************************************************************/
    case HostDeviceInterface::GET_PEERNAME: {
      socket_client_->sendCMD(HostDeviceInterface::GET_PEERNAME, false, host_device_interface->int_val1);
      
      string result = socket_client_->getResult<string>(HostDeviceInterface::GET_PEERNAME);
      
      strcpy(const_cast<char *>(host_device_interface->str_val1), result.c_str());
      // Set result available for GPU Kernel
      host_device_interface->is_result_available = true;
      
      if (host_device_interface->is_debugging) {
        printf("HostMonitor got result: %s result_available: %s\n",
               host_device_interface->str_val1,
               (host_device_interface->is_result_available) ? "true" : "false");
      }
      
      // block until result was consumed
      while (host_device_interface->is_result_available) {}
      
      if (host_device_interface->is_debugging) {
        printf("HostMonitor result was consumed\n");
      }
      break;
    }
    
    /***********************************************************************/
    case HostDeviceInterface::GET_ALL_PEERNAME: {
      socket_client_->sendCMD(HostDeviceInterface::GET_ALL_PEERNAME, false);
      
      vector<string> results = socket_client_->getVectorResult<string>(HostDeviceInterface::GET_ALL_PEERNAME);
      
      // Set result available for GPU Kernel
      host_device_interface->int_val1 = results.size();
      host_device_interface->use_int_val1 = true;
      
      if (host_device_interface->is_debugging) {
        printf("HostMonitor got results.len: %d\n",
               host_device_interface->int_val1);
      }
      
      int index = 0;
      while (index < host_device_interface->int_val1) {
        
        if (index < host_device_interface->int_val1) {
          strcpy(const_cast<char *>(host_device_interface->str_val1), results.at(index).c_str());
          host_device_interface->use_str_val1 = true;
          index++;
        }
        
        if (index < host_device_interface->int_val1) {
          strcpy(const_cast<char *>(host_device_interface->str_val2), results.at(index).c_str());
          host_device_interface->use_str_val2 = true;
          index++;
        }
        
        if (index < host_device_interface->int_val1) {
          strcpy(const_cast<char *>(host_device_interface->str_val3), results.at(index).c_str());
          host_device_interface->use_str_val3 = true;
          index++;
        }
        
        host_device_interface->is_result_available = true;
        
        // block until result was consumed
        while (host_device_interface->is_result_available) {}
      }
      
      host_device_interface->use_int_val1 = false;
      host_device_interface->is_result_available = true;
      
      // block until result was consumed
      while (host_device_interface->is_result_available) {}
      
      if (host_device_interface->is_debugging) {
        printf("HostMonitor all results were consumed\n");
      }
      break;
    }
    
    /***********************************************************************/
    case HostDeviceInterface::GET_PEER_INDEX: {
      socket_client_->sendCMD(HostDeviceInterface::GET_PEER_INDEX, false);
      
      host_device_interface->int_val1 = socket_client_->getResult<int32_t>(HostDeviceInterface::GET_PEER_INDEX);
      // Set result available for GPU Kernel
      host_device_interface->is_result_available = true;
      
      if (host_device_interface->is_debugging) {
        printf("HostMonitor got result: %d result_available: %s\n",
               host_device_interface->int_val1,
               (host_device_interface->is_result_available) ? "true" : "false");
      }
      
      // block until result was consumed
      while (host_device_interface->is_result_available) {}
      
      if (host_device_interface->is_debugging) {
        printf("HostMonitor result was consumed\n");
      }
      break;
    }
    
    /***********************************************************************/
    case HostDeviceInterface::GET_PEER_COUNT: {
      socket_client_->sendCMD(HostDeviceInterface::GET_PEER_COUNT, false);
      
      host_device_interface->int_val1 = socket_client_->getResult<int32_t>(HostDeviceInterface::GET_PEER_COUNT);
      // Set result available for GPU Kernel
      host_device_interface->is_result_available = true;
      
      if (host_device_interface->is_debugging) {
        printf("HostMonitor got result: %d result_available: %s\n",
               host_device_interface->int_val1,
               (host_device_interface->is_result_available) ? "true" : "false");
      }
      
      // block until result was consumed
      while (host_device_interface->is_result_available) {}
      
      if (host_device_interface->is_debugging) {
        printf("HostMonitor result was consumed\n");
      }
      break;
    }
    
    /***********************************************************************/
    case HostDeviceInterface::CLEAR: {
      bool response = socket_client_->sendCMD(HostDeviceInterface::CLEAR, true);
      
      if (host_device_interface->is_debugging) {
        printf("HostMonitor sent CLEAR\n");
      }
      
      if (response == false) {
        // TODO throw CudaException?
        printf("HostDeviceInterface::CLEAR got wrong response command!\n");
      }
      
      host_device_interface->int_val1 = 0;
      // Set result available for GPU Kernel
      host_device_interface->is_result_available = true;
      
      // block until result was consumed
      while (host_device_interface->is_result_available) {}
      break;
    }
    
    /***********************************************************************/
    case HostDeviceInterface::REOPEN_INPUT: {
      bool response = socket_client_->sendCMD(HostDeviceInterface::REOPEN_INPUT, true);
      
      if (host_device_interface->is_debugging) {
        printf("HostMonitor sent REOPEN_INPUT\n");
      }
      
      if (response == false) {
        // TODO throw CudaException?
        printf("HostDeviceInterface::REOPEN_INPUT got wrong response command!\n");
      }
      
      host_device_interface->int_val1 = 0;
      // Set result available for GPU Kernel
      host_device_interface->is_result_available = true;
      
      // block until result was consumed
      while (host_device_interface->is_result_available) {}
      break;
    }
    
    /***********************************************************************/
    case HostDeviceInterface::READ_KEYVALUE:
    case HostDeviceInterface::SEQFILE_READNEXT: {
      if (host_device_interface->command == HostDeviceInterface::READ_KEYVALUE) {
        socket_client_->sendCMD(HostDeviceInterface::READ_KEYVALUE, false);
      } else {
        socket_client_->sendCMD(HostDeviceInterface::SEQFILE_READNEXT, false,
                                host_device_interface->int_val1); // file_id
      }
      
      // Check key and value type
      // Variation with repetition (n=6,k=2) -> 6^2 - 1 (null,null) variations
      /***********************************************************************/
      // (null,int)
      if ( (host_device_interface->key_type == HostDeviceInterface::NULL_TYPE) &&
          (host_device_interface->value_type == HostDeviceInterface::INT) ) {
        KeyValuePair<int32_t,int32_t> key_value_pair = socket_client_->getKeyValueResult<int32_t,int32_t>(
                                                                                                          host_device_interface->command, false, true);
        if (!key_value_pair.is_empty) {
          host_device_interface->end_of_data = false;
          host_device_interface->int_val2 = key_value_pair.second;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key: 'NULL' value: '%d' \n", host_device_interface->int_val2);
          }
        } else {
          host_device_interface->end_of_data = true;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key_value_pair is empty!\n");
          }
        }
        // (null,long)
      } else if ( (host_device_interface->key_type == HostDeviceInterface::NULL_TYPE) &&
                 (host_device_interface->value_type == HostDeviceInterface::LONG) ) {
        KeyValuePair<int32_t,int64_t> key_value_pair = socket_client_->getKeyValueResult<int32_t,int64_t>(
                                                                                                          host_device_interface->command, false, true);
        if (!key_value_pair.is_empty) {
          host_device_interface->end_of_data = false;
          host_device_interface->long_val2 = key_value_pair.second;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key: 'NULL' value: '%lld' \n", host_device_interface->long_val2);
          }
        } else {
          host_device_interface->end_of_data = true;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key_value_pair is empty!\n");
          }
        }
        // (null,float)
      } else if ( (host_device_interface->key_type == HostDeviceInterface::NULL_TYPE) &&
                 (host_device_interface->value_type == HostDeviceInterface::FLOAT) ) {
        KeyValuePair<int32_t,float> key_value_pair = socket_client_->getKeyValueResult<int32_t,float>(
                                                                                                      host_device_interface->command, false, true);
        if (!key_value_pair.is_empty) {
          host_device_interface->end_of_data = false;
          host_device_interface->float_val2 = key_value_pair.second;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key: 'NULL' value: '%f' \n", host_device_interface->float_val2);
          }
        } else {
          host_device_interface->end_of_data = true;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key_value_pair is empty!\n");
          }
        }
        // (null,double)
      } else if ( (host_device_interface->key_type == HostDeviceInterface::NULL_TYPE) &&
                 (host_device_interface->value_type == HostDeviceInterface::DOUBLE) ) {
        KeyValuePair<int32_t,double> key_value_pair = socket_client_->getKeyValueResult<int32_t,double>(
                                                                                                        host_device_interface->command, false, true);
        if (!key_value_pair.is_empty) {
          host_device_interface->end_of_data = false;
          host_device_interface->double_val2 = key_value_pair.second;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key: 'NULL' value: '%f' \n", host_device_interface->double_val2);
          }
        } else {
          host_device_interface->end_of_data = true;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key_value_pair is empty!\n");
          }
        }
        // (null,string)
      } else if ( (host_device_interface->key_type == HostDeviceInterface::NULL_TYPE) &&
                 (host_device_interface->value_type == HostDeviceInterface::STRING) ) {
        KeyValuePair<int32_t,string> key_value_pair = socket_client_->getKeyValueResult<int32_t,string>(
                                                                                                        host_device_interface->command, false, true);
        if (!key_value_pair.is_empty) {
          host_device_interface->end_of_data = false;
          strcpy(const_cast<char *>(host_device_interface->str_val2), key_value_pair.second.c_str());
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key: 'NULL' value: '%s' \n", host_device_interface->str_val2);
          }
        } else {
          host_device_interface->end_of_data = true;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key_value_pair is empty!\n");
          }
        }
      }
      /***********************************************************************/
      // (int,null)
      else if ( (host_device_interface->key_type == HostDeviceInterface::INT) &&
               (host_device_interface->value_type == HostDeviceInterface::NULL_TYPE) ) {
        KeyValuePair<int32_t,int32_t> key_value_pair = socket_client_->getKeyValueResult<int32_t,int32_t>(
                                                                                                          host_device_interface->command, true, false);
        if (!key_value_pair.is_empty) {
          host_device_interface->end_of_data = false;
          host_device_interface->int_val1 = key_value_pair.first;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key: '%d' value: 'NULL' \n", host_device_interface->int_val1);
          }
        } else {
          host_device_interface->end_of_data = true;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key_value_pair is empty!\n");
          }
        }
        // (int,int)
      } else if ( (host_device_interface->key_type == HostDeviceInterface::INT) &&
                 (host_device_interface->value_type == HostDeviceInterface::INT) ) {
        KeyValuePair<int32_t,int32_t> key_value_pair = socket_client_->getKeyValueResult<int32_t,int32_t>(
                                                                                                          host_device_interface->command, true, true);
        if (!key_value_pair.is_empty) {
          host_device_interface->end_of_data = false;
          host_device_interface->int_val1 = key_value_pair.first;
          host_device_interface->int_val2 = key_value_pair.second;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key: '%d' value: '%d' \n", host_device_interface->int_val1,
                   host_device_interface->int_val2);
          }
        } else {
          host_device_interface->end_of_data = true;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key_value_pair is empty!\n");
          }
        }
        // (int,long)
      } else if ( (host_device_interface->key_type == HostDeviceInterface::INT) &&
                 (host_device_interface->value_type == HostDeviceInterface::LONG) ) {
        KeyValuePair<int32_t,int64_t> key_value_pair = socket_client_->getKeyValueResult<int32_t,int64_t>(
                                                                                                          host_device_interface->command, true, true);
        if (!key_value_pair.is_empty) {
          host_device_interface->end_of_data = false;
          host_device_interface->int_val1 = key_value_pair.first;
          host_device_interface->long_val2 = key_value_pair.second;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key: '%d' value: '%lld' \n", host_device_interface->int_val1,
                   host_device_interface->long_val2);
          }
        } else {
          host_device_interface->end_of_data = true;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key_value_pair is empty!\n");
          }
        }
        // (int,float)
      } else if ( (host_device_interface->key_type == HostDeviceInterface::INT) &&
                 (host_device_interface->value_type == HostDeviceInterface::FLOAT) ) {
        KeyValuePair<int32_t,float> key_value_pair = socket_client_->getKeyValueResult<int32_t,float>(
                                                                                                      host_device_interface->command, true, true);
        if (!key_value_pair.is_empty) {
          host_device_interface->end_of_data = false;
          host_device_interface->int_val1 = key_value_pair.first;
          host_device_interface->float_val2 = key_value_pair.second;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key: '%d' value: '%f' \n", host_device_interface->int_val1,
                   host_device_interface->float_val2);
          }
        } else {
          host_device_interface->end_of_data = true;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key_value_pair is empty!\n");
          }
        }
        // (int,double)
      } else if ( (host_device_interface->key_type == HostDeviceInterface::INT) &&
                 (host_device_interface->value_type == HostDeviceInterface::DOUBLE) ) {
        KeyValuePair<int32_t,double> key_value_pair = socket_client_->getKeyValueResult<int32_t,double>(
                                                                                                        host_device_interface->command, true, true);
        if (!key_value_pair.is_empty) {
          host_device_interface->end_of_data = false;
          host_device_interface->int_val1 = key_value_pair.first;
          host_device_interface->double_val2 = key_value_pair.second;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key: '%d' value: '%f' \n", host_device_interface->int_val1,
                   host_device_interface->double_val2);
          }
        } else {
          host_device_interface->end_of_data = true;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key_value_pair is empty!\n");
          }
        }
        // (int,string)
      } else if ( (host_device_interface->key_type == HostDeviceInterface::INT) &&
                 (host_device_interface->value_type == HostDeviceInterface::STRING) ) {
        KeyValuePair<int32_t,string> key_value_pair = socket_client_->getKeyValueResult<int32_t,string>(
                                                                                                        host_device_interface->command, true, true);
        if (!key_value_pair.is_empty) {
          host_device_interface->end_of_data = false;
          host_device_interface->int_val1 = key_value_pair.first;
          strcpy(const_cast<char *>(host_device_interface->str_val2), key_value_pair.second.c_str());
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key: '%d' value: '%s' \n", host_device_interface->int_val1,
                   host_device_interface->str_val2);
          }
        } else {
          host_device_interface->end_of_data = true;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key_value_pair is empty!\n");
          }
        }
      }
      /***********************************************************************/
      // (long,null)
      else if ( (host_device_interface->key_type == HostDeviceInterface::LONG) &&
               (host_device_interface->value_type == HostDeviceInterface::NULL_TYPE) ) {
        KeyValuePair<int64_t,int32_t> key_value_pair = socket_client_->getKeyValueResult<int64_t,int32_t>(
                                                                                                          host_device_interface->command, true, false);
        if (!key_value_pair.is_empty) {
          host_device_interface->end_of_data = false;
          host_device_interface->long_val1 = key_value_pair.first;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key: '%lld' value: 'NULL' \n", host_device_interface->long_val1);
          }
        } else {
          host_device_interface->end_of_data = true;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key_value_pair is empty!\n");
          }
        }
        // (long,int)
      } else if ( (host_device_interface->key_type == HostDeviceInterface::LONG) &&
                 (host_device_interface->value_type == HostDeviceInterface::INT) ) {
        KeyValuePair<int64_t,int32_t> key_value_pair = socket_client_->getKeyValueResult<int64_t,int32_t>(
                                                                                                          host_device_interface->command, true, true);
        if (!key_value_pair.is_empty) {
          host_device_interface->end_of_data = false;
          host_device_interface->long_val1 = key_value_pair.first;
          host_device_interface->int_val2 = key_value_pair.second;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key: '%lld' value: '%d' \n", host_device_interface->long_val1,
                   host_device_interface->int_val2);
          }
        } else {
          host_device_interface->end_of_data = true;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key_value_pair is empty!\n");
          }
        }
        // (long,long)
      } else if ( (host_device_interface->key_type == HostDeviceInterface::LONG) &&
                 (host_device_interface->value_type == HostDeviceInterface::LONG) ) {
        KeyValuePair<int64_t,int64_t> key_value_pair = socket_client_->getKeyValueResult<int64_t,int64_t>(
                                                                                                          host_device_interface->command, true, true);
        if (!key_value_pair.is_empty) {
          host_device_interface->end_of_data = false;
          host_device_interface->long_val1 = key_value_pair.first;
          host_device_interface->long_val2 = key_value_pair.second;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key: '%lld' value: '%lld' \n", host_device_interface->long_val1,
                   host_device_interface->long_val2);
          }
        } else {
          host_device_interface->end_of_data = true;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key_value_pair is empty!\n");
          }
        }
        // (long,float)
      } else if ( (host_device_interface->key_type == HostDeviceInterface::LONG) &&
                 (host_device_interface->value_type == HostDeviceInterface::FLOAT) ) {
        KeyValuePair<int64_t,float> key_value_pair = socket_client_->getKeyValueResult<int64_t,float>(
                                                                                                      host_device_interface->command, true, true);
        if (!key_value_pair.is_empty) {
          host_device_interface->end_of_data = false;
          host_device_interface->long_val1 = key_value_pair.first;
          host_device_interface->float_val2 = key_value_pair.second;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key: '%lld' value: '%f' \n", host_device_interface->long_val1,
                   host_device_interface->float_val2);
          }
        } else {
          host_device_interface->end_of_data = true;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key_value_pair is empty!\n");
          }
        }
        // (long,double)
      } else if ( (host_device_interface->key_type == HostDeviceInterface::LONG) &&
                 (host_device_interface->value_type == HostDeviceInterface::DOUBLE) ) {
        KeyValuePair<int64_t,double> key_value_pair = socket_client_->getKeyValueResult<int64_t,double>(
                                                                                                        host_device_interface->command, true, true);
        if (!key_value_pair.is_empty) {
          host_device_interface->end_of_data = false;
          host_device_interface->long_val1 = key_value_pair.first;
          host_device_interface->double_val2 = key_value_pair.second;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key: '%lld' value: '%f' \n", host_device_interface->long_val1,
                   host_device_interface->double_val2);
          }
        } else {
          host_device_interface->end_of_data = true;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key_value_pair is empty!\n");
          }
        }
        // (long,string)
      } else if ( (host_device_interface->key_type == HostDeviceInterface::LONG) &&
                 (host_device_interface->value_type == HostDeviceInterface::STRING) ) {
        KeyValuePair<int64_t,string> key_value_pair = socket_client_->getKeyValueResult<int64_t,string>(
                                                                                                        host_device_interface->command, true, true);
        if (!key_value_pair.is_empty) {
          host_device_interface->end_of_data = false;
          host_device_interface->long_val1 = key_value_pair.first;
          strcpy(const_cast<char *>(host_device_interface->str_val2), key_value_pair.second.c_str());
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key: '%lld' value: '%s' \n", host_device_interface->long_val1,
                   host_device_interface->str_val2);
          }
        } else {
          host_device_interface->end_of_data = true;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key_value_pair is empty!\n");
          }
        }
      }
      /***********************************************************************/
      // (float,null)
      else if ( (host_device_interface->key_type == HostDeviceInterface::FLOAT) &&
               (host_device_interface->value_type == HostDeviceInterface::NULL_TYPE) ) {
        KeyValuePair<float,int32_t> key_value_pair = socket_client_->getKeyValueResult<float,int32_t>(
                                                                                                      host_device_interface->command, true, false);
        if (!key_value_pair.is_empty) {
          host_device_interface->end_of_data = false;
          host_device_interface->float_val1 = key_value_pair.first;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key: '%f' value: 'NULL' \n", host_device_interface->float_val1);
          }
        } else {
          host_device_interface->end_of_data = true;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key_value_pair is empty!\n");
          }
        }
        // (float,int)
      } else if ( (host_device_interface->key_type == HostDeviceInterface::FLOAT) &&
                 (host_device_interface->value_type == HostDeviceInterface::INT) ) {
        KeyValuePair<float,int32_t> key_value_pair = socket_client_->getKeyValueResult<float,int32_t>(
                                                                                                      host_device_interface->command, true, true);
        if (!key_value_pair.is_empty) {
          host_device_interface->end_of_data = false;
          host_device_interface->float_val1 = key_value_pair.first;
          host_device_interface->int_val2 = key_value_pair.second;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key: '%f' value: '%d' \n", host_device_interface->float_val1,
                   host_device_interface->int_val2);
          }
        } else {
          host_device_interface->end_of_data = true;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key_value_pair is empty!\n");
          }
        }
        // (float,long)
      } else if ( (host_device_interface->key_type == HostDeviceInterface::FLOAT) &&
                 (host_device_interface->value_type == HostDeviceInterface::LONG) ) {
        KeyValuePair<float,int64_t> key_value_pair = socket_client_->getKeyValueResult<float,int64_t>(
                                                                                                      host_device_interface->command, true, true);
        if (!key_value_pair.is_empty) {
          host_device_interface->end_of_data = false;
          host_device_interface->float_val1 = key_value_pair.first;
          host_device_interface->long_val2 = key_value_pair.second;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key: '%f' value: '%lld' \n", host_device_interface->float_val1,
                   host_device_interface->long_val2);
          }
        } else {
          host_device_interface->end_of_data = true;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key_value_pair is empty!\n");
          }
        }
        // (float,float)
      } else if ( (host_device_interface->key_type == HostDeviceInterface::FLOAT) &&
                 (host_device_interface->value_type == HostDeviceInterface::FLOAT) ) {
        KeyValuePair<float,float> key_value_pair = socket_client_->getKeyValueResult<float,float>(
                                                                                                  host_device_interface->command, true, true);
        if (!key_value_pair.is_empty) {
          host_device_interface->end_of_data = false;
          host_device_interface->float_val1 = key_value_pair.first;
          host_device_interface->float_val2 = key_value_pair.second;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key: '%f' value: '%f' \n", host_device_interface->float_val1,
                   host_device_interface->float_val2);
          }
        } else {
          host_device_interface->end_of_data = true;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key_value_pair is empty!\n");
          }
        }
        // (float,double)
      } else if ( (host_device_interface->key_type == HostDeviceInterface::FLOAT) &&
                 (host_device_interface->value_type == HostDeviceInterface::DOUBLE) ) {
        KeyValuePair<float,double> key_value_pair = socket_client_->getKeyValueResult<float,double>(
                                                                                                    host_device_interface->command, true, true);
        if (!key_value_pair.is_empty) {
          host_device_interface->end_of_data = false;
          host_device_interface->float_val1 = key_value_pair.first;
          host_device_interface->double_val2 = key_value_pair.second;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key: '%f' value: '%f' \n", host_device_interface->float_val1,
                   host_device_interface->double_val2);
          }
        } else {
          host_device_interface->end_of_data = true;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key_value_pair is empty!\n");
          }
        }
        // (float,string)
      } else if ( (host_device_interface->key_type == HostDeviceInterface::FLOAT) &&
                 (host_device_interface->value_type == HostDeviceInterface::STRING) ) {
        KeyValuePair<float,string> key_value_pair = socket_client_->getKeyValueResult<float,string>(
                                                                                                    host_device_interface->command, true, true);
        if (!key_value_pair.is_empty) {
          host_device_interface->end_of_data = false;
          host_device_interface->float_val1 = key_value_pair.first;
          strcpy(const_cast<char *>(host_device_interface->str_val2), key_value_pair.second.c_str());
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key: '%f' value: '%s' \n", host_device_interface->float_val1,
                   host_device_interface->str_val2);
          }
        } else {
          host_device_interface->end_of_data = true;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key_value_pair is empty!\n");
          }
        }
      }
      /***********************************************************************/
      // (double,null)
      else if ( (host_device_interface->key_type == HostDeviceInterface::DOUBLE) &&
               (host_device_interface->value_type == HostDeviceInterface::NULL_TYPE) ) {
        KeyValuePair<double,int32_t> key_value_pair = socket_client_->getKeyValueResult<double,int32_t>(
                                                                                                        host_device_interface->command, true, false);
        if (!key_value_pair.is_empty) {
          host_device_interface->end_of_data = false;
          host_device_interface->double_val1 = key_value_pair.first;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key: '%f' value: 'NULL' \n", host_device_interface->double_val1);
          }
        } else {
          host_device_interface->end_of_data = true;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key_value_pair is empty!\n");
          }
        }
        // (double,int)
      } else if ( (host_device_interface->key_type == HostDeviceInterface::DOUBLE) &&
                 (host_device_interface->value_type == HostDeviceInterface::INT) ) {
        KeyValuePair<double,int32_t> key_value_pair = socket_client_->getKeyValueResult<double,int32_t>(
                                                                                                        host_device_interface->command, true, true);
        if (!key_value_pair.is_empty) {
          host_device_interface->end_of_data = false;
          host_device_interface->double_val1 = key_value_pair.first;
          host_device_interface->int_val2 = key_value_pair.second;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key: '%f' value: '%d' \n", host_device_interface->double_val1,
                   host_device_interface->int_val2);
          }
        } else {
          host_device_interface->end_of_data = true;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key_value_pair is empty!\n");
          }
        }
        // (double,long)
      } else if ( (host_device_interface->key_type == HostDeviceInterface::DOUBLE) &&
                 (host_device_interface->value_type == HostDeviceInterface::LONG) ) {
        KeyValuePair<double,int64_t> key_value_pair = socket_client_->getKeyValueResult<double,int64_t>(
                                                                                                        host_device_interface->command, true, true);
        if (!key_value_pair.is_empty) {
          host_device_interface->end_of_data = false;
          host_device_interface->double_val1 = key_value_pair.first;
          host_device_interface->long_val2 = key_value_pair.second;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key: '%f' value: '%lld' \n", host_device_interface->double_val1,
                   host_device_interface->long_val2);
          }
        } else {
          host_device_interface->end_of_data = true;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key_value_pair is empty!\n");
          }
        }
        // (double,float)
      } else if ( (host_device_interface->key_type == HostDeviceInterface::DOUBLE) &&
                 (host_device_interface->value_type == HostDeviceInterface::FLOAT) ) {
        KeyValuePair<double,float> key_value_pair = socket_client_->getKeyValueResult<double,float>(
                                                                                                    host_device_interface->command, true, true);
        if (!key_value_pair.is_empty) {
          host_device_interface->end_of_data = false;
          host_device_interface->double_val1 = key_value_pair.first;
          host_device_interface->float_val2 = key_value_pair.second;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key: '%f' value: '%f' \n", host_device_interface->double_val1,
                   host_device_interface->float_val2);
          }
        } else {
          host_device_interface->end_of_data = true;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key_value_pair is empty!\n");
          }
        }
        // (double,double)
      } else if ( (host_device_interface->key_type == HostDeviceInterface::DOUBLE) &&
                 (host_device_interface->value_type == HostDeviceInterface::DOUBLE) ) {
        KeyValuePair<double,double> key_value_pair = socket_client_->getKeyValueResult<double,double>(
                                                                                                      host_device_interface->command, true, true);
        if (!key_value_pair.is_empty) {
          host_device_interface->end_of_data = false;
          host_device_interface->double_val1 = key_value_pair.first;
          host_device_interface->double_val2 = key_value_pair.second;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key: '%f' value: '%f' \n", host_device_interface->double_val1,
                   host_device_interface->double_val2);
          }
        } else {
          host_device_interface->end_of_data = true;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key_value_pair is empty!\n");
          }
        }
        // (double,string)
      } else if ( (host_device_interface->key_type == HostDeviceInterface::DOUBLE) &&
                 (host_device_interface->value_type == HostDeviceInterface::STRING) ) {
        KeyValuePair<double,string> key_value_pair = socket_client_->getKeyValueResult<double,string>(
                                                                                                      host_device_interface->command, true, true);
        if (!key_value_pair.is_empty) {
          host_device_interface->end_of_data = false;
          host_device_interface->double_val1 = key_value_pair.first;
          strcpy(const_cast<char *>(host_device_interface->str_val2), key_value_pair.second.c_str());
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key: '%f' value: '%s' \n", host_device_interface->double_val1,
                   host_device_interface->str_val2);
          }
        } else {
          host_device_interface->end_of_data = true;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key_value_pair is empty!\n");
          }
        }
      }
      /***********************************************************************/
      // (string,null)
      else if ( (host_device_interface->key_type == HostDeviceInterface::STRING) &&
               (host_device_interface->value_type == HostDeviceInterface::NULL_TYPE) ) {
        KeyValuePair<string,int32_t> key_value_pair = socket_client_->getKeyValueResult<string,int32_t>(
                                                                                                        host_device_interface->command, true, false);
        if (!key_value_pair.is_empty) {
          host_device_interface->end_of_data = false;
          strcpy(const_cast<char *>(host_device_interface->str_val1), key_value_pair.first.c_str());
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key: '%s' value: 'NULL' \n", host_device_interface->str_val1);
          }
        } else {
          host_device_interface->end_of_data = true;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key_value_pair is empty!\n");
          }
        }
        // (string,int)
      } else if ( (host_device_interface->key_type == HostDeviceInterface::STRING) &&
                 (host_device_interface->value_type == HostDeviceInterface::INT) ) {
        KeyValuePair<string,int32_t> key_value_pair = socket_client_->getKeyValueResult<string,int32_t>(
                                                                                                        host_device_interface->command, true, true);
        if (!key_value_pair.is_empty) {
          host_device_interface->end_of_data = false;
          strcpy(const_cast<char *>(host_device_interface->str_val1), key_value_pair.first.c_str());
          host_device_interface->int_val2 = key_value_pair.second;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key: '%s' value: '%d' \n", host_device_interface->str_val1,
                   host_device_interface->int_val2);
          }
        } else {
          host_device_interface->end_of_data = true;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key_value_pair is empty!\n");
          }
        }
        // (string,long)
      } else if ( (host_device_interface->key_type == HostDeviceInterface::STRING) &&
                 (host_device_interface->value_type == HostDeviceInterface::LONG) ) {
        KeyValuePair<string,int64_t> key_value_pair = socket_client_->getKeyValueResult<string,int64_t>(
                                                                                                        host_device_interface->command, true, true);
        if (!key_value_pair.is_empty) {
          host_device_interface->end_of_data = false;
          strcpy(const_cast<char *>(host_device_interface->str_val1), key_value_pair.first.c_str());
          host_device_interface->long_val2 = key_value_pair.second;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key: '%s' value: '%lld' \n", host_device_interface->str_val1,
                   host_device_interface->long_val2);
          }
        } else {
          host_device_interface->end_of_data = true;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key_value_pair is empty!\n");
          }
        }
        // (string,float)
      } else if ( (host_device_interface->key_type == HostDeviceInterface::STRING) &&
                 (host_device_interface->value_type == HostDeviceInterface::FLOAT) ) {
        KeyValuePair<string,float> key_value_pair = socket_client_->getKeyValueResult<string,float>(
                                                                                                    host_device_interface->command, true, true);
        if (!key_value_pair.is_empty) {
          host_device_interface->end_of_data = false;
          strcpy(const_cast<char *>(host_device_interface->str_val1), key_value_pair.first.c_str());
          host_device_interface->float_val2 = key_value_pair.second;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key: '%s' value: '%f' \n", host_device_interface->str_val1,
                   host_device_interface->float_val2);
          }
        } else {
          host_device_interface->end_of_data = true;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key_value_pair is empty!\n");
          }
        }
        // (string,double)
      } else if ( (host_device_interface->key_type == HostDeviceInterface::STRING) &&
                 (host_device_interface->value_type == HostDeviceInterface::DOUBLE) ) {
        KeyValuePair<string,double> key_value_pair = socket_client_->getKeyValueResult<string,double>(
                                                                                                      host_device_interface->command, true, true);
        if (!key_value_pair.is_empty) {
          host_device_interface->end_of_data = false;
          strcpy(const_cast<char *>(host_device_interface->str_val1), key_value_pair.first.c_str());
          host_device_interface->double_val2 = key_value_pair.second;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key: '%s' value: '%f' \n", host_device_interface->str_val1,
                   host_device_interface->double_val2);
          }
        } else {
          host_device_interface->end_of_data = true;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key_value_pair is empty!\n");
          }
        }
        // (string,string)
      } else if ( (host_device_interface->key_type == HostDeviceInterface::STRING) &&
                 (host_device_interface->value_type == HostDeviceInterface::STRING) ) {
        KeyValuePair<string,string> key_value_pair = socket_client_->getKeyValueResult<string,string>(
                                                                                                      host_device_interface->command, true, true);
        if (!key_value_pair.is_empty) {
          host_device_interface->end_of_data = false;
          strcpy(const_cast<char *>(host_device_interface->str_val1), key_value_pair.first.c_str());
          strcpy(const_cast<char *>(host_device_interface->str_val2), key_value_pair.second.c_str());
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key: '%s' value: '%s' \n", host_device_interface->str_val1,
                   host_device_interface->str_val2);
          }
        } else {
          host_device_interface->end_of_data = true;
          if (host_device_interface->is_debugging) {
            printf("HostMonitor got key_value_pair is empty!\n");
          }
        }
      }
      /***********************************************************************/
      
      // Set result available for GPU Kernel
      host_device_interface->is_result_available = true;
      
      // block until result was consumed
      while (host_device_interface->is_result_available) {}
      
      if (host_device_interface->is_debugging) {
        printf("HostMonitor result was consumed\n");
      }
      break;
    }
    
    /***********************************************************************/
    case HostDeviceInterface::WRITE_KEYVALUE:
    case HostDeviceInterface::SEQFILE_APPEND: {
      int response = 0; // false
      // Variation with repetition (n=6,k=2) -> 6^2 - 1 (null,null) variations
      /***********************************************************************/
      // (null,int)
      if ( (host_device_interface->key_type == HostDeviceInterface::NULL_TYPE) &&
          (host_device_interface->use_int_val2) ) {
        
        if (host_device_interface->command == HostDeviceInterface::WRITE_KEYVALUE) {
          response = socket_client_->sendCMD(HostDeviceInterface::WRITE_KEYVALUE, true,
                                             host_device_interface->int_val2);
        } else {
          socket_client_->sendCMD(HostDeviceInterface::SEQFILE_APPEND, false,
                                  host_device_interface->int_val3, // file_id
                                  host_device_interface->int_val2);
          response = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_APPEND);
        }
      }
      // (null,long)
      else if ( (host_device_interface->key_type == HostDeviceInterface::NULL_TYPE) &&
               (host_device_interface->use_long_val2) ) {
        if (host_device_interface->command == HostDeviceInterface::WRITE_KEYVALUE) {
          response = socket_client_->sendCMD(HostDeviceInterface::WRITE_KEYVALUE, true,
                                             host_device_interface->long_val2);
        } else {
          socket_client_->sendCMD(HostDeviceInterface::SEQFILE_APPEND, false,
                                  host_device_interface->int_val3, // file_id
                                  host_device_interface->long_val2);
          response = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_APPEND);
        }
      }
      // (null,float)
      else if ( (host_device_interface->key_type == HostDeviceInterface::NULL_TYPE) &&
               (host_device_interface->use_float_val2) ) {
        if (host_device_interface->command == HostDeviceInterface::WRITE_KEYVALUE) {
          response = socket_client_->sendCMD(HostDeviceInterface::WRITE_KEYVALUE, true,
                                             host_device_interface->float_val2);
        } else {
          socket_client_->sendCMD(HostDeviceInterface::SEQFILE_APPEND, false,
                                  host_device_interface->int_val3, // file_id
                                  host_device_interface->float_val2);
          response = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_APPEND);
        }
      }
      // (null,double)
      else if ( (host_device_interface->key_type == HostDeviceInterface::NULL_TYPE) &&
               (host_device_interface->use_double_val2) ) {
        if (host_device_interface->command == HostDeviceInterface::WRITE_KEYVALUE) {
          response = socket_client_->sendCMD(HostDeviceInterface::WRITE_KEYVALUE, true,
                                             host_device_interface->double_val2);
        } else {
          socket_client_->sendCMD(HostDeviceInterface::SEQFILE_APPEND, false,
                                  host_device_interface->int_val3, // file_id
                                  host_device_interface->double_val2);
          response = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_APPEND);
        }
      }
      // (null,string)
      else if ( (host_device_interface->key_type == HostDeviceInterface::NULL_TYPE) &&
               (host_device_interface->use_str_val2) ) {
        if (host_device_interface->command == HostDeviceInterface::WRITE_KEYVALUE) {
          response = socket_client_->sendCMD(HostDeviceInterface::WRITE_KEYVALUE, true,
                                             string(const_cast<char *>(host_device_interface->str_val2)));
        } else {
          socket_client_->sendCMD(HostDeviceInterface::SEQFILE_APPEND, false,
                                  host_device_interface->int_val3, // file_id
                                  string(const_cast<char *>(host_device_interface->str_val2)));
          response = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_APPEND);
        }
      }
      /***********************************************************************/
      // (int,null)
      else if ( (host_device_interface->use_int_val1) &&
               (host_device_interface->value_type == HostDeviceInterface::NULL_TYPE) ) {
        
        if (host_device_interface->command == HostDeviceInterface::WRITE_KEYVALUE) {
          response = socket_client_->sendCMD(HostDeviceInterface::WRITE_KEYVALUE, true,
                                             host_device_interface->int_val1);
        } else {
          socket_client_->sendCMD(HostDeviceInterface::SEQFILE_APPEND, false,
                                  host_device_interface->int_val3, // file_id
                                  host_device_interface->int_val1);
          response = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_APPEND);
        }
      }
      // (int,int)
      else if ( (host_device_interface->use_int_val1) &&
               (host_device_interface->use_int_val2) ) {
        
        if (host_device_interface->command == HostDeviceInterface::WRITE_KEYVALUE) {
          response = socket_client_->sendCMD(HostDeviceInterface::WRITE_KEYVALUE, true,
                                             host_device_interface->int_val1,
                                             host_device_interface->int_val2);
        } else {
          socket_client_->sendCMD(HostDeviceInterface::SEQFILE_APPEND, false,
                                  host_device_interface->int_val3, // file_id
                                  host_device_interface->int_val1,
                                  host_device_interface->int_val2);
          response = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_APPEND);
        }
      }
      // (int,long)
      else if ( (host_device_interface->use_int_val1) &&
               (host_device_interface->use_long_val2) ) {
        if (host_device_interface->command == HostDeviceInterface::WRITE_KEYVALUE) {
          response = socket_client_->sendCMD(HostDeviceInterface::WRITE_KEYVALUE, true,
                                             host_device_interface->int_val1,
                                             host_device_interface->long_val2);
        } else {
          socket_client_->sendCMD(HostDeviceInterface::SEQFILE_APPEND, false,
                                  host_device_interface->int_val3, // file_id
                                  host_device_interface->int_val1,
                                  host_device_interface->long_val2);
          response = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_APPEND);
        }
      }
      // (int,float)
      else if ( (host_device_interface->use_int_val1) &&
               (host_device_interface->use_float_val2) ) {
        if (host_device_interface->command == HostDeviceInterface::WRITE_KEYVALUE) {
          response = socket_client_->sendCMD(HostDeviceInterface::WRITE_KEYVALUE, true,
                                             host_device_interface->int_val1,
                                             host_device_interface->float_val2);
        } else {
          socket_client_->sendCMD(HostDeviceInterface::SEQFILE_APPEND, false,
                                  host_device_interface->int_val3, // file_id
                                  host_device_interface->int_val1,
                                  host_device_interface->float_val2);
          response = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_APPEND);
        }
      }
      // (int,double)
      else if ( (host_device_interface->use_int_val1) &&
               (host_device_interface->use_double_val2) ) {
        if (host_device_interface->command == HostDeviceInterface::WRITE_KEYVALUE) {
          response = socket_client_->sendCMD(HostDeviceInterface::WRITE_KEYVALUE, true,
                                             host_device_interface->int_val1,
                                             host_device_interface->double_val2);
        } else {
          socket_client_->sendCMD(HostDeviceInterface::SEQFILE_APPEND, false,
                                  host_device_interface->int_val3, // file_id
                                  host_device_interface->int_val1,
                                  host_device_interface->double_val2);
          response = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_APPEND);
        }
      }
      // (int,string)
      else if ( (host_device_interface->use_int_val1) &&
               (host_device_interface->use_str_val2) ) {
        if (host_device_interface->command == HostDeviceInterface::WRITE_KEYVALUE) {
          response = socket_client_->sendCMD(HostDeviceInterface::WRITE_KEYVALUE, true,
                                             host_device_interface->int_val1,
                                             string(const_cast<char *>(host_device_interface->str_val2)));
        } else {
          socket_client_->sendCMD(HostDeviceInterface::SEQFILE_APPEND, false,
                                  host_device_interface->int_val3, // file_id
                                  host_device_interface->int_val1,
                                  string(const_cast<char *>(host_device_interface->str_val2)));
          response = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_APPEND);
        }
      }
      /***********************************************************************/
      // (long,null)
      else if ( (host_device_interface->use_long_val1) &&
               (host_device_interface->value_type == HostDeviceInterface::NULL_TYPE) ) {
        if (host_device_interface->command == HostDeviceInterface::WRITE_KEYVALUE) {
          response = socket_client_->sendCMD(HostDeviceInterface::WRITE_KEYVALUE, true,
                                             host_device_interface->long_val1);
        } else {
          socket_client_->sendCMD(HostDeviceInterface::SEQFILE_APPEND, false,
                                  host_device_interface->int_val3, // file_id
                                  host_device_interface->long_val1);
          response = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_APPEND);
        }
      }
      // (long,int)
      else if ( (host_device_interface->use_long_val1) &&
               (host_device_interface->use_int_val2) ) {
        if (host_device_interface->command == HostDeviceInterface::WRITE_KEYVALUE) {
          response = socket_client_->sendCMD(HostDeviceInterface::WRITE_KEYVALUE, true,
                                             host_device_interface->long_val1,
                                             host_device_interface->int_val2);
        } else {
          socket_client_->sendCMD(HostDeviceInterface::SEQFILE_APPEND, false,
                                  host_device_interface->int_val3, // file_id
                                  host_device_interface->long_val1,
                                  host_device_interface->int_val2);
          response = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_APPEND);
        }
      }
      // (long,long)
      else if ( (host_device_interface->use_long_val1) &&
               (host_device_interface->use_long_val2) ) {
        if (host_device_interface->command == HostDeviceInterface::WRITE_KEYVALUE) {
          response = socket_client_->sendCMD(HostDeviceInterface::WRITE_KEYVALUE, true,
                                             host_device_interface->long_val1,
                                             host_device_interface->long_val2);
        } else {
          socket_client_->sendCMD(HostDeviceInterface::SEQFILE_APPEND, false,
                                  host_device_interface->int_val3, // file_id
                                  host_device_interface->long_val1,
                                  host_device_interface->long_val2);
          response = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_APPEND);
        }
      }
      // (long,float)
      else if ( (host_device_interface->use_long_val1) &&
               (host_device_interface->use_float_val2) ) {
        if (host_device_interface->command == HostDeviceInterface::WRITE_KEYVALUE) {
          response = socket_client_->sendCMD(HostDeviceInterface::WRITE_KEYVALUE, true,
                                             host_device_interface->long_val1,
                                             host_device_interface->float_val2);
        } else {
          socket_client_->sendCMD(HostDeviceInterface::SEQFILE_APPEND, false,
                                  host_device_interface->int_val3, // file_id
                                  host_device_interface->long_val1,
                                  host_device_interface->float_val2);
          response = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_APPEND);
        }
      }
      // (long,double)
      else if ( (host_device_interface->use_long_val1) &&
               (host_device_interface->use_double_val2) ) {
        if (host_device_interface->command == HostDeviceInterface::WRITE_KEYVALUE) {
          response = socket_client_->sendCMD(HostDeviceInterface::WRITE_KEYVALUE, true,
                                             host_device_interface->long_val1,
                                             host_device_interface->double_val2);
        } else {
          socket_client_->sendCMD(HostDeviceInterface::SEQFILE_APPEND, false,
                                  host_device_interface->int_val3, // file_id
                                  host_device_interface->long_val1,
                                  host_device_interface->double_val2);
          response = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_APPEND);
        }
      }
      // (long,string)
      else if ( (host_device_interface->use_long_val1) &&
               (host_device_interface->use_str_val2) ) {
        if (host_device_interface->command == HostDeviceInterface::WRITE_KEYVALUE) {
          response = socket_client_->sendCMD(HostDeviceInterface::WRITE_KEYVALUE, true,
                                             host_device_interface->long_val1,
                                             string(const_cast<char *>(host_device_interface->str_val2)));
        } else {
          socket_client_->sendCMD(HostDeviceInterface::SEQFILE_APPEND, false,
                                  host_device_interface->int_val3, // file_id
                                  host_device_interface->long_val1,
                                  string(const_cast<char *>(host_device_interface->str_val2)));
          response = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_APPEND);
        }
      }
      /***********************************************************************/
      // (float,null)
      else if ( (host_device_interface->use_float_val1) &&
               (host_device_interface->value_type == HostDeviceInterface::NULL_TYPE) ) {
        if (host_device_interface->command == HostDeviceInterface::WRITE_KEYVALUE) {
          response = socket_client_->sendCMD(HostDeviceInterface::WRITE_KEYVALUE, true,
                                             host_device_interface->float_val1);
        } else {
          socket_client_->sendCMD(HostDeviceInterface::SEQFILE_APPEND, false,
                                  host_device_interface->int_val3, // file_id
                                  host_device_interface->float_val1);
          response = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_APPEND);
        }
      }
      // (float,int)
      else if ( (host_device_interface->use_float_val1) &&
               (host_device_interface->use_int_val2) ) {
        if (host_device_interface->command == HostDeviceInterface::WRITE_KEYVALUE) {
          response = socket_client_->sendCMD(HostDeviceInterface::WRITE_KEYVALUE, true,
                                             host_device_interface->float_val1,
                                             host_device_interface->int_val2);
        } else {
          socket_client_->sendCMD(HostDeviceInterface::SEQFILE_APPEND, false,
                                  host_device_interface->int_val3, // file_id
                                  host_device_interface->float_val1,
                                  host_device_interface->int_val2);
          response = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_APPEND);
        }
      }
      // (float,long)
      else if ( (host_device_interface->use_float_val1) &&
               (host_device_interface->use_long_val2) ) {
        if (host_device_interface->command == HostDeviceInterface::WRITE_KEYVALUE) {
          response = socket_client_->sendCMD(HostDeviceInterface::WRITE_KEYVALUE, true,
                                             host_device_interface->float_val1,
                                             host_device_interface->long_val2);
        } else {
          socket_client_->sendCMD(HostDeviceInterface::SEQFILE_APPEND, false,
                                  host_device_interface->int_val3, // file_id
                                  host_device_interface->float_val1,
                                  host_device_interface->long_val2);
          response = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_APPEND);
        }
      }
      // (float,float)
      else if ( (host_device_interface->use_float_val1) &&
               (host_device_interface->use_float_val2) ) {
        if (host_device_interface->command == HostDeviceInterface::WRITE_KEYVALUE) {
          response = socket_client_->sendCMD(HostDeviceInterface::WRITE_KEYVALUE, true,
                                             host_device_interface->float_val1,
                                             host_device_interface->float_val2);
        } else {
          socket_client_->sendCMD(HostDeviceInterface::SEQFILE_APPEND, false,
                                  host_device_interface->int_val3, // file_id
                                  host_device_interface->float_val1,
                                  host_device_interface->float_val2);
          response = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_APPEND);
        }
      }
      // (float,double)
      else if ( (host_device_interface->use_float_val1) &&
               (host_device_interface->use_double_val2) ) {
        if (host_device_interface->command == HostDeviceInterface::WRITE_KEYVALUE) {
          response = socket_client_->sendCMD(HostDeviceInterface::WRITE_KEYVALUE, true,
                                             host_device_interface->float_val1,
                                             host_device_interface->double_val2);
        } else {
          socket_client_->sendCMD(HostDeviceInterface::SEQFILE_APPEND, false,
                                  host_device_interface->int_val3, // file_id
                                  host_device_interface->float_val1,
                                  host_device_interface->double_val2);
          response = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_APPEND);
        }
      }
      // (float,string)
      else if ( (host_device_interface->use_float_val1) &&
               (host_device_interface->use_str_val2) ) {
        if (host_device_interface->command == HostDeviceInterface::WRITE_KEYVALUE) {
          response = socket_client_->sendCMD(HostDeviceInterface::WRITE_KEYVALUE, true,
                                             host_device_interface->float_val1,
                                             string(const_cast<char *>(host_device_interface->str_val2)));
        } else {
          socket_client_->sendCMD(HostDeviceInterface::SEQFILE_APPEND, false,
                                  host_device_interface->int_val3, // file_id
                                  host_device_interface->float_val1,
                                  string(const_cast<char *>(host_device_interface->str_val2)));
          response = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_APPEND);
        }
      }
      /***********************************************************************/
      // (double,null)
      else if ( (host_device_interface->use_double_val1) &&
               (host_device_interface->value_type == HostDeviceInterface::NULL_TYPE) ) {
        if (host_device_interface->command == HostDeviceInterface::WRITE_KEYVALUE) {
          response = socket_client_->sendCMD(HostDeviceInterface::WRITE_KEYVALUE, true,
                                             host_device_interface->double_val1);
        } else {
          socket_client_->sendCMD(HostDeviceInterface::SEQFILE_APPEND, false,
                                  host_device_interface->int_val3, // file_id
                                  host_device_interface->double_val1);
          response = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_APPEND);
        }
      }
      // (double,int)
      else if ( (host_device_interface->use_double_val1) &&
               (host_device_interface->use_int_val2) ) {
        if (host_device_interface->command == HostDeviceInterface::WRITE_KEYVALUE) {
          response = socket_client_->sendCMD(HostDeviceInterface::WRITE_KEYVALUE, true,
                                             host_device_interface->double_val1,
                                             host_device_interface->int_val2);
        } else {
          socket_client_->sendCMD(HostDeviceInterface::SEQFILE_APPEND, false,
                                  host_device_interface->int_val3, // file_id
                                  host_device_interface->double_val1,
                                  host_device_interface->int_val2);
          response = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_APPEND);
        }
      }
      // (double,long)
      else if ( (host_device_interface->use_double_val1) &&
               (host_device_interface->use_long_val2) ) {
        if (host_device_interface->command == HostDeviceInterface::WRITE_KEYVALUE) {
          response = socket_client_->sendCMD(HostDeviceInterface::WRITE_KEYVALUE, true,
                                             host_device_interface->double_val1,
                                             host_device_interface->long_val2);
        } else {
          socket_client_->sendCMD(HostDeviceInterface::SEQFILE_APPEND, false,
                                  host_device_interface->int_val3, // file_id
                                  host_device_interface->double_val1,
                                  host_device_interface->long_val2);
          response = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_APPEND);
        }
      }
      // (double,float)
      else if ( (host_device_interface->use_double_val1) &&
               (host_device_interface->use_float_val2) ) {
        if (host_device_interface->command == HostDeviceInterface::WRITE_KEYVALUE) {
          response = socket_client_->sendCMD(HostDeviceInterface::WRITE_KEYVALUE, true,
                                             host_device_interface->double_val1,
                                             host_device_interface->float_val2);
        } else {
          socket_client_->sendCMD(HostDeviceInterface::SEQFILE_APPEND, false,
                                  host_device_interface->int_val3, // file_id
                                  host_device_interface->double_val1,
                                  host_device_interface->float_val2);
          response = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_APPEND);
        }
      }
      // (double,double)
      else if ( (host_device_interface->use_double_val1) &&
               (host_device_interface->use_double_val2) ) {
        if (host_device_interface->command == HostDeviceInterface::WRITE_KEYVALUE) {
          response = socket_client_->sendCMD(HostDeviceInterface::WRITE_KEYVALUE, true,
                                             host_device_interface->double_val1,
                                             host_device_interface->double_val2);
        } else {
          socket_client_->sendCMD(HostDeviceInterface::SEQFILE_APPEND, false,
                                  host_device_interface->int_val3, // file_id
                                  host_device_interface->double_val1,
                                  host_device_interface->double_val2);
          response = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_APPEND);
        }
      }
      // (double,string)
      else if ( (host_device_interface->use_double_val1) &&
               (host_device_interface->use_str_val2) ) {
        if (host_device_interface->command == HostDeviceInterface::WRITE_KEYVALUE) {
          response = socket_client_->sendCMD(HostDeviceInterface::WRITE_KEYVALUE, true,
                                             host_device_interface->double_val1,
                                             string(const_cast<char *>(host_device_interface->str_val2)));
        } else {
          socket_client_->sendCMD(HostDeviceInterface::SEQFILE_APPEND, false,
                                  host_device_interface->int_val3, // file_id
                                  host_device_interface->double_val1,
                                  string(const_cast<char *>(host_device_interface->str_val2)));
          response = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_APPEND);
        }
      }
      /***********************************************************************/
      // (string,null)
      else if ( (host_device_interface->use_str_val1) &&
               (host_device_interface->value_type == HostDeviceInterface::NULL_TYPE) ) {
        if (host_device_interface->command == HostDeviceInterface::WRITE_KEYVALUE) {
          response = socket_client_->sendCMD(HostDeviceInterface::WRITE_KEYVALUE, true,
                                             string(const_cast<char *>(host_device_interface->str_val1)));
        } else {
          socket_client_->sendCMD(HostDeviceInterface::SEQFILE_APPEND, false,
                                  host_device_interface->int_val3, // file_id
                                  string(const_cast<char *>(host_device_interface->str_val1)));
          response = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_APPEND);
        }
      }
      // (string,int)
      else if ( (host_device_interface->use_str_val1) &&
               (host_device_interface->use_int_val2) ) {
        if (host_device_interface->command == HostDeviceInterface::WRITE_KEYVALUE) {
          response = socket_client_->sendCMD(HostDeviceInterface::WRITE_KEYVALUE, true,
                                             string(const_cast<char *>(host_device_interface->str_val1)),
                                             host_device_interface->int_val2);
        } else {
          socket_client_->sendCMD(HostDeviceInterface::SEQFILE_APPEND, false,
                                  host_device_interface->int_val3, // file_id
                                  string(const_cast<char *>(host_device_interface->str_val1)),
                                  host_device_interface->int_val2);
          response = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_APPEND);
        }
      }
      // (string,long)
      else if ( (host_device_interface->use_str_val1) &&
               (host_device_interface->use_long_val2) ) {
        if (host_device_interface->command == HostDeviceInterface::WRITE_KEYVALUE) {
          response = socket_client_->sendCMD(HostDeviceInterface::WRITE_KEYVALUE, true,
                                             string(const_cast<char *>(host_device_interface->str_val1)),
                                             host_device_interface->long_val2);
        } else {
          socket_client_->sendCMD(HostDeviceInterface::SEQFILE_APPEND, false,
                                  host_device_interface->int_val3, // file_id
                                  string(const_cast<char *>(host_device_interface->str_val1)),
                                  host_device_interface->long_val2);
          response = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_APPEND);
        }
      }
      // (string,float)
      else if ( (host_device_interface->use_str_val1) &&
               (host_device_interface->use_float_val2) ) {
        if (host_device_interface->command == HostDeviceInterface::WRITE_KEYVALUE) {
          response = socket_client_->sendCMD(HostDeviceInterface::WRITE_KEYVALUE, true,
                                             string(const_cast<char *>(host_device_interface->str_val1)),
                                             host_device_interface->float_val2);
        } else {
          socket_client_->sendCMD(HostDeviceInterface::SEQFILE_APPEND, false,
                                  host_device_interface->int_val3, // file_id
                                  string(const_cast<char *>(host_device_interface->str_val1)),
                                  host_device_interface->float_val2);
          response = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_APPEND);
        }
      }
      // (string,double)
      else if ( (host_device_interface->use_str_val1) &&
               (host_device_interface->use_double_val2) ) {
        if (host_device_interface->command == HostDeviceInterface::WRITE_KEYVALUE) {
          response = socket_client_->sendCMD(HostDeviceInterface::WRITE_KEYVALUE, true,
                                             string(const_cast<char *>(host_device_interface->str_val1)),
                                             host_device_interface->double_val2);
        } else {
          socket_client_->sendCMD(HostDeviceInterface::SEQFILE_APPEND, false,
                                  host_device_interface->int_val3, // file_id
                                  string(const_cast<char *>(host_device_interface->str_val1)),
                                  host_device_interface->double_val2);
          response = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_APPEND);
        }
      }
      // (string,string)
      else if ( (host_device_interface->use_str_val1) &&
               (host_device_interface->use_str_val2) ) {
        if (host_device_interface->command == HostDeviceInterface::WRITE_KEYVALUE) {
          response = socket_client_->sendCMD(HostDeviceInterface::WRITE_KEYVALUE, true,
                                             string(const_cast<char *>(host_device_interface->str_val1)),
                                             string(const_cast<char *>(host_device_interface->str_val2)));
        } else {
          socket_client_->sendCMD(HostDeviceInterface::SEQFILE_APPEND, false,
                                  host_device_interface->int_val3, // file_id
                                  string(const_cast<char *>(host_device_interface->str_val1)),
                                  string(const_cast<char *>(host_device_interface->str_val2)));
          response = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_APPEND);
        }
      }
      /***********************************************************************/
      
      if ( (host_device_interface->command == HostDeviceInterface::WRITE_KEYVALUE) && (response == 0) ) {
        // TODO throw CudaException?
        printf("HostDeviceInterface::WRITE_KEYVALUE got wrong response command!\n");
      }
      
      host_device_interface->int_val1 = response;
      // Set result available for GPU Kernel
      host_device_interface->is_result_available = true;
      
      if (host_device_interface->is_debugging) {
        printf("HostMonitor got result: %d result_available: %s\n",
               host_device_interface->int_val1,
               (host_device_interface->is_result_available) ? "true" : "false");
      }
      
      // block until result was consumed
      while (host_device_interface->is_result_available) {}
      
      if (host_device_interface->is_debugging) {
        printf("HostMonitor result was consumed\n");
      }
      break;
    }
    
    /***********************************************************************/
    case HostDeviceInterface::SEQFILE_OPEN: {
      string values[] = {string(const_cast<char *>(host_device_interface->str_val1)),
        string(1, (char) host_device_interface->int_val1),
        string(const_cast<char *>(host_device_interface->str_val2)),
        string(const_cast<char *>(host_device_interface->str_val3))
      };
      
      socket_client_->sendCMD(HostDeviceInterface::SEQFILE_OPEN, values, 4);
      
      host_device_interface->int_val1 = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_OPEN);
      // Set result available for GPU Kernel
      host_device_interface->is_result_available = true;
      
      if (host_device_interface->is_debugging) {
        printf("HostMonitor got result: %d result_available: %s\n",
               host_device_interface->int_val1,
               (host_device_interface->is_result_available) ? "true" : "false");
      }
      
      // block until result was consumed
      while (host_device_interface->is_result_available) {}
      
      if (host_device_interface->is_debugging) {
        printf("HostMonitor result was consumed\n");
      }
      break;
    }
    
    /***********************************************************************/
    case HostDeviceInterface::SEQFILE_CLOSE: {
      socket_client_->sendCMD(HostDeviceInterface::SEQFILE_CLOSE, false, host_device_interface->int_val1);
      
      host_device_interface->int_val1 = socket_client_->getResult<int32_t>(HostDeviceInterface::SEQFILE_CLOSE);
      // Set result available for GPU Kernel
      host_device_interface->is_result_available = true;
      
      if (host_device_interface->is_debugging) {
        printf("HostMonitor got result: %d result_available: %s\n",
               host_device_interface->int_val1,
               (host_device_interface->is_result_available) ? "true" : "false");
      }
      
      // block until result was consumed
      while (host_device_interface->is_result_available) {}
      
      if (host_device_interface->is_debugging) {
        printf("HostMonitor result was consumed\n");
      }
      break;
    }
  }
}

