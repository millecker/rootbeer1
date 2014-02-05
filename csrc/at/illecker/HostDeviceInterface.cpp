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

#include "HostDeviceInterface.h"

HostDeviceInterface::HostDeviceInterface() {
  init();
}

HostDeviceInterface::~HostDeviceInterface() {}

void HostDeviceInterface::init() {
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
  end_of_data = true;
  is_result_available = false;
}
