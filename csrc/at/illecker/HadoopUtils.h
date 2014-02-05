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

#ifndef HADOOP_UTILS_H
#define HADOOP_UTILS_H

#include <errno.h>
#include <typeinfo> /* typeid */
#include <string>
#include <string.h>
#include <sstream> /* ostringstream */
#include <rpc/types.h>
#include <rpc/xdr.h>

using std::string;

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

namespace HadoopUtils {

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
  template <> string toString<string>(const string& t);

  /**
   * Convert an integer to a string.
   */
  int toInt(const string& val);

  /**
   * Convert the string to a float.
   * @throws Error if the string is not a valid float
   */
  float toFloat(const string& val);

  /**
   * Convert the string to a double.
   * @throws Error if the string is not a valid double
   */
  double toDouble(const string& val);

  /**
   * Convert the string to a boolean.
   * @throws Error if the string is not a valid boolean value
   */
  bool toBool(const string& val);

  /*****************************************************************************/
  // Error
  /*****************************************************************************/
  /**
   * A simple exception class that records a message for the user.
   */
  class Error {
  private:
    std::string error;
  public:
    
    /**
     * Create an error object with the given message.
     */
    Error(const std::string& msg);
    
    /**
     * Construct an error object with the given message that was created on
     * the given file, line, and functino.
     */
    Error(const std::string& msg,
          const std::string& file, int line, const std::string& function);
    
    /**
     * Get the error message.
     */
    const std::string& getMessage() const;
  };
  
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
      FileInStream();
      ~FileInStream();
  
      bool open(const std::string& name);
      bool open(FILE* file);
      void read(void *buf, size_t len);
      bool skip(size_t nbytes);
      bool close();
  };

  /*****************************************************************************/
  // FileOutStream
  /*****************************************************************************/
  class FileOutStream {
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
      FileOutStream();
      ~FileOutStream();
      bool open(const std::string& name, bool overwrite);
      bool open(FILE* file);
      void write(const void* buf, size_t len);
      bool advance(size_t nbytes);
      bool close();
      void flush();
  };

  /*****************************************************************************/
  // Serialization and Deserialization
  /*****************************************************************************/
  /**
   * Generic serialization
   */
  template<class T>
  void serialize(T t, FileOutStream& stream) {
    serialize<string>(toString<T>(t), stream);
  }

  template <> void serialize<int64_t>(int64_t t, FileOutStream& stream);
  template <> void serialize<int32_t>(int32_t t, FileOutStream& stream);
  template <> void serialize<float>(float t, FileOutStream& stream);
  template <> void serialize<double>(double t, FileOutStream& stream);
  template <> void serialize<string>(string t, FileOutStream& stream);

  /**
   * Generic deserialization
   */
  template<class T>
  T deserialize(FileInStream& stream) {
    string str = "Not able to deserialize type: ";
    throw Error(str.append(typeid(T).name()));
  }

  template <> int64_t deserialize<int64_t>(FileInStream& stream);
  template <> int32_t deserialize<int32_t>(FileInStream& stream);
  template <> float deserialize<float>(FileInStream& stream);
  template <> double deserialize<double>(FileInStream& stream);
  template <> string deserialize<string>(FileInStream& stream);
}

#endif

