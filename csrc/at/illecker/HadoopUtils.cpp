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

#include "HadoopUtils.h"

namespace HadoopUtils {
  
  /**
   * Generic toString template specializations
   */
  template <> string toString<string>(const string& t) {
    return t;
  }

  /**
   * Convert an integer to a string.
   */
  int toInt(const string& val) {
    int result;
    char trash;
    int num = sscanf(val.c_str(), "%d%c", &result, &trash);
    HADOOP_ASSERT(num == 1,
                  "Problem converting " + val + " to integer.");
    return result;
  }
  
  /**
   * Convert the string to a float.
   * @throws Error if the string is not a valid float
   */
  float toFloat(const string& val) {
    float result;
    char trash;
    int num = sscanf(val.c_str(), "%f%c", &result, &trash);
    HADOOP_ASSERT(num == 1,
                  "Problem converting " + val + " to float.");
    return result;
  }
  
  /**
   * Convert the string to a double.
   * @throws Error if the string is not a valid double
   */
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
  
  /**
   * Convert the string to a boolean.
   * @throws Error if the string is not a valid boolean value
   */
  bool toBool(const string& val) {
    if (val == "true") {
      return true;
    } else if (val == "false") {
      return false;
    } else {
      HADOOP_ASSERT(false, "Problem converting " + val + " to boolean.");
    }
  }
  
  /*****************************************************************************/
  // Error
  /*****************************************************************************/
  Error::Error(const string& msg): error(msg) {
  }
  
  Error::Error(const string& msg,
        const string& file, int line,
        const string& function) {
    error = msg + " at " + file + ":" + toString<int32_t>(line) +
    " in " + function;
  }
  
  const string& Error::getMessage() const {
    return error;
  }

  /*****************************************************************************/
  // FileInStream
  /*****************************************************************************/
  FileInStream::FileInStream() {
    mFile = NULL;
    isOwned = false;
  }
  
  FileInStream::~FileInStream() {
    if (mFile != NULL) {
      close();
    }
  }
  
  bool FileInStream::open(const std::string& name) {
    mFile = fopen(name.c_str(), "rb");
    isOwned = true;
    return (mFile != NULL);
  }
  
  bool FileInStream::open(FILE* file) {
    mFile = file;
    isOwned = false;
    return (mFile != NULL);
  }
  
  void FileInStream::read(void *buf, size_t len) {
    size_t result = fread(buf, len, 1, mFile);
    if (result == 0) {
      if (feof(mFile)) {
        HADOOP_ASSERT(false, "end of file");
      } else {
        HADOOP_ASSERT(false, string("read error on file: ") + strerror(errno));
      }
    }
  }
  
  bool FileInStream::skip(size_t nbytes) {
    return (0==fseek(mFile, nbytes, SEEK_CUR));
  }
  
  bool FileInStream::close() {
    int ret = 0;
    if (mFile != NULL && isOwned) {
      ret = fclose(mFile);
    }
    mFile = NULL;
    return (ret==0);
  }

  /*****************************************************************************/
  // FileOutStream
  /*****************************************************************************/
  FileOutStream::FileOutStream() {
    mFile = NULL;
    isOwned = false;
  }
  
  FileOutStream::~FileOutStream() {
    if (mFile != NULL) {
      close();
    }
  }
  
  bool FileOutStream::open(const std::string& name, bool overwrite) {
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
  
  bool FileOutStream::open(FILE* file) {
    mFile = file;
    isOwned = false;
    return (mFile != NULL);
  }
  
  void FileOutStream::write(const void* buf, size_t len) {
    size_t result = fwrite(buf, len, 1, mFile);
    HADOOP_ASSERT(result == 1,
                  string("write error to file: ") + strerror(errno));
  }
  
  bool FileOutStream::advance(size_t nbytes) {
    return (0==fseek(mFile, nbytes, SEEK_CUR));
  }
  
  bool FileOutStream::close() {
    int ret = 0;
    if (mFile != NULL && isOwned) {
      ret = fclose(mFile);
    }
    mFile = NULL;
    return (ret == 0);
  }
  
  void FileOutStream::flush() {
    fflush(mFile);
  }

  /*****************************************************************************/
  // Serialization and Deserialization
  /*****************************************************************************/
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

}

