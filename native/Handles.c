#include "edu_syr_pcpratts_rootbeer_runtime2_cuda_Handles.h"
#include <cuda.h>

/*
 * Class:     edu_syr_pcpratts_rootbeer_runtime2_cuda_Handles
 * Method:    doWriteLong
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_edu_syr_pcpratts_rootbeer_runtime2_cuda_Handles_doWriteLong
  (JNIEnv *env, jobject obj, jlong base_addr, jint offset, jlong value){

  value = value >> 4;
  jint * mem = (jint *) base_addr;
  mem[offset] = value;
}

/*
 * Class:     edu_syr_pcpratts_rootbeer_runtime2_cuda_Handles
 * Method:    doReadLong
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_edu_syr_pcpratts_rootbeer_runtime2_cuda_Handles_doReadLong
  (JNIEnv *env, jobject obj, jlong base_addr, jint offset){


  jint * mem = (jint *) base_addr;
  jint ret = mem[offset];
  jlong long_ret = ret;
  long_ret = long_ret << 4;
  return long_ret;
}
