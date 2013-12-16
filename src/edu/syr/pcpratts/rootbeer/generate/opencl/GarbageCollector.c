#define GC_OBJ_TYPE_COUNT char
#define GC_OBJ_TYPE_COLOR char
#define GC_OBJ_TYPE_TYPE int
#define GC_OBJ_TYPE_CTOR_USED char
#define GC_OBJ_TYPE_SIZE int

#define COLOR_GREY 0
#define COLOR_BLACK 1
#define COLOR_WHITE 2

$$__device__$$ void edu_syr_pcpratts_gc_collect($$__global$$ char * gc_info);
$$__device__$$ void edu_syr_pcpratts_gc_assign($$__global$$ char * gc_info, int * lhs, int rhs);
$$__device__$$ $$__global$$ char * edu_syr_pcpratts_gc_deref($$__global$$ char * gc_info, int handle);
$$__device__$$ int edu_syr_pcpratts_gc_malloc($$__global$$ char * gc_info, int size);
$$__device__$$ unsigned long long edu_syr_pcpratts_gc_malloc_no_fail($$__global$$ char * gc_info, int size);
$$__device__$$ int edu_syr_pcpratts_classConstant(int type_num);
$$__device__$$ long long java_lang_System_nanoTime($$__global$$ char * gc_info, int * exception);

#define CACHE_SIZE_BYTES 32
#define CACHE_SIZE_INTS (CACHE_SIZE_BYTES / sizeof(int))
#define CACHE_ENTRY_SIZE 4

#define TO_SPACE_OFFSET               0
#define TO_SPACE_FREE_POINTER_OFFSET  8

$$__device__$$
void edu_syr_pcpratts_exitMonitorRef($$__global$$ char * gc_info, int thisref, int old){
  char * mem = edu_syr_pcpratts_gc_deref(gc_info, thisref); 
  mem += 16;
  if(old == -1){    
    edu_syr_pcpratts_threadfence();  
    atomicExch((int *) mem, -1); 
  }
}

$$__device__$$
void edu_syr_pcpratts_exitMonitorMem($$__global$$ char * gc_info, char * mem, int old){
  if(old == -1){   
    edu_syr_pcpratts_threadfence(); 
    atomicExch((int *) mem, -1);
  }
}

$$__device__$$ 
long long java_lang_Double_doubleToLongBits($$__global$$ char * gc_info, double value, int * exception){
  long long ret = *((long long *) ((double *) &value));
  return ret;
}

$$__device__$$ 
double java_lang_Double_longBitsToDouble($$__global$$ char * gc_info, long long value, int * exception){
  double ret = *((double *) ((long long *) &value));
  return ret;
}

$$__device__$$
int java_lang_Float_floatToIntBits($$__global$$ char * gc_info, float value, int * exception){
  int ret = *((int *) ((float *) &value));
  return ret;
}  

$$__device__$$
float java_lang_Float_intBitsToFloat($$__global$$ char * gc_info, int value, int * exception){
  float ret = *((float *) ((int *) &value));
  return ret;
}

$$__device__$$ double java_lang_StrictMath_exp( char * gc_info , double parameter0 , int * exception ) { 
  return exp(parameter0); 
} 

$$__device__$$ double java_lang_StrictMath_log( char * gc_info , double parameter0 , int * exception ) { 
  return log(parameter0); 
} 

$$__device__$$ double java_lang_StrictMath_log10( char * gc_info , double parameter0 , int * exception ) { 
  return log10(parameter0); 
} 

$$__device__$$ double java_lang_StrictMath_sqrt( char * gc_info , double parameter0 , int * exception ) { 
  return sqrt(parameter0); 
} 

$$__device__$$ double java_lang_StrictMath_cbrt( char * gc_info , double parameter0 , int * exception ) { 
  //2.2204460492503131e-16 is DBL_EPSILON
  if (fabs(parameter0) < 2.2204460492503131e-16){
    return 0.0;
  }

  if (parameter0 > 0.0) {
    return pow(parameter0, 1.0/3.0);
  }

  return -pow(-parameter0, 1.0/3.0);
} 

$$__device__$$ double java_lang_StrictMath_IEEEremainder( char * gc_info , double parameter0 , double parameter1, int * exception ) { 
  return remainder(parameter0, parameter1); 
} 

$$__device__$$ double java_lang_StrictMath_ceil( char * gc_info , double parameter0 , int * exception ) { 
  return ceil(parameter0); 
} 

$$__device__$$ double java_lang_StrictMath_floor( char * gc_info , double parameter0 , int * exception ) { 
  return floor(parameter0); 
} 

$$__device__$$ double java_lang_StrictMath_sin( char * gc_info , double parameter0 , int * exception ) { 
  return sin(parameter0); 
} 

$$__device__$$ double java_lang_StrictMath_cos( char * gc_info , double parameter0 , int * exception ) { 
  return cos(parameter0); 
} 

$$__device__$$ double java_lang_StrictMath_tan( char * gc_info , double parameter0 , int * exception ) { 
  return tan(parameter0); 
} 

$$__device__$$ double java_lang_StrictMath_asin( char * gc_info , double parameter0 , int * exception ) { 
  return asin(parameter0); 
} 

$$__device__$$ double java_lang_StrictMath_acos( char * gc_info , double parameter0 , int * exception ) { 
  return acos(parameter0); 
} 

$$__device__$$ double java_lang_StrictMath_atan( char * gc_info , double parameter0 , int * exception ) { 
  return atan(parameter0); 
} 

$$__device__$$ double java_lang_StrictMath_atan2( char * gc_info , double parameter0 , double parameter1, int * exception ) { 
  return atan2(parameter0, parameter1); 
} 

$$__device__$$ double java_lang_StrictMath_pow( char * gc_info , double parameter0 , double parameter1, int * exception ) { 
  return pow(parameter0, parameter1); 
} 

$$__device__$$ double java_lang_StrictMath_sinh( char * gc_info , double parameter0 , int * exception ) { 
  return sinh(parameter0); 
} 

$$__device__$$ double java_lang_StrictMath_cosh( char * gc_info , double parameter0 , int * exception ) { 
  return cosh(parameter0); 
} 

$$__device__$$ double java_lang_StrictMath_tanh( char * gc_info , double parameter0 , int * exception ) { 
  return tanh(parameter0); 
} 

$$__device__$$ 
void edu_syr_pcpratts_rootbeer_runtime_GpuStopwatch_start($$__global$$ char * gc_info, int thisref, int * exception){
  long long int time;
  
  time = clock64();
  instance_setter_edu_syr_pcpratts_rootbeer_runtime_GpuStopwatch_m_start(gc_info, thisref, time, exception);
}

$$__device__$$ 
void edu_syr_pcpratts_rootbeer_runtime_GpuStopwatch_stop($$__global$$ char * gc_info, int thisref, int * exception){
  long long int time;
  
  time = clock64();
  instance_setter_edu_syr_pcpratts_rootbeer_runtime_GpuStopwatch_m_stop(gc_info, thisref, time, exception);
}

$$__device__$$ 
char edu_syr_pcpratts_rootbeer_runtime_RootbeerGpu_isOnGpu($$__global$$ char * gc_info, int * exception){
  return 1;
}

$$__device__$$ 
int edu_syr_pcpratts_rootbeer_runtime_RootbeerGpu_getThreadId($$__global$$ char * gc_info, int * exception){
  return getThreadId();
}

$$__device__$$ 
int edu_syr_pcpratts_rootbeer_runtime_RootbeerGpu_getThreadIdxx($$__global$$ char * gc_info, int * exception){
  return getThreadIdxx();
}

$$__device__$$ 
int edu_syr_pcpratts_rootbeer_runtime_RootbeerGpu_getBlockIdxx($$__global$$ char * gc_info, int * exception){
  return getBlockIdxx();
}

$$__device__$$ 
int edu_syr_pcpratts_rootbeer_runtime_RootbeerGpu_getBlockDimx($$__global$$ char * gc_info, int * exception){
  return getBlockDimx();
}

$$__device__$$ 
int edu_syr_pcpratts_rootbeer_runtime_RootbeerGpu_getGridDimx($$__global$$ char * gc_info, int * exception){
  return getGridDimx();
}


$$__device__$$ 
long long edu_syr_pcpratts_rootbeer_runtime_RootbeerGpu_getRef($$__global$$ char * gc_info, int ref, int * exception){
  return ref;
}

$$__device__$$
char edu_syr_pcpratts_rootbeer_runtime_RootbeerGpu_getSharedByte($$__global$$ char * gc_info, int index, int * exception){
#ifdef ARRAY_CHECKS
  if(index < 0 || index >= %%shared_mem_size%%){
    *exception = edu_syr_pcpratts_rootbeer_runtimegpu_GpuException_arrayOutOfBounds(gc_info, 
      index, 0, %%shared_mem_size%%, exception);
    return 0;
  }
#endif
  return m_shared[index]; 
}

$$__device__$$
void edu_syr_pcpratts_rootbeer_runtime_RootbeerGpu_setSharedByte($$__global$$ char * gc_info, int index, char value, int * exception){
#ifdef ARRAY_CHECKS
  if(index < 0 || index >= %%shared_mem_size%%){
    *exception = edu_syr_pcpratts_rootbeer_runtimegpu_GpuException_arrayOutOfBounds(gc_info, 
      index, 0, %%shared_mem_size%%, exception);
    return;
  }
#endif
  m_shared[index] = value;
}
  
$$__device__$$
char edu_syr_pcpratts_rootbeer_runtime_RootbeerGpu_getSharedChar($$__global$$ char * gc_info, int index, int * exception){
#ifdef ARRAY_CHECKS
  if(index < 0 || index >= %%shared_mem_size%%){
    *exception = edu_syr_pcpratts_rootbeer_runtimegpu_GpuException_arrayOutOfBounds(gc_info, 
      index, 0, %%shared_mem_size%%, exception);
    return 0;
  }
#endif
  return m_shared[index]; 
}

$$__device__$$
void edu_syr_pcpratts_rootbeer_runtime_RootbeerGpu_setSharedChar($$__global$$ char * gc_info, int index, char value, int * exception){
#ifdef ARRAY_CHECKS
  if(index < 0 || index >= %%shared_mem_size%%){
    *exception = edu_syr_pcpratts_rootbeer_runtimegpu_GpuException_arrayOutOfBounds(gc_info, 
      index, 0, %%shared_mem_size%%, exception);
    return;
  }
#endif
  m_shared[index] = value;
}
  
$$__device__$$
char edu_syr_pcpratts_rootbeer_runtime_RootbeerGpu_getSharedBoolean($$__global$$ char * gc_info, int index, int * exception){
#ifdef ARRAY_CHECKS
  if(index < 0 || index >= %%shared_mem_size%%){
    *exception = edu_syr_pcpratts_rootbeer_runtimegpu_GpuException_arrayOutOfBounds(gc_info, 
      index, 0, %%shared_mem_size%%, exception);
    return 0;
  }
#endif
  return m_shared[index]; 
}

$$__device__$$
void edu_syr_pcpratts_rootbeer_runtime_RootbeerGpu_setSharedBoolean($$__global$$ char * gc_info, int index, char value, int * exception){
#ifdef ARRAY_CHECKS
  if(index < 0 || index >= %%shared_mem_size%%){
    *exception = edu_syr_pcpratts_rootbeer_runtimegpu_GpuException_arrayOutOfBounds(gc_info, 
      index, 0, %%shared_mem_size%%, exception);
    return;
  }
#endif
  m_shared[index] = value;
}
  
$$__device__$$
short edu_syr_pcpratts_rootbeer_runtime_RootbeerGpu_getSharedShort($$__global$$ char * gc_info, int index, int * exception){
#ifdef ARRAY_CHECKS
  if(index < 0 || index + 2 >= %%shared_mem_size%%){
    *exception = edu_syr_pcpratts_rootbeer_runtimegpu_GpuException_arrayOutOfBounds(gc_info, 
      index, 0, %%shared_mem_size%%, exception);
    return 0;
  }  
#endif
  short ret = 0;
  ret |= m_shared[index] & 0xff;
  ret |= (m_shared[index + 1] << 8) & 0xff00;
  return ret;
}

$$__device__$$
void edu_syr_pcpratts_rootbeer_runtime_RootbeerGpu_setSharedShort($$__global$$ char * gc_info, int index, short value, int * exception){
#ifdef ARRAY_CHECKS
  if(index < 0 || index + 2 >= %%shared_mem_size%%){
    *exception = edu_syr_pcpratts_rootbeer_runtimegpu_GpuException_arrayOutOfBounds(gc_info, 
      index, 0, %%shared_mem_size%%, exception);
    return;
  }
#endif
  m_shared[index] = (char) (value & 0xff);
  m_shared[index + 1] = (char) ((value >> 8) & 0xff);
}

$$__device__$$
int edu_syr_pcpratts_rootbeer_runtime_RootbeerGpu_getSharedInteger($$__global$$ char * gc_info, int index, int * exception){
#ifdef ARRAY_CHECKS
  if(index < 0 || index + 4 >= %%shared_mem_size%%){
    *exception = edu_syr_pcpratts_rootbeer_runtimegpu_GpuException_arrayOutOfBounds(gc_info, 
      index, 0, %%shared_mem_size%%, exception);
    return 0;
  }
#endif
  int ret = m_shared[index] & 0x000000ff;
  ret |= (m_shared[index + 1] << 8)  & 0x0000ff00;
  ret |= (m_shared[index + 2] << 16) & 0x00ff0000;
  ret |= (m_shared[index + 3] << 24) & 0xff000000; 
  return ret;
}

$$__device__$$
void edu_syr_pcpratts_rootbeer_runtime_RootbeerGpu_setSharedInteger($$__global$$ char * gc_info, int index, int value, int * exception){  
#ifdef ARRAY_CHECKS
  if(index < 0 || index + 4 >= %%shared_mem_size%%){
    *exception = edu_syr_pcpratts_rootbeer_runtimegpu_GpuException_arrayOutOfBounds(gc_info, 
      index, 0, %%shared_mem_size%%, exception);
    return;
  }
#endif
  m_shared[index] = (char) (value & 0xff);
  m_shared[index + 1] = (char) ((value >> 8)  & 0xff);
  m_shared[index + 2] = (char) ((value >> 16) & 0xff);
  m_shared[index + 3] = (char) ((value >> 24) & 0xff);
}

$$__device__$$
long long edu_syr_pcpratts_rootbeer_runtime_RootbeerGpu_getSharedLong($$__global$$ char * gc_info, int index, int * exception){
#ifdef ARRAY_CHECKS
  if(index < 0 || index + 8 >= %%shared_mem_size%%){
    *exception = edu_syr_pcpratts_rootbeer_runtimegpu_GpuException_arrayOutOfBounds(gc_info, 
      index, 0, %%shared_mem_size%%, exception);
    return 0;
  }
#endif
  long long ret = 0;
  ret |=  ((long long) m_shared[index]) & 0x00000000000000ffL;
  ret |= ((long long) m_shared[index + 1] << 8)  & 0x000000000000ff00L;
  ret |= ((long long) m_shared[index + 2] << 16) & 0x0000000000ff0000L;
  ret |= ((long long) m_shared[index + 3] << 24) & 0x00000000ff000000L;
  ret |= ((long long) m_shared[index + 4] << 32) & 0x000000ff00000000L;
  ret |= ((long long) m_shared[index + 5] << 40) & 0x0000ff0000000000L;
  ret |= ((long long) m_shared[index + 6] << 48) & 0x00ff000000000000L;
  ret |= ((long long) m_shared[index + 7] << 56) & 0xff00000000000000L;
  return ret;
}

$$__device__$$
void edu_syr_pcpratts_rootbeer_runtime_RootbeerGpu_setSharedLong($$__global$$ char * gc_info, int index, long long value, int * exception){
#ifdef ARRAY_CHECKS
  if(index < 0 || index + 8 >= %%shared_mem_size%%){
    *exception = edu_syr_pcpratts_rootbeer_runtimegpu_GpuException_arrayOutOfBounds(gc_info, 
      index, 0, %%shared_mem_size%%, exception);
    return;
  }
#endif
  m_shared[index] = (char) (value & 0x00000000000000ffL);
  m_shared[index + 1] = (char) ((value >> 8)  & 0x00000000000000ffL);
  m_shared[index + 2] = (char) ((value >> 16) & 0x00000000000000ffL);
  m_shared[index + 3] = (char) ((value >> 24) & 0x00000000000000ffL);
  m_shared[index + 4] = (char) ((value >> 32) & 0x00000000000000ffL);
  m_shared[index + 5] = (char) ((value >> 40) & 0x00000000000000ffL);
  m_shared[index + 6] = (char) ((value >> 48) & 0x00000000000000ffL);
  m_shared[index + 7] = (char) ((value >> 56) & 0x00000000000000ffL);
}
  
$$__device__$$
float edu_syr_pcpratts_rootbeer_runtime_RootbeerGpu_getSharedFloat($$__global$$ char * gc_info, int index, int * exception){
  int int_value = edu_syr_pcpratts_rootbeer_runtime_RootbeerGpu_getSharedInteger(gc_info, index, exception);
  return *((float *) &int_value);
}

$$__device__$$
void edu_syr_pcpratts_rootbeer_runtime_RootbeerGpu_setSharedFloat($$__global$$ char * gc_info, int index, float value, int * exception){
  int int_value = *((int *) &value);
  edu_syr_pcpratts_rootbeer_runtime_RootbeerGpu_setSharedInteger(gc_info, index, int_value, exception);
}
  
$$__device__$$
double edu_syr_pcpratts_rootbeer_runtime_RootbeerGpu_getSharedDouble($$__global$$ char * gc_info, int index, int * exception){
  long long long_value = edu_syr_pcpratts_rootbeer_runtime_RootbeerGpu_getSharedLong(gc_info, index, exception);
  return *((double *) &long_value);
}

$$__device__$$
void edu_syr_pcpratts_rootbeer_runtime_RootbeerGpu_setSharedDouble($$__global$$ char * gc_info, int index, double value, int * exception){
  long long long_value = *((long long *) &value);
  edu_syr_pcpratts_rootbeer_runtime_RootbeerGpu_setSharedLong(gc_info, index, long_value, exception);
}

$$__device__$$
int edu_syr_pcpratts_rootbeer_runtime_RootbeerGpu_getSharedObject($$__global$$ char * gc_info, int index, int * exception){
  return edu_syr_pcpratts_rootbeer_runtime_RootbeerGpu_getSharedInteger(gc_info, index, exception);
}

$$__device__$$
void edu_syr_pcpratts_rootbeer_runtime_RootbeerGpu_setSharedObject($$__global$$ char * gc_info, int index, int value, int * exception){
  edu_syr_pcpratts_rootbeer_runtime_RootbeerGpu_setSharedInteger(gc_info, index, value, exception);
}
  
$$__device__$$
void java_io_PrintStream_println0_9_($$__global$$ char * gc_info, int thisref, int str_ret, int * exception){
  int valueref;
  int count;
  int offset;
  int i;
  int curr_offset;

  char * valueref_deref;

  valueref = instance_getter_java_lang_String_value(gc_info, str_ret, exception);  
  if(*exception != 0){
    return; 
  } 
  count = instance_getter_java_lang_String_count(gc_info, str_ret, exception);
  if(*exception != 0){
    return; 
  } 
  offset = instance_getter_java_lang_String_offset(gc_info, str_ret, exception);
  if(*exception != 0){
    return; 
  } 
  valueref_deref = (char *) edu_syr_pcpratts_gc_deref(gc_info, valueref);
  for(i = offset; i < count; ++i){
    curr_offset = 32 + (i * 4);
    printf("%c", valueref_deref[curr_offset]);
  }
  printf("\n");
}

$$__device__$$
void java_io_PrintStream_println0_($$__global$$ char * gc_info, int thisref, int * exception){
  printf("\n");
}

$$__device__$$
void java_io_PrintStream_println0_1_($$__global$$ char * gc_info, int thisref, int value, int * exception){
  printf("%d\n", value);
}

$$__device__$$
void java_io_PrintStream_println0_2_($$__global$$ char * gc_info, int thisref, char value, int * exception){
  printf("%d\n", value);
}

$$__device__$$
void java_io_PrintStream_println0_3_($$__global$$ char * gc_info, int thisref, char value, int * exception){
  printf("%c\n", value);
}

$$__device__$$
void java_io_PrintStream_println0_4_($$__global$$ char * gc_info, int thisref, short value, int * exception){
  printf("%d\n", value);
}

$$__device__$$
void java_io_PrintStream_println0_5_($$__global$$ char * gc_info, int thisref, int value, int * exception){
  printf("%d\n", value);
}

$$__device__$$
void java_io_PrintStream_println0_6_($$__global$$ char * gc_info, int thisref, long long value, int * exception){
  printf("%lld\n", value);
}

$$__device__$$
void java_io_PrintStream_println0_7_($$__global$$ char * gc_info, int thisref, float value, int * exception){
  printf("%e\n", value);
}

$$__device__$$
void java_io_PrintStream_println0_8_($$__global$$ char * gc_info, int thisref, double value, int * exception){
  printf("%e\n", value);
}

$$__device__$$
void java_io_PrintStream_print0_9_($$__global$$ char * gc_info, int thisref, int str_ret, int * exception){
  int valueref;
  int count;
  int offset;
  int i;
  int curr_offset;

  char * valueref_deref;

  valueref = instance_getter_java_lang_String_value(gc_info, str_ret, exception);  
  if(*exception != 0){
    return; 
  } 
  count = instance_getter_java_lang_String_count(gc_info, str_ret, exception);
  if(*exception != 0){
    return; 
  } 
  offset = instance_getter_java_lang_String_offset(gc_info, str_ret, exception);
  if(*exception != 0){
    return; 
  } 
  valueref_deref = (char *) edu_syr_pcpratts_gc_deref(gc_info, valueref);
  for(i = offset; i < count; ++i){
    curr_offset = 32 + (i * 4);
    printf("%c", valueref_deref[curr_offset]);
  }
}

$$__device__$$
void java_io_PrintStream_print0_1_($$__global$$ char * gc_info, int thisref, int value, int * exception){
  printf("%d", value);
}

$$__device__$$
void java_io_PrintStream_print0_2_($$__global$$ char * gc_info, int thisref, char value, int * exception){
  printf("%d", value);
}

$$__device__$$
void java_io_PrintStream_print0_3_($$__global$$ char * gc_info, int thisref, char value, int * exception){
  printf("%c", value);
}

$$__device__$$
void java_io_PrintStream_print0_4_($$__global$$ char * gc_info, int thisref, short value, int * exception){
  printf("%d", value);
}

$$__device__$$
void java_io_PrintStream_print0_5_($$__global$$ char * gc_info, int thisref, int value, int * exception){
  printf("%d", value);
}

$$__device__$$
void java_io_PrintStream_print0_6_($$__global$$ char * gc_info, int thisref, long long value, int * exception){
  printf("%lld", value);
}

$$__device__$$
void java_io_PrintStream_print0_7_($$__global$$ char * gc_info, int thisref, float value, int * exception){
  printf("%e", value);
}

$$__device__$$
void java_io_PrintStream_print0_8_($$__global$$ char * gc_info, int thisref, double value, int * exception){
  printf("%e", value);
}

$$__device__$$
int edu_syr_pcpratts_rootbeer_runtime_RootbeerAtomicInt_atomicAdd($$__global$$ char * gc_info, int thisref, int value, int * exception){
  char * thisref_deref;
  int * array;

  thisref_deref = edu_syr_pcpratts_gc_deref ( gc_info , thisref ) ;
  thisref_deref += 32;
  array = (int *) thisref_deref;
  return atomicAdd(array, value);
}

$$__device__$$
double edu_syr_pcpratts_rootbeer_runtime_RootbeerGpu_sin($$__global$$ char * gc_info, double value, int * exception){
  return sin(value);
}

$$__device__$$ 
void edu_syr_pcpratts_rootbeer_runtime_RootbeerGpu_syncthreads($$__global$$ char * gc_info, int * exception){
  edu_syr_pcpratts_syncthreads();
}

$$__device__$$ 
void edu_syr_pcpratts_rootbeer_runtime_RootbeerGpu_threadfence($$__global$$ char * gc_info, int * exception){
  edu_syr_pcpratts_threadfence();
}

$$__device__$$ 
void edu_syr_pcpratts_rootbeer_runtime_RootbeerGpu_threadfenceBlock($$__global$$ char * gc_info, int * exception){
  edu_syr_pcpratts_threadfence_block();
}

$$__device__$$ char
edu_syr_pcpratts_cmp(long long lhs, long long rhs){
  if(lhs > rhs)
    return 1;
  if(lhs < rhs)
    return -1;
  return 0;
}

$$__device__$$ char
edu_syr_pcpratts_cmpl(double lhs, double rhs){
  if(lhs > rhs)
    return 1;
  if(lhs < rhs)
    return -1;
  if(lhs == rhs)
    return 0;
  return -1;
}

$$__device__$$ char
edu_syr_pcpratts_cmpg(double lhs, double rhs){
  if(lhs > rhs)
    return 1;
  if(lhs < rhs)
    return -1;
  if(lhs == rhs)
    return 0;
  return 1;
}


$$__device__$$ void
edu_syr_pcpratts_gc_memcpy($$__global$$ char * dest, $$__global$$ char * src, int len) {
  int i;
  for(i = 0; i < len; ++i){
    dest[i] = src[i];
  }
}

$$__device__$$ double edu_syr_pcpratts_modulus(double a, double b)
{
  long result = (long) ( a / b );
  return a - ((double) result) * b;
}

$$__device__$$ int
edu_syr_pcpratts_gc_get_loc($$__global$$ char * mem_loc, int count){
  mem_loc += sizeof(GC_OBJ_TYPE_COUNT) + sizeof(GC_OBJ_TYPE_COLOR) +
    sizeof(char) + sizeof(GC_OBJ_TYPE_CTOR_USED) + sizeof(GC_OBJ_TYPE_SIZE) +
    sizeof(GC_OBJ_TYPE_TYPE) + count * sizeof(int);
  return (($$__global$$ int *) mem_loc)[0];
}

$$__device__$$ void
edu_syr_pcpratts_gc_set_count($$__global$$ char * mem_loc, GC_OBJ_TYPE_COUNT value){
  mem_loc[0] = value;
}

$$__device__$$ GC_OBJ_TYPE_COUNT
edu_syr_pcpratts_gc_get_count($$__global$$ char * mem_loc){
  return mem_loc[0];
}

$$__device__$$ void
edu_syr_pcpratts_gc_set_color($$__global$$ char * mem_loc, GC_OBJ_TYPE_COLOR value){
  mem_loc += sizeof(GC_OBJ_TYPE_COUNT);
  mem_loc[0] = value;
}

$$__device__$$ void
edu_syr_pcpratts_gc_init_monitor($$__global$$ char * mem_loc){
  int * addr;
  mem_loc += 16;
  addr = (int *) mem_loc;
  *addr = -1;
}

$$__device__$$ GC_OBJ_TYPE_COLOR
edu_syr_pcpratts_gc_get_color($$__global$$ char * mem_loc){
  mem_loc += sizeof(GC_OBJ_TYPE_COUNT);
  return mem_loc[0];
}

$$__device__$$ void
edu_syr_pcpratts_gc_set_type($$__global$$ char * mem_loc, GC_OBJ_TYPE_TYPE value){
  mem_loc += sizeof(GC_OBJ_TYPE_COUNT) + sizeof(GC_OBJ_TYPE_COLOR) + sizeof(char) +
    sizeof(GC_OBJ_TYPE_CTOR_USED);
  *(($$__global$$ GC_OBJ_TYPE_TYPE *) &mem_loc[0]) = value;
}

$$__device__$$ GC_OBJ_TYPE_TYPE
edu_syr_pcpratts_gc_get_type($$__global$$ char * mem_loc){
  mem_loc += sizeof(GC_OBJ_TYPE_COUNT) + sizeof(GC_OBJ_TYPE_COLOR) + sizeof(char) +
    sizeof(GC_OBJ_TYPE_CTOR_USED);
  return *(($$__global$$ GC_OBJ_TYPE_TYPE *) &mem_loc[0]);
}

$$__device__$$ void
edu_syr_pcpratts_gc_set_ctor_used($$__global$$ char * mem_loc, GC_OBJ_TYPE_CTOR_USED value){
  mem_loc += sizeof(GC_OBJ_TYPE_COUNT) + sizeof(GC_OBJ_TYPE_COLOR) + sizeof(char);
  mem_loc[0] = value;
}

$$__device__$$ GC_OBJ_TYPE_CTOR_USED
edu_syr_pcpratts_gc_get_ctor_used($$__global$$ char * mem_loc){
  mem_loc += sizeof(GC_OBJ_TYPE_COUNT) + sizeof(GC_OBJ_TYPE_COLOR) + sizeof(char);
  return mem_loc[0];
}

$$__device__$$ void
edu_syr_pcpratts_gc_set_size($$__global$$ char * mem_loc, GC_OBJ_TYPE_SIZE value){
  mem_loc += sizeof(GC_OBJ_TYPE_COUNT) + sizeof(GC_OBJ_TYPE_COLOR) + sizeof(char) + 
    sizeof(GC_OBJ_TYPE_CTOR_USED) + sizeof(GC_OBJ_TYPE_TYPE);
  *(($$__global$$ GC_OBJ_TYPE_SIZE *) &mem_loc[0]) = value;
}

$$__device__$$ GC_OBJ_TYPE_SIZE
edu_syr_pcpratts_gc_get_size($$__global$$ char * mem_loc){
  mem_loc += sizeof(GC_OBJ_TYPE_COUNT) + sizeof(GC_OBJ_TYPE_COLOR) + sizeof(char) + 
    sizeof(GC_OBJ_TYPE_CTOR_USED) + sizeof(GC_OBJ_TYPE_TYPE);
  return *(($$__global$$ GC_OBJ_TYPE_SIZE *) &mem_loc[0]);
}

$$__device__$$ char edu_syr_pcpratts_getchar($$__global$$ char * buffer, int pos){
  return buffer[pos];
}

$$__device__$$ void edu_syr_pcpratts_setchar($$__global$$ char * buffer, int pos, char value){
  buffer[pos] = value;
}

$$__device__$$ short edu_syr_pcpratts_getshort($$__global$$ char * buffer, int pos){
  return *(($$__global$$ short *) &buffer[pos]);
}

$$__device__$$ void edu_syr_pcpratts_setshort($$__global$$ char * buffer, int pos, short value){
  *(($$__global$$ short *) &buffer[pos]) = value;
}

$$__device__$$ int edu_syr_pcpratts_getint($$__global$$ char * buffer, int pos){
  return *(($$__global$$ int *) &buffer[pos]);
}

$$__device__$$ void edu_syr_pcpratts_setint($$__global$$ char * buffer, int pos, int value){
  *(($$__global$$ int *) &buffer[pos]) = value;
}

$$__device__$$ long long edu_syr_pcpratts_getlong($$__global$$ char * buffer, int pos){
  return *(($$__global$$ long *) &buffer[pos]);
}

$$__device__$$ void edu_syr_pcpratts_setlong($$__global$$ char * buffer, int pos, long long value){
  *(($$__global$$ long long *) &buffer[pos]) = value;
}

$$__device__$$ size_t edu_syr_pcpratts_getsize_t($$__global$$ char * buffer, int pos){
  return *(($$__global$$ size_t *) &buffer[pos]);
}

$$__device__$$ void edu_syr_pcpratts_setsize_t($$__global$$ char * buffer, int pos, size_t value){
  *(($$__global$$ size_t *) &buffer[pos]) = value;
}

$$__device__$$ void
edu_syr_pcpratts_gc_set_to_space_address($$__global$$ char * gc_info, $$__global$$ char * value){
  edu_syr_pcpratts_setlong(gc_info, TO_SPACE_OFFSET, (long long) value);
}

$$__device__$$ $$__global$$ long long *
edu_syr_pcpratts_gc_get_to_space_address($$__global$$ char * gc_info){
  long long value = edu_syr_pcpratts_getlong(gc_info, TO_SPACE_OFFSET);
  return ($$__global$$ long long *) value;
}

$$__device__$$ long long
edu_syr_pcpratts_gc_get_to_space_free_ptr($$__global$$ char * gc_info){
  return edu_syr_pcpratts_getlong(gc_info, TO_SPACE_FREE_POINTER_OFFSET);
}

$$__device__$$ void
edu_syr_pcpratts_gc_set_to_space_free_ptr($$__global$$ char * gc_info, long long value){
  edu_syr_pcpratts_setlong(gc_info, TO_SPACE_FREE_POINTER_OFFSET, value);
}

$$__device__$$ int
edu_syr_pcpratts_gc_get_space_size($$__global$$ char * gc_info){
  return edu_syr_pcpratts_getint(gc_info, SPACE_SIZE_OFFSET);
}

$$__device__$$ int
edu_syr_pcpratts_strlen(char * str_constant){
  int ret = 0;
  while(1){
    if(str_constant[ret] != '\0'){
      ret++;
    } else {
      return ret;
    }
  }
}

$$__device__$$ int
edu_syr_pcpratts_array_length($$__global$$ char * gc_info, int thisref){
  //if(thisref & 0x1000000000000000L){
  //  thisref &= 0x0fffffffffffffffL;
  //  thisref += 8;
  //  return edu_syr_pcpratts_cache_get_int(thisref);
  //} else {
    $$__global$$ char * thisref_deref = edu_syr_pcpratts_gc_deref(gc_info, thisref);
    int ret = edu_syr_pcpratts_getint(thisref_deref, 12);
    return ret;
  //}
}

$$__device__$$
int java_lang_StringBuilder_initab850b60f96d11de8a390800200c9a660_(char * gc_info, int * exception){ 
  int thisref;
  char * thisref_deref;
  int chars;

  thisref = edu_syr_pcpratts_gc_malloc(gc_info , 48);
  if(thisref == -1){
    *exception = %%java_lang_NullPointerException_TypeNumber%%; 
    return -1; 
  }

  thisref_deref = edu_syr_pcpratts_gc_deref(gc_info, thisref);
  edu_syr_pcpratts_gc_set_count(thisref_deref, 1); 
  edu_syr_pcpratts_gc_set_color(thisref_deref, COLOR_GREY); 
  edu_syr_pcpratts_gc_set_type(thisref_deref, %%java_lang_String_TypeNumber%%); 
  edu_syr_pcpratts_gc_set_ctor_used(thisref_deref, 1); 
  edu_syr_pcpratts_gc_set_size(thisref_deref, 48); 
  edu_syr_pcpratts_gc_init_monitor(thisref_deref); 

  chars = char__array_new(gc_info, 0, exception);
  instance_setter_java_lang_AbstractStringBuilder_value(gc_info, thisref, chars, exception); 
  instance_setter_java_lang_AbstractStringBuilder_count(gc_info, thisref, 0, exception);
  return thisref; 
}

$$__device__$$
int java_lang_String_initab850b60f96d11de8a390800200c9a66(char * gc_info, int parameter0, int * exception) { 
  int r0 = -1; 
  int r1 = -1; 
  int i0; 
  int $r2 = -1; 
  int thisref; 
  char * thisref_deref; 
  int i;
  int len;
  int characters_copy;
  char ch;
  
  thisref = -1; 
  edu_syr_pcpratts_gc_assign(gc_info, &thisref, edu_syr_pcpratts_gc_malloc(gc_info, 48)); 
  if(thisref == -1) { 
    *exception = %%java_lang_NullPointerException_TypeNumber%%; 
    return -1; 
  } 
  thisref_deref = edu_syr_pcpratts_gc_deref(gc_info, thisref); 
  edu_syr_pcpratts_gc_set_count(thisref_deref, 1); 
  edu_syr_pcpratts_gc_set_color(thisref_deref, COLOR_GREY); 
  edu_syr_pcpratts_gc_set_type(thisref_deref, %%java_lang_String_TypeNumber%%); 
  edu_syr_pcpratts_gc_set_ctor_used(thisref_deref, 1); 
  edu_syr_pcpratts_gc_set_size(thisref_deref, 48); 
  edu_syr_pcpratts_gc_init_monitor(thisref_deref); 

  len = edu_syr_pcpratts_array_length(gc_info, parameter0);
  characters_copy = char__array_new(gc_info, len, exception);
  for(i = 0; i < len; ++i){
    ch = char__array_get(gc_info, parameter0, i, exception);
    char__array_set(gc_info, characters_copy, i, ch, exception);
  }
  instance_setter_java_lang_String_value(gc_info, thisref, characters_copy, exception); 
  instance_setter_java_lang_String_count(gc_info, thisref, len, exception); 
  instance_setter_java_lang_String_offset(gc_info, thisref, 0, exception); 
  return thisref; 
} 

$$__device__$$ int 
char__array_new($$__global$$ char * gc_info, int size, int * exception);

$$__device__$$ void 
char__array_set($$__global$$ char * gc_info, int thisref, int parameter0, char parameter1, int * exception);

$$__device__$$ int
edu_syr_pcpratts_string_constant($$__global$$ char * gc_info, char * str_constant, int * exception){
  int i;
  int len = edu_syr_pcpratts_strlen(str_constant);
  int characters = char__array_new(gc_info, len, exception);
  unsigned long long * addr = (unsigned long long *) (gc_info + TO_SPACE_FREE_POINTER_OFFSET);
  for(i = 0; i < len; ++i){
    char__array_set(gc_info, characters, i, str_constant[i], exception);
  }

  return java_lang_String_initab850b60f96d11de8a390800200c9a66(gc_info, characters, exception);
}

$$__device__$$ void
edu_syr_pcpratts_gc_assign($$__global$$ char * gc_info, int * lhs_ptr, int rhs){
  *lhs_ptr = rhs;
}

$$__device__$$ void
edu_syr_pcpratts_gc_assign_global($$__global$$ char * gc_info, $$__global$$ int * lhs_ptr, int rhs){
  *lhs_ptr = rhs;
}
 
$$__device__$$ int java_lang_StackTraceElement__array_get($$__global$$ char * gc_info, int thisref, int parameter0, int * exception);
$$__device__$$ void java_lang_StackTraceElement__array_set($$__global$$ char * gc_info, int thisref, int parameter0, int parameter1, int * exception);
$$__device__$$ int java_lang_StackTraceElement__array_new($$__global$$ char * gc_info, int size, int * exception);
$$__device__$$ int java_lang_StackTraceElement_initab850b60f96d11de8a390800200c9a660_3_3_3_4_($$__global$$ char * gc_info, int parameter0, int parameter1, int parameter2, int parameter3, int * exception);
$$__device__$$ void instance_setter_java_lang_RuntimeException_stackDepth($$__global$$ char * gc_info, int thisref, int parameter0);
$$__device__$$ int instance_getter_java_lang_RuntimeException_stackDepth($$__global$$ char * gc_info, int thisref);
$$__device__$$ int java_lang_StackTraceElement__array_get($$__global$$ char * gc_info, int thisref, int parameter0, int * exception);
$$__device__$$ int instance_getter_java_lang_Throwable_stackTrace($$__global$$ char * gc_info, int thisref, int * exception);
$$__device__$$ void instance_setter_java_lang_Throwable_stackTrace($$__global$$ char * gc_info, int thisref, int parameter0, int * exception);

$$__device__$$ int java_lang_Throwable_fillInStackTrace($$__global$$ char * gc_info, int thisref, int * exception){
  //int trace = java_lang_StackTraceElement__array_new(gc_info, 8, exception);
  //instance_setter_java_lang_Throwable_stackTrace(gc_info, thisref, trace, exception);
  return thisref;
}

$$__device__$$ int java_lang_Throwable_getStackTraceElement($$__global$$ char * gc_info, int thisref, int parameter0, int * exception){
  //int array = instance_getter_java_lang_Throwable_stackTrace(gc_info, thisref, exception);
  //return java_lang_StackTraceElement__array_get(gc_info, array, parameter0, exception);
  return -1;
}

$$__device__$$ int java_lang_Throwable_getStackTraceDepth($$__global$$ char * gc_info, int thisref, int * exception){
  return 0;
}

$$__device__$$ void edu_syr_pcpratts_fillInStackTrace($$__global$$ char * gc_info, int exception, char * class_name, char * method_name){
}

$$__device__$$ void instance_setter_java_lang_Throwable_cause($$__global$$ char * gc_info, int thisref, int parameter0, int * exception);
$$__device__$$ void instance_setter_java_lang_Throwable_detailMessage($$__global$$ char * gc_info, int thisref, int parameter0, int * exception);
$$__device__$$ void instance_setter_java_lang_Throwable_stackDepth($$__global$$ char * gc_info, int thisref, int parameter0, int * exception);
$$__device__$$ void java_lang_VirtualMachineError_initab850b60f96d11de8a390800200c9a66_body0_($$__global$$ char * gc_info, int thisref, int * exception);

$$__device__$$ int java_lang_OutOfMemoryError_initab850b60f96d11de8a390800200c9a66($$__global$$ char * gc_info, int * exception){
  int r0 = -1;
  int thisref = edu_syr_pcpratts_gc_malloc(gc_info, 40);
  char * thisref_deref = edu_syr_pcpratts_gc_deref(gc_info, thisref);

  //class info
  edu_syr_pcpratts_gc_set_count(thisref_deref, 0);
  edu_syr_pcpratts_gc_set_color(thisref_deref, COLOR_GREY);
  edu_syr_pcpratts_gc_set_type(thisref_deref, 9);
  edu_syr_pcpratts_gc_set_ctor_used(thisref_deref, 1);
  edu_syr_pcpratts_gc_set_size(thisref_deref, 40);

  instance_setter_java_lang_Throwable_cause(gc_info, thisref, -1, exception);
  instance_setter_java_lang_Throwable_detailMessage(gc_info, thisref, -1, exception);
  instance_setter_java_lang_Throwable_stackTrace(gc_info, thisref, -1, exception);

  //r0 := @this: java.lang.OutOfMemoryError
  edu_syr_pcpratts_gc_assign(gc_info, & r0 ,  thisref );

  //specialinvoke r0.<java.lang.VirtualMachineError: void <init>()>()
  java_lang_VirtualMachineError_initab850b60f96d11de8a390800200c9a66_body0_(gc_info,
   thisref, exception);
  return thisref;
}


$$__device__$$ int
java_lang_Object_hashCode($$__global$$ char * gc_info, int thisref, int * exception){
  return thisref;
}

$$__device__$$ int
java_lang_Class_getName( char * gc_info , int thisref , int * exception ) { 
  int $r1 =-1 ; 
  $r1 = instance_getter_java_lang_Class_name ( gc_info , thisref , exception ) ; 
  if ( * exception != 0 ) { 
    return 0 ; 
  } 
  return $r1;
}

$$__device__$$ int
java_lang_Object_getClass( char * gc_info , int thisref, int * exception ) { 
  char * mem_loc = edu_syr_pcpratts_gc_deref(gc_info, thisref);
  int type = edu_syr_pcpratts_gc_get_type(mem_loc);
  return edu_syr_pcpratts_classConstant(type);
}

$$__device__$$ int
java_lang_StringValue_from( char * gc_info , int thisref, int * exception ) { 
  int i, size, new_ref;
  char * mem_loc, * new_mem_loc;
  
  mem_loc = edu_syr_pcpratts_gc_deref(gc_info, thisref);
  size = edu_syr_pcpratts_gc_get_size(mem_loc);
  new_ref = edu_syr_pcpratts_gc_malloc(gc_info, size);
  new_mem_loc = edu_syr_pcpratts_gc_deref(gc_info, new_ref);
  
  for(i = 0; i < size; ++i){
    new_mem_loc[i] = mem_loc[i];  
  }
  
  return new_ref;
}

$$__device__$$ int
java_util_Arrays_copyOf(char * gc_info, int object_array, int new_size, int * exception ){
  int ret;
  char * ret_deref;
  char * object_array_deref;
  int length;
  int i;
  
  ret = edu_syr_pcpratts_gc_malloc(gc_info, 32 + (4 * new_size));
  ret_deref = edu_syr_pcpratts_gc_deref(gc_info, ret);
  object_array_deref = edu_syr_pcpratts_gc_deref(gc_info, object_array);
    
  for(i = 0; i < 32; ++i){
    ret_deref[i] = object_array_deref[i];
  }

  length = edu_syr_pcpratts_getint(object_array_deref, 12);
  edu_syr_pcpratts_setint(ret_deref, 8, 32 + (4 * new_size));
  edu_syr_pcpratts_setint(ret_deref, 12, new_size);

  if(length < new_size){
    for(i = 0; i < length * 4; ++i){
      ret_deref[32+i], object_array_deref[32+i];
    }
    int diff = new_size - length;
    for(i = 0; i < diff; ++i){
      * ((int *) &ret_deref[32 + (length * 4) + (i * 4)]) = -1;
    }
  } else {
    for(i = 0; i < new_size * 4; ++i){
      ret_deref[32+i], object_array_deref[32+i];
    }
  }

  return ret; 
}

$$__device__$$ 
int java_lang_StringBuilder_initab850b60f96d11de8a390800200c9a6610_9_(char * gc_info, 
  int str ,int * exception){
 
  int r0 = -1; 
  int thisref; 
  int str_value;
  int str_count;  

  char * thisref_deref; 
  thisref = -1;
  edu_syr_pcpratts_gc_assign ( gc_info , & thisref , edu_syr_pcpratts_gc_malloc ( gc_info , 48 ) ) ; 
  if ( thisref ==-1 ) { 
    * exception = %%java_lang_NullPointerException_TypeNumber%%; 
    return-1 ; 
  } 
  thisref_deref = edu_syr_pcpratts_gc_deref ( gc_info , thisref ) ; 
  edu_syr_pcpratts_gc_set_count ( thisref_deref , 0 ) ; 
  edu_syr_pcpratts_gc_set_color ( thisref_deref , COLOR_GREY ) ; 
  edu_syr_pcpratts_gc_set_type ( thisref_deref , %%java_lang_StringBuilder_TypeNumber%% ) ; 
  edu_syr_pcpratts_gc_set_ctor_used ( thisref_deref , 1 ) ; 
  edu_syr_pcpratts_gc_set_size ( thisref_deref , 44 ) ; 
  edu_syr_pcpratts_gc_init_monitor ( thisref_deref ) ; 

  str_value = instance_getter_java_lang_String_value(gc_info, str, exception);
  str_count = instance_getter_java_lang_String_count(gc_info, str, exception);

  instance_setter_java_lang_AbstractStringBuilder_value(gc_info, thisref, str_value, exception); 
  instance_setter_java_lang_AbstractStringBuilder_count(gc_info, thisref, str_count, exception); 
  return thisref; 
} 

//<java.lang.StringBuilder: java.lang.StringBuilder append(java.lang.String)>
$$__device__$$ 
int java_lang_StringBuilder_append10_9_(char * gc_info, int thisref,
  int parameter0, int * exception){

  int sb_value;
  int sb_count;
  int str_value;
  int str_count;
  int new_count;
  int new_sb_value;
  int i;
  char ch;
  int new_str;

  //get string builder value and count
  sb_value = instance_getter_java_lang_AbstractStringBuilder_value(gc_info, thisref,
    exception);

  sb_count = instance_getter_java_lang_AbstractStringBuilder_count(gc_info, thisref,
    exception);

  //get string value and count
  str_value = instance_getter_java_lang_String_value(gc_info, parameter0,
    exception);

  str_count = instance_getter_java_lang_String_count(gc_info, parameter0,
    exception);

  new_count = sb_count + str_count;
  new_sb_value = char__array_new(gc_info, new_count, exception);
  for(i = 0; i < sb_count; ++i){
    ch = char__array_get(gc_info, sb_value, i, exception);
    char__array_set(gc_info, new_sb_value, i, ch, exception);
  }
  for(i = 0; i < str_count; ++i){
    ch = char__array_get(gc_info, str_value, i, exception);
    char__array_set(gc_info, new_sb_value, sb_count + i, ch, exception);
  }

  //make new String
  new_str = java_lang_String_initab850b60f96d11de8a390800200c9a66(gc_info, 
    new_sb_value, exception);

  //return new StringBuilder from String
  return java_lang_StringBuilder_initab850b60f96d11de8a390800200c9a6610_9_(gc_info,
    new_str, exception);
}

//<java.lang.StringBuilder: java.lang.StringBuilder append(boolean)>
$$__device__$$ 
int java_lang_StringBuilder_append10_1_(char * gc_info, int thisref,
  bool parameter0, int * exception){
  
  int str = java_lang_Boolean_toString9_1_(gc_info, parameter0, exception);
  return java_lang_StringBuilder_append10_9_(gc_info, thisref, str, exception);
}

//<java.lang.StringBuilder: java.lang.StringBuilder append(char)>
$$__device__$$ 
int java_lang_StringBuilder_append10_3_(char * gc_info, int thisref,
  int parameter0, int * exception){
  
  int str = java_lang_Character_toString9_3_(gc_info, parameter0, exception);
  return java_lang_StringBuilder_append10_9_(gc_info, thisref, str, exception);
}

//<java.lang.StringBuilder: java.lang.StringBuilder append(double)>
$$__device__$$ 
int java_lang_StringBuilder_append10_8_(char * gc_info, int thisref,
  double parameter0, int * exception){
  
  int str = java_lang_Double_toString9_8_(gc_info, parameter0, exception);
  return java_lang_StringBuilder_append10_9_(gc_info, thisref, str, exception);
}

//<java.lang.StringBuilder: java.lang.StringBuilder append(float)>
$$__device__$$ 
int java_lang_StringBuilder_append10_7_(char * gc_info, int thisref,
  float parameter0, int * exception){
  
  int str = java_lang_Float_toString9_7_(gc_info, parameter0, exception);
  return java_lang_StringBuilder_append10_9_(gc_info, thisref, str, exception);
}

//<java.lang.StringBuilder: java.lang.StringBuilder append(int)>
$$__device__$$ 
int java_lang_StringBuilder_append10_5_(char * gc_info, int thisref,
  int parameter0, int * exception){

  int str = java_lang_Integer_toString9_5_(gc_info, parameter0, exception);
  return java_lang_StringBuilder_append10_9_(gc_info, thisref, str, exception);
}

//<java.lang.StringBuilder: java.lang.StringBuilder append(long)>
$$__device__$$ 
int java_lang_StringBuilder_append10_6_(char * gc_info, int thisref,
  long long parameter0, int * exception){

  int str = java_lang_Long_toString9_6_(gc_info, parameter0, exception);
  return java_lang_StringBuilder_append10_9_(gc_info, thisref, str, exception);
}

//<java.lang.StringBuilder: java.lang.String toString()>
$$__device__$$ 
int java_lang_StringBuilder_toString9_(char * gc_info, int thisref,
  int * exception){
 
  int value = instance_getter_java_lang_AbstractStringBuilder_value(gc_info, thisref,
    exception);
  return java_lang_String_initab850b60f96d11de8a390800200c9a66(gc_info, value, 
    exception);
}

/*****************************************************************************/
/* local methods */

// string length using volatile argument
$$__device__$$
int at_illecker_strlen(volatile char * str_constant) {
  int ret = 0;
  while(1) {
    if(str_constant[ret] != '\0') {
      ret++;
    } else {
      return ret;
    }
  }
}

// char* to String using volatile argument
$$__device__$$
int at_illecker_string_constant(char * gc_info, volatile char * str_constant, int * exception) {
  if (str_constant == 0) {
    return 0;
  }
  int i;
  int len = at_illecker_strlen(str_constant);
  int characters = char__array_new(gc_info, len, exception);
  
  printf("at_illecker_string_constant str: '"); 
  for(i = 0; i < len; ++i) {
    char__array_set(gc_info, characters, i, str_constant[i], exception);
    printf("%c",str_constant[i]);
  }
  printf("'\n");  

  // make new String
  return java_lang_String_initab850b60f96d11de8a390800200c9a66(gc_info, characters, exception);
}

/*****************************************************************************/
/* toString methods */

$$__device__$$
double at_illecker_abs_val(double value) {
  double result = value;
  if (value < 0) {
    result = -value;
  }
  return result;
}

$$__device__$$
double at_illecker_pow10(int exp) {
  double result = 1;
  while (exp) {
    result *= 10;
    exp--;
  }
  return result;
}

$$__device__$$
long at_illecker_round(double value) {
  long intpart;
  intpart = value;
  value = value - intpart;
  if (value >= 0.5) {
    intpart++;
  }
  return intpart;
}

$$__device__$$
void at_illecker_set_char(char *buffer, int *currlen, int maxlen, char c) {
  if (*currlen < maxlen) {
    buffer[(*currlen)++] = c;
  }
}

// local double to string method
// http://www.opensource.apple.com/source/srm/srm-6/srm/lib/snprintf.c
$$__device__$$
int at_illecker_double_to_string(char * gc_info, double fvalue, int max, int * exception) {
  int signvalue = 0;
  double ufvalue;
  long intpart;
  long fracpart;
  char iconvert[20];
  char fconvert[20];
  int iplace = 0;
  int fplace = 0;
  int zpadlen = 0; // lasting zeros

  char buffer[64];
  int maxlen = 64;
  int currlen = 0;

  // DEBUG
  // printf("at_illecker_doubleToString: fvalue: %f max: %d\n", fvalue, max);

  // Max digits after decimal point, default is 6
  if (max < 0) {
    max = 6;
  }
  // Sorry, we only support 9 digits past the decimal because of our 
  // conversion method
  if (max > 9) {
    max = 9;
  }

  // Set sign if negative
  if (fvalue < 0) {
    signvalue = '-';
  }

  ufvalue = at_illecker_abs_val(fvalue);
  intpart = ufvalue;

  // We "cheat" by converting the fractional part to integer by
  // multiplying by a factor of 10
  fracpart = at_illecker_round(at_illecker_pow10(max) * (ufvalue - intpart));

  if (fracpart >= at_illecker_pow10(max)) {
    intpart++;
    fracpart -= at_illecker_pow10(max);
  }

  // DEBUG
  // printf("at_illecker_doubleToString: %f =? %d.%d\n", fvalue, intpart, fracpart);

  // Convert integer part
  do {
    iconvert[iplace++] = "0123456789abcdef"[intpart % 10];
    intpart = (intpart / 10);
  } while(intpart && (iplace < 20));

  if (iplace == 20) {
    iplace--;
  }
  iconvert[iplace] = 0;

  // Convert fractional part
  do {
    fconvert[fplace++] = "0123456789abcdef"[fracpart % 10];
    fracpart = (fracpart / 10);
  } while(fracpart && (fplace < 20));
  
  if (fplace == 20) {
    fplace--;
  }
  fconvert[fplace] = 0;

  // Calc lasting zeros for padding
  zpadlen = max - fplace;
  if (zpadlen < 0) {
    zpadlen = 0;
  }

  //  DEBUG
  // printf("at_illecker_doubleToString: zpadlen: %d\n", zpadlen);

  // Set sign
  if (signvalue) {
    at_illecker_set_char(buffer, &currlen, maxlen, signvalue);
  }

  // Set integer part
  while (iplace > 0) {
    at_illecker_set_char(buffer, &currlen, maxlen, iconvert[--iplace]);
  }

  // Check if decimal point is needed
  if (max > 0) {
    // Set decimal point
    // This should probably use locale to find the correct
    // char to print out.
    at_illecker_set_char(buffer, &currlen, maxlen, '.');

    while (fplace > 0) {
      at_illecker_set_char(buffer, &currlen, maxlen, fconvert[--fplace]);
    }
  }

  // Add lasting zeros
  while (zpadlen > 0) {
    at_illecker_set_char(buffer, &currlen, maxlen, '0');
    --zpadlen;
  }

  // Terminate string
  if (currlen < maxlen - 1) {
    buffer[currlen] = '\0';
  } else {
    buffer[maxlen - 1] = '\0';
  }

  return at_illecker_string_constant(gc_info, buffer, exception);
}

//<java.lang.Double: java.lang.String toString(double)>
$$__device__$$ 
int java_lang_Double_toString9_8_(char * gc_info, double double_val, int * exception) {

  // Default is 6 digits after decimal point
  return at_illecker_double_to_string(gc_info, double_val, 6, exception);
}

//<java.lang.Float: java.lang.String toString(float)>
$$__device__$$ 
int java_lang_Float_toString9_7_(char * gc_info, float float_val, int * exception){

  // Default is 6 digits after decimal point
  return at_illecker_double_to_string(gc_info, (double)float_val, 6, exception);
}

/*****************************************************************************/
/* String.indexOf methods */

// Returns the position of the first character of the first match.
// If no matches were found, the function returns -1
$$__device__$$
int at_illecker_strpos(char * gc_info, int str_value, int str_count, 
                       int sub_str_value, int sub_str_count, 
                       int start_pos, int * exception) {

  if ( (str_count == 0) || (sub_str_count == 0) || 
       (start_pos > str_count)) {
    return -1;
  }

  for (int i = start_pos; i < str_count; i++) {
    if (char__array_get(gc_info, str_value, i, exception) != 
        char__array_get(gc_info, sub_str_value, 0, exception)) {
      continue;
    }
    int found_pos = i;
    int found_sub_string = true;
    for (int j = 1; j < sub_str_count; j++) {
      i++;
      if (char__array_get(gc_info, str_value, i, exception) != 
          char__array_get(gc_info, sub_str_value, j, exception)) {
        found_sub_string = false;
        break;
      }
    }
    if (found_sub_string) {
      return found_pos;
    }
  }
  return -1;
}

//<java.lang.String: int indexOf(java.lang.String)>
$$__device__$$
int java_lang_String_indexOf(char * gc_info, int str_obj_ref, 
                             int search_str_obj_ref, int * exception) {
  int str_value = 0;
  int str_count = 0;
  int search_str_value = 0;
  int search_str_count = 0;
  
  str_value = instance_getter_java_lang_String_value(gc_info, str_obj_ref, exception);
  str_count = instance_getter_java_lang_String_count(gc_info, str_obj_ref, exception);
  search_str_value = instance_getter_java_lang_String_value(gc_info, search_str_obj_ref, exception);
  search_str_count = instance_getter_java_lang_String_count(gc_info, search_str_obj_ref, exception);

  return at_illecker_strpos(gc_info, str_value, str_count, search_str_value, search_str_count, 0, exception);
}

//<java.lang.String: int indexOf(java.lang.String, int fromIndex)>
$$__device__$$
int java_lang_String_indexOf(char * gc_info, int str_obj_ref, 
                             int search_str_obj_ref, int from_index, int * exception) {
  int str_value = 0;
  int str_count = 0;
  int search_str_value = 0;
  int search_str_count = 0;
  
  str_value = instance_getter_java_lang_String_value(gc_info, str_obj_ref, exception);
  str_count = instance_getter_java_lang_String_count(gc_info, str_obj_ref, exception);
  search_str_value = instance_getter_java_lang_String_value(gc_info, search_str_obj_ref, exception);
  search_str_count = instance_getter_java_lang_String_count(gc_info, search_str_obj_ref, exception);

  return at_illecker_strpos(gc_info, str_value, str_count, search_str_value, search_str_count, from_index, exception);
}

/*****************************************************************************/
/* String.substring methods */

// Returns a substring from given start index
$$__device__$$
int at_illecker_substring(char * gc_info, int str_value, int str_count, 
                       int begin_index, int end_index, int * exception) {
  int new_length = 0;
  int new_string = -1;

  // set new length
  if (end_index == -1) { // copy to end
    new_length = str_count - begin_index;
  } else {
    if (end_index < str_count) {
      new_length = end_index - begin_index;
    } else {
      new_length = str_count - begin_index;
    }
  }
 
  // printf("at_illecker_substring begin_index: %d, end_index: %d, new_length: %d\n", begin_index, end_index, new_length);
  new_string = char__array_new(gc_info, new_length, exception);

  for(int i = 0; i < new_length; i++) {
    char__array_set(gc_info, new_string, i, char__array_get(gc_info, str_value, begin_index, exception), exception);
    begin_index++;
  }

  return java_lang_String_initab850b60f96d11de8a390800200c9a66(gc_info, new_string, exception);
}

//<java.lang.String: java.lang.String substring(int)>
$$__device__$$
int java_lang_String_substring(char * gc_info, int str_obj_ref, int begin_index, int * exception) {
  int str_value = 0;
  int str_count = 0;

  str_value = instance_getter_java_lang_String_value(gc_info, str_obj_ref, exception);
  str_count = instance_getter_java_lang_String_count(gc_info, str_obj_ref, exception);

  return at_illecker_substring(gc_info, str_value, str_count, begin_index, -1, exception);
}

//<java.lang.String: java.lang.String substring(int,int)>
$$__device__$$
int java_lang_String_substring(char * gc_info, int str_obj_ref, int begin_index, 
                               int end_index, int * exception) {
  int str_value = 0;
  int str_count = 0;

  str_value = instance_getter_java_lang_String_value(gc_info, str_obj_ref, exception);
  str_count = instance_getter_java_lang_String_count(gc_info, str_obj_ref, exception);

  return at_illecker_substring(gc_info, str_value, str_count, begin_index, end_index, exception);
}


/*****************************************************************************/
/* String.split methods */

// Returns the amount of occurrences of substring in string
// If no matches were found, the function returns 0
$$__device__$$
int at_illecker_strcnt(char * gc_info, int str_value, int str_count, 
                       int sub_str_value, int sub_str_count, int * exception) {
  int occurrences = 0;

  if ( (str_count == 0) || (sub_str_count == 0) ) {
    return 0;
  }

  for (int i = 0; i < str_count; i++) {
    if (char__array_get(gc_info, str_value, i, exception) != 
        char__array_get(gc_info, sub_str_value, 0, exception)) {
      continue;
    }
    bool found_sub_string = true;
    for (int j = 1; j < sub_str_count; j++) {
      i++;
      if (char__array_get(gc_info, str_value, i, exception) != 
          char__array_get(gc_info, sub_str_value, j, exception)) {
        found_sub_string = false;
        break;
      }
    }
    if (found_sub_string) {
      occurrences++;
    }
  }
  return occurrences;
}

// local split method
$$__device__$$
int at_illecker_split(char * gc_info, int str_value, int str_count, 
                      int delim_str_value, int delim_str_count,
                      int delim_occurrences, int * exception) {
  int return_obj = -1;
  int start = 0;
  int end = 0;

  // printf("at_illecker_split: delim_occurrences: %d\n", delim_occurrences);

  return_obj = java_lang_String__array_new(gc_info, delim_occurrences + 1, exception);

  for (int i = 0; i < delim_occurrences; i++) {
    end = at_illecker_strpos(gc_info, str_value, str_count, 
                             delim_str_value, delim_str_count, start, exception);

    if (end == -1) {
      break;
    }

    // add token - substring(start, end - start)
    java_lang_String__array_set(gc_info, return_obj, i,
      at_illecker_substring(gc_info, str_value, str_count, start, end, exception), exception);

    // Exclude the delimiter in the next search
    start = end + delim_str_count;
  }

  // add last token
  if ( (delim_occurrences > 0) && (end != -1) ) {

    // substring(start, END_OF_STRING)
    java_lang_String__array_set(gc_info, return_obj, delim_occurrences,
      at_illecker_substring(gc_info, str_value, str_count, start, -1, exception), exception);
  }

  //TODO if delim_occurrences > current delimiters -> add emtpy string
  //TODO if delim_occurrences < current delimiters -> parse last token until next delimiter

  return return_obj;
}

//<java.lang.String: java.lang.String[] split(java.lang.String,int)>
$$__device__$$
int java_lang_String_split(char * gc_info, int str_obj_ref, int delim_str_obj_ref, int limit, int * exception) {
  int str_value = 0;
  int str_count = 0;
  int delim_str_value = 0;
  int delim_str_count = 0;
  
  str_value = instance_getter_java_lang_String_value(gc_info, str_obj_ref, exception);
  str_count = instance_getter_java_lang_String_count(gc_info, str_obj_ref, exception);
  delim_str_value = instance_getter_java_lang_String_value(gc_info, delim_str_obj_ref, exception);
  delim_str_count = instance_getter_java_lang_String_count(gc_info, delim_str_obj_ref, exception);

  printf("java_lang_String_split: limit: %d\n", limit);

  return at_illecker_split(gc_info, str_value, str_count, delim_str_value, delim_str_count, limit-1, exception);
}

//<java.lang.String: java.lang.String[] split(java.lang.String)>
$$__device__$$
int java_lang_String_split(char * gc_info, int str_obj_ref, int delim_str_obj_ref, int * exception) {
  int occurrences = 0;
  int str_value = 0;
  int str_count = 0;
  int delim_str_value = 0;
  int delim_str_count = 0;
  
  str_value = instance_getter_java_lang_String_value(gc_info, str_obj_ref, exception);
  str_count = instance_getter_java_lang_String_count(gc_info, str_obj_ref, exception);
  delim_str_value = instance_getter_java_lang_String_value(gc_info, delim_str_obj_ref, exception);
  delim_str_count = instance_getter_java_lang_String_count(gc_info, delim_str_obj_ref, exception);
 
  occurrences = at_illecker_strcnt(gc_info, str_value, str_count, 
                  delim_str_value, delim_str_count, exception);

  // TODO check if occurrences == 0

  printf("java_lang_String_split: occurrences: %d\n", occurrences);

  return at_illecker_split(gc_info, str_value, str_count, delim_str_value, delim_str_count, occurrences, exception);
}

/*****************************************************************************/
/* valueOf methods */

//<java.lang.Integer: java.lang.Integer valueOf(int)>
$$__device__$$
int java_lang_Integer_valueOf(char * gc_info, int int_value, int * exception) {
  int return_obj = -1;
  
  edu_syr_pcpratts_gc_assign (gc_info, 
    &return_obj, java_lang_Integer_initab850b60f96d11de8a390800200c9a660_5_(gc_info,
    int_value , exception));
  
  if(*exception != 0) {
    return 0; 
  }
  return return_obj;
}

/*****************************************************************************/
/* Parse methods */
$$__device__$$
bool at_illecker_is_digit(unsigned char c) {
  return ((c)>='0' && (c)<='9');
}

$$__device__$$
bool at_illecker_is_space(unsigned char c) {
  return ((c)==' ' || (c)=='\f' || (c)=='\n' || (c)=='\r' || (c)=='\t' || (c)=='\v');
}

// local string to unsigned long method
// http://www.opensource.apple.com/source/tcl/tcl-14/tcl/compat/strtoul.c
//
/* Argument1: String of ASCII digits, possibly
 * preceded by white space.  For bases
 * greater than 10, either lower- or
 * upper-case digits may be used.
 */
/* Argument2: Where to store address of terminating
 * character, or NULL.
 */
/* Argument3: Base for conversion.  Must be less
 * than 37.  If 0, then the base is chosen
 * from the leading characters of string:
 * "0x" means hex, "0" means octal, anything
 * else means decimal.
 */
$$__device__$$
unsigned long int at_illecker_strtoul(const char *string, char **end_ptr, int base) {
  register const char *p;
  register unsigned long int result = 0;
  register unsigned digit;
  int anyDigits = 0;
  int negative=0;
  int overflow=0;

  char cvtIn[] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,		/* '0' - '9' */
    100, 100, 100, 100, 100, 100, 100,		/* punctuation */
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,	/* 'A' - 'Z' */
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    30, 31, 32, 33, 34, 35,
    100, 100, 100, 100, 100, 100,		/* punctuation */
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,	/* 'a' - 'z' */
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    30, 31, 32, 33, 34, 35
  };

  // Skip any leading blanks.
  p = string;
  while (at_illecker_is_space((unsigned char) (*p))) {
    p += 1;
  }
  // Check for a sign.
  if (*p == '-') {
    negative = 1;
    p += 1;
  } else {
    if (*p == '+') {
      p += 1;
    }
  }

  // If no base was provided, pick one from the leading characters
  // of the string.
  if (base == 0) {
    if (*p == '0') {
      p += 1;
      if ((*p == 'x') || (*p == 'X')) {
        p += 1;
        base = 16;
      } else {
        // Must set anyDigits here, otherwise "0" produces a
        // "no digits" error.
        anyDigits = 1;
        base = 8;
      }
    } else {
      base = 10;
    }
  } else if (base == 16) {
    // Skip a leading "0x" from hex numbers.
    if ((p[0] == '0') && ((p[1] == 'x') || (p[1] == 'X'))) {
      p += 2;
    }
  }

  // Sorry this code is so messy, but speed seems important. Do
  // different things for base 8, 10, 16, and other.
  if (base == 8) {
    unsigned long maxres = 0xFFFFFFFFUL >> 3; // ULONG_MAX = 0xFFFFFFFFUL
    for ( ; ; p += 1) {
      digit = *p - '0';
      if (digit > 7) {
        break;
      }
      if (result > maxres) { 
        overflow = 1;
      }
      result = (result << 3);
      if (digit > (0xFFFFFFFFUL - result)) { 
        overflow = 1;
      }
      result += digit;
      anyDigits = 1;
    }
  } else if (base == 10) {
    unsigned long maxres = 0xFFFFFFFFUL / 10; // ULONG_MAX = 0xFFFFFFFFUL
    for ( ; ; p += 1) {
      digit = *p - '0';
      if (digit > 9) {
        break;
      }
      if (result > maxres) { 
        overflow = 1;
      }
      result *= 10;
      if (digit > (0xFFFFFFFFUL - result)) { 
        overflow = 1;
      }
      result += digit;
      anyDigits = 1;
    }
  } else if (base == 16) {
    unsigned long maxres = 0xFFFFFFFFUL >> 4;
    for ( ; ; p += 1) {
      digit = *p - '0';
      if (digit > ('z' - '0')) {
        break;
      }
      digit = cvtIn[digit];
      if (digit > 15) {
        break;
      }
      if (result > maxres) { 
        overflow = 1;
      }
      result = (result << 4);
      if (digit > (0xFFFFFFFFUL - result)) { 
        overflow = 1;
      }
      result += digit;
      anyDigits = 1;
    }
  } else if ( base >= 2 && base <= 36 ) {
    unsigned long maxres = 0xFFFFFFFFUL / base;
    for ( ; ; p += 1) {
      digit = *p - '0';
      if (digit > ('z' - '0')) {
        break;
      }
      digit = cvtIn[digit];
      if (digit >= ( (unsigned) base )) {
        break;
      }
      if (result > maxres) { 
        overflow = 1;
      }
      result *= base;
      if (digit > (0xFFFFFFFFUL - result)) {
        overflow = 1;
      }
      result += digit;
      anyDigits = 1;
    }
  }

  // See if there were any digits at all.
  if (!anyDigits) {
    p = string;
  }

  if (end_ptr != 0) {
    /* unsafe, but required by the strtoul prototype */
    *end_ptr = (char *) p;
  }

  if (overflow) {
    // TODO
    return 0xFFFFFFFFUL;
  } 

  if (negative) {
    return -result;
  }
  return result;
}

// local string to long method
// http://www.opensource.apple.com/source/tcl/tcl-14/tcl/compat/strtol.c
//
/* Argument1: String of ASCII digits, possibly
 * preceded by white space.  For bases
 * greater than 10, either lower- or
 * upper-case digits may be used.
 */
/* Argument2: Where to store address of terminating
 * character, or NULL.
 */
/* Argument3: Base for conversion.  Must be less
 * than 37.  If 0, then the base is chosen
 * from the leading characters of string:
 * "0x" means hex, "0" means octal, anything
 * else means decimal.
 */
$$__device__$$
long int at_illecker_strtol(const char *string, char **end_ptr, int base) {
  register const char *p;
  long result;

  // Skip any leading blanks.
  p = string;
  while (at_illecker_is_space((unsigned char) (*p))) {
    p += 1;
  }
  // Check for a sign.
  if (*p == '-') {
    p += 1;
    result = -(at_illecker_strtoul(p, end_ptr, base));
  } else {
    if (*p == '+') {
      p += 1;
    }
    result = at_illecker_strtoul(p, end_ptr, base);
  }
  if ((result == 0) && (end_ptr != 0) && (*end_ptr == p)) {
    *end_ptr = (char *) string;
  }
  return result;
}

// local string to double method
// http://www.opensource.apple.com/source/tcl/tcl-14/tcl/compat/strtod.c
$$__device__$$
double at_illecker_strtod(const char *string) {
  int sign = 0; // FALSE
  int expSign = 0; // FALSE
  double fraction, dblExp, *d;
  register const char *p;
  register int c;
  int exp = 0;
  int fracExp = 0;
  int mantSize;
  int decPt;
  const char *pExp;

  int maxExponent = 511;
  double powersOf10[] = {
    10.,
    100.,
    1.0e4,
    1.0e8,
    1.0e16,
    1.0e32,
    1.0e64,
    1.0e128,
    1.0e256
  };

  // Strip off leading blanks and check for a sign.
  p = string;
  while (at_illecker_is_space((unsigned char) (*p))) {
    p += 1;
  }
  
  if (*p == '-') {
    sign = 1; // TRUE
    p += 1;
  } else {
    if (*p == '+') {
      p += 1;
    }
    sign = 0; // FALSE
  }

  // Count the number of digits in the mantissa (including the decimal
  // point), and also locate the decimal point.
  decPt = -1;
  for (mantSize = 0; ; mantSize += 1) {
    c = *p;
    if (!at_illecker_is_digit(c)) {
      if ((c != '.') || (decPt >= 0)) {
        break;
      }
      decPt = mantSize;
    }
    p += 1;
  }

  // Now suck up the digits in the mantissa.  Use two integers to
  // collect 9 digits each (this is faster than using floating-point).
  // If the mantissa has more than 18 digits, ignore the extras, since
  // they can't affect the value anyway.
  pExp  = p;
  p -= mantSize;
  if (decPt < 0) {
    decPt = mantSize;
  } else {
    mantSize -= 1;
  }
  if (mantSize > 18) {
    fracExp = decPt - 18;
    mantSize = 18;
  } else {
    fracExp = decPt - mantSize;
  }
  if (mantSize == 0) {
    fraction = 0.0;
    p = string;
    goto done;
  } else {
    int frac1, frac2;
    frac1 = 0;
    for ( ; mantSize > 9; mantSize -= 1) {
      c = *p;
      p += 1;
      if (c == '.') {
        c = *p;
        p += 1;
      }
      frac1 = 10*frac1 + (c - '0');
    }
    frac2 = 0;
    for (; mantSize > 0; mantSize -= 1) {
      c = *p;
      p += 1;
      if (c == '.') {
        c = *p;
        p += 1;
      }
      frac2 = 10*frac2 + (c - '0');
    }
    fraction = (1.0e9 * frac1) + frac2;
  }

  // Skim off the exponent.
  p = pExp;
  if ((*p == 'E') || (*p == 'e')) {
    p += 1;
    if (*p == '-') {
      expSign = 1; // TRUE
      p += 1;
    } else {
      if (*p == '+') {
        p += 1;
      }
      expSign = 0; // FALSE
    }
    if (!at_illecker_is_digit((unsigned char) (*p))) {
      p = pExp;
      goto done;
    }
    while (at_illecker_is_digit((unsigned char) (*p))) {
      exp = exp * 10 + (*p - '0');
      p += 1;
    }
  }
  if (expSign) {
    exp = fracExp - exp;
  } else {
    exp = fracExp + exp;
  }

  // Generate a floating-point number that represents the exponent.
  // Do this by processing the exponent one bit at a time to combine
  // many powers of 2 of 10. Then combine the exponent with the
  // fraction.
  if (exp < 0) {
    expSign = 1; // TRUE
    exp = -exp;
  } else {
    expSign = 0; // FALSE
  }
  if (exp > maxExponent) {
    exp = maxExponent;
    // TODO 
    // errno = ERANGE;
  }
  dblExp = 1.0;
  for (d = powersOf10; exp != 0; exp >>= 1, d += 1) {
    if (exp & 01) {
      dblExp *= *d;
    }
  }
  if (expSign) {
    fraction /= dblExp;
  } else {
    fraction *= dblExp;
  }

done:
  if (sign) {
    return -fraction;
  }
  return fraction;
}

//<java.lang.Long: long parseLong(java.lang.String)>
$$__device__$$
long java_lang_Long_parseLong(char * gc_info, int str_obj_ref, int * exception) {
  int str_value = 0;
  int str_count = 0;
  char str_val[255];
  long return_val = 0;

  str_value = instance_getter_java_lang_String_value(gc_info, str_obj_ref, exception);
  str_count = instance_getter_java_lang_String_count(gc_info, str_obj_ref, exception);

  // convert string to char[]
  // TODO check if str_count > 255
  for(int i = 0; i < str_count; i++){
    str_val[i] = char__array_get(gc_info, str_value, i, exception);
  }
  str_val[str_count] = '\0';

  // printf("java_lang_Long_parseLong str: '%s'\n", str_val);
  return_val = at_illecker_strtol(str_val, 0, 0);
  // printf("java_lang_Long_parseLong int: '%ld'\n", return_val);

  return return_val;
}

//<java.lang.Integer: int parseInt(java.lang.String)>
$$__device__$$
int java_lang_Integer_parseInt(char * gc_info, int str_obj_ref, int * exception) {
  return java_lang_Long_parseLong(gc_info, str_obj_ref, exception);
}

//<java.lang.Double: double parseDouble(java.lang.String)>
$$__device__$$
double java_lang_Double_parseDouble(char * gc_info, int str_obj_ref, int * exception) {
  int str_value = 0;
  int str_count = 0;
  char str_val[255];
  double return_val = 0;

  str_value = instance_getter_java_lang_String_value(gc_info, str_obj_ref, exception);
  str_count = instance_getter_java_lang_String_count(gc_info, str_obj_ref, exception);

  // convert string to char[]
  // TODO check if str_count > 255
  for(int i = 0; i < str_count; i++){
    str_val[i] = char__array_get(gc_info, str_value, i, exception);
  }
  str_val[str_count] = '\0';

  // printf("java_lang_Double_parseDouble str: '%s'\n", str_val);
  return_val = at_illecker_strtod(str_val);
  // printf("java_lang_Double_parseDouble double: '%f'\n", return_val);

  return return_val;
}

//<java.lang.Float: float parseFloat(java.lang.String)>
$$__device__$$
float java_lang_Float_parseFloat(char * gc_info, int str_obj_ref, int * exception) {
  return java_lang_Double_parseDouble(gc_info, str_obj_ref, exception);
}

/*****************************************************************************/
/* local typeof methods */

// typeof_Integer
__device__ bool at_illecker_typeof_Integer(char * gc_info, int thisref){
  char * thisref_deref;
  GC_OBJ_TYPE_TYPE type;
  if(thisref == -1){
    return false;
  }
  thisref_deref = edu_syr_pcpratts_gc_deref(gc_info, thisref);
  type = edu_syr_pcpratts_gc_get_type(thisref_deref);
  if(type==%%java_lang_Integer_TypeNumber%%) {
    return true;
  }
  return false;
}

// typeof_Long
__device__ bool at_illecker_typeof_Long(char * gc_info, int thisref){
  char * thisref_deref;
  GC_OBJ_TYPE_TYPE type;
  if(thisref == -1){
    return false;
  }
  thisref_deref = edu_syr_pcpratts_gc_deref(gc_info, thisref);
  type = edu_syr_pcpratts_gc_get_type(thisref_deref);
  if(type==%%java_lang_Long_TypeNumber%%) {
    return true;
  }
  return false;
}
// typeof_Float
__device__ bool at_illecker_typeof_Float(char * gc_info, int thisref){
  char * thisref_deref;
  GC_OBJ_TYPE_TYPE type;
  if(thisref == -1){
    return false;
  }
  thisref_deref = edu_syr_pcpratts_gc_deref(gc_info, thisref);
  type = edu_syr_pcpratts_gc_get_type(thisref_deref);
  if(type==%%java_lang_Float_TypeNumber%%) {
    return true;
  }
  return false;
}

// typeof_Double
__device__ bool at_illecker_typeof_Double(char * gc_info, int thisref){
  char * thisref_deref;
  GC_OBJ_TYPE_TYPE type;
  if(thisref == -1){
    return false;
  }
  thisref_deref = edu_syr_pcpratts_gc_deref(gc_info, thisref);
  type = edu_syr_pcpratts_gc_get_type(thisref_deref);
  if(type==%%java_lang_Double_TypeNumber%%) {
    return true;
  }
  return false;
}

// typeof_String
__device__ bool at_illecker_typeof_String(char * gc_info, int thisref){
  char * thisref_deref;
  GC_OBJ_TYPE_TYPE type;
  if(thisref == -1){
    return false;
  }
  thisref_deref = edu_syr_pcpratts_gc_deref(gc_info, thisref);
  type = edu_syr_pcpratts_gc_get_type(thisref_deref);
  if(type==%%java_lang_String_TypeNumber%%) {
    return true;
  }
  return false;
}

/*****************************************************************************/
// getResult
// is used to communicate with the host (HostMonitor) via pinned memory
// object HostDeviceInterface and fetches results
template<class T>
$$__device__$$
T at_illecker_getResult($$__global$$ char * gc_info, 
    HostDeviceInterface::MESSAGE_TYPE cmd, 
    HostDeviceInterface::TYPE return_type, bool use_return_value,
    int key_value_pair_ref, HostDeviceInterface::TYPE key_type, HostDeviceInterface::TYPE value_type,
    int int_param1, bool use_int_param1,
    int int_param2, bool use_int_param2,
    int int_param3, bool use_int_param3,
    long long long_param1, bool use_long_param1,
    long long long_param2, bool use_long_param2,
    float float_param1, bool use_float_param1,
    float float_param2, bool use_float_param2,
    double double_param1, bool use_double_param1,
    double double_param2, bool use_double_param2,
    int str_param1, bool use_str_param1,
    int str_param2, bool use_str_param2,
    int str_param3, bool use_str_param3,
    int * exception) {

  T return_value = 0;

  int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  int count = 0;
  int timeout = 0;
  bool done = false;

  int str_param1_value = 0;
  int str_param1_count = 0;
  int str_param2_value = 0;
  int str_param2_count = 0;
  int str_param3_value = 0;
  int str_param3_count = 0;

  int key_obj_ref = 0;
  int value_obj_ref = 0;
  char * key_obj_deref;
  char * value_obj_deref;

  // loop until done == true
  while (count < 100) {

    // TODO timeout to break infinite loop
    if (++timeout > 100000) {
      break;
    }
    __syncthreads();
    
    if (done) {
      break;
    }

    // (lock_thread_id == -1 ? thread_id : lock_thread_id)
    int old = atomicCAS((int *) &host_device_interface->lock_thread_id, -1, thread_id);

    // printf("Thread %d old: %d\n", thread_id, old);

    if (old == -1 || old == thread_id) {
      //do critical section code
      // thread won race condition

      printf("gpu_Thread %d GOT LOCK lock_thread_id: %d\n", thread_id,
             host_device_interface->lock_thread_id);

      /***********************************************************************/
      // wait for possible old task to end
      int inner_timeout = 0;
      while (host_device_interface->has_task) {
        // TODO timeout to break infinite loop
        if (++inner_timeout > 10000) {
          break;
        }
      }

      /***********************************************************************/
      // Setup command
      host_device_interface->command = cmd;
      host_device_interface->return_type = return_type;

      // Setup transfer variable as parameters
      if (use_int_param1) {
        host_device_interface->use_int_val1 = true;
        host_device_interface->int_val1 = int_param1;
      }
      if (use_int_param2) {
        host_device_interface->use_int_val2 = true;
        host_device_interface->int_val2 = int_param2;
      }
      if (use_int_param3) {
        host_device_interface->use_int_val3 = true;
        host_device_interface->int_val3 = int_param3;
      }
      if (use_long_param1) {
        host_device_interface->use_long_val1 = true;
        host_device_interface->long_val1 = long_param1;
      }
      if (use_long_param2) {
        host_device_interface->use_long_val2 = true;
        host_device_interface->long_val2 = long_param2;
      }
      if (use_float_param1) {
        host_device_interface->use_float_val1 = true;
        host_device_interface->float_val1 = float_param1;
      }
      if (use_float_param2) {
        host_device_interface->use_float_val2 = true;
        host_device_interface->float_val2 = float_param2;
      }
      if (use_double_param1) {
        host_device_interface->use_double_val1 = true;
        host_device_interface->double_val1 = double_param1;
      }
      if (use_double_param2) {
        host_device_interface->use_double_val2 = true;
        host_device_interface->double_val2 = double_param2;
      }
      if (use_str_param1) {
        str_param1_value = instance_getter_java_lang_String_value(gc_info, str_param1,
                          exception);
        str_param1_count = instance_getter_java_lang_String_count(gc_info, str_param1,
                          exception);

        // TODO - check for max str_val1 size
        for(int i = 0; i < str_param1_count; i++) {
          host_device_interface->str_val1[i] = char__array_get(gc_info, str_param1_value, i, exception);
        }
        host_device_interface->use_str_val1 = true;
        host_device_interface->str_val1[str_param1_count] = '\0';
      }
      if (use_str_param2) {
        str_param2_value = instance_getter_java_lang_String_value(gc_info, str_param2,
                           exception);
        str_param2_count = instance_getter_java_lang_String_count(gc_info, str_param2,
                           exception);

        // TODO - check for max str_val2 size
        for(int i = 0; i < str_param2_count; i++) {
          host_device_interface->str_val2[i] = char__array_get(gc_info, str_param2_value, i, exception);
        }
        host_device_interface->use_str_val2 = true;
        host_device_interface->str_val2[str_param2_count] = '\0';
      }
      if (use_str_param3) {
        str_param3_value = instance_getter_java_lang_String_value(gc_info, str_param3,
                           exception);
        str_param3_count = instance_getter_java_lang_String_count(gc_info, str_param3,
                           exception);

        // TODO - check for max str_val3 size(255)
        for(int i = 0; i < str_param3_count; i++) {
          host_device_interface->str_val3[i] = char__array_get(gc_info, str_param3_value, i, exception);
        }
        host_device_interface->use_str_val3 = true;
        host_device_interface->str_val3[str_param3_count] = '\0';
      }

      if (return_type == HostDeviceInterface::KEY_VALUE_PAIR) {
        host_device_interface->key_type = key_type;
        host_device_interface->value_type = value_type;
      }

      /***********************************************************************/
      // Activate task for HostMonitor
      host_device_interface->has_task = true;
      __threadfence_system();
      //__threadfence();

      /***********************************************************************/
      // wait for socket communication to end
      inner_timeout = 0;
      while (!host_device_interface->is_result_available) {
        __threadfence_system();
        //__threadfence();
	// TODO timeout to break infinite loop 
        if (++inner_timeout > 30000) {
          break;
        }
      }

      /***********************************************************************/
      // Get result from host device interface

      if (return_type == HostDeviceInterface::KEY_VALUE_PAIR) {
        // Update KeyValuePair object

        // Update key
        key_obj_ref = instance_getter_edu_syr_pcpratts_rootbeer_runtime_KeyValuePair_m_key(gc_info, 
                      key_value_pair_ref, exception);
        key_obj_deref = edu_syr_pcpratts_gc_deref(gc_info, key_obj_ref);
        
        if (key_type == HostDeviceInterface::INT) {
          *(( int *) &key_obj_deref[32]) = host_device_interface->int_val1;
        } else if (key_type == HostDeviceInterface::LONG) {
          *(( long long *) &key_obj_deref[32]) = host_device_interface->long_val1;
        } else if (key_type == HostDeviceInterface::FLOAT) {
          *(( float *) &key_obj_deref[32]) = host_device_interface->float_val1;
        } else if (key_type == HostDeviceInterface::DOUBLE) {
          *(( double *) &key_obj_deref[32]) = host_device_interface->double_val1;
        } else if (key_type == HostDeviceInterface::STRING) {
          int i;
          int len = at_illecker_strlen(host_device_interface->str_val1);
          int characters = char__array_new(gc_info, len, exception);
          for(i = 0; i < len; ++i) {
            char__array_set(gc_info, characters, i, host_device_interface->str_val1[i], exception);
          }
          // Set new value
          *(( int *) &key_obj_deref[32]) = characters;
          // Set new length
          *(( int *) &key_obj_deref[40]) = len;
          // Set new offset to 0
          *(( int *) &key_obj_deref[44]) = 0;
        }

        // Update value
        value_obj_ref = instance_getter_edu_syr_pcpratts_rootbeer_runtime_KeyValuePair_m_value(gc_info, 
                        key_value_pair_ref, exception);
        value_obj_deref = edu_syr_pcpratts_gc_deref(gc_info, value_obj_ref);
        
        if (value_type == HostDeviceInterface::INT) {
          *(( int *) &value_obj_deref[32]) = host_device_interface->int_val2;
        } else if (value_type == HostDeviceInterface::LONG) {
          *(( long long *) &value_obj_deref[32]) = host_device_interface->long_val2;
        } else if (value_type == HostDeviceInterface::FLOAT) {
          *(( float *) &value_obj_deref[32]) = host_device_interface->float_val2;
        } else if (value_type == HostDeviceInterface::DOUBLE) {
          *(( double *) &value_obj_deref[32]) = host_device_interface->double_val2;
        } else if (value_type == HostDeviceInterface::STRING) {
          int i;
          int len = at_illecker_strlen(host_device_interface->str_val2);
          int characters = char__array_new(gc_info, len, exception);
          for(i = 0; i < len; ++i) {
            char__array_set(gc_info, characters, i, host_device_interface->str_val2[i], exception);
          }
          // Set new value
          *(( int *) &value_obj_deref[32]) = characters;
          // Set new length
          *(( int *) &value_obj_deref[40]) = len;
          // Set new offset to 0
          *(( int *) &value_obj_deref[44]) = 0;
        }

        // true if more data is available
        return_value = !host_device_interface->end_of_data;

      } else if (use_return_value) { // Update return_value

        // Get right return type
        if (return_type == HostDeviceInterface::INT) {
          return_value = host_device_interface->int_val1;

        } else if (return_type == HostDeviceInterface::LONG) {
          return_value = host_device_interface->long_val1;

        } else if (return_type == HostDeviceInterface::FLOAT) {
          return_value = host_device_interface->float_val1;

        } else if (return_type == HostDeviceInterface::DOUBLE) {
          return_value = host_device_interface->double_val1;

        } else if (return_type == HostDeviceInterface::STRING) {
          // make new String object
          edu_syr_pcpratts_gc_assign(gc_info, (int*)&return_value,
            at_illecker_string_constant(gc_info, host_device_interface->str_val1, exception));
        }
      }

      /***********************************************************************/
      // Reset transfer variables
      if ( (use_int_param1) || (return_type == HostDeviceInterface::INT) ) {
        host_device_interface->int_val1 = 0;
        host_device_interface->use_int_val1 = false;
      }
      if (use_int_param2) {
        host_device_interface->int_val2 = 0;
        host_device_interface->use_int_val2 = false;
      }
      if (use_int_param3) {
        host_device_interface->int_val3 = 0;
        host_device_interface->use_int_val3 = false;
      }
      if ( (use_long_param1) || (return_type == HostDeviceInterface::LONG) ) {
        host_device_interface->long_val1 = 0;
        host_device_interface->use_long_val1 = false;
      }
      if (use_long_param1) {
        host_device_interface->long_val2 = 0;
        host_device_interface->use_long_val2 = false;
      }
      if ( (use_float_param1) || (return_type == HostDeviceInterface::FLOAT) ) {
        host_device_interface->float_val1 = 0;
        host_device_interface->use_float_val1 = false;
      }
      if (use_float_param2) {
        host_device_interface->float_val2 = 0;
        host_device_interface->use_float_val2 = false;
      }
      if ( (use_double_param1) || (return_type == HostDeviceInterface::DOUBLE) ) {
        host_device_interface->double_val1 = 0;
        host_device_interface->use_double_val1 = false;
      }
      if (use_double_param2) {
        host_device_interface->double_val2 = 0;
        host_device_interface->use_double_val2 = false;
      }
      if ( (use_str_param1) || (return_type == HostDeviceInterface::STRING) ) {
        host_device_interface->str_val1[0] = '\0';
        host_device_interface->use_str_val1 = false;
      }
      if (use_str_param2) {
        host_device_interface->str_val2[0] = '\0';
        host_device_interface->use_str_val2 = false;
      }
      if (use_str_param3) {
        host_device_interface->str_val3[0] = '\0';
        host_device_interface->use_str_val3 = false;
      }
      if (return_type == HostDeviceInterface::KEY_VALUE_PAIR) {
        host_device_interface->key_type = HostDeviceInterface::NOT_AVAILABLE;
        host_device_interface->value_type = HostDeviceInterface::NOT_AVAILABLE;
      }

      host_device_interface->command = HostDeviceInterface::UNDEFINED;
      host_device_interface->return_type = HostDeviceInterface::NOT_AVAILABLE;

      /***********************************************************************/ 
      // Notify HostMonitor that result was received
      host_device_interface->is_result_available = false;
      host_device_interface->lock_thread_id = -1;
      
      __threadfence_system();
      //__threadfence();

      /***********************************************************************/ 
      // exit infinite loop
      done = true; // finished work

    } else {
      count++;
      if (count > 50) {
        count = 0;
      }
    }
  }
  return return_value;
}

/*****************************************************************************/
/* Hama Peer public methods */

// HamaPeer.send
//<edu.syr.pcpratts.rootbeer.runtime.HamaPeer: void send(String peerName, Object message)>
$$__device__$$
void edu_syr_pcpratts_rootbeer_runtime_HamaPeer_send($$__global$$ char * gc_info,
     int peer_name_str_ref, int message_obj_ref, int * exception) {

  int int_value = 0;
  bool use_int_value = false;
  long long long_value = 0;
  bool use_long_value = false;
  float float_value = 0;
  bool use_float_value = false;
  double double_value = 0;
  bool use_double_value = false;
  int string_value = 0;
  bool use_string_value = false;
  char * message_obj_deref;
  
  // check message type
  if (at_illecker_typeof_Integer(gc_info, message_obj_ref)) {
    message_obj_deref = edu_syr_pcpratts_gc_deref(gc_info, message_obj_ref);
    int_value = *(( int *) &message_obj_deref[32]);
    use_int_value = true;
    
  } else if (at_illecker_typeof_Long(gc_info, message_obj_ref)) {
    message_obj_deref = edu_syr_pcpratts_gc_deref(gc_info, message_obj_ref);
    long_value = *(( long long *) &message_obj_deref[32]);
    use_long_value = true;
    
  } else if (at_illecker_typeof_Float(gc_info, message_obj_ref)) {
    message_obj_deref = edu_syr_pcpratts_gc_deref(gc_info, message_obj_ref);
    float_value = *(( float *) &message_obj_deref[32]);
    use_float_value = true;
    
  } else if (at_illecker_typeof_Double(gc_info, message_obj_ref)) {
    message_obj_deref = edu_syr_pcpratts_gc_deref(gc_info, message_obj_ref);
    double_value = *(( double *) &message_obj_deref[32]);
    use_double_value = true;
    
  } else if (at_illecker_typeof_String(gc_info, message_obj_ref)) {
    string_value = message_obj_ref;
    use_string_value = true;

  } else {
    // TODO throw CudaException unsupported Type
    printf("HamaPeer.send Exception: unsupported Type\n");
    return;
  }
  
  at_illecker_getResult<int>(gc_info, HostDeviceInterface::SEND_MSG,
    HostDeviceInterface::NOT_AVAILABLE, false, // do not use the return value
    0, HostDeviceInterface::NOT_AVAILABLE, HostDeviceInterface::NOT_AVAILABLE,
    int_value, use_int_value,
    0, false,
    0, false,
    long_value, use_long_value,
    0, false,
    float_value, use_float_value,
    0, false,
    double_value, use_double_value,
    0, false,
    peer_name_str_ref, true,
    string_value, use_string_value,
    0, false,
    exception);
}

// HamaPeer.getCurrentIntMessage
//<edu.syr.pcpratts.rootbeer.runtime.HamaPeer: int getCurrentIntMessage()>
$$__device__$$
int edu_syr_pcpratts_rootbeer_runtime_HamaPeer_getCurrentIntMessage($$__global$$ char * gc_info, 
    int * exception) {

  return at_illecker_getResult<int>(gc_info, HostDeviceInterface::GET_MSG,
           HostDeviceInterface::INT, true, // expecting integer return value
           0, HostDeviceInterface::NOT_AVAILABLE, HostDeviceInterface::NOT_AVAILABLE,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           exception);
}

// HamaPeer.getCurrentLongMessage
//<edu.syr.pcpratts.rootbeer.runtime.HamaPeer: long getCurrentLongMessage()>
$$__device__$$
long edu_syr_pcpratts_rootbeer_runtime_HamaPeer_getCurrentLongMessage($$__global$$ char * gc_info, 
    int * exception) {

  return at_illecker_getResult<long>(gc_info, HostDeviceInterface::GET_MSG,
           HostDeviceInterface::LONG, true, // expecting long return value
           0, HostDeviceInterface::NOT_AVAILABLE, HostDeviceInterface::NOT_AVAILABLE,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           exception);
}

// HamaPeer.getCurrentFloatMessage
//<edu.syr.pcpratts.rootbeer.runtime.HamaPeer: float getCurrentFloatMessage()>
$$__device__$$
float edu_syr_pcpratts_rootbeer_runtime_HamaPeer_getCurrentFloatMessage($$__global$$ char * gc_info, 
    int * exception) {

  return at_illecker_getResult<float>(gc_info, HostDeviceInterface::GET_MSG,
           HostDeviceInterface::FLOAT, true, // expecting float return value
           0, HostDeviceInterface::NOT_AVAILABLE, HostDeviceInterface::NOT_AVAILABLE,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           exception);
}

// HamaPeer.getCurrentDoubleMessage
//<edu.syr.pcpratts.rootbeer.runtime.HamaPeer: double getCurrentDoubleMessage()>
$$__device__$$
double edu_syr_pcpratts_rootbeer_runtime_HamaPeer_getCurrentDoubleMessage($$__global$$ char * gc_info, 
    int * exception) {

  return at_illecker_getResult<double>(gc_info, HostDeviceInterface::GET_MSG,
           HostDeviceInterface::DOUBLE, true, // expecting double return value
           0, HostDeviceInterface::NOT_AVAILABLE, HostDeviceInterface::NOT_AVAILABLE,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           exception);
}

// HamaPeer.getCurrentStringMessage
//<edu.syr.pcpratts.rootbeer.runtime.HamaPeer: String getCurrentStringMessage()>
$$__device__$$
int edu_syr_pcpratts_rootbeer_runtime_HamaPeer_getCurrentStringMessage($$__global$$ char * gc_info, 
    int * exception) {

  return at_illecker_getResult<int>(gc_info, HostDeviceInterface::GET_MSG,
           HostDeviceInterface::STRING, true, // expecting string return value
           0, HostDeviceInterface::NOT_AVAILABLE, HostDeviceInterface::NOT_AVAILABLE,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           exception);
}

// HamaPeer.getNumCurrentMessages
//<edu.syr.pcpratts.rootbeer.runtime.HamaPeer: int getNumCurrentMessages()>
$$__device__$$
int edu_syr_pcpratts_rootbeer_runtime_HamaPeer_getNumCurrentMessages($$__global$$ char * gc_info, 
    int * exception) {

  return at_illecker_getResult<int>(gc_info, HostDeviceInterface::GET_MSG_COUNT,
           HostDeviceInterface::INT, true, // expecting integer return value
           0, HostDeviceInterface::NOT_AVAILABLE, HostDeviceInterface::NOT_AVAILABLE,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           exception);
}

// HamaPeer.sync
// This method blocks.
//<edu.syr.pcpratts.rootbeer.runtime.HamaPeer: void sync()>
$$__device__$$
void edu_syr_pcpratts_rootbeer_runtime_HamaPeer_sync($$__global$$ char * gc_info, 
     int * exception) {

  at_illecker_getResult<int>(gc_info, HostDeviceInterface::SYNC,
    HostDeviceInterface::NOT_AVAILABLE, false, // do not use return value
    0, HostDeviceInterface::NOT_AVAILABLE, HostDeviceInterface::NOT_AVAILABLE,
    0, false,
    0, false,
    0, false,
    0, false,
    0, false,
    0, false,
    0, false,
    0, false,
    0, false,
    0, false,
    0, false,
    0, false,
    exception);
}

// HamaPeer.getSuperstepCount
//<edu.syr.pcpratts.rootbeer.runtime.HamaPeer: long getSuperstepCount()>
$$__device__$$
long edu_syr_pcpratts_rootbeer_runtime_HamaPeer_getSuperstepCount($$__global$$ char * gc_info, 
    int * exception) {

  return at_illecker_getResult<long>(gc_info, HostDeviceInterface::GET_SUPERSTEP_COUNT,
           HostDeviceInterface::LONG, true, // expecting long return value
           0, HostDeviceInterface::NOT_AVAILABLE, HostDeviceInterface::NOT_AVAILABLE,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           exception);
}

// HamaPeer.getPeerName
// Returns own PeerName
//<edu.syr.pcpratts.rootbeer.runtime.HamaPeer: String getPeerName()>
$$__device__$$
int edu_syr_pcpratts_rootbeer_runtime_HamaPeer_getPeerName($$__global$$ char * gc_info, 
    int * exception) {

  return at_illecker_getResult<int>(gc_info, HostDeviceInterface::GET_PEERNAME, 
           HostDeviceInterface::STRING, true, // expecting string return value
           0, HostDeviceInterface::NOT_AVAILABLE, HostDeviceInterface::NOT_AVAILABLE,
           -1, true, // -1 for own peername
           0, false,
           0, false,
           0, false,
           0, false, 
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false, 
           0, false,
           exception);
}

// HamaPeer.getPeerName
//<edu.syr.pcpratts.rootbeer.runtime.HamaPeer: String getPeerName(int index)>
$$__device__$$
int edu_syr_pcpratts_rootbeer_runtime_HamaPeer_getPeerName($$__global$$ char * gc_info, 
    int index, int * exception) {

  return at_illecker_getResult<int>(gc_info, HostDeviceInterface::GET_PEERNAME,
           HostDeviceInterface::STRING, true, // expecting string return value
           0, HostDeviceInterface::NOT_AVAILABLE, HostDeviceInterface::NOT_AVAILABLE,
           index, true,
           0, false,
           0, false,
           0, false,
           0, false, 
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           exception);
}

// HamaPeer.getPeerIndex
//<edu.syr.pcpratts.rootbeer.runtime.HamaPeer: int getPeerIndex()>
$$__device__$$
int edu_syr_pcpratts_rootbeer_runtime_HamaPeer_getPeerIndex($$__global$$ char * gc_info, 
    int * exception) {

  return at_illecker_getResult<int>(gc_info, HostDeviceInterface::GET_PEER_INDEX,
           HostDeviceInterface::INT, true, // expecting integer return value
           0, HostDeviceInterface::NOT_AVAILABLE, HostDeviceInterface::NOT_AVAILABLE,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           exception);
}

// HamaPeer.getAllPeerNames
//<edu.syr.pcpratts.rootbeer.runtime.HamaPeer: String[] getAllPeerNames()>
$$__device__$$
int edu_syr_pcpratts_rootbeer_runtime_HamaPeer_getAllPeerNames($$__global$$ char * gc_info, 
    int * exception) {

  // TODO
  return 0;
}

// HamaPeer.getNumPeers
//<edu.syr.pcpratts.rootbeer.runtime.HamaPeer: int getNumPeers()>
$$__device__$$
int edu_syr_pcpratts_rootbeer_runtime_HamaPeer_getNumPeers($$__global$$ char * gc_info, 
    int * exception) {

  return at_illecker_getResult<int>(gc_info, HostDeviceInterface::GET_PEER_COUNT,
           HostDeviceInterface::INT, true, // expecting integer return value
           0, HostDeviceInterface::NOT_AVAILABLE, HostDeviceInterface::NOT_AVAILABLE,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false, 
           0, false,
           exception);
}

// HamaPeer.clear
//<edu.syr.pcpratts.rootbeer.runtime.HamaPeer: void clear()>
$$__device__$$
void edu_syr_pcpratts_rootbeer_runtime_HamaPeer_clear($$__global$$ char * gc_info, 
     int * exception) {

  at_illecker_getResult<int>(gc_info, HostDeviceInterface::CLEAR,
           HostDeviceInterface::NOT_AVAILABLE, false, // do not use return value
           0, HostDeviceInterface::NOT_AVAILABLE, HostDeviceInterface::NOT_AVAILABLE,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           exception);
}

// HamaPeer.reopenInput
//<edu.syr.pcpratts.rootbeer.runtime.HamaPeer: void reopenInput()>
$$__device__$$
void edu_syr_pcpratts_rootbeer_runtime_HamaPeer_reopenInput($$__global$$ char * gc_info, 
     int * exception) {

  at_illecker_getResult<int>(gc_info, HostDeviceInterface::REOPEN_INPUT,
           HostDeviceInterface::NOT_AVAILABLE, false, // do not use return value
           0, HostDeviceInterface::NOT_AVAILABLE, HostDeviceInterface::NOT_AVAILABLE,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           exception);
}

// HamaPeer.readNext
//<edu.syr.pcpratts.rootbeer.runtime.HamaPeer: boolean readNext(KeyValuePair key_value_pair)>
$$__device__$$
bool edu_syr_pcpratts_rootbeer_runtime_HamaPeer_readNext($$__global$$ char * gc_info, 
     int key_value_pair_ref, int * exception) {

  int key_obj_ref;
  int value_obj_ref;
  HostDeviceInterface::TYPE key_type;
  HostDeviceInterface::TYPE value_type;

  key_obj_ref = instance_getter_edu_syr_pcpratts_rootbeer_runtime_KeyValuePair_m_key(gc_info, 
                key_value_pair_ref, exception);
  value_obj_ref = instance_getter_edu_syr_pcpratts_rootbeer_runtime_KeyValuePair_m_value(gc_info, 
                  key_value_pair_ref, exception);

  // check key type
  if (at_illecker_typeof_Integer(gc_info, key_obj_ref)) {
    key_type = HostDeviceInterface::INT;
  } else if (at_illecker_typeof_Long(gc_info, key_obj_ref)) {
    key_type = HostDeviceInterface::LONG;
  } else if (at_illecker_typeof_Float(gc_info, key_obj_ref)) {
    key_type = HostDeviceInterface::FLOAT;
  } else if (at_illecker_typeof_Double(gc_info, key_obj_ref)) {
    key_type = HostDeviceInterface::DOUBLE;
  } else if (at_illecker_typeof_String(gc_info, key_obj_ref)) {
    key_type = HostDeviceInterface::STRING;
  } else {
    // TODO throw CudaException unsupported Type
    printf("Exception: unsupported Key Type\n");
    return false;
  }

  // check value type
  if (at_illecker_typeof_Integer(gc_info, value_obj_ref)) {
    value_type = HostDeviceInterface::INT;
  } else if (at_illecker_typeof_Long(gc_info, value_obj_ref)) {
    value_type = HostDeviceInterface::LONG;
  } else if (at_illecker_typeof_Float(gc_info, value_obj_ref)) {
    value_type = HostDeviceInterface::FLOAT;
  } else if (at_illecker_typeof_Double(gc_info, value_obj_ref)) {
    value_type = HostDeviceInterface::DOUBLE;
  } else if (at_illecker_typeof_String(gc_info, value_obj_ref)) {
    value_type = HostDeviceInterface::STRING;
  } else {
    // TODO throw CudaException unsupported Type
    printf("Exception: unsupported Value Type\n");
    return false;
  }

  return at_illecker_getResult<int>(gc_info, HostDeviceInterface::READ_KEYVALUE,
           HostDeviceInterface::KEY_VALUE_PAIR, false, // do not use return value, because key_value_pair obj will be modified
           key_value_pair_ref, key_type, value_type,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           exception);
}

// HamaPeer.write
//<edu.syr.pcpratts.rootbeer.runtime.HamaPeer: void write(Object key, Object value)>
$$__device__$$
void edu_syr_pcpratts_rootbeer_runtime_HamaPeer_write($$__global$$ char * gc_info, 
     int key_obj_ref, int value_obj_ref, int * exception) {

  // key values
  int int_val1 = 0;
  bool use_int_val1 = false;
  long long long_val1 = 0;
  bool use_long_val1 = false;
  float float_val1 = 0;
  bool use_float_val1 = false;
  double double_val1 = 0;
  bool use_double_val1 = false;
  int string_val1 = 0;
  bool use_string_val1 = false;

  // value values
  int int_val2 = 0;
  bool use_int_val2 = false;
  long long long_val2 = 0;
  bool use_long_val2 = false;
  float float_val2 = 0;
  bool use_float_val2 = false;
  double double_val2 = 0;
  bool use_double_val2 = false;
  int string_val2 = 0;
  bool use_string_val2 = false;

  char * key_obj_deref;
  char * value_obj_deref;

  key_obj_deref = edu_syr_pcpratts_gc_deref(gc_info, key_obj_ref);
  value_obj_deref = edu_syr_pcpratts_gc_deref(gc_info, value_obj_ref);

  // check key type
  if (at_illecker_typeof_Integer(gc_info, key_obj_ref)) {
    int_val1 = *(( int *) &key_obj_deref[32]);
    use_int_val1 = true;
  } else if (at_illecker_typeof_Long(gc_info, key_obj_ref)) {
    long_val1 = *(( long long *) &key_obj_deref[32]);
    use_long_val1 = true;
  } else if (at_illecker_typeof_Float(gc_info, key_obj_ref)) {
    float_val1 = *(( float *) &key_obj_deref[32]);
    use_float_val1 = true;
  } else if (at_illecker_typeof_Double(gc_info, key_obj_ref)) {
    double_val1 = *(( double *) &key_obj_deref[32]);
    use_double_val1 = true;
  } else if (at_illecker_typeof_String(gc_info, key_obj_ref)) {
    string_val1 = key_obj_ref;
    use_string_val1 = true;
  } else {
    // TODO throw CudaException unsupported Type
    printf("Exception: unsupported Key Type\n");
    return;
  }

  // check value type
  if (at_illecker_typeof_Integer(gc_info, value_obj_ref)) {
    int_val2 = *(( int *) &value_obj_deref[32]);
    use_int_val2 = true;
  } else if (at_illecker_typeof_Long(gc_info, value_obj_ref)) {
    long_val2 = *(( long long *) &value_obj_deref[32]);
    use_long_val2 = true;
  } else if (at_illecker_typeof_Float(gc_info, value_obj_ref)) {
    float_val2 = *(( float *) &value_obj_deref[32]);
    use_float_val2 = true;
  } else if (at_illecker_typeof_Double(gc_info, value_obj_ref)) {
    double_val2 = *(( double *) &value_obj_deref[32]);
    use_double_val2 = true;
  } else if (at_illecker_typeof_String(gc_info, value_obj_ref)) {
    string_val2 = value_obj_ref;
    use_string_val2 = true;
  } else {
    // TODO throw CudaException unsupported Type
    printf("Exception: unsupported Value Type\n");
    return;
  }

  at_illecker_getResult<int>(gc_info, HostDeviceInterface::WRITE_KEYVALUE,
    HostDeviceInterface::NOT_AVAILABLE, false, // do not use the return value
    0, HostDeviceInterface::NOT_AVAILABLE, HostDeviceInterface::NOT_AVAILABLE,
    int_val1, use_int_val1,
    int_val2, use_int_val2,
    0, false,
    long_val1, use_long_val1,
    long_val2, use_long_val2,
    float_val1, use_float_val1,
    float_val2, use_float_val2,
    double_val1, use_double_val1,
    double_val2, use_double_val2,
    string_val1, use_string_val1,
    string_val2, use_string_val2,
    0, false,
    exception);
}

// HamaPeer.sequenceFileReadNext
//<edu.syr.pcpratts.rootbeer.runtime.HamaPeer: boolean sequenceFileReadNext(int file_id, KeyValuePair key_value_pair)>
$$__device__$$
bool edu_syr_pcpratts_rootbeer_runtime_HamaPeer_sequenceFileReadNext($$__global$$ char * gc_info, 
     int file_id, int key_value_pair_ref, int * exception) {

  int key_obj_ref;
  int value_obj_ref;
  HostDeviceInterface::TYPE key_type;
  HostDeviceInterface::TYPE value_type;

  key_obj_ref = instance_getter_edu_syr_pcpratts_rootbeer_runtime_KeyValuePair_m_key(gc_info, 
                key_value_pair_ref, exception);
  value_obj_ref = instance_getter_edu_syr_pcpratts_rootbeer_runtime_KeyValuePair_m_value(gc_info, 
                  key_value_pair_ref, exception);

  // check key type
  if (at_illecker_typeof_Integer(gc_info, key_obj_ref)) {
    key_type = HostDeviceInterface::INT;
  } else if (at_illecker_typeof_Long(gc_info, key_obj_ref)) {
    key_type = HostDeviceInterface::LONG;
  } else if (at_illecker_typeof_Float(gc_info, key_obj_ref)) {
    key_type = HostDeviceInterface::FLOAT;
  } else if (at_illecker_typeof_Double(gc_info, key_obj_ref)) {
    key_type = HostDeviceInterface::DOUBLE;
  } else if (at_illecker_typeof_String(gc_info, key_obj_ref)) {
    key_type = HostDeviceInterface::STRING;
  } else {
    // TODO throw CudaException unsupported Type
    printf("Exception: unsupported Key Type\n");
    return false;
  }

  // check value type
  if (at_illecker_typeof_Integer(gc_info, value_obj_ref)) {
    value_type = HostDeviceInterface::INT;
  } else if (at_illecker_typeof_Long(gc_info, value_obj_ref)) {
    value_type = HostDeviceInterface::LONG;
  } else if (at_illecker_typeof_Float(gc_info, value_obj_ref)) {
    value_type = HostDeviceInterface::FLOAT;
  } else if (at_illecker_typeof_Double(gc_info, value_obj_ref)) {
    value_type = HostDeviceInterface::DOUBLE;
  } else if (at_illecker_typeof_String(gc_info, value_obj_ref)) {
    value_type = HostDeviceInterface::STRING;
  } else {
    // TODO throw CudaException unsupported Type
    printf("Exception: unsupported Value Type\n");
    return false;
  }

  return at_illecker_getResult<int>(gc_info, HostDeviceInterface::SEQFILE_READNEXT,
           HostDeviceInterface::KEY_VALUE_PAIR, false, // do not use return value, because obj is modified
           key_value_pair_ref, key_type, value_type,
           file_id, true,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           exception);
}

// HamaPeer.sequenceFileAppend
//<edu.syr.pcpratts.rootbeer.runtime.HamaPeer: boolean sequenceFileAppend(int file_id, Object key, Object value)>
$$__device__$$
bool edu_syr_pcpratts_rootbeer_runtime_HamaPeer_sequenceFileAppend($$__global$$ char * gc_info, 
     int file_id, int key_obj_ref, int value_obj_ref, int * exception) {

  // key values
  int int_val1 = 0;
  bool use_int_val1 = false;
  long long long_val1 = 0;
  bool use_long_val1 = false;
  float float_val1 = 0;
  bool use_float_val1 = false;
  double double_val1 = 0;
  bool use_double_val1 = false;
  int string_val1 = 0;
  bool use_string_val1 = false;

  // value values
  int int_val2 = 0;
  bool use_int_val2 = false;
  long long long_val2 = 0;
  bool use_long_val2 = false;
  float float_val2 = 0;
  bool use_float_val2 = false;
  double double_val2 = 0;
  bool use_double_val2 = false;
  int string_val2 = 0;
  bool use_string_val2 = false;

  char * key_obj_deref;
  char * value_obj_deref;

  key_obj_deref = edu_syr_pcpratts_gc_deref(gc_info, key_obj_ref);
  value_obj_deref = edu_syr_pcpratts_gc_deref(gc_info, value_obj_ref);

  // check key type
  if (at_illecker_typeof_Integer(gc_info, key_obj_ref)) {
    int_val1 = *(( int *) &key_obj_deref[32]);
    use_int_val1 = true;
  } else if (at_illecker_typeof_Long(gc_info, key_obj_ref)) {
    long_val1 = *(( long long *) &key_obj_deref[32]);
    use_long_val1 = true;
  } else if (at_illecker_typeof_Float(gc_info, key_obj_ref)) {
    float_val1 = *(( float *) &key_obj_deref[32]);
    use_float_val1 = true;
  } else if (at_illecker_typeof_Double(gc_info, key_obj_ref)) {
    double_val1 = *(( double *) &key_obj_deref[32]);
    use_double_val1 = true;
  } else if (at_illecker_typeof_String(gc_info, key_obj_ref)) {
    string_val1 = key_obj_ref;
    use_string_val1 = true;
  } else {
    // TODO throw CudaException unsupported Type
    printf("Exception: unsupported Key Type\n");
    return;
  }
  
  // check value type
  if (at_illecker_typeof_Integer(gc_info, value_obj_ref)) {
    int_val2 = *(( int *) &value_obj_deref[32]);
    use_int_val2 = true;
  } else if (at_illecker_typeof_Long(gc_info, value_obj_ref)) {
    long_val2 = *(( long long *) &value_obj_deref[32]);
    use_long_val2 = true;
  } else if (at_illecker_typeof_Float(gc_info, value_obj_ref)) {
    float_val2 = *(( float *) &value_obj_deref[32]);
    use_float_val2 = true;
  } else if (at_illecker_typeof_Double(gc_info, value_obj_ref)) {
    double_val2 = *(( double *) &value_obj_deref[32]);
    use_double_val2 = true;
  } else if (at_illecker_typeof_String(gc_info, value_obj_ref)) {
    string_val2 = value_obj_ref;
    use_string_val2 = true;
  } else {
    // TODO throw CudaException unsupported Type
    printf("Exception: unsupported Value Type\n");
    return;
  }
  
  return at_illecker_getResult<int>(gc_info, HostDeviceInterface::SEQFILE_APPEND,
           HostDeviceInterface::INT, true, // expecting integer return value
           0, HostDeviceInterface::NOT_AVAILABLE, HostDeviceInterface::NOT_AVAILABLE,
           int_val1, use_int_val1,
           int_val2, use_int_val2,
           file_id, true,
           long_val1, use_long_val1,
           long_val2, use_long_val2,
           float_val1, use_float_val1,
           float_val2, use_float_val2,
           double_val1, use_double_val1,
           double_val2, use_double_val2,
           string_val1, use_string_val1,
           string_val2, use_string_val2,
           0, false,
           exception);
}

// HamaPeer.sequenceFileOpen
//<edu.syr.pcpratts.rootbeer.runtime.HamaPeer: int sequenceFileOpen(String path, char option, String keyType, String valueType)>
$$__device__$$
int edu_syr_pcpratts_rootbeer_runtime_HamaPeer_sequenceFileOpen($$__global$$ char * gc_info, 
     int path_str_ref, char option, 
     int key_type_str_ref, int value_type_str_ref, 
     int * exception) {

  return at_illecker_getResult<int>(gc_info, HostDeviceInterface::SEQFILE_OPEN,
           HostDeviceInterface::INT, true, // expecting integer return value
           0, HostDeviceInterface::NOT_AVAILABLE, HostDeviceInterface::NOT_AVAILABLE,
           option, true,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           path_str_ref, true,
           key_type_str_ref, true,
           value_type_str_ref, true,
           exception);
}

// HamaPeer.sequenceFileClose
//<edu.syr.pcpratts.rootbeer.runtime.HamaPeer: boolean sequenceFileClose(int file_id)>
$$__device__$$
bool edu_syr_pcpratts_rootbeer_runtime_HamaPeer_sequenceFileClose($$__global$$ char * gc_info, 
     int file_id, int * exception) {
  
  return at_illecker_getResult<int>(gc_info, HostDeviceInterface::SEQFILE_CLOSE,
           HostDeviceInterface::INT, true, // expecting integer return value
           0, HostDeviceInterface::NOT_AVAILABLE, HostDeviceInterface::NOT_AVAILABLE,
           file_id, true,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           0, false,
           exception);
}

