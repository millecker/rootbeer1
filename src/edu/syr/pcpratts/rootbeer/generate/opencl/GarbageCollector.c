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

//<java.lang.Double: java.lang.String toString(double)>
$$__device__$$ 
int java_lang_Double_toString9_8_(char * gc_info, double parameter0, int * exception){

  long long long_value;
  long long fraction;
  int part1;
  int part2;
  int part3;
  int string_builder;

  long_value = (long) parameter0;
  long_value *= 10000000;
  fraction = (long) (parameter0 * 10000000);
  fraction -= long_value;
    
  part1 = java_lang_Long_toString9_6_(gc_info, long_value, exception);
  part2 = edu_syr_pcpratts_string_constant(gc_info, ".", exception);
  part3 = java_lang_Long_toString9_6_(gc_info, fraction, exception);

  string_builder = java_lang_StringBuilder_initab850b60f96d11de8a390800200c9a6610_9_(gc_info,
    part1, exception);
  java_lang_StringBuilder_append10_9_(gc_info, string_builder, part2, exception);
  java_lang_StringBuilder_append10_9_(gc_info, string_builder, part3, exception);

  return java_lang_StringBuilder_toString9_(gc_info, string_builder, exception);
}

//<java.lang.Float: java.lang.String toString(float)>
$$__device__$$ 
int java_lang_Float_toString9_7_(char * gc_info, float parameter0, int * exception){

  long long long_value;
  long long fraction;
  int part1;
  int part2;
  int part3;
  int string_builder;

  long_value = (long) parameter0;
  long_value *= 10000000;
  fraction = (long) (parameter0 * 10000000);
  fraction -= long_value;
    
  part1 = java_lang_Long_toString9_6_(gc_info, long_value, exception);
  part2 = edu_syr_pcpratts_string_constant(gc_info, ".", exception);
  part3 = java_lang_Long_toString9_6_(gc_info, fraction, exception);

  string_builder = java_lang_StringBuilder_initab850b60f96d11de8a390800200c9a6610_9_(gc_info,
    part1, exception);
  java_lang_StringBuilder_append10_9_(gc_info, string_builder, part2, exception);
  java_lang_StringBuilder_append10_9_(gc_info, string_builder, part3, exception);

  return java_lang_StringBuilder_toString9_(gc_info, string_builder, exception);
}

// Hama Peer implementation

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

$$__device__$$
int at_illecker_string_constant( char * gc_info, volatile char * str_constant, int * exception) {
  int i;
  int len = at_illecker_strlen(str_constant);
  int characters = char__array_new(gc_info, len, exception);
  unsigned long long * addr = (unsigned long long *) (gc_info + TO_SPACE_FREE_POINTER_OFFSET);
  printf("at_illecker_string_constant str: '"); 
  for(i = 0; i < len; ++i) {
    char__array_set(gc_info, characters, i, str_constant[i], exception);
    printf("%c",str_constant[i]);
  }
  printf("'\n");  

  return java_lang_String_initab850b60f96d11de8a390800200c9a66(gc_info, characters, exception);
}

$$__device__$$
int at_illecker_getIntResult($$__global$$ char * gc_info, HostDeviceInterface::MESSAGE_TYPE cmd, 
    int * exception) {

  int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  int count = 0;
  int timeout = 0;
  bool done = false;
  int return_value = -1;

  while (count < 100) {

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

      int inner_timeout = 0;
      // wait for possible old task to end
      while (host_device_interface->has_task) {
        if (++inner_timeout > 10000) {
	  break;
	}
      }
		
      // Setup command
      host_device_interface->command = cmd;
      host_device_interface->has_task = true;
      __threadfence_system();
      //__threadfence();

      inner_timeout = 0;
      // wait for socket communication to end
      while (!host_device_interface->is_result_available) {
        __threadfence_system();
        //__threadfence();
	      
        if (++inner_timeout > 30000) {
	  break;
        }
      }

      return_value = host_device_interface->result_int;
      
      host_device_interface->is_result_available = false;
      host_device_interface->lock_thread_id = -1;

      __threadfence_system();
      //__threadfence();

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

$$__device__$$
int at_illecker_getLongResult($$__global$$ char * gc_info, HostDeviceInterface::MESSAGE_TYPE cmd, 
    int * exception) {

  int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  int count = 0;
  int timeout = 0;
  bool done = false;
  long return_value = -1;

  while (count < 100) {

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

      int inner_timeout = 0;
      // wait for possible old task to end
      while (host_device_interface->has_task) {
        if (++inner_timeout > 10000) {
	  break;
	}
      }
		
      // Setup command
      host_device_interface->command = cmd;
      host_device_interface->has_task = true;
      __threadfence_system();
      //__threadfence();

      inner_timeout = 0;
      // wait for socket communication to end
      while (!host_device_interface->is_result_available) {
        __threadfence_system();
        //__threadfence();
	      
        if (++inner_timeout > 30000) {
	  break;
        }
      }

      return_value = host_device_interface->result_long;
      
      host_device_interface->is_result_available = false;
      host_device_interface->lock_thread_id = -1;

      __threadfence_system();
      //__threadfence();

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

$$__device__$$
int at_illecker_getStringResult($$__global$$ char * gc_info, HostDeviceInterface::MESSAGE_TYPE cmd, 
    int int_parameter, bool use_int_parameter, int * exception) {

  int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  int count = 0;
  int timeout = 0;
  bool done = false;
  int return_value = 0;

  while (count < 100) {

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


      int inner_timeout = 0;
      // wait for possible old task to end
      while (host_device_interface->has_task) {
        if (++inner_timeout > 10000) {
	  break;
	}
      }
		
      // Setup command
      host_device_interface->command = cmd;
      if (use_int_parameter) {
        host_device_interface->param1 = int_parameter;
      }
      host_device_interface->has_task = true;
      __threadfence_system();
      //__threadfence();

      inner_timeout = 0;
      // wait for socket communication to end
      while (!host_device_interface->is_result_available) {
        __threadfence_system();
        //__threadfence();
	      
        if (++inner_timeout > 30000) {
	  break;
        }
      }

      // make new String object
      edu_syr_pcpratts_gc_assign(gc_info, &return_value,
        at_illecker_string_constant(gc_info, host_device_interface->result_string, exception));

      host_device_interface->is_result_available = false;
      host_device_interface->lock_thread_id = -1;
      
      __threadfence_system();
      //__threadfence();

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


// HamaPeer.send
// public static void send(String peerName, String msg)
$$__device__$$
void edu_syr_pcpratts_rootbeer_runtime_HamaPeer_send($$__global$$ char * gc_info, 
     int peer_name_str_ref, int message_str_ref, int * exception) {

  // TODO
}

// HamaPeer.getCurrentMessage
// public static String getCurrentMessage()
$$__device__$$
int edu_syr_pcpratts_rootbeer_runtime_HamaPeer_getCurrentMessage($$__global$$ char * gc_info, 
    int * exception) {

  return at_illecker_getStringResult(gc_info, HostDeviceInterface::GET_MSG, 0, false, exception);
}

// HamaPeer.getNumCurrentMessages
// public static int getNumCurrentMessages()
$$__device__$$
int edu_syr_pcpratts_rootbeer_runtime_HamaPeer_getNumCurrentMessages($$__global$$ char * gc_info, 
    int * exception) {

  return at_illecker_getIntResult(gc_info, HostDeviceInterface::GET_MSG_COUNT, exception);
}

// HamaPeer.sync
// public static void sync()
$$__device__$$
void edu_syr_pcpratts_rootbeer_runtime_HamaPeer_sync($$__global$$ char * gc_info, 
     int * exception) {

  at_illecker_getIntResult(gc_info, HostDeviceInterface::SYNC, exception);
}

// HamaPeer.getSuperstepCount
// public static long getSuperstepCount()
$$__device__$$
long edu_syr_pcpratts_rootbeer_runtime_HamaPeer_getSuperstepCount($$__global$$ char * gc_info, 
    int * exception) {

  return at_illecker_getLongResult(gc_info, HostDeviceInterface::GET_SUPERSTEP_COUNT, exception);
}

// HamaPeer.getPeerName
// public static String getPeerName()
$$__device__$$
int edu_syr_pcpratts_rootbeer_runtime_HamaPeer_getPeerName($$__global$$ char * gc_info, 
    int * exception) {

  return at_illecker_getStringResult(gc_info, HostDeviceInterface::GET_PEERNAME, -1, true, exception);
}

// HamaPeer.getPeerName
// public static String getPeerName(int index)
$$__device__$$
int edu_syr_pcpratts_rootbeer_runtime_HamaPeer_getPeerName($$__global$$ char * gc_info, 
    int index, int * exception) {

  return at_illecker_getStringResult(gc_info, HostDeviceInterface::GET_PEERNAME, index, true, exception);
}

// HamaPeer.getPeerIndex
// public static int getPeerIndex()
$$__device__$$
int edu_syr_pcpratts_rootbeer_runtime_HamaPeer_getPeerIndex($$__global$$ char * gc_info, 
    int * exception) {

  return at_illecker_getIntResult(gc_info, HostDeviceInterface::GET_PEER_INDEX, exception);
}

// HamaPeer.getAllPeerNames
// public static String[] getAllPeerNames()
$$__device__$$
int edu_syr_pcpratts_rootbeer_runtime_HamaPeer_getAllPeerNames($$__global$$ char * gc_info, 
    int * exception) {
  // TODO
  return 0;
}

// HamaPeer.getNumPeers
// public static int getNumPeers()
$$__device__$$
int edu_syr_pcpratts_rootbeer_runtime_HamaPeer_getNumPeers($$__global$$ char * gc_info, 
    int * exception) {

  return at_illecker_getIntResult(gc_info, HostDeviceInterface::GET_PEER_COUNT, exception);
}

// HamaPeer.clear
// public static void clear()
$$__device__$$
void edu_syr_pcpratts_rootbeer_runtime_HamaPeer_clear($$__global$$ char * gc_info, 
     int * exception) {

  at_illecker_getIntResult(gc_info, HostDeviceInterface::CLEAR, exception);
}

// HamaPeer.write
// public static void write(String key, String value)
$$__device__$$
void edu_syr_pcpratts_rootbeer_runtime_HamaPeer_write($$__global$$ char * gc_info, 
     int peer_name_str_ref, int message_str_ref, int * exception) {
  // TODO
}

// HamaPeer.readNext
// public static boolean readNext(String key, String value)
$$__device__$$
bool edu_syr_pcpratts_rootbeer_runtime_HamaPeer_readNext($$__global$$ char * gc_info, 
     int peer_name_str_ref, int message_str_ref, int * exception) {
  // TODO
  return true;
}

// HamaPeer.reopenInput
// public static void reopenInput() {
$$__device__$$
void edu_syr_pcpratts_rootbeer_runtime_HamaPeer_reopenInput($$__global$$ char * gc_info, 
     int * exception) {

  at_illecker_getIntResult(gc_info, HostDeviceInterface::REOPEN_INPUT, exception);
}

