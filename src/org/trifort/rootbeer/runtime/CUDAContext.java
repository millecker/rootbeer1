package org.trifort.rootbeer.runtime;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.trifort.rootbeer.configuration.Configuration;
import org.trifort.rootbeer.runtime.util.Stopwatch;
import org.trifort.rootbeer.runtimegpu.GpuException;
import org.trifort.rootbeer.util.ResourceReader;

public class CUDAContext implements Context, Runnable {

  private GpuDevice m_device;
  private boolean m_32bit;
  private Map<String, byte[]> m_cubinFiles;

  private List<StatsRow> m_stats;
  private Stopwatch m_writeBlocksStopwatch;
  private Stopwatch m_runStopwatch;
  private Stopwatch m_runOnGpuStopwatch;
  private Stopwatch m_readBlocksStopwatch;
  private long m_serializationTime;
  private long m_executionTime;
  private long m_deserializationTime;
  private long m_overallTime;
  
  private Memory m_objectMemory;
  private Memory m_textureMemory;
  private Memory m_handlesMemory;
  private Memory m_exceptionsMemory;
  private Memory m_classMemory;
  private boolean m_usingKernelTemplates;
  
  private Map<Kernel, Long> m_handles;
  
  private Thread m_thread;
  private BlockingQueue<KernelLaunch> m_toThread;
  private BlockingQueue<KernelLaunch> m_fromThread;
  
  private boolean m_usingUncheckedMemory;
  
  private HamaPeer m_hamaPeer = null;
  
  public CUDAContext(GpuDevice device){
    m_usingUncheckedMemory = false;
    m_device = device;    
       
    String arch = System.getProperty("os.arch");
    m_32bit = arch.equals("x86") || arch.equals("i386");
    
    m_cubinFiles = new HashMap<String, byte[]>();
    
    m_stats = new ArrayList<StatsRow>();
    m_writeBlocksStopwatch = new Stopwatch();
    m_runStopwatch = new Stopwatch();
    m_runOnGpuStopwatch = new Stopwatch();
    m_readBlocksStopwatch = new Stopwatch();
    
    m_handles = new HashMap<Kernel, Long>();
    m_textureMemory = new CheckedFixedMemory(64);
    
    m_toThread = new BlockingQueue<KernelLaunch>();
    m_fromThread = new BlockingQueue<KernelLaunch>();
    m_thread = new Thread(this);
    m_thread.setDaemon(true);
    m_thread.start();
  }

  @Override
  public void init(long memory_size) {
    if(m_usingUncheckedMemory){
      m_objectMemory = new FixedMemory(memory_size);
    } else {
      m_objectMemory = new CheckedFixedMemory(memory_size);
    }
  }

  @Override
  public void init() {
    long free_mem_size_gpu = m_device.getFreeGlobalMemoryBytes();
    long free_mem_size_cpu = Runtime.getRuntime().freeMemory();
    long free_mem_size = Math.min(free_mem_size_gpu, free_mem_size_cpu);
    
    //space for instructions
    free_mem_size -= 64 * 1024 * 1024;
    
    free_mem_size -= m_handlesMemory.getSize();
    free_mem_size -= m_exceptionsMemory.getSize();
    free_mem_size -= m_classMemory.getSize();
    
    if(free_mem_size <= 0){
      StringBuilder error = new StringBuilder();
      error.append("OutOfMemory while allocating Java CPU and GPU memory.\n");
      error.append("  Try increasing the max Java Heap Size using -Xmx and the initial Java Heap Size using -Xms.\n");
      error.append("  Try reducing the number of threads you are using.\n");
      error.append("  Try using kernel templates.\n");
      error.append("  Debugging Output:\n");
      error.append("    GPU_SIZE: "+free_mem_size_gpu+"\n");
      error.append("    CPU_SIZE: "+free_mem_size_cpu+"\n");
      error.append("    HANDLES_SIZE: "+m_handlesMemory.getSize()+"\n");
      error.append("    EXCEPTIONS_SIZE: "+m_exceptionsMemory.getSize()+"\n");
      error.append("    CLASS_MEMORY_SIZE: "+m_classMemory.getSize());
      throw new RuntimeException(error.toString());
    }
    m_objectMemory = new CheckedFixedMemory(free_mem_size);
  }
  
  @Override
  public void close() {
    m_toThread.put(new KernelLaunch(true));
    m_fromThread.take();
    
    m_objectMemory.close();
    m_objectMemory = null;
    m_handlesMemory.close();
    m_exceptionsMemory.close();
    m_classMemory.close();
  }
  
  public long size(){
    return m_objectMemory.getSize();
  }
  
  public List<StatsRow> getStats(){
    return m_stats;
  }
  
  @Override
  public GpuDevice getDevice() {
    return m_device;
  }
  
  @Override
  public void setHamaPeer(HamaPeer hamaPeer) {
    this.m_hamaPeer = hamaPeer;
  }

  @Override
  public void run(Kernel template, ThreadConfig thread_config) {
    m_usingKernelTemplates = true;
    m_runStopwatch.start();
    CompiledKernel compiled_kernel = (CompiledKernel) template;
    
    String filename;
    if(m_32bit){
      filename = compiled_kernel.getCubin32();
    } else {
      filename = compiled_kernel.getCubin64();
    }
    
    if(filename.endsWith(".error")){
      throw new RuntimeException("CUDA code compiled with error");
    }
    
    byte[] cubin_file;
    
    if(m_cubinFiles.containsKey(filename)){
      cubin_file = m_cubinFiles.get(filename);
    } else {
      cubin_file = readCubinFile(filename);
      m_cubinFiles.put(filename, cubin_file);
    }
    if(m_usingUncheckedMemory){
      m_handlesMemory = new FixedMemory(4);
      m_classMemory = new FixedMemory(1024);
      m_exceptionsMemory = new FixedMemory(getExceptionsMemSize(thread_config));
    } else {
      m_handlesMemory = new CheckedFixedMemory(4);
      m_exceptionsMemory = new CheckedFixedMemory(getExceptionsMemSize(thread_config));
      m_classMemory = new CheckedFixedMemory(1024);
    }
    if(m_objectMemory == null){
      init();
    }
    
    writeBlocksTemplate(compiled_kernel, thread_config);
    runBlocks(thread_config, cubin_file);
    readBlocksTemplate(compiled_kernel, thread_config);
    
    m_runStopwatch.stop();
    m_overallTime = m_runStopwatch.elapsedTimeMillis();
    
    m_stats.add(new StatsRow(m_serializationTime, m_executionTime, 
        m_deserializationTime, m_overallTime,
        thread_config.getGridShapeX(), thread_config.getBlockShapeX()));
  }

  private long getExceptionsMemSize(ThreadConfig thread_config) {
    if(Configuration.runtimeInstance().getExceptions()){
      return 4L*thread_config.getNumThreads();
    } else {
      return 4;
    }
  }

  @Override
  public void run(List<Kernel> work, ThreadConfig thread_config) {
    m_usingKernelTemplates = false;
    m_runStopwatch.start();
    CompiledKernel compiled_kernel = (CompiledKernel) work.get(0);
    
    String filename;
    if(m_32bit){
      filename = compiled_kernel.getCubin32();
    } else {
      filename = compiled_kernel.getCubin64();
    }
    
    if(filename.endsWith(".error")){
      throw new RuntimeException("CUDA code compiled with error");
    }
    
    byte[] cubin_file;
    
    if(m_cubinFiles.containsKey(filename)){
      cubin_file = m_cubinFiles.get(filename);
    } else {
      cubin_file = readCubinFile(filename);
      m_cubinFiles.put(filename, cubin_file);
    }
    
    if(m_usingUncheckedMemory){
      m_handlesMemory = new FixedMemory(4*thread_config.getNumThreads());
      m_exceptionsMemory = new FixedMemory(getExceptionsMemSize(thread_config));
      m_classMemory = new FixedMemory(1024);
    } else {
      m_handlesMemory = new CheckedFixedMemory(4*thread_config.getNumThreads());
      m_exceptionsMemory = new CheckedFixedMemory(getExceptionsMemSize(thread_config));
      m_classMemory = new CheckedFixedMemory(1024);
    }
    if(m_objectMemory == null){
      init();
    }

    writeBlocks(work);
    runBlocks(thread_config, cubin_file);
    readBlocks(work);
    
    m_runStopwatch.stop();
    m_overallTime = m_runStopwatch.elapsedTimeMillis();
    
    m_stats.add(new StatsRow(m_serializationTime, m_executionTime, 
        m_deserializationTime, m_overallTime,
        thread_config.getGridShapeX(), thread_config.getBlockShapeX()));
  }
  
  private void writeBlocks(List<Kernel> work){
    m_writeBlocksStopwatch.start();
    m_objectMemory.clearHeapEndPtr();
    m_handlesMemory.clearHeapEndPtr();
    m_handles.clear();
    
    CompiledKernel compiled_kernel = (CompiledKernel) work.get(0);
    Serializer serializer = compiled_kernel.getSerializer(m_objectMemory, m_textureMemory);
    serializer.writeStaticsToHeap();
    
    for(Kernel kernel : work){
      long handle = serializer.writeToHeap(kernel);
      m_handlesMemory.writeRef(handle);
      m_handles.put(kernel, handle);
    }
    m_objectMemory.align16();
    
    if(Configuration.getPrintMem()){
      BufferPrinter printer = new BufferPrinter();
      printer.print(m_objectMemory, 0, 256);
    }
    m_writeBlocksStopwatch.stop();
    m_serializationTime = m_writeBlocksStopwatch.elapsedTimeMillis();
  }

  private void writeBlocksTemplate(CompiledKernel compiled_kernel,
    ThreadConfig thread_config){
    
    m_writeBlocksStopwatch.start();
    m_objectMemory.clearHeapEndPtr();
    m_handlesMemory.clearHeapEndPtr();
    
    Serializer serializer = compiled_kernel.getSerializer(m_objectMemory, m_textureMemory);
    serializer.writeStaticsToHeap();
    
    long handle = serializer.writeToHeap(compiled_kernel);
    m_handlesMemory.writeRef(handle);
    m_objectMemory.align16();
   
    
    if(Configuration.getPrintMem()){
      BufferPrinter printer = new BufferPrinter();
      printer.print(m_objectMemory, 0, 256);
    }
    m_writeBlocksStopwatch.stop();
    m_serializationTime = m_writeBlocksStopwatch.elapsedTimeMillis();
  }
  
  private void readBlocks(List<Kernel> work){
    m_readBlocksStopwatch.start();
    m_objectMemory.setAddress(0);
    m_handlesMemory.setAddress(0);
    m_exceptionsMemory.setAddress(0);
    
    CompiledKernel compiled_kernel = (CompiledKernel) work.get(0);
    Serializer serializer = compiled_kernel.getSerializer(m_objectMemory, m_textureMemory);

    if(Configuration.runtimeInstance().getExceptions()){
      for(int i = 0; i < work.size(); ++i){
        long ref = m_exceptionsMemory.readRef();
        if(ref != 0){
          long ref_num = ref >> 4;
          if(ref_num == compiled_kernel.getNullPointerNumber()){
            throw new NullPointerException(); 
          } else if(ref_num == compiled_kernel.getOutOfMemoryNumber()){
            throw new OutOfMemoryError();
          }
          
          m_objectMemory.setAddress(ref);          
          Object except = serializer.readFromHeap(null, true, ref);
          if(except instanceof Error){
            Error except_th = (Error) except;
            throw except_th;
          } else if(except instanceof GpuException){
            GpuException gpu_except = (GpuException) except;
            throw new ArrayIndexOutOfBoundsException("array_index: "+gpu_except.m_arrayIndex+
              " array_length: "+gpu_except.m_arrayLength+" array: "+gpu_except.m_array);
          } else {
            throw new RuntimeException((Throwable) except);
          }
        }
      }
    }
    
    serializer.readStaticsFromHeap();
    for(Kernel kernel : work){
      long handle = m_handles.get(kernel);
      serializer.readFromHeap(kernel, true, handle);
    }
    
    if(Configuration.getPrintMem()){
      BufferPrinter printer = new BufferPrinter();
      printer.print(m_objectMemory, 0, 256);
    }
    m_readBlocksStopwatch.stop();
    m_deserializationTime = m_readBlocksStopwatch.elapsedTimeMillis();
  }
  
  private void readBlocksTemplate(CompiledKernel compiled_kernel, ThreadConfig thread_config){
    m_readBlocksStopwatch.start();
    m_objectMemory.setAddress(0);
    m_handlesMemory.setAddress(0);
    m_exceptionsMemory.setAddress(0);
    
    Serializer serializer = compiled_kernel.getSerializer(m_objectMemory, m_textureMemory);
    
    if(Configuration.runtimeInstance().getExceptions()){
      for(long i = 0; i < thread_config.getNumThreads(); ++i){
        long ref = m_exceptionsMemory.readRef();
        if(ref != 0){
          long ref_num = ref >> 4;
          if(ref_num == compiled_kernel.getNullPointerNumber()){
            throw new NullPointerException(); 
          } else if(ref_num == compiled_kernel.getOutOfMemoryNumber()){
            throw new OutOfMemoryError();
          }
          
          m_objectMemory.setAddress(ref);           
          Object except = serializer.readFromHeap(null, true, ref);
          if(except instanceof Error){
            Error except_th = (Error) except;
            throw except_th;
          } else if(except instanceof GpuException){
            GpuException gpu_except = (GpuException) except;
            throw new ArrayIndexOutOfBoundsException("array_index: "+gpu_except.m_arrayIndex+
                " array_length: "+gpu_except.m_arrayLength+" array: "+gpu_except.m_array);
          } else {
            throw new RuntimeException((Throwable) except);
          }
        }
      }    
    }
    
    serializer.readStaticsFromHeap();
    serializer.readFromHeap(compiled_kernel, true, m_handlesMemory.readRef());
    
    if(Configuration.getPrintMem()){
      BufferPrinter printer = new BufferPrinter();
      printer.print(m_objectMemory, 0, 256);
    }
    m_readBlocksStopwatch.stop();
    m_deserializationTime = m_readBlocksStopwatch.elapsedTimeMillis();
  }
  
  private void runBlocks(ThreadConfig thread_config, byte[] cubin_file){    
    m_runOnGpuStopwatch.start();
    
    KernelLaunch item = new KernelLaunch(m_device.getDeviceId(), cubin_file, 
      cubin_file.length, thread_config.getBlockShapeX(), 
      thread_config.getGridShapeX(), thread_config.getNumThreads(), 
      m_objectMemory, m_handlesMemory, m_exceptionsMemory, m_classMemory,
      m_usingKernelTemplates, m_hamaPeer);
    
    m_toThread.put(item);
    item = m_fromThread.take();
    
    m_runOnGpuStopwatch.stop();
    m_executionTime = m_runOnGpuStopwatch.elapsedTimeMillis();
    
    if(item.getException() != null){
      Exception exception = item.getException();
      if(exception instanceof RuntimeException){
        RuntimeException run_except = (RuntimeException) exception;
        throw run_except;
      } else {
        throw new RuntimeException(item.getException());
      }
    }
  }  

  private byte[] readCubinFile(String filename) {
    try {
      List<byte[]> buffer = ResourceReader.getResourceArray(filename);
      int total_len = 0;
      for(byte[] sub_buffer : buffer){
        total_len += sub_buffer.length;
      }
      byte[] cubin_file = new byte[total_len];
      int pos = 0;
      for(byte[] small_buffer : buffer){
        System.arraycopy(small_buffer, 0, cubin_file, pos, small_buffer.length);
        pos += small_buffer.length;
      }
      return cubin_file;
    } catch(Exception ex){
      throw new RuntimeException(ex);
    }
  }
  
  @Override
  public void run() {
    while(true){        
      
      //here we are changing to our own thread before launching.
      //cuda assigns the context to the calling thread, so each
      //launch will be on its own thread and have it's own context
      KernelLaunch item = m_toThread.take();
      
      try {
        if(item.quit()){
          m_fromThread.put(item);
          return;
        }
        
        cudaRun(item.getDeviceIndex(), item.getCubinFile(), item.getCubinLength(),
          item.getBlockShapeX(), item.getGridShapeX(), item.getNumThreads(), 
          item.getObjectMem(), item.getHandlesMem(), item.getExceptionsMem(),
          item.getClassMem(), b2i(item.getUsingKernelTemplates()),
          b2i(Configuration.runtimeInstance().getExceptions()),
          item.getHamaPeer());
        
        m_fromThread.put(item);
      } catch(Exception ex){
        item.setException(ex);
        m_fromThread.put(item);
      }
    }
  }
  
  private int b2i(boolean value){
    if(value){
      return 1;
    } else {
      return 0;
    }
  }
  
  private native void cudaRun(int device_index, byte[] cubin_file, int cubin_length,
    int block_shape_x, int grid_shape_x, int num_threads, Memory object_mem,
    Memory handles_mem, Memory exceptions_mem, Memory class_mem, 
    int usingKernelTemplates, int usingExceptions, HamaPeer hama_peer);
}
