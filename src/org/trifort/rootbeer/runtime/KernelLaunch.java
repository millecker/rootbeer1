package org.trifort.rootbeer.runtime;

public class KernelLaunch {

  private int m_deviceIndex;
  private byte[] m_cubinFile;
  private int m_cubinFileLength;
  private int m_blockShapeX;
  private int m_gridShapeX;
  private long m_numThreads;
  private Memory m_objectMem;
  private Memory m_handlesMem;
  private Memory m_exceptionsMem;
  private Memory m_classMem;
<<<<<<< HEAD
  private HamaPeer m_hamaPeer;
=======
  private boolean m_usingKernelTemplates;
>>>>>>> 36575e07a2395316c69a17b7ffb41fe069ccde4c
  private boolean m_quit;
  private Exception m_exception;
  
  public KernelLaunch(int device_index, byte[] cubin_file, int cubin_file_length, 
<<<<<<< HEAD
    int block_shape_x, int grid_shape_x, int num_threads, Memory object_mem, 
    Memory handles_mem, Memory exceptions_mem, Memory class_mem, HamaPeer hama_peer) {
=======
    int block_shape_x, int grid_shape_x, long num_threads, Memory object_mem, 
    Memory handles_mem, Memory exceptions_mem, Memory class_mem, 
    boolean usingKernelTemplates){
>>>>>>> 36575e07a2395316c69a17b7ffb41fe069ccde4c
    
    m_deviceIndex = device_index;
    m_cubinFile = cubin_file;
    m_cubinFileLength = cubin_file_length;
    m_blockShapeX = block_shape_x;
    m_gridShapeX = grid_shape_x;
    m_numThreads = num_threads;
    m_objectMem = object_mem;
    m_handlesMem = handles_mem;
    m_exceptionsMem = exceptions_mem;
    m_classMem = class_mem;
<<<<<<< HEAD
    m_hamaPeer = hama_peer;
=======
    m_usingKernelTemplates = usingKernelTemplates;
>>>>>>> 36575e07a2395316c69a17b7ffb41fe069ccde4c
    m_quit = false;
  }
  
  public KernelLaunch(boolean quit){
    m_quit = quit;
  }
  
  public boolean quit() {
    return m_quit;
  }

  public int getDeviceIndex() {
    return m_deviceIndex;
  }

  public byte[] getCubinFile() {
    return m_cubinFile;
  }

  public int getCubinLength() {
    return m_cubinFileLength;
  }

  public int getBlockShapeX() {
    return m_blockShapeX;
  }

  public int getGridShapeX() {
    return m_gridShapeX;
  }

  public int getNumThreads() {
    return (int) m_numThreads;
  }

  public Memory getObjectMem() {
    return m_objectMem;
  }

  public Memory getHandlesMem() {
    return m_handlesMem;
  }

  public Memory getExceptionsMem() {
    return m_exceptionsMem;
  }

  public Memory getClassMem() {
    return m_classMem;
  }
  
<<<<<<< HEAD
  public HamaPeer getHamaPeer() {
    return m_hamaPeer;
=======
  public boolean getUsingKernelTemplates(){
    return m_usingKernelTemplates;
>>>>>>> 36575e07a2395316c69a17b7ffb41fe069ccde4c
  }
  
  public void setException(Exception exception){
    m_exception = exception;
  }
  
  public Exception getException(){
    return m_exception;
  }
}
