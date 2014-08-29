/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.testcases.rootbeertest.kerneltemplate;

import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.ThreadConfig;
import org.trifort.rootbeer.test.TestKernelTemplate;

public class SharedMemoryTest implements TestKernelTemplate {

  private int m_blockSize;
  private int m_gridSize;
  private int[] m_array;

  public SharedMemoryTest() {
    m_blockSize = 1024;
    m_gridSize = 1;
    m_array = new int[m_blockSize];
  }

  public Kernel create() {
    Kernel ret = new SharedMemoryRunOnGpu(m_array, m_gridSize);
    return ret;
  }

  public ThreadConfig getThreadConfig() {
    ThreadConfig ret = new ThreadConfig(m_blockSize, m_gridSize,
        (long) m_blockSize * (long) m_gridSize);
    return ret;
  }

  public boolean compare(Kernel original, Kernel from_heap) {
    SharedMemoryRunOnGpu lhs = (SharedMemoryRunOnGpu) original;
    SharedMemoryRunOnGpu rhs = (SharedMemoryRunOnGpu) from_heap;
    return lhs.compare(rhs);
  }

}
