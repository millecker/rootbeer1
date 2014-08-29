package org.trifort.rootbeer.testcases.rootbeertest.kerneltemplate;

/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.RootbeerGpu;

public class SharedMemoryRunOnGpu implements Kernel {

  private int[] m_array;
  private int m_blockSize;

  public SharedMemoryRunOnGpu(int[] array, int blockSize) {
    m_array = array;
    m_blockSize = blockSize;
  }

  public void gpuMethod() {
    int blockSize = m_blockSize;
    int thread_idxx = RootbeerGpu.getThreadIdxx();

    int value1 = RootbeerGpu.getThreadId();
    int value2 = RootbeerGpu.getThreadId();

    // Store value1 in shared memory
    RootbeerGpu.setSharedInteger(thread_idxx * 4, value1);

    // Store value2 in shared memory
    RootbeerGpu.setSharedInteger(1024 + (thread_idxx * 4), value2);

    // Sync all threads and make sure shared memory is consistent
    RootbeerGpu.syncthreads();

    int sum = 0;
    for (int i = 0; i < blockSize; i++) {
      // Shift fetch index
      int fetchIndex = (thread_idxx + i) % blockSize;

      // Fetch value1 from shared memory
      value1 = RootbeerGpu.getSharedInteger(fetchIndex * 4);

      // Fetch value2 from shared memory
      value2 = RootbeerGpu.getSharedInteger(1024 + (fetchIndex * 4));

      sum += value1 * value2;
    }

    m_array[thread_idxx] = sum;
  }

  public boolean compare(SharedMemoryRunOnGpu rhs) {
    for (int i = 0; i < m_blockSize; i++) {
      if (m_array[i] != rhs.m_array[i]) {
        System.out.println("m_array error at index: " + i + " " + m_array[i]
            + " != " + rhs.m_array[i]);
        return false;
      }
    }
    return true;
  }
}
