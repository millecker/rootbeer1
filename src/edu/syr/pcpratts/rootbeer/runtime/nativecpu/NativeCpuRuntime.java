/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.runtime.nativecpu;

import edu.syr.pcpratts.rootbeer.runtime.ParallelRuntime;
import edu.syr.pcpratts.rootbeer.runtime.PartiallyCompletedParallelJob;
import edu.syr.pcpratts.rootbeer.runtime.Rootbeer;
import edu.syr.pcpratts.rootbeer.runtime.ThreadConfig;
import java.util.Iterator;

public class NativeCpuRuntime<T> implements ParallelRuntime<T> {

  private static NativeCpuRuntime m_Instance = null;
  
  public static <T> NativeCpuRuntime<T> v(){
    if(m_Instance == null)
      m_Instance = new NativeCpuRuntime<T>();
    return m_Instance;
  }
  
  NativeCpuDevice<T> m_Device;
  
  private NativeCpuRuntime(){
    m_Device = new NativeCpuDevice<T>();
  }
  
  public PartiallyCompletedParallelJob<T> run(Iterator<T> blocks, Rootbeer rootbeer, ThreadConfig thread_config) {
    return m_Device.run(blocks);
  }
  
  public void run(T kernel_template, Rootbeer rootbeer, ThreadConfig thread_config) {
    m_Device.run(kernel_template, thread_config);
  }

  public boolean isGpuPresent() {
    return true;
  }
  
}
