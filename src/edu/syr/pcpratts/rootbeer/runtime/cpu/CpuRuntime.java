/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.runtime.cpu;

import edu.syr.pcpratts.rootbeer.runtime.*;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class CpuRuntime<T> implements ParallelRuntime<T> {

  private static CpuRuntime mInstance = null;
  private List<CpuCore<T>> m_Cores;

  public static <T> CpuRuntime v(){
    if(mInstance == null)
      mInstance = new CpuRuntime<T>();
    return mInstance;
  }

  private CpuRuntime(){
    m_Cores = new ArrayList<CpuCore<T>>();
    int num_cores = Runtime.getRuntime().availableProcessors();
    for(int i = 0; i < num_cores; ++i){
      m_Cores.add(new CpuCore<T>());
    }
  }

  public PartiallyCompletedParallelJob<T> run(Iterator<T> jobs, Rootbeer rootbeer, ThreadConfig thread_config) throws Exception {
    PartiallyCompletedParallelJob<T> ret = new PartiallyCompletedParallelJob<T>(jobs);
    int enqueued = 0;
    for(int i = 0; i < m_Cores.size(); ++i){
      if(jobs.hasNext()){
        T job = jobs.next();
        m_Cores.get(i).enqueue(job);
        enqueued++;
      }
    }

    for(int i = 0; i < enqueued; ++i){
      T curr = m_Cores.get(i).getResult();
      ret.enqueueJob(curr);
    }
    return ret;
  }

  public boolean isGpuPresent() {
    return true;
  }


}
