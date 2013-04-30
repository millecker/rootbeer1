/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.runtime.cpu;

import edu.syr.pcpratts.rootbeer.runtime.Kernel;
import java.util.Queue;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.LinkedBlockingQueue;

class CpuCore<T> implements Runnable{

  private LinkedBlockingQueue<T> m_InQueue;
  private LinkedBlockingQueue<T> m_OutQueue;
  public CpuCore(){
    m_InQueue = new LinkedBlockingQueue<T>();
    m_OutQueue = new LinkedBlockingQueue<T>();
    Thread t = new Thread(this);
    t.setDaemon(true);
    t.start();
  }

  public void run() {
    while(true){
      try {
        T job = m_InQueue.take();
        job.gpuMethod();
        m_OutQueue.put(job);
      } catch(Exception ex){
        //ignore
      }
    }
  }

  void enqueue(T job) {
    while(true){
      try {
        m_InQueue.put(job);
        return;
      } catch(Exception ex){
        //ignore
      }
    }
  }

  T getResult() {
    while(true){
      try {
        return m_OutQueue.take();
      } catch(Exception ex){
        //ignore
      }
    }
  }
}
