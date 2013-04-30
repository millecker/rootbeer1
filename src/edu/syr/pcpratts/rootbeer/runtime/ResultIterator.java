/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.runtime;

import java.util.Iterator;
import java.util.List;

public class ResultIterator<T> implements Iterator<T> {

  private Iterator<T> m_currIter;
  private Iterator<T> m_jobsToEnqueue;
  private ParallelRuntime<T> m_runtime;
  private Rootbeer m_rootbeer;

  public ResultIterator(PartiallyCompletedParallelJob<T> partial, ParallelRuntime<T> runtime, Rootbeer rootbeer){
    readPartial(partial);
    m_runtime = runtime;
    m_rootbeer = rootbeer;
  }

  private void readPartial(PartiallyCompletedParallelJob<T> partial){
    List<T> active_jobs = partial.getActiveJobs();
    m_currIter = active_jobs.iterator();
    m_jobsToEnqueue = partial.getJobsToEnqueue();
  }

  public boolean hasNext() {
    if(m_currIter.hasNext())
      return true;
    if(m_jobsToEnqueue.hasNext() == false)
      return false;
    try {
      readPartial(m_runtime.run(m_jobsToEnqueue, m_rootbeer, null));
    } catch(Exception ex){
      ex.printStackTrace();
      return false;
    }
    return m_currIter.hasNext();
  }

  public T next() {
    return m_currIter.next();
  }

  public void remove() {
    throw new UnsupportedOperationException("Not supported yet.");
  }

}
