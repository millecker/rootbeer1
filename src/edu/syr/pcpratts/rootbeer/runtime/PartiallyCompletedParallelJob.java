/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.runtime;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

public class PartiallyCompletedParallelJob<T> {

  private Iterator<T> m_RemainingJobs;
  private List<T> m_ActiveJobs;
  private List<T> m_NotWritten;

  public PartiallyCompletedParallelJob(Iterator<T> remaining_jobs) {
    m_RemainingJobs = remaining_jobs;
    m_ActiveJobs = new LinkedList<T>();
    m_NotWritten = new ArrayList<T>();
  }

  public List<T> getActiveJobs() {
    return m_ActiveJobs;
  }

  public Iterator<T> getJobsToEnqueue(){
    return new CompositeIterator<T>(m_NotWritten, m_RemainingJobs);
  }

  public void enqueueJob(T job){
    m_ActiveJobs.add(job);
  }

  public void enqueueJobs(List<T> items) {
    m_ActiveJobs.addAll(items);
  }

  public void addNotWritten(List<T> not_written) {
    m_NotWritten = new ArrayList<T>();
    m_NotWritten.addAll(not_written);
  }
  
  public class CompositeIterator<E> implements Iterator<E> {

    private Iterator<E> m_NotWritten;
    private Iterator<E> m_Remaining;
    
    private CompositeIterator(List<E> not_written, Iterator<E> remaining) {
      m_NotWritten = not_written.iterator();
      m_Remaining = remaining;
    }    

    public boolean hasNext() {
      if(m_NotWritten.hasNext())
        return true;
      if(m_Remaining.hasNext())
        return true;
      return false;
    }

    public E next() {
      if(m_NotWritten.hasNext())
        return m_NotWritten.next();
      if(m_Remaining.hasNext())
        return m_Remaining.next();
      throw new RuntimeException("out of items");
    }

    public void remove() {
      throw new UnsupportedOperationException("Not supported yet.");
    }
    
  }
}
