/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.runtime2.cuda;

public class Handles {
  
  private long m_addr;
  private int m_offset;
  
  public Handles(long base_address){
    m_addr = base_address;
  }

  
  public void resetPointer(){
    m_offset = 0;
  }
  
  public void writeLong(long value){
    doWriteLong(m_addr, m_offset, value);
    ++m_offset;
  }
  
  public long readLong(){
    long ret = doReadLong(m_addr, m_offset);
    ++m_offset;
    return ret;
  }
  
  private native void doWriteLong(long base_addr, int offset, long value);
  private native long doReadLong(long base_addr, int offset);
}
