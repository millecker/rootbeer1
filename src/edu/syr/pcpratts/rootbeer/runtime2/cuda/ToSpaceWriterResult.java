/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.runtime2.cuda;

import java.util.List;

public class ToSpaceWriterResult<T> {
  
  private List<Long> m_Handles;
  private List<T> m_Items;
  private List<T> m_NotWrittenItems;
  
  public ToSpaceWriterResult(List<Long> handles, List<T> items,
    List<T> not_written){
    
    m_Handles = handles;
    m_Items = items;
    m_NotWrittenItems = not_written;
  }
  
  public List<Long> getHandles(){
    return m_Handles;
  }
  
  public List<T> getItems(){
    return m_Items;
  }
  
  public List<T> getNotWrittenItems(){
    return m_NotWrittenItems; 
  }
}
