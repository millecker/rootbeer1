/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.runtime2.cuda;

import edu.syr.pcpratts.rootbeer.runtime.Serializer;
import java.util.List;

public class ToSpaceReader<T> {

  private BlockingQueue<InputItem<T>> m_InputQueue;
  private BlockingQueue<InputItem<T>> m_OutputQueue;
  private Thread m_Thread;
  
  public ToSpaceReader(){
    m_InputQueue = new BlockingQueue<InputItem<T>>();
    m_OutputQueue = new BlockingQueue<InputItem<T>>();
    ReadThreadProc<T> proc = new ReadThreadProc<T>(m_InputQueue, m_OutputQueue);
    m_Thread = new Thread(proc);
    m_Thread.setDaemon(true);
    m_Thread.start();
  }
  
  public void read(List<T> items, List<Long> handles, Serializer visitor){
    InputItem<T> item = new InputItem<T>();
    item.m_Items = items;
    item.m_HandlesCache = handles;
    item.m_Visitor = visitor;
    m_InputQueue.put(item);
  }
  
  public void join(){
    m_OutputQueue.take();
  }
  
  private class InputItem<E> {
    public List<E> m_Items;
    public List<Long> m_HandlesCache;
    public Serializer m_Visitor;
  }
    
  private class ReadThreadProc<E> implements Runnable {
       
    private BlockingQueue<InputItem<E>> m_InputQueue;
    private BlockingQueue<InputItem<E>> m_OutputQueue;
  
    public ReadThreadProc(BlockingQueue<InputItem<E>> input_queue,
      BlockingQueue<InputItem<E>> output_queue){
      
      m_InputQueue = input_queue;
      m_OutputQueue = output_queue;
    }
        
    public void run(){
      while(true){
        InputItem<E> input_item = m_InputQueue.take();
        for(int i = 0; i < input_item.m_Items.size(); ++i){
          E item = input_item.m_Items.get(i);
          long handle = input_item.m_HandlesCache.get(i);
          
          E new_item = (E) input_item.m_Visitor.readFromHeap(item, true, handle);
          input_item.m_Items.set(i, new_item);
        }
        m_OutputQueue.put(input_item);
      }
    }
  } 
}
