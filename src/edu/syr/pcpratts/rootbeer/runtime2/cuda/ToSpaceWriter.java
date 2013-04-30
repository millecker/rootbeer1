/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.runtime2.cuda;

import edu.syr.pcpratts.rootbeer.runtime.Serializer;
import java.util.ArrayList;
import java.util.List;

public class ToSpaceWriter<T> {
  
  private BlockingQueue<InputItem<T>> m_InputQueue;
  private BlockingQueue<ToSpaceWriterResult<T>> m_OutputQueue;
  private Thread m_Thread;
  
  public ToSpaceWriter(){
    m_InputQueue = new BlockingQueue<InputItem<T>>();
    m_OutputQueue = new BlockingQueue<ToSpaceWriterResult<T>>();
    
    WriteThreadProc<T> proc = new WriteThreadProc<T>(m_InputQueue, m_OutputQueue);
    m_Thread = new Thread(proc);
    m_Thread.setDaemon(true);
    m_Thread.start();
  }
  
  public void write(List<T> items, Serializer visitor){
    InputItem<T> item = new InputItem<T>();
    item.m_Items = items;
    item.m_Visitor = visitor;
    m_InputQueue.put(item);
  }
  
  public ToSpaceWriterResult<T> join(){
    return m_OutputQueue.take();  
  }
  
  private class InputItem<E> {
    public List<E> m_Items;
    public Serializer m_Visitor;
  }  
  
  private class WriteThreadProc<E> implements Runnable {

    private BlockingQueue<InputItem<E>> m_InputQueue;
    private BlockingQueue<ToSpaceWriterResult<E>> m_OutputQueue;
  
    public WriteThreadProc(BlockingQueue<InputItem<E>> input_queue,
      BlockingQueue<ToSpaceWriterResult<E>> output_queue){
      
      m_InputQueue = input_queue;
      m_OutputQueue = output_queue;
    }
    
    public void run() {
      while(true){  
        List<Long> handles = new ArrayList<Long>();   
        List<E> items = new ArrayList<E>();
        List<E> not_written = new ArrayList<E>();
        try {
          InputItem<E> input_item = m_InputQueue.take();
          not_written.addAll(input_item.m_Items);
          for(E item : input_item.m_Items){
            long handle = input_item.m_Visitor.writeToHeap(item);
            handles.add(handle);
            items.add(item);
            not_written.remove(0);
          }     
        } catch(OutOfMemoryError ex){
          ex.printStackTrace();
          System.exit(1);
        } finally {
          m_OutputQueue.put(new ToSpaceWriterResult<E>(handles, items, not_written));     
        }
      }
    }
    
  }

}
