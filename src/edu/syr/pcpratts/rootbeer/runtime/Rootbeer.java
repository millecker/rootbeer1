/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.runtime;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class Rootbeer implements IRootbeer {

  private IRootbeerInternal m_Rootbeer;
  private List<StatsRow> m_stats;
  private boolean m_ranGpu;
  private ThreadConfig m_threadConfig;
  
  public Rootbeer(){
    RootbeerFactory factory = new RootbeerFactory();
    m_Rootbeer = factory.create(this);
  }
  
  public Rootbeer(Map<String, String> env){
    this();
    int port = Integer.parseInt(env.get("hama.pipes.command.port"));
    System.out.println("Starting Rootbeer using port: " + port);
    //HamaPeer.init(port);
  }
  
  public static void init(){
    try {
      Class c = Class.forName("edu.syr.pcpratts.rootbeer.runtime2.cuda.CudaRuntime2");
      Method v_method = c.getMethod("v");
      v_method.invoke(null);
    } catch(Exception ex){
      //ignore
    }
  }
  
  public void setThreadConfig(int block_shape_x, int grid_shape_x, int numThreads){
    m_threadConfig = new ThreadConfig(block_shape_x, grid_shape_x, numThreads);
  }
  
  public void runAll(Kernel job_template){
    if(job_template instanceof CompiledKernel == false){
      System.out.println("m_ranGpu = false #1");
      m_ranGpu = false;
    }
    //this must happen above Rootbeer.runAll in case exceptions are thrown
    m_ranGpu = true;
      
    m_stats = new ArrayList<StatsRow>();
    if(m_threadConfig != null){
      m_Rootbeer.setThreadConfig(m_threadConfig);
      m_threadConfig = null;
    } else {
      m_Rootbeer.clearThreadConfig();
    }
    m_Rootbeer.runAll(job_template);
  }
  
  public void runAll(List<Kernel> jobs) {
    if(jobs.isEmpty()){
      System.out.println("m_ranGpu = false #2");
      m_ranGpu = false;
      return;
    }
    if(jobs.get(0) instanceof CompiledKernel == false){
      for(Kernel job : jobs){
        job.gpuMethod();
      }
      Kernel first = jobs.get(0);
      Class cls = first.getClass();
      Class[] ifaces = cls.getInterfaces();
      for(Class iface : ifaces){
        System.out.println("iface: "+iface.getName());
      }
      System.out.println("m_ranGpu = false 3");
      m_ranGpu = false;
    } else {
      //this must happen above Rootbeer.runAll in case exceptions are thrown
      m_ranGpu = true;
      
      m_stats = new ArrayList<StatsRow>();
      if(m_threadConfig != null){
        m_Rootbeer.setThreadConfig(m_threadConfig);
        m_threadConfig = null;
      } else {
        m_Rootbeer.clearThreadConfig();
      }
      m_Rootbeer.runAll(jobs);
    }
  }
  
  public void printMem(int start, int len){
    m_Rootbeer.printMem(start, len);
  }

  public boolean getRanGpu(){
    return m_ranGpu;  
  }
  
  public Iterator<Kernel> run(Iterator<Kernel> jobs) {
    return m_Rootbeer.run(jobs);
  }
  
  public void addStatsRow(StatsRow row) {
    m_stats.add(row);
  }
  
  public List<StatsRow> getStats(){
    return m_stats;
  }
}
