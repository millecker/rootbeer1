/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.runtime;

import java.util.Iterator;
import java.util.List;

public interface IRootbeerInternal<T> {
  public void runAll(T jobs);
  public void runAll(List<T> jobs);
  public Iterator<T> run(Iterator<T> jobs);
  public void setThreadConfig(ThreadConfig thread_config);
  public void clearThreadConfig();
  public void printMem(int start, int len);
  public List<GpuCard> getGpuCards();
  public void setCurrentGpuCard(GpuCard gpuCard);
  public GpuCard getCurrentGpuCard();
}
