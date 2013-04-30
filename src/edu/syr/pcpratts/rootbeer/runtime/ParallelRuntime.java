/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.runtime;

import java.util.Iterator;

public interface ParallelRuntime<T> {

  public PartiallyCompletedParallelJob<T> run(Iterator<T> blocks, Rootbeer rootbeer, ThreadConfig thread_config) throws Exception;
  public boolean isGpuPresent();
}
