/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.testcases.rootbeertest.serialization;

import java.util.ArrayList;
import java.util.List;

import edu.syr.pcpratts.rootbeer.runtime.Kernel;
import edu.syr.pcpratts.rootbeer.test.TestSerialization;

public class StringToLongTest implements TestSerialization {

  public List<Kernel> create() {
    List<Kernel> ret = new ArrayList<Kernel>();
    for(long l = 0; l < 5; ++l) {
      ret.add(new StringToLongRunOnGpu(Long.toString(l)));
    }
    return ret;
  }

  public boolean compare(Kernel original, Kernel from_heap) {
    StringToLongRunOnGpu lhs = (StringToLongRunOnGpu) original;
    StringToLongRunOnGpu rhs = (StringToLongRunOnGpu) from_heap;
    return lhs.compare(rhs);
  }
}
