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

public class IntegerToStringTest implements TestSerialization {

  public List<Kernel> create() {
    List<Kernel> ret = new ArrayList<Kernel>();
    for(int i = 0; i < 5; ++i){
      ret.add(new IntegerToStringRunOnGpu(i));
    }
    return ret;
  }

  public boolean compare(Kernel original, Kernel from_heap) {
    IntegerToStringRunOnGpu lhs = (IntegerToStringRunOnGpu) original;
    IntegerToStringRunOnGpu rhs = (IntegerToStringRunOnGpu) from_heap;
    return lhs.compare(rhs);
  }
}
