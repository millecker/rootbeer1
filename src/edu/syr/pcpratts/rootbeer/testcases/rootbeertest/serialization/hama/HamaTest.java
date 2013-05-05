/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */
package edu.syr.pcpratts.rootbeer.testcases.rootbeertest.serialization.hama;

import java.util.ArrayList;
import java.util.List;

import edu.syr.pcpratts.rootbeer.runtime.Kernel;
import edu.syr.pcpratts.rootbeer.test.TestSerialization;

public class HamaTest implements TestSerialization {

  @Override
  public List<Kernel> create(){
    TestBSP testBSP = new TestBSP();
    List<Kernel> ret = new ArrayList<Kernel>();
    ret.add(testBSP.new SetupKernel(null));
    ret.add(testBSP.new BspKernel(null));
    ret.add(testBSP.new CleanupKernel(null));
    return ret;
  }

  @Override
  public boolean compare(Kernel original, Kernel from_heap){
    return true;
  }

}
