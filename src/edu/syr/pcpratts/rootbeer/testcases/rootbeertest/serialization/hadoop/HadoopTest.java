/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.testcases.rootbeertest.serialization.hadoop;

import java.util.ArrayList;
import java.util.List;

import edu.syr.pcpratts.rootbeer.runtime.Kernel;
import edu.syr.pcpratts.rootbeer.test.TestSerialization;

public class HadoopTest implements TestSerialization {

  @Override
  public List<Kernel> create(){
    TestGpuMapper testGpuMapper = new TestGpuMapper();
    // TestGpuReducer testGpuReducer = new TestGpuReducer();
    
    List<Kernel> ret = new ArrayList<Kernel>();
    /* All Kernels must have the same type
    --> The compiler uses the first element in
    the list to inspect for code generation. */
    
    ret.add(testGpuMapper.new MapperKernel(null,null,null,null));
    // ret.add(testGpuReducer.new ReducerKernel(null,null,null,null));
    return ret;
  }

  @Override
  public boolean compare(Kernel original, Kernel from_heap){
    return true;
  }

}
