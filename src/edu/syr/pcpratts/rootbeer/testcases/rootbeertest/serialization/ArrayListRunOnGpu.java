/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.testcases.rootbeertest.serialization;

import edu.syr.pcpratts.rootbeer.runtime.Kernel;
import java.util.List;

public class ArrayListRunOnGpu implements Kernel {

  private List<ArrayListTestObject> m_arrayList;
  
  public ArrayListRunOnGpu(List<ArrayListTestObject> array_list){
    m_arrayList = array_list;
  }
  
  public void gpuMethod() {
    for(int i = 0; i < 4; ++i){
      ArrayListTestObject new_object = new ArrayListTestObject();
      new_object.m_value = i;
      m_arrayList.add(new_object);
    }
  }

  public boolean compare(ArrayListRunOnGpu rhs) {
    List<ArrayListTestObject> lhs_list = m_arrayList;
    List<ArrayListTestObject> rhs_list = rhs.m_arrayList;
    
    if(lhs_list.size() != rhs_list.size()){
      System.out.println("size");
      System.out.println("lhs: "+lhs_list.size());
      System.out.println("rhs: "+rhs_list.size());
      return false;
    }
    
    boolean passed = true;
    for(int i = 0; i < lhs_list.size(); ++i){
      ArrayListTestObject lhs_obj = lhs_list.get(i);
      ArrayListTestObject rhs_obj = rhs_list.get(i);
      if(lhs_obj == null){
        System.out.println("lhs_obj null at "+i);
<<<<<<< HEAD
        return false;
      }
      if(rhs_obj == null){
        System.out.println("rhs_obj null at "+i);
        return false;
      }
      if(lhs_obj.m_value != rhs_obj.m_value){
        System.out.println("obj.m_value at "+i);
        return false;
=======
        passed = false;
      } else if(rhs_obj == null){
        System.out.println("rhs_obj null at "+i);
        passed = false;
      } else if(lhs_obj.m_value != rhs_obj.m_value){
        System.out.println("obj.m_value at "+i);
        passed = false;
>>>>>>> 56f1a04b81d80e4356d3decc3e22ef176f2fd6c7
      }
    }
    return passed;
  }

}
