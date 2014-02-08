/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.testcases.rootbeertest.serialization;

import edu.syr.pcpratts.rootbeer.runtime.Kernel;

public class StringToIntegerRunOnGpu implements Kernel {

  private String m_strValue;
  private int m_toInteger;
  
  public StringToIntegerRunOnGpu(String value) {
    m_strValue = value;
  }
  
  public void gpuMethod() {
    m_toInteger = Integer.parseInt(m_strValue);
  }

  public boolean compare(StringToIntegerRunOnGpu rhs) {
    if(rhs.m_toInteger != m_toInteger) {
      System.out.println("m_toInteger");
      System.out.println("  lhs: "+m_toInteger);
      System.out.println("  rhs: "+rhs.m_toInteger);
      return false;
    }
    return true;
  }
}
