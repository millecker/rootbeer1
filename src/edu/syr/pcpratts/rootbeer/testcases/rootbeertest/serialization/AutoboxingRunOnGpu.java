package edu.syr.pcpratts.rootbeer.testcases.rootbeertest.serialization;

import edu.syr.pcpratts.rootbeer.runtime.Kernel;

public class AutoboxingRunOnGpu implements Kernel {

  private Double m_double_result;
  private Integer m_interger_result;
  
  public void gpuMethod() {
    m_double_result = returnDouble();
    m_interger_result = returnInteger();
  }

  private double returnDouble() {
    return 10;
  }
  
  private int returnInteger() {
    return 0; // values between -128 and 0 will fail
  }
  
  public double getDoubleResult(){
    return m_double_result;
  }
  
  public double getIntegerResult(){
    return m_interger_result;
  }

  public boolean compare(AutoboxingRunOnGpu rhs) {
    try {
      if(getDoubleResult() != rhs.getDoubleResult()){
        System.out.println("m_double_result");
        System.out.println("lhs: "+getDoubleResult());
        System.out.println("rhs: "+rhs.getDoubleResult());
        return false;
      }
      if(getIntegerResult() != rhs.getIntegerResult()){
        System.out.println("m_interger_result");
        System.out.println("lhs: "+getIntegerResult());
        System.out.println("rhs: "+rhs.getIntegerResult());
        return false;
      }
      return true;
    } catch(Exception ex){
      System.out.println("exception thrown");
      return false;
    }
  }
  
}