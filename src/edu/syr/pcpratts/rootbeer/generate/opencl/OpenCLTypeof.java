/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.generate.opencl;

import soot.SootClass;
import soot.jimple.InstanceOfExpr;
import edu.syr.pcpratts.rootbeer.generate.opencl.tweaks.Tweaks;

public class OpenCLTypeof {
    
  private SootClass m_class;
  
  public OpenCLTypeof(SootClass soot_class) {
    m_class = soot_class;
  }

  public String getPrototype() {
    return getDecl()+";\n";
  }
  
  private String getDecl(){
    String device = Tweaks.v().getDeviceFunctionQualifier();
    String global = Tweaks.v().getGlobalAddressSpaceQualifier();
    
    String ret = device+" bool "+getMethodName();
    ret += "("+global+" char * gc_info, int thisref)";
    return ret;
  }
  
  private String getMethodName(){
    return "edu_syr_pcpratts_rootbeer_typeof_"+m_class.getShortName();
  }

  public String getBody() {
    String ret = getDecl();
    ret += "{\n";
    ret += "  char * thisref_deref;\n";
    ret += "  GC_OBJ_TYPE_TYPE type;\n";
    ret += "  if(thisref == -1){\n";
    ret += "    return false;\n";
    ret += "  }\n";
    ret += "  thisref_deref = edu_syr_pcpratts_gc_deref(gc_info, thisref);\n";
    ret += "  type = edu_syr_pcpratts_gc_get_type(thisref_deref);\n";
    ret += "  if(type=="+OpenCLScene.v().getClassType(m_class)+"){\n";
    ret += "      return true;\n";
    ret += "  }\n";
    ret += "  return false;\n";
    ret += "}\n";
    return ret;
  }
  
  public String invokeExpr(InstanceOfExpr arg0){
    String ret = getMethodName();
    ret += "(gc_info, "+arg0.getOp().toString()+")";
    return ret;
  }
  
  @Override
  public boolean equals(Object other){
    if(other == null){
      return false;
    }
    if(other instanceof OpenCLTypeof){
      OpenCLTypeof rhs = (OpenCLTypeof) other;
      return m_class.equals(rhs.m_class);
    } else {
      return false;
    }
  }

  @Override
  public int hashCode() {
    int hash = 5;
    hash = 29 * hash + (this.m_class != null ? this.m_class.hashCode() : 0);
    return hash;
  }
}