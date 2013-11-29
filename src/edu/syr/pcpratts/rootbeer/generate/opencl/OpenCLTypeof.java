/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.generate.opencl;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import soot.rbclassload.NumberedType;
import edu.syr.pcpratts.rootbeer.generate.opencl.tweaks.Tweaks;

public class OpenCLTypeof {
  
  private Set<NumberedType> m_numberedTypes;
  
  public OpenCLTypeof() {
    m_numberedTypes = new HashSet<NumberedType>();
  }
  
  public void addNumberedType(List<NumberedType> numberedTypes) {
    for (NumberedType type : numberedTypes) {
      if(m_numberedTypes.contains(type) == false){
        m_numberedTypes.add(type);
        // System.out.println("type: "+type.getType()+" number: "+type.getNumber());
      }
    }
  }

  public String getPrototype() {
    return getDecl()+";\n";
  }
  
  private String getDecl(){
    String device = Tweaks.v().getDeviceFunctionQualifier();
    String global = Tweaks.v().getGlobalAddressSpaceQualifier();
    
    String ret = device+" bool "+getMethodName();
    ret += "("+global+" char * gc_info, int thisref,  char * type_name)";
    return ret;
  }
  
  private String getMethodName(){
    return "edu_syr_pcpratts_rootbeer_typeof";
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
    int i=0;
    for(NumberedType ntype : m_numberedTypes){
      if (i==0) {
        ret += "  if((edu_syr_pcpratts_cmpstr(\""+ntype.getType()+"\",type_name)) && ("+ntype.getNumber()+"==type)==0) {\n";
      } else {
        ret += "  else if((edu_syr_pcpratts_cmpstr(\""+ntype.getType()+"\",type_name)) && ("+ntype.getNumber()+"==type)==0) {\n";
      }
      ret += "    return true;\n";
      ret += "  }\n";
      i++;
    }
    ret += "  return false;\n";
    ret += "}\n";
    return ret;
  }
}