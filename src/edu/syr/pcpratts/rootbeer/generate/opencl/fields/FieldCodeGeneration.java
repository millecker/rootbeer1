/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.generate.opencl.fields;

import edu.syr.pcpratts.rootbeer.entry.ForcedFields;
import edu.syr.pcpratts.rootbeer.generate.bytecode.FieldReadWriteInspector;
import edu.syr.pcpratts.rootbeer.generate.opencl.FieldPackingSorter;
import edu.syr.pcpratts.rootbeer.generate.opencl.OpenCLClass;
import edu.syr.pcpratts.rootbeer.generate.opencl.OpenCLScene;

import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import soot.RefType;
import soot.SootField;
import soot.rbclassload.FieldSignatureUtil;

public class FieldCodeGeneration {
  
  private FieldReadWriteInspector m_Inspector;
  private FieldTypeSwitch m_TypeSwitch;
 
  public String prototypes(Map<String, OpenCLClass> classes, FieldReadWriteInspector inspector) {
    m_Inspector = inspector;
    Set<String> set = new HashSet<String>();
    List<CompositeField> fields = OpenCLScene.v().getCompositeFields();
    for(CompositeField field : fields){
      set.addAll(getFieldPrototypes(field));
    }
    // Force fields to be generated
    FieldSignatureUtil util = new FieldSignatureUtil();
    for(String field_sig : new ForcedFields().get()){
      util.parse(field_sig);
      OpenCLClass field_class = classes.get(util.getDeclaringClass());
      OpenCLField field = field_class.getField(util.getName());
      set.add(field.getGetterSetterPrototypes());
    }
    return setToString(set);
  }
  
  public String bodies(Map<String, OpenCLClass> classes, FieldReadWriteInspector inspector, FieldTypeSwitch type_switch) {
    m_Inspector = inspector;
    m_TypeSwitch = type_switch;
    Set<String> set = new HashSet<String>();
    List<CompositeField> fields = OpenCLScene.v().getCompositeFields();
    for(CompositeField field : fields){
      set.addAll(getFieldBodies(field));
    }
    // Force fields to be generated
    FieldSignatureUtil util = new FieldSignatureUtil();
    for(String field_sig : new ForcedFields().get()){
      util.parse(field_sig);
      OpenCLClass field_class = classes.get(util.getDeclaringClass());
      OpenCLField field = field_class.getField(util.getName());
      CompositeField composite = new CompositeField();
      SootField soot_field = util.getSootField();
      if(soot_field.getType() instanceof RefType){
        composite.addRefField(field, soot_field.getDeclaringClass());
      } else {
        composite.addNonRefField(field, soot_field.getDeclaringClass());
      }
      set.add(field.getGetterSetterBodies(composite, true, m_TypeSwitch));
    }
    return setToString(set);
  }
  
  private Set<String> getFieldBodies(CompositeField composite){
    Set<String> ret = new HashSet<String>();
    FieldPackingSorter sorter = new FieldPackingSorter();
    List<OpenCLField> ref_sorted = sorter.sort(composite.getRefFields());
    List<OpenCLField> nonref_sorted = sorter.sort(composite.getNonRefFields());
    for(OpenCLField field : ref_sorted){
      boolean writable = m_Inspector.fieldIsWrittenOnGpu(field);
      ret.add(field.getGetterSetterBodies(composite, writable, m_TypeSwitch));
    }
    for(OpenCLField field : nonref_sorted){
      boolean writable = m_Inspector.fieldIsWrittenOnGpu(field);
      ret.add(field.getGetterSetterBodies(composite, writable, m_TypeSwitch));
    }
    return ret;
  }

  private Set<String> getFieldPrototypes(CompositeField composite){
    Set<String> ret = new HashSet<String>();
    for(OpenCLField field : composite.getRefFields()){
      ret.add(field.getGetterSetterPrototypes());
    }
    for(OpenCLField field : composite.getNonRefFields()){
      ret.add(field.getGetterSetterPrototypes());
    }
    return ret;
  }
  
  private String setToString(Set<String> set){
    String ret = "";
    Iterator<String> iter = set.iterator();
    while(iter.hasNext()){
      ret += iter.next()+"\n";
    }
    return ret;
  }
}
