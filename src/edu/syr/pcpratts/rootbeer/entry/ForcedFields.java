/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.entry;

import java.util.ArrayList;
import java.util.List;

public class ForcedFields {
  
  private List<String> m_fields;
  
  public ForcedFields(){
    m_fields = new ArrayList<String>();
    m_fields.add("<edu.syr.pcpratts.rootbeer.runtime.KeyValuePair: java.lang.Object m_key>");
    m_fields.add("<edu.syr.pcpratts.rootbeer.runtime.KeyValuePair: java.lang.Object m_value>");
  }
  
  public List<String> get(){
    return m_fields;
  }
}
