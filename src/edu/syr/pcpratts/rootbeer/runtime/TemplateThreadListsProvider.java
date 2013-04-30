/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */
package edu.syr.pcpratts.rootbeer.runtime;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class TemplateThreadListsProvider<T> {

  private List<TemplateThread<T>> sleeping = Collections.synchronizedList(new ArrayList<TemplateThread<T>>());
  private List<TemplateThread<T>> computing = Collections.synchronizedList(new ArrayList<TemplateThread<T>>());

  public TemplateThreadListsProvider() {
    for (int i = 0; i < Runtime.getRuntime().availableProcessors(); ++i) {
      TemplateThread<T> t = new TemplateThread<T>(this);
      t.start();
      sleeping.add(t);
    }
  }
  
  public List<TemplateThread<T>> getSleeping() {
    return sleeping;
  }

  public List<TemplateThread<T>> getComputing() {
    return computing;
  }
}
