/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.configuration;

import edu.syr.pcpratts.rootbeer.entry.DontDfsMethods;
import edu.syr.pcpratts.rootbeer.util.ResourceReader;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import soot.rbclassload.ListClassTester;

public class Configuration {

  public static final int MODE_GPU = 0;
  public static final int MODE_NEMU = 1;
  public static final int MODE_JEMU = 2;
  
  private static Configuration m_Instance;
  
  public static Configuration compilerInstance(){
    if(m_Instance == null){
      m_Instance = new Configuration();
    }
    return m_Instance;
  }
  
  public static Configuration runtimeInstance(){
    if(m_Instance == null){
      m_Instance = new Configuration(true);
    } else if(m_Instance.m_compilerInstance){
      m_Instance = new Configuration(true);
    }
    return m_Instance;
  }

  private int m_mode;
  private boolean m_compilerInstance;
  private static boolean m_runAll;
  private static boolean m_printMem;
  private boolean m_remapAll;
  private boolean m_maxRegCountSet;
  private int m_maxRegCount;
  private boolean m_arrayChecks;
  private boolean m_doubles;
  private boolean m_recursion;
  private Set<String> m_loadClasses;
  private ListClassTester m_ignores;
  private DontDfsMethods m_dont_dfs_methods;
  
  static {
    m_printMem = false;
  }
  
  private Configuration(){
    m_compilerInstance = true;
    m_remapAll = true;
    m_maxRegCountSet = false;
    m_arrayChecks = true;
    m_doubles = true;
    m_recursion = true;
    m_loadClasses = new HashSet<String>();
    m_ignores = getIgnorePackages();    
    m_dont_dfs_methods = new DontDfsMethods();
  }

  private Configuration(boolean load) {
    m_compilerInstance = false;
    try {
      List<byte[]> data = ResourceReader.getResourceArray("edu/syr/pcpratts/rootbeer/runtime/config.txt");
      int mode = data.get(0)[0];
      m_mode = mode;
    } catch(Exception ex){
      m_mode = MODE_GPU;
    }
  }
  
  private ListClassTester getIgnorePackages(){
    ListClassTester ignore_packages = new ListClassTester();
    ignore_packages.addPackage("edu.syr.pcpratts.compressor.");
    ignore_packages.addPackage("edu.syr.pcpratts.deadmethods.");
    ignore_packages.addPackage("edu.syr.pcpratts.jpp.");
    ignore_packages.addPackage("edu.syr.pcpratts.rootbeer.compiler.");
    ignore_packages.addPackage("edu.syr.pcpratts.rootbeer.configuration.");
    ignore_packages.addPackage("edu.syr.pcpratts.rootbeer.entry.");
    ignore_packages.addPackage("edu.syr.pcpratts.rootbeer.generate.");
    ignore_packages.addPackage("edu.syr.pcpratts.rootbeer.test.");
    ignore_packages.addPackage("edu.syr.pcpratts.rootbeer.util.");
    ignore_packages.addPackage("pack.");
    ignore_packages.addPackage("jasmin.");
    ignore_packages.addPackage("soot.");
    ignore_packages.addPackage("beaver.");
    ignore_packages.addPackage("polyglot.");
    ignore_packages.addPackage("org.antlr.");
    ignore_packages.addPackage("java_cup.");
    ignore_packages.addPackage("ppg.");
    ignore_packages.addPackage("antlr.");
    ignore_packages.addPackage("jas.");
    ignore_packages.addPackage("scm.");
    ignore_packages.addPackage("org.xmlpull.v1.");
    ignore_packages.addPackage("android.util.");
    ignore_packages.addPackage("android.content.res.");
    ignore_packages.addPackage("org.apache.commons.codec.");
    // Hadoop library
    ignore_packages.addPackage("org.apache.hadoop.mapred.gpu.");
    // Hama library
    ignore_packages.addPackage("org.apache.hama.bsp.gpu.");
    return ignore_packages;
  }
  
  public void setMode(int mode) {
    m_mode = mode;
  }
  
  public int getMode(){
    return m_mode;
  }
  
  public void setRemapSparse(){
    m_remapAll = false;
  }
  
  public boolean getRemapAll(){
    return m_remapAll;
  }
  
  public static void setRunAllTests(boolean run_all){
    m_runAll = run_all;
  }
  
  public static boolean getRunAllTests(){
    return m_runAll;
  }
  
  public static boolean getPrintMem() {
    return m_printMem;
  }
  
  public static void setPrintMem(boolean print){
    m_printMem = print;
  }

  public void setMaxRegCount(int value) {
    m_maxRegCount = value;
    m_maxRegCountSet = true;
  }
  
  public boolean isMaxRegCountSet(){
    return m_maxRegCountSet;
  }
  
  public int getMaxRegCount(){
    return m_maxRegCount;
  }

  public void setArrayChecks(boolean value) {
    m_arrayChecks = value;
  }
  
  public boolean getArrayChecks(){
    return m_arrayChecks;
  }
  
  public void setDoubles(boolean value){
    m_doubles = value;
  }

  public void setRecursion(boolean value){
    m_recursion = value;
  }
  
  public boolean getDoubles() {
    return m_doubles;
  }

  public boolean getRecursion() {
    return m_recursion;
  }

  public Set<String> getLoadClasses(){
    return m_loadClasses;
  }

  public boolean addLoadClasses(String className){
    return m_loadClasses.add(className);
  }
  
  public void addIgnorePackage(String packageName){
    m_ignores.addPackage(packageName);
  }
  
  public boolean removeIgnorePackage(String packageName){
    return m_ignores.removePackage(packageName);
  }
  
  public ListClassTester getIgnoreTester(){
    return m_ignores;
  }
  
  public boolean addDontDfsMethod(String methodSignature){
    return m_dont_dfs_methods.add(methodSignature);
  }
  
  public DontDfsMethods getDontDfsMethods(){
    return m_dont_dfs_methods;
  }
}
