/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.entry;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import soot.Scene;
import soot.SootClass;
import soot.SootMethod;
import soot.options.Options;
import soot.rbclassload.ListClassTester;
import soot.rbclassload.ListMethodTester;
import soot.rbclassload.RootbeerClassLoader;
import edu.syr.pcpratts.rootbeer.compiler.Transform2;
import edu.syr.pcpratts.rootbeer.configuration.RootbeerPaths;
import edu.syr.pcpratts.rootbeer.util.CurrJarName;

public class DefaultRootbeerCompiler extends RootbeerCompiler {

  public DefaultRootbeerCompiler() {
    super();
  }

  @Override
  protected void setupSoot(String jar_filename, String rootbeer_jar,
      boolean runtests){

    RootbeerClassLoader.v().setUserJar(jar_filename);
    extractJar(jar_filename);

    List<String> proc_dir = new ArrayList<String>();
    proc_dir.add(RootbeerPaths.v().getJarContentsFolder());

    Options.v().set_allow_phantom_refs(true);
    Options.v().set_rbclassload(true);
    Options.v().set_prepend_classpath(true);
    Options.v().set_process_dir(proc_dir);
    if(m_enableClassRemapping) {
      Options.v().set_rbclassload_buildcg(true);
    }
    if(rootbeer_jar.equals("") == false) {
      Options.v().set_soot_classpath(rootbeer_jar);
    }

    // Options.v().set_rbcl_remap_all(Configuration.compilerInstance().getRemapAll());
    Options.v().set_rbcl_remap_all(false);
    Options.v().set_rbcl_remap_prefix(
        "edu.syr.pcpratts.rootbeer.runtime.remap.");

    RootbeerClassLoader.v().addEntryMethodTester(m_entryDetector);

    ListClassTester ignore_packages = new ListClassTester();
    ignore_packages.addPackage("edu.syr.pcpratts.compressor.");
    ignore_packages.addPackage("edu.syr.pcpratts.deadmethods.");
    ignore_packages.addPackage("edu.syr.pcpratts.jpp.");
    ignore_packages.addPackage("edu.syr.pcpratts.rootbeer.compiler.");
    ignore_packages.addPackage("edu.syr.pcpratts.rootbeer.configuration.");
    ignore_packages.addPackage("edu.syr.pcpratts.rootbeer.entry.");
    ignore_packages.addPackage("edu.syr.pcpratts.rootbeer.generate.");
    ignore_packages.addPackage("edu.syr.pcpratts.rootbeer.test.");
    if(!runtests) {
      ignore_packages.addPackage("edu.syr.pcpratts.rootbeer.testcases.");
    }
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
    RootbeerClassLoader.v().addDontFollowClassTester(ignore_packages);

    ListClassTester keep_packages = new ListClassTester();
    for(String runtime_class : m_runtimePackages) {
      keep_packages.addPackage(runtime_class);
    }
    RootbeerClassLoader.v().addToSignaturesClassTester(keep_packages);

    RootbeerClassLoader.v().addNewInvoke("java.lang.StringBuilder");

    ListMethodTester follow_tester = new ListMethodTester();
    follow_tester.addSignature("<java.lang.String: void <init>(char[])>");
    follow_tester.addSignature("<java.lang.StringBuilder: void <init>()>");
    follow_tester
        .addSignature("<java.lang.StringBuilder: java.lang.StringBuilder append(java.lang.String)>");
    follow_tester
        .addSignature("java.lang.StringBuilder: java.lang.String toString()>");
    follow_tester
        .addSignature("<edu.syr.pcpratts.rootbeer.runtime.Sentinal: void <init>()>");
    follow_tester
        .addSignature("<edu.syr.pcpratts.rootbeer.runtimegpu.GpuException: void <init>()>");
    follow_tester
        .addSignature("<edu.syr.pcpratts.rootbeer.runtimegpu.GpuException: edu.syr.pcpratts.rootbeer.runtimegpu.GpuException arrayOutOfBounds(int,int,int)>");
    follow_tester
        .addSignature("<edu.syr.pcpratts.rootbeer.runtime.Serializer: void <init>(edu.syr.pcpratts.rootbeer.runtime.memory.Memory,edu.syr.pcpratts.rootbeer.runtime.memory.Memory)>");
    follow_tester
        .addSignature("<edu.syr.pcpratts.rootbeer.testcases.rootbeertest.serialization.CovarientTest: void <init>()>");
    RootbeerClassLoader.v().addFollowMethodTester(follow_tester);

    RootbeerClassLoader.v().addFollowClassTester(new TestCaseFollowTester());

    RootbeerClassLoader.v().addConditionalCudaEntry(
        new StringConstantCudaEntry());

    DontDfsMethods dont_dfs_methods = new DontDfsMethods();
    ListMethodTester dont_dfs_tester = new ListMethodTester();
    Set<String> dont_dfs_set = dont_dfs_methods.get();
    for(String dont_dfs : dont_dfs_set) {
      dont_dfs_tester.addSignature(dont_dfs);
    }
    RootbeerClassLoader.v().addDontFollowMethodTester(dont_dfs_tester);

    RootbeerClassLoader.v().loadField(
        "<java.lang.Class: java.lang.String name>");

    RootbeerClassLoader.v().loadNecessaryClasses();

  }

  @Override
  public void compile(String jar_filename, String outname, boolean run_tests)
      throws Exception{

    m_entryDetector = new KernelEntryPointDetector();
    CurrJarName jar_name = new CurrJarName();
    setupSoot(jar_filename, jar_name.get(), run_tests);

    List<SootMethod> kernel_methods = RootbeerClassLoader.v().getEntryPoints();
    compileForKernels(outname, kernel_methods);
  }

  @Override
  protected void compileForKernels(String outname,
      List<SootMethod> kernel_methods) throws Exception{

    if(kernel_methods.isEmpty()) {
      System.out
          .println("There are no kernel classes. Please implement the following interface to use rootbeer:");
      System.out.println("edu.syr.pcpratts.rootbeer.runtime.Kernel");
      System.exit(0);
    }

    Transform2 transform2 = new Transform2();
    for(SootMethod kernel_method : kernel_methods) {
      System.out.println("running transform2 on: "
          + kernel_method.getSignature() + "...");
      RootbeerClassLoader.v().loadDfsInfo(kernel_method);
      SootClass soot_class = kernel_method.getDeclaringClass();
      transform2.run(soot_class.getName(),"void gpuMethod()");
    }

    System.out.println("writing classes out...");

    Iterator<SootClass> iter = Scene.v().getClasses().iterator();
    while(iter.hasNext()) {
      SootClass soot_class = iter.next();
      if(soot_class.isLibraryClass()) {
        continue;
      }
      String class_name = soot_class.getName();
      boolean write = true;
      for(String runtime_class : m_runtimePackages) {
        if(class_name.startsWith(runtime_class)) {
          write = false;
          break;
        }
      }
      Iterator<SootClass> ifaces = soot_class.getInterfaces().iterator();
      while(ifaces.hasNext()) {
        SootClass iface = ifaces.next();
        if(iface.getName().startsWith("edu.syr.pcpratts.rootbeer.test.")) {
          write = false;
        }
      }
      if(write) {
        writeClassFile(class_name);
        writeJimpleFile(class_name);
      }
    }

    makeOutJar();
    pack(outname);
  }

}
