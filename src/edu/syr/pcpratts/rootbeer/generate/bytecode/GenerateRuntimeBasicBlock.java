/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.generate.bytecode;

import edu.syr.pcpratts.deadmethods2.DeadMethods;
import edu.syr.pcpratts.rootbeer.configuration.Configuration;
import edu.syr.pcpratts.rootbeer.configuration.RootbeerPaths;
import edu.syr.pcpratts.rootbeer.generate.codesegment.CodeSegment;
import edu.syr.pcpratts.rootbeer.generate.codesegment.LoopCodeSegment;
import edu.syr.pcpratts.rootbeer.generate.codesegment.MethodCodeSegment;
import edu.syr.pcpratts.rootbeer.generate.opencl.OpenCLScene;
import edu.syr.pcpratts.rootbeer.generate.misc.BasicBlock;
import edu.syr.pcpratts.rootbeer.generate.opencl.tweaks.CompileResult;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import soot.*;
import soot.jimple.IntConstant;
import soot.jimple.Jimple;
import soot.jimple.JimpleBody;
import soot.jimple.StringConstant;
import soot.options.Options;
import soot.rbclassload.RootbeerClassLoader;

public class GenerateRuntimeBasicBlock {
  private CodeSegment codeSegment;
  private SootClass mSootClass;
  private List<Local> mFirstIterationLocals;
  private Jimple jimple;
  private String runtimeBasicBlockClassName;
  private String gcObjectVisitorClassName;

  public GenerateRuntimeBasicBlock(BasicBlock block, String uuid) {
    codeSegment = new LoopCodeSegment(block);
    jimple = Jimple.v();
    mFirstIterationLocals = new ArrayList<Local>();
  }

  public GenerateRuntimeBasicBlock(SootMethod method, String uuid){
    jimple = Jimple.v();
    mFirstIterationLocals = new ArrayList<Local>();
    mSootClass = method.getDeclaringClass();
    codeSegment = new MethodCodeSegment(method);
  }

  public Type getType(){
    return mSootClass.getType();
  }

  public void makeClass() throws Exception {
    gcObjectVisitorClassName = codeSegment.getSootClass().getName()+"GcObjectVisitor";
    
    makeCpuBody();
    makeGpuBody();
    makeIsUsingGarbageCollectorBody();
    makeIsReadOnly();    
    makeExceptionNumbers();
                            
    GcHeapReadWriteAdder adder = new GcHeapReadWriteAdder();
    adder.add(codeSegment);
  }

  private void makeCpuBody() {
    codeSegment.makeCpuBodyForRuntimeBasicBlock(mSootClass);
  }  
  
  private void makeGetCodeMethodThatReturnsBytes(boolean m32, String filename) {
    BytecodeLanguage bcl = new BytecodeLanguage();
    bcl.openClass(mSootClass);
    SootClass string = Scene.v().getSootClass("java.lang.String");
    bcl.startMethod("getCubin"  + (m32 ? "32" : "64"), string.getType());
    Local thisref = bcl.refThis();
    bcl.returnValue(StringConstant.v(filename));
    bcl.endMethod();
  }
  
  private void makeGetCodeMethodThatReturnsString(String gpu_code, boolean unix){    
    //make the getCode method with the results of the opencl code generation
    String name = "getCode";
    if(unix){
      name += "Unix";
    } else {
      name += "Windows";
    }
    SootMethod getCode = new SootMethod(name, new ArrayList(), RefType.v("java.lang.String"), Modifier.PUBLIC);
    getCode.setDeclaringClass(mSootClass);
    mSootClass.addMethod(getCode);
    
    RootbeerClassLoader.v().addGeneratedMethod(getCode.getSignature());

    JimpleBody body = jimple.newBody(getCode);
    UnitAssembler assembler = new UnitAssembler();

    //create an instance of self
    Local thislocal = jimple.newLocal("this0", mSootClass.getType());
    Unit thisid = jimple.newIdentityStmt(thislocal, jimple.newThisRef(mSootClass.getType()));
    assembler.add(thisid);

    //java string constants encoded in a class file have a maximum size of 65535...
    //$r1 = new java.lang.StringBuilder;
    SootClass string_builder_soot_class = Scene.v().getSootClass("java.lang.StringBuilder");
    Local r1 = jimple.newLocal("r1", string_builder_soot_class.getType());
    Value r1_assign_rhs = jimple.newNewExpr(string_builder_soot_class.getType());
    Unit r1_assign = jimple.newAssignStmt(r1, r1_assign_rhs);
    assembler.add(r1_assign);

    //specialinvoke $r1.<java.lang.StringBuilder: void <init>()>();
    SootMethod string_builder_ctor = string_builder_soot_class.getMethod("void <init>()");
    Value r1_ctor = jimple.newSpecialInvokeExpr(r1, string_builder_ctor.makeRef(), new ArrayList());
    Unit r1_ctor_unit = jimple.newInvokeStmt(r1_ctor);
    assembler.add(r1_ctor_unit);
    
    //r2 = $r1;
    Local r2 = jimple.newLocal("r2", string_builder_soot_class.getType());
    Unit r2_assign_r1 = jimple.newAssignStmt(r2, r1);
    assembler.add(r2_assign_r1);
    
    SootClass string_class = Scene.v().getSootClass("java.lang.String");
    SootMethod string_builder_append = string_builder_soot_class.getMethod("java.lang.StringBuilder append(java.lang.String)");

    GpuCodeSplitter splitter = new GpuCodeSplitter();
    List<String> blocks = splitter.split(gpu_code);

    for(String block : blocks){
      Value curr_string_constant = StringConstant.v(block);
        
      //virtualinvoke r2.<java.lang.StringBuilder: java.lang.StringBuilder append(java.lang.String)>("gpu code");
      List args = new ArrayList();
      args.add(curr_string_constant);
      Value invoke_expr = jimple.newVirtualInvokeExpr(r2, string_builder_append.makeRef(), args);
      Unit invoke_stmt = jimple.newInvokeStmt(invoke_expr);
      assembler.add(invoke_stmt);
    }

    //$r5 = virtualinvoke r2.<java.lang.StringBuilder: java.lang.String toString()>();
    Local r5 = jimple.newLocal("r5", string_class.getType());
    SootMethod to_string = string_builder_soot_class.getMethod("java.lang.String toString()");
    Value r5_rhs = jimple.newVirtualInvokeExpr(r2, to_string.makeRef());
    Unit r5_assign = jimple.newAssignStmt(r5, r5_rhs);
    assembler.add(r5_assign);

    assembler.add(jimple.newReturnStmt(r5));

    assembler.assemble(body);
    getCode.setActiveBody(body);
  }

  private void makeGpuBody() throws Exception {
    OpenCLScene.v().addCodeSegment(codeSegment);
    if(Configuration.compilerInstance().getMode() == Configuration.MODE_GPU){
      CompileResult[] result = OpenCLScene.v().getCudaCode();
      for (CompileResult res : result) {
        String suffix = res.is32Bit() ? "-32" : "-64";
        if (res.getBinary() == null) {
          makeGetCodeMethodThatReturnsBytes(res.is32Bit(), cubinFilename(false, suffix) + ".error");
        } else {
          List<byte[]> bytes = res.getBinary();
          writeBytesToFile(bytes, cubinFilename(true, suffix));
          makeGetCodeMethodThatReturnsBytes(res.is32Bit(), cubinFilename(false, suffix));
        }
      }
      makeGetCodeMethodThatReturnsString("", true);
      makeGetCodeMethodThatReturnsString("", false);
    } else {
      String[] code = OpenCLScene.v().getOpenCLCode();
      //code[0] is unix
      //code[1] is windows
      
      PrintWriter writer = new PrintWriter(RootbeerPaths.v().getRootbeerHome()+"pre_dead_unix.c");
      writer.println(code[0]);
      writer.flush();
      writer.close();
      
      writer = new PrintWriter(RootbeerPaths.v().getRootbeerHome()+"pre_dead_windows.c");
      writer.println(code[1]);
      writer.flush();
      writer.close();
      
      System.out.println("removing dead methods...");
      DeadMethods dead_methods = new DeadMethods();
      dead_methods.parseString(code[0]);
      code[0] = dead_methods.getResult();
      dead_methods.parseString(code[1]);
      code[1] = dead_methods.getResult();
      
      //jpp can't handle declspec very well
      code[1] = code[1].replace("void entry(char * gc_info_space,", "__declspec(dllexport)\nvoid entry(char * gc_info_space,");
      
      makeGetCodeMethodThatReturnsString(code[0], true);
      makeGetCodeMethodThatReturnsString(code[1], false);
      makeGetCodeMethodThatReturnsBytes(true, "");
      makeGetCodeMethodThatReturnsBytes(false, "");
    }
  }
  
  private String cubinFilename(boolean use_class_folder, String suffix){
    String class_name = File.separator +
            gcObjectVisitorClassName.replace(".", File.separator) +
            suffix + ".cubin";
    if(use_class_folder)
      return RootbeerPaths.v().getOutputClassFolder() + class_name;
    else
      return class_name;
  }
  
  private void writeBytesToFile(List<byte[]> bytes, String filename) {
    try {
      File file = new File(filename);
      File parent = file.getParentFile();
      parent.mkdirs();
      OutputStream os = new FileOutputStream(filename);
      for(byte[] buffer : bytes){
        os.write(buffer);
      }
      os.flush();
      os.close();
    } catch(Exception ex){
      ex.printStackTrace();
    }
  }
  
  public SootField getField(String name, Type type){
    return mSootClass.getField(name, type);
  }

  public void addFirstIterationLocal(Local local) {
    mFirstIterationLocals.add(local);
  }

  private void makeIsUsingGarbageCollectorBody() {
    BytecodeLanguage bcl = new BytecodeLanguage();
    bcl.openClass(mSootClass);
    bcl.startMethod("isUsingGarbageCollector", BooleanType.v());
    bcl.refThis();
    if(OpenCLScene.v().getUsingGarbageCollector())
      bcl.returnValue(IntConstant.v(1));
    else
      bcl.returnValue(IntConstant.v(0));
    bcl.endMethod();
  }

  public String getRuntimeBasicBlockName() {
    return runtimeBasicBlockClassName;
  }

  public String getGcObjectVisitorName() {
    return gcObjectVisitorClassName;
  }

  private void makeIsReadOnly() {
    BytecodeLanguage bcl = new BytecodeLanguage();
    bcl.openClass(mSootClass);
    bcl.startMethod("isReadOnly", BooleanType.v());
    bcl.refThis();
    if(OpenCLScene.v().getReadOnlyTypes().isRootReadOnly())
      bcl.returnValue(IntConstant.v(1));
    else
      bcl.returnValue(IntConstant.v(0));
    bcl.endMethod();
  }

  private void makeExceptionNumbers() {
    String prefix = Options.v().rbcl_remap_prefix();
    if(Options.v().rbcl_remap_all() == false){
      prefix = "";
    }
    makeExceptionMethod("getNullPointerNumber", prefix+"java.lang.NullPointerException");
    makeExceptionMethod("getOutOfMemoryNumber", prefix+"java.lang.OutOfMemoryError");
  }
  
  private void makeExceptionMethod(String method_name, String cls_name) {
    SootClass soot_class = Scene.v().getSootClass(cls_name);
    int number = RootbeerClassLoader.v().getClassNumber(soot_class);
    
    BytecodeLanguage bcl = new BytecodeLanguage();
    bcl.openClass(mSootClass);
    bcl.startMethod(method_name, IntType.v());
    bcl.refThis();
    bcl.returnValue(IntConstant.v(number));
    bcl.endMethod();
  }

}
