/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.generate.opencl.tweaks;

import edu.syr.pcpratts.rootbeer.util.WindowsCompile;
import edu.syr.pcpratts.compressor.Compressor;
import edu.syr.pcpratts.deadmethods2.DeadMethods;
import edu.syr.pcpratts.rootbeer.configuration.Configuration;
import edu.syr.pcpratts.rootbeer.configuration.RootbeerPaths;
import edu.syr.pcpratts.rootbeer.generate.opencl.tweaks.GencodeOptions.CompileArchitecture;
import edu.syr.pcpratts.rootbeer.util.CompilerRunner;
import edu.syr.pcpratts.rootbeer.util.CudaPath;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class CudaTweaks extends Tweaks {

  @Override
  public String getGlobalAddressSpaceQualifier() {
    return "";
  }

  @Override
  public String getUnixHeaderPath() {
    return "/edu/syr/pcpratts/rootbeer/generate/opencl/CudaHeader.c";
  }
  
  @Override
  public String getWindowsHeaderPath() {
    return "/edu/syr/pcpratts/rootbeer/generate/opencl/CudaHeader.c";
  }
  
  @Override
  public String getBothHeaderPath() {
    return null;
  }
  
  @Override
  public String getBarrierPath() {
    return null;
  }
  
  @Override
  public String getGarbageCollectorPath() {
    return "/edu/syr/pcpratts/rootbeer/generate/opencl/GarbageCollector.c";
  }

  @Override
  public String getUnixKernelPath() {
    return "/edu/syr/pcpratts/rootbeer/generate/opencl/CudaKernel.c";
  }
  
  @Override
  public String getWindowsKernelPath() {
    return "/edu/syr/pcpratts/rootbeer/generate/opencl/CudaKernel.c";
  }

  @Override
  public String getBothKernelPath() {
    return null;
  }
      
  /**
   * Compiles CUDA code.
   *
   * @param cuda_code string containing code.
   * @param compileArch determine if we need to build 32bit, 64bit or both.
   * @return an array containing compilation results. You can use <tt>is32Bit()</tt> on each element 
   * to determine if it is 32 bit or 64bit code. If compilation for an architecture fails, only the 
   * offending element is returned.
   */
  public CompileResult[] compileProgram(String cuda_code, CompileArchitecture compileArch) {
    PrintWriter writer;
    try {
      writer = new PrintWriter(RootbeerPaths.v().getRootbeerHome() + "pre_dead.cu");
      writer.println(cuda_code);
      writer.flush();
      writer.close();

      // Ignore custom code, because template kernel functions are not supported
      // Check for custom Hama Pipes code and remove before tweaks (DeadMethod check)
      
      // Check custom Header Code in CudaHeader.c
      String start_str = "/*HAMA_PIPES_HEADER_CODE_IGNORE_IN_TWEAKS_START*/";
      String end_str = "/*HAMA_PIPES_HEADER_CODE_IGNORE_IN_TWEAKS_END*/";
      String template_header_str = "/*HAMA_PIPES_HEADER_CODE*/";
      int start_pos = cuda_code.indexOf(start_str);
      int end_pos = cuda_code.indexOf(end_str);
      String hama_custom_header_code = "";
      if ( (start_pos>0) && (end_pos>0) ) {
        hama_custom_header_code = cuda_code.substring(start_pos, 
            end_pos + end_str.length());
        
        cuda_code = cuda_code.substring(0, start_pos) 
            + template_header_str + "\n"
            + cuda_code.substring(end_pos + end_str.length());
      }
      
      // Do tweaks
      DeadMethods dead_methods = new DeadMethods();
      dead_methods.parseString(cuda_code);
      cuda_code = dead_methods.getResult();

      //Compressor compressor = new Compressor();
      //cuda_code = compressor.compress(cuda_code);

      // Add custom Hama Pipes code after tweaks (DeadMethod check)
      if (cuda_code.indexOf(template_header_str)>0) {
        cuda_code = cuda_code.replace(template_header_str, hama_custom_header_code);
      }
      
      File generated = new File(RootbeerPaths.v().getRootbeerHome() + "generated.cu");
      writer = new PrintWriter(generated);
      writer.println(cuda_code.toString());
      writer.flush();
      writer.close();

      CudaPath cuda_path = new CudaPath();
      GencodeOptions options_gen = new GencodeOptions();
      String gencode_options = options_gen.getOptions();
      
      ParallelCompile parallel_compile = new ParallelCompile();
      return parallel_compile.compile(generated, cuda_path, gencode_options, compileArch);
    } catch (Exception ex) {
      throw new RuntimeException("Failed to compile cuda code.", ex);
    }
  }

  @Override
  public String getDeviceFunctionQualifier() {
    return "__device__";
  }

}
