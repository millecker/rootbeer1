/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package rootbeer.examples.gtc2013;

import edu.syr.pcpratts.rootbeer.runtime.Kernel;
import edu.syr.pcpratts.rootbeer.runtime.RootbeerGpu;

public class PrintlnKernel implements Kernel {

  @Override
  public void gpuMethod(){
    RootbeerGpu.println("hello world");
  }

}
