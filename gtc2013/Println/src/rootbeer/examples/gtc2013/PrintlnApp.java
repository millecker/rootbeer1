/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package rootbeer.examples.gtc2013;

import java.util.List;
import java.util.ArrayList;
import edu.syr.pcpratts.rootbeer.runtime.Rootbeer;
import edu.syr.pcpratts.rootbeer.runtime.Kernel;

public class PrintlnApp {

  public void run(){
    List<Kernel> kernels = new ArrayList<Kernel>();
    for(int i = 0; i < 5; ++i){
      kernels.add(new PrintlnKernel());
    }
    Rootbeer rootbeer = new Rootbeer();
    rootbeer.runAll(kernels);
  }

  public static void main(String[] args){
    PrintlnApp app = new PrintlnApp();
    app.run();
  }
}
