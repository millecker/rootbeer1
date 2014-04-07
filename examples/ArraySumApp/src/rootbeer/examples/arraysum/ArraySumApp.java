
package rootbeer.examples.arraysum;

import java.util.List;
import java.util.ArrayList;
import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.Rootbeer;
import org.trifort.rootbeer.runtime.Context;

public class ArraySumApp {

  public int[] sumArrays(List<int[]> arrays){

     List<Kernel> jobs = new ArrayList<Kernel>();
     int[] ret = new int[arrays.size()];
     for(int i = 0; i < arrays.size(); ++i){
        jobs.add(new ArraySum(arrays.get(i), ret, i));
     }

     Rootbeer rootbeer = new Rootbeer();
     Context context = rootbeer.createDefaultContext();
     context.init(1024*1024*4);
     rootbeer.run(jobs, context);
     return ret;
  }
  
  public static void main(String[] args){
    ArraySumApp app = new ArraySumApp();
    List<int[]> arrays = new ArrayList<int[]>();
    
    //you want 1000s of threads to run on the GPU all at once for speedups
    for(int i = 0; i < 1024; ++i){
      int[] array = new int[512];
      for(int j = 0; j < array.length; ++j){
        array[j] = j;
      }
      arrays.add(array);
    }
    
    int[] sums = app.sumArrays(arrays);
    for(int i = 0; i < sums.length; ++i){
      System.out.println("sum for arrays["+i+"]: "+sums[i]);
    }
  }
}
