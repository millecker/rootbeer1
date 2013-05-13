/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.testcases.rootbeertest.serialization.hama;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hama.bsp.BSPPeer;
import org.apache.hama.bsp.gpu.GpuBSP;
import org.apache.hama.bsp.sync.SyncException;

public class TestGpuBSP extends GpuBSP<Text, Text, Text, Integer, IntWritable> {

  public TestGpuBSP() {
    super();
  }

  @Override
  public void setupGPU(BSPPeer<Text, Text, Text, Integer, IntWritable> peer)
      throws IOException, SyncException, InterruptedException{
    System.out.println("Rootbeer TestGpuBSP setupGPU!");
  }

  @Override
  public void bspGPU(BSPPeer<Text, Text, Text, Integer, IntWritable> peer)
      throws IOException, SyncException, InterruptedException{
    System.out.println("Rootbeer TestGpuBSP bspGPU!");
  }

  @Override
  public void cleanupGPU(BSPPeer<Text, Text, Text, Integer, IntWritable> peer)
      throws IOException{
    System.out.println("Rootbeer TestGpuBSP cleanupGPU!");
  }
}
