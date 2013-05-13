/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.testcases.rootbeertest.serialization.hadoop;

import java.io.IOException;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.gpu.GpuMapper;

public class TestGpuMapper extends GpuMapper<Text, Text, Text, Text> {

  @Override
  public void configure(JobConf arg0){
    // TODO Auto-generated method stub

  }

  @Override
  public void close() throws IOException{
    // TODO Auto-generated method stub

  }

  @Override
  public void mapGpu(Text arg0, Text arg1, OutputCollector<Text, Text> arg2,
      Reporter reporter){
    System.out.println("Rootbeer TestGpuMapper mapGpu!");

  }

}
