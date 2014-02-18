/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.testcases.rootbeertest.serialization;

import org.trifort.rootbeer.runtime.HamaPeer;
import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.KeyValuePair;

public class HamaPeerRunOnGpu implements Kernel {

  private String[] m_temp;
  
  public HamaPeerRunOnGpu(){
  }
  
  public void gpuMethod(){
    // fix error: identifier "java_lang_String__array_new" is undefined
    m_temp = new String[2];

    
    HamaPeer.send("", "");
    
    HamaPeer.getCurrentIntMessage();
    HamaPeer.getCurrentLongMessage();
    HamaPeer.getCurrentFloatMessage();
    HamaPeer.getCurrentDoubleMessage();
    HamaPeer.getCurrentStringMessage();

    HamaPeer.getNumCurrentMessages();

    HamaPeer.sync();

    HamaPeer.getSuperstepCount();

    HamaPeer.getPeerName();
    HamaPeer.getPeerName(0);

    HamaPeer.getPeerIndex();

    HamaPeer.getAllPeerNames();

    HamaPeer.getNumPeers();

    HamaPeer.clear();

    HamaPeer.reopenInput();
    
    HamaPeer.readNext(new KeyValuePair(0, 0));

    HamaPeer.write("", "");

    HamaPeer.sequenceFileOpen("", 'w', "", "");

    HamaPeer.sequenceFileReadNext(0, new KeyValuePair(0, 0));

    HamaPeer.sequenceFileAppend(0, "", "");

    HamaPeer.sequenceFileClose(0);
  }

  public boolean compare(HamaPeerRunOnGpu rhs) {
    return true;
  }
}
