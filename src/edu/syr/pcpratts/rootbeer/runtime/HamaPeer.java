/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package edu.syr.pcpratts.rootbeer.runtime;


public class HamaPeer {

  /**
   * Send a data with a tag to another BSPSlave corresponding to hostname.
   * Messages sent by this method are not guaranteed to be received in a sent
   * order.
   * 
   * @param peerName
   * @param Object message
   */
  public static void send(String peerName, Object message) {
  }

  /**
   * Returns a message from the peer's received messages queue (a FIFO).
   */
  public static void getCurrentMessage(Object message) {
  }

  /**
   * @return The number of messages in the peer's received messages queue.
   */
  public static int getNumCurrentMessages() {
    return 0;
  }
  
  /**
   * Barrier Synchronization.
   * 
   * Sends all the messages in the outgoing message queues to the corresponding
   * remote peers.
   */
  public static void sync() {
  }

  /**
   * @return the count of current super-step
   */
  public static long getSuperstepCount() {
    return 0;
  }

  /**
   * @return the name of this peer in the format "hostname:port".
   */
  public static String getPeerName() {
    return null;
  }

  /**
   * @return the name of n-th peer from sorted array by name.
   */
  public static String getPeerName(int index) {
    return null;
  }

  /**
   * @return the index of this peer from sorted array by name.
   */
  public static int getPeerIndex() {
    return 0;
  }

  /**
   * @return the names of all the peers executing tasks from the same job
   *         (including this peer).
   */
  public static String[] getAllPeerNames() {
    return null;
  }

  /**
   * @return the number of peers
   */
  public static int getNumPeers() {
    return 0;
  }

  /**
   * Clears all queues entries.
   */
  public static void clear() {
  }

  /**
   * Writes a key/value pair to the output collector.
   * 
   * @param key your key object
   * @param value your value object
   */
  public static void write(Object key, Object value) {
  }

  /**
   * Deserializes the next input key value into the given objects.
   * 
   * @param key
   * @param value
   * @return false if there are no records to read anymore
   */
  public static boolean readNext(Object key, Object value) {
    return false;
  }
  
  /**
   * Closes the input and opens it right away, so that the file pointer is at
   * the beginning again.
   */
  public static void reopenInput() {
  }

}
