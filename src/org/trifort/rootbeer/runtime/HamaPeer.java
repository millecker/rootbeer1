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
package org.trifort.rootbeer.runtime;

public class HamaPeer {

  private int m_port;
  private boolean m_isDebugging;
  private long m_hostMonitor; // native pointer

  public HamaPeer(int port, boolean isDebugging) {
    this.m_port = port;
    this.m_isDebugging = isDebugging;
    if (m_isDebugging) {
      System.out.println("HamaPeer uses port: " + m_port + " debugging: "
          + m_port);
    }
    m_hostMonitor = connect(m_port, m_isDebugging);
  }

  public HamaPeer(int port) {
    this(port, false);
  }

  /**
   * Init socket connection to Hama Pipes
   * 
   * @param port of socket connection
   * @param debugging write debug outputs
   */
  private native long connect(int port, boolean is_debugging);

  /**
   * Send a data with a tag to another BSPSlave corresponding to hostname.
   * Messages sent by this method are not guaranteed to be received in a sent
   * order.
   * 
   * @param peerName target peerName which will receive the message
   * @param Object message to send
   */
  public static void send(String peerName, Object message) {
  }

  /**
   * @return A int message from the peer's received messages queue (a FIFO).
   */
  public static int getCurrentIntMessage() {
    return 0;
  }

  /**
   * @return A long message from the peer's received messages queue (a FIFO).
   */
  public static long getCurrentLongMessage() {
    return 0;
  }

  /**
   * @return A float message from the peer's received messages queue (a FIFO).
   */
  public static float getCurrentFloatMessage() {
    return 0;
  }

  /**
   * @return A double message from the peer's received messages queue (a FIFO).
   */
  public static double getCurrentDoubleMessage() {
    return 0;
  }

  /**
   * @return A string message from the peer's received messages queue (a FIFO).
   */
  public static String getCurrentStringMessage() {
    return null;
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
   * remote peers. This method blocks.
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
   * Closes the input and opens it right away, so that the file pointer is at
   * the beginning again.
   */
  public static void reopenInput() {
  }

  /**
   * Deserializes the next input key value into the given objects.
   * 
   * @param KeyValuePair is an out parameter that contains key and value
   * @return false if there are no records to read anymore
   */
  public static boolean readNext(KeyValuePair keyValuePair) {
    return false;
  }

  /**
   * Writes a key/value pair to the output collector.
   * 
   * @param key to write
   * @param value to write
   */
  public static void write(Object key, Object value) {
  }

  /**
   * Opens a SequenceFile with option "r" or "w", key/value type and returns the
   * corresponding FileID.
   * 
   * @param path of the SequenceFile
   * @param option "r" for read or "w" for write
   * @param keyType Type of the key
   * @param valueType Type of the value
   * @return FileID of the SequencFile Reader or Writer
   */
  public static int sequenceFileOpen(String path, char option, String keyType,
      String valueType) {
    return 0;
  }

  /**
   * Reads the next key/value pair from the SequenceFile.
   * 
   * @param fileID of the SequenceFile Reader
   * @param keyValuePair is an out parameter that contains key and value
   * @return false if there are no records to read anymore
   */
  public static boolean sequenceFileReadNext(int fileID,
      KeyValuePair keyValuePair) {
    return false;
  }

  /**
   * Appends the next key/value pair to the SequenceFile.
   * 
   * @param fileID of the SequenceFile Writer
   * @param key to write
   * @param value to write
   * @return true if the key/value pair was sucessfully added
   */
  public static boolean sequenceFileAppend(int fileID, Object key, Object value) {
    return false;
  }

  /**
   * Closes a SequenceFile.
   * 
   * @param fileID of the SequenceFile Reader or Writer
   * @return true if the SequenceFile Reader or Writer was sucessfully closed
   */
  public static boolean sequenceFileClose(int fileID) {
    return false;
  }

}
