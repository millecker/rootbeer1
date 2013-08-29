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

  private static HamaPeer m_instance = new HamaPeer();

  private HamaPeer() {
  }

  public static HamaPeer getInstance() {
    return m_instance;
  }

  public void init(int port) {

  }

  /**
   * Send a data with a tag to another BSPSlave corresponding to hostname.
   * Messages sent by this method are not guaranteed to be received in a sent
   * order.
   * 
   * @param peerName
   * @param msg
   */
  public static void send(String peerName, String msg) {

  }

  /**
   * @return A message from the peer's received messages queue (a FIFO).
   */
  public static String getCurrentMessage() {
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
  public static void write(String key, String value) {

  }

  /**
   * Deserializes the next input key value into the given objects.
   * 
   * @param key
   * @param value
   * @return false if there are no records to read anymore
   */
  public static boolean readNext(String key, String value) {
    return false;
  }

  /**
   * Reads the next key value pair and returns it as a pair. It may reuse a
   * {@link KeyValuePair} instance to save garbage collection time.
   * 
   * @return null if there are no records left.
   */
  // public static KeyValuePair<K1, V1> readNext() {
  // return null;
  // }

  /**
   * Closes the input and opens it right away, so that the file pointer is at
   * the beginning again.
   */
  public static void reopenInput() {

  }

  /**
   * @return the jobs configuration
   */
  // public static Configuration getConfiguration();

  /**
   * Get the {@link Counter} of the given group with the given name.
   * 
   * @param name counter name
   * @return the <code>Counter</code> of the given group/name.
   */
  // public Counter getCounter(Enum<?> name);

  /**
   * Get the {@link Counter} of the given group with the given name.
   * 
   * @param group counter group
   * @param name counter name
   * @return the <code>Counter</code> of the given group/name.
   */
  // public Counter getCounter(String group, String name);

  /**
   * Increments the counter identified by the key, which can be of any
   * {@link Enum} type, by the specified amount.
   * 
   * @param key key to identify the counter to be incremented. The key can be be
   *          any <code>Enum</code>.
   * @param amount A non-negative amount by which the counter is to be
   *          incremented.
   */
  // public void incrementCounter(Enum<?> key, long amount);

  /**
   * Increments the counter identified by the group and counter name by the
   * specified amount.
   * 
   * @param group name to identify the group of the counter to be incremented.
   * @param counter name to identify the counter within the group.
   * @param amount A non-negative amount by which the counter is to be
   *          incremented.
   */
  // public void incrementCounter(String group, String counter, long amount);

  /**
   * @return the size of assigned split
   */
  // public long getSplitSize();

  /**
   * @return the current position of the file read pointer
   * @throws IOException
   */
  // public long getPos() throws IOException;

  /**
   * @return the task id of this task.
   */
  // public TaskAttemptID getTaskId();

}
