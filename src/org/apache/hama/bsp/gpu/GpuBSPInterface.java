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
package org.apache.hama.bsp.gpu;

import org.apache.hadoop.io.Writable;
import org.apache.hama.bsp.BSPPeer;

/**
 * The {@link GpuBSPInterface} defines the basic operations needed to implement
 * a GPU BSP based algorithm. The implementing algorithm takes {@link BSPPeer}s
 * as parameters which are responsible for communication, reading K1-V1 inputs,
 * collecting k2-V2 outputs and exchanging messages of type M.
 */
public interface GpuBSPInterface<K1, V1, K2, V2, M extends Writable> {

	/**
	 * This method is your computation method, the main work of your GPU BSP
	 * should be done here.
	 * 
	 * @param peer
	 *            Your BSPPeer instance.
	 */
	public void bspGPU(BSPPeer<K1, V1, K2, V2, M> peer);

	/**
	 * This method is called before the BSP method. It can be used for setup
	 * purposes.
	 * 
	 * @param peer
	 *            Your BSPPeer instance.
	 */
	public void setupGPU(BSPPeer<K1, V1, K2, V2, M> peer);

	/**
	 * This method is called after the BSP method. It can be used for cleanup
	 * purposes. Cleanup is guranteed to be called after the BSP runs, even in
	 * case of exceptions.
	 * 
	 * @param peer
	 *            Your BSPPeer instance.
	 */
	public void cleanupGPU(BSPPeer<K1, V1, K2, V2, M> peer);

	public void setBspKernelCount(long bspKernelCount);
}
