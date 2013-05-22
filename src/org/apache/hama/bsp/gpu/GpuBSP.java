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

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.io.Writable;
import org.apache.hama.bsp.BSP;
import org.apache.hama.bsp.BSPPeer;
import org.apache.hama.bsp.sync.SyncException;

import edu.syr.pcpratts.rootbeer.runtime.Kernel;
import edu.syr.pcpratts.rootbeer.runtime.Rootbeer;
import edu.syr.pcpratts.rootbeer.runtime.StatsRow;

public abstract class GpuBSP<K1, V1, K2, V2, M extends Writable> extends
		BSP<K1, V1, K2, V2, M> implements GpuBSPInterface<K1, V1, K2, V2, M> {

	private Rootbeer rootbeer = new Rootbeer();

	private long setupKernelCount = 1;
	private long bspKernelCount = 1;
	private long cleanupKernelCount = 1;
	
	@Override
	public void setBspKernelCount(long bspKernelCount) {
		this.bspKernelCount = bspKernelCount;
	}

	@Override
	public final void setup(BSPPeer<K1, V1, K2, V2, M> peer)
			throws IOException, SyncException, InterruptedException {

		// TODO
		// CHECK which GPU is available?
		// OR execute on CPU?
        System.out.println("GpuBSP setup started...");
        
		List<Kernel> jobs = new ArrayList<Kernel>();
		for (int i = 0; i < setupKernelCount; i++)
			jobs.add(new SetupKernel(peer));
		rootbeer.runAll(jobs);
		
		/*
		for (StatsRow stats : rootbeer.getStats()){
			System.out.println("Rootbeer SetupKernel - InitTime: "+stats.getInitTime());
			System.out.println("Rootbeer SetupKernel - NumBlocks: "+stats.getNumBlocks());
			System.out.println("Rootbeer SetupKernel - NumThreads: "+stats.getNumThreads());
			System.out.println("Rootbeer SetupKernel - SerializationTime: "+stats.getSerializationTime());
			System.out.println("Rootbeer SetupKernel - DeserializationTime: "+stats.getDeserializationTime());
			System.out.println("Rootbeer SetupKernel - ExecutionTime: "+stats.getExecutionTime());
		}
		*/
		
		System.out.println("GpuBSP setup ended...");
	}

	@Override
	public final void bsp(BSPPeer<K1, V1, K2, V2, M> peer) throws IOException,
			SyncException, InterruptedException {

		System.out.println("GpuBSP bsp started...");
        
		List<Kernel> jobs = new ArrayList<Kernel>();

		for (int i = 0; i < bspKernelCount; i++)
			jobs.add(new BspKernel(peer));

		rootbeer.runAll(jobs);
		
		/*
		for (StatsRow stats : rootbeer.getStats()){
			System.out.println("Rootbeer BspKernel - InitTime: "+stats.getInitTime());
			System.out.println("Rootbeer BspKernel - NumBlocks: "+stats.getNumBlocks());
			System.out.println("Rootbeer BspKernel - NumThreads: "+stats.getNumThreads());
			System.out.println("Rootbeer BspKernel - SerializationTime: "+stats.getSerializationTime());
			System.out.println("Rootbeer BspKernel - DeserializationTime: "+stats.getDeserializationTime());
			System.out.println("Rootbeer BspKernel - ExecutionTime: "+stats.getExecutionTime());
		}
		*/
		System.out.println("GpuBSP bsp ended...");
	}

	@Override
	public final void cleanup(BSPPeer<K1, V1, K2, V2, M> peer)
			throws IOException {

		System.out.println("GpuBSP cleanup started...");
        
		List<Kernel> jobs = new ArrayList<Kernel>();
		for (int i = 0; i < cleanupKernelCount; i++)
			jobs.add(new CleanupKernel(peer));
		rootbeer.runAll(jobs);
		
		/*
		for (StatsRow stats : rootbeer.getStats()){
			System.out.println("Rootbeer CleanupKernel - InitTime: "+stats.getInitTime());
			System.out.println("Rootbeer CleanupKernel - NumBlocks: "+stats.getNumBlocks());
			System.out.println("Rootbeer CleanupKernel - NumThreads: "+stats.getNumThreads());
			System.out.println("Rootbeer CleanupKernel - SerializationTime: "+stats.getSerializationTime());
			System.out.println("Rootbeer CleanupKernel - DeserializationTime: "+stats.getDeserializationTime());
			System.out.println("Rootbeer CleanupKernel - ExecutionTime: "+stats.getExecutionTime());
		}
		*/
		
		System.out.println("GpuBSP cleanup ended...");
	}

	@Override
	public void setupGPU(BSPPeer<K1, V1, K2, V2, M> peer) {
		// possible empty implementation
		//System.out.println("GpuBSP setupGPU empty implementation...");
	}

	@Override
	public abstract void bspGPU(BSPPeer<K1, V1, K2, V2, M> peer);

	// Use rootbeer object for
	// RootbeerGpu.getThreadIdxx() and RootbeerGpu.getBlockIdxx()
	// in Kernel Template

	@Override
	public void cleanupGPU(BSPPeer<K1, V1, K2, V2, M> peer) {
		// possible empty implementation
		//System.out.println("GpuBSP setupGPU empty implementation...");
	}

	public class SetupKernel implements Kernel {
		private BSPPeer<K1, V1, K2, V2, M> peer;

		public SetupKernel(BSPPeer<K1, V1, K2, V2, M> peer) {
			this.peer = peer;
		}

		@Override
		public void gpuMethod() {
			setupGPU(peer);
		}
	}

	public class BspKernel implements Kernel {
		private BSPPeer<K1, V1, K2, V2, M> peer;

		public BspKernel(BSPPeer<K1, V1, K2, V2, M> peer) {
			this.peer = peer;
		}

		@Override
		public void gpuMethod() {
			bspGPU(peer);
		}
	}

	public class CleanupKernel implements Kernel {
		private BSPPeer<K1, V1, K2, V2, M> peer;

		public CleanupKernel(BSPPeer<K1, V1, K2, V2, M> peer) {
			this.peer = peer;
		}

		@Override
		public void gpuMethod() {
			cleanupGPU(peer);
		}
	}
}
