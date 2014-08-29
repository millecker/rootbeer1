package org.trifort.rootbeer.testcases.rootbeertest.kerneltemplate;

/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.RootbeerGpu;

public class MatrixMultiplicationRunOnGpu implements Kernel {

  private double[] m_matrixA; // matrix A is transposed
  private double[] m_matrixB;
  public double[] m_matrixC;
  private int m_N;
  private int m_M;
  private int m_L;
  private int m_gridSize;
  private int m_blockSize;
  private int m_tileWidth;
  private int m_subMatricesPerThread;

  public MatrixMultiplicationRunOnGpu(double[] transposedmatrixA,
      double[] matrixB, double[] matrixC, int n, int m, int l, int gridSize,
      int blockSize, int tileWidth, int subMatricesPerThread) {
    m_matrixA = transposedmatrixA; // m x n
    m_matrixB = matrixB; // m x l
    m_matrixC = matrixC; // n x l
    m_N = n;
    m_M = m;
    m_L = l;
    m_gridSize = gridSize;
    m_blockSize = blockSize;
    m_tileWidth = tileWidth; // 32 by default
    m_subMatricesPerThread = subMatricesPerThread;
  }

  // SharedMemory per block
  // blockSize = 1024
  // => 12 (needed by Rootbeer) + (2 * 1024 * 8 (double)) = 16396 bytes
  public void gpuMethod() {
    // get local blockIdx and threadIdx
    int block_idxx = RootbeerGpu.getBlockIdxx();
    int thread_idxx = RootbeerGpu.getThreadIdxx();

    // store fields into local variables
    int M = m_M;
    int L = m_L;
    int tileWidth = m_tileWidth;
    int subMatricesPerThread = m_subMatricesPerThread;
    double[] matrixA = m_matrixA;
    double[] matrixB = m_matrixB;
    double[] matrixC = m_matrixC;

    // Convert block_idxx to a two dimensional index
    int blockRow = block_idxx / (L / tileWidth);
    int blockCol = block_idxx % (L / tileWidth);

    // Convert thread_idxx to a two dimensional index within submatrix
    int threadRow = thread_idxx / tileWidth;
    int threadCol = thread_idxx % tileWidth;

    // Calculate the index of the destination row and col within submatrix
    int destRow = (blockRow * tileWidth) + threadRow;
    int destCol = (blockCol * tileWidth) + threadCol;

    double sum = 0;

    // Loop over all the sub-matrices
    for (int m = 0; m < subMatricesPerThread; m++) {
      int aRowIndex = (m * tileWidth) + threadRow;
      int aColIndex = (blockRow * tileWidth) + threadCol;
      int aValueIndex = (aRowIndex * M) + aColIndex;

      int bRowIndex = (m * tileWidth) + threadRow;
      int bColIndex = destCol;
      int bValueIndex = (bRowIndex * L) + bColIndex;

      double aValue = matrixA[aValueIndex];
      double bValue = matrixB[bValueIndex];

      // store the aValue into shared memory at location
      RootbeerGpu.setSharedDouble(thread_idxx * 8, aValue);
      // store the bValue into shared memory at location
      // 1024 is the offset for the row of matrix A
      RootbeerGpu.setSharedDouble(1024 + (thread_idxx * 8), bValue);

      // sync threads within a block to make sure the sub-matrices are loaded
      RootbeerGpu.syncthreads();

      // loop over all of aValues and bValues
      for (int k = 0; k < tileWidth; k++) {
        // read the aValue from shared memory
        aValue = RootbeerGpu.getSharedDouble((k * tileWidth + threadRow) * 8);
        // read the bValue from shared memory
        bValue = RootbeerGpu
            .getSharedDouble(1024 + (k * tileWidth + threadCol) * 8);

        // multiply aValue and bValue and accumulate
        sum += aValue * bValue;
      }

      // sync threads within a block to make sure that computations have been
      // finished
      RootbeerGpu.syncthreads();
    }

    // update the target cValue with the sum
    int cValueIndex = destRow * L + destCol;
    matrixC[cValueIndex] = sum;
  }

  public boolean compare(MatrixMultiplicationRunOnGpu rhs) {
    for (int i = 0; i < m_N; ++i) {
      for (int j = 0; j < m_L; ++j) {
        if (m_matrixC[i * m_L + j] != rhs.m_matrixC[i * m_L + j]) {
          System.out.println("Verify error at [" + i + "," + j + "]: "
              + m_matrixC[i * m_L + j] + " != " + rhs.m_matrixC[i * m_L + j]);
          return false;
        }
      }
    }
    return true;
  }
}
