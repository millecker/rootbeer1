/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.testcases.rootbeertest.kerneltemplate;

import java.util.Random;

import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.ThreadConfig;
import org.trifort.rootbeer.test.TestKernelTemplate;

public class MatrixMultiplicationTest implements TestKernelTemplate {

  private int m_N; // rows of matrix A
  private int m_M; // cols of matrix A and rows of matrix B
  private int m_L; // cols of matrix B
  private double[] m_matrixA; // N x M
  private double[] m_transposedMatrixAgpu; // M x N
  private double[] m_matrixB; // M x L
  private double[] m_matrixC; // N x L

  private int m_tileWidth;
  private int m_blockSize;
  private int m_gridSize;
  private int m_subMatricesPerThread;

  public MatrixMultiplicationTest() {
    m_N = 1024;
    m_M = 1024;
    m_L = 1024;
    m_tileWidth = 32;

    int subMatrixSize = m_tileWidth * m_tileWidth;
    m_blockSize = subMatrixSize;

    int numberOfSubMatrices = divup(m_N * m_L, subMatrixSize);
    m_gridSize = numberOfSubMatrices;

    m_subMatricesPerThread = divup(m_M, m_tileWidth);

    m_matrixA = createRandomMatrix(m_N, m_M, new Random(42L));
    m_transposedMatrixAgpu = transposeMatrix(m_matrixA, m_N, m_M);
    m_matrixB = createRandomMatrix(m_M, m_L, new Random(1337L));
    m_matrixC = new double[m_N * m_L];
  }

  private int divup(int x, int y) {
    if (x % y != 0) {
      return ((x + y - 1) / y); // round up
    } else {
      return x / y;
    }
  }

  private double[] createRandomMatrix(int n, int m, Random rand) {
    final double matrix[] = new double[n * m];
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        // matrix[i * m + j] = rand.nextDouble();
        matrix[i * m + j] = rand.nextInt(9) + 1; // between 1 and 10
      }
    }
    return matrix;
  }

  private double[] transposeMatrix(double[] matrix, int n, int m) {
    final double transposedMatrix[] = new double[m * n];
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        transposedMatrix[i * n + j] = matrix[j * m + i]; // M[i][j] = M[j][i]
      }
    }
    return transposedMatrix;
  }

  public Kernel create() {
    Kernel ret = new MatrixMultiplicationRunOnGpu(m_transposedMatrixAgpu,
        m_matrixB, m_matrixC, m_N, m_M, m_L, m_gridSize, m_blockSize,
        m_tileWidth, m_subMatricesPerThread);
    return ret;
  }

  public ThreadConfig getThreadConfig() {
    ThreadConfig ret = new ThreadConfig(m_blockSize, m_gridSize,
        (long) m_blockSize * (long) m_gridSize);
    return ret;
  }

  public boolean compare(Kernel original, Kernel from_heap) {
    MatrixMultiplicationRunOnGpu lhs = (MatrixMultiplicationRunOnGpu) original;
    MatrixMultiplicationRunOnGpu rhs = (MatrixMultiplicationRunOnGpu) from_heap;
    return lhs.compare(rhs);
  }

}
