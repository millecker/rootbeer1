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
    m_N = 4; // 1024;
    m_M = 4; // 1024;
    m_L = 4; // 1024;
    m_tileWidth = 4; // 32;

    int subMatrixSize = m_tileWidth * m_tileWidth;
    m_blockSize = subMatrixSize;

    int numberOfSubMatrices = divup(m_N * m_L, subMatrixSize);
    m_gridSize = numberOfSubMatrices;

    m_subMatricesPerThread = divup(m_M, m_tileWidth);

    m_matrixA = createRandomMatrix(m_N, m_M, new Random(42L));
    m_transposedMatrixAgpu = transposeMatrix(m_matrixA, m_N, m_M);
    m_matrixB = createRandomMatrix(m_M, m_L, new Random(1337L));
    m_matrixC = new double[m_N * m_L];

    System.out.println("tileWidth: " + m_tileWidth);
    System.out.println("gridSize: " + m_gridSize);
    System.out.println("blockSize: " + m_blockSize);
    System.out.println("n: " + m_N);
    System.out.println("m: " + m_M);
    System.out.println("l: " + m_L);
    System.out.println("subMatricesPerThread: " + m_subMatricesPerThread);
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

  private double[] multiply(double[] matrixA, double[] matrixB, int n, int m,
      int l) {
    final double matrix[] = new double[n * l];
    for (int i = 0; i < n; i++) { // for each row of A
      for (int j = 0; j < l; j++) { // for each col of B
        int sum = 0;
        for (int k = 0; k < m; k++) { // for each col of A and row of B
          sum += (matrixA[i * m + k] * matrixB[k * l + j]); // A[i][k] * B[k][j]
        }
        matrix[i * l + j] = sum; // C[i][j] += A[i][k] * B[k][j]
      }
    }
    return matrix;
  }

  private boolean verify(double[] matrixA, double[] matrixB, int n, int l) {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < l; ++j) {
        if (matrixA[i * l + j] != matrixB[i * l + j]) {
          System.out.println("Verify error at [" + i + "," + j + "]: "
              + matrixA[i * l + j] + " != " + matrixB[i * l + j]);
          return false;
        }
      }
    }
    return true;
  }

  private void printMatrix(double[] matrix, int n, int m) {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        if (j == m - 1) {
          System.out.println(matrix[i * m + j]);
        } else {
          System.out.print(matrix[i * m + j] + ",");
        }
      }
    }
    System.out.println();
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
    if (lhs.compare(rhs) == false) {
      return false;
    }

    // compute matrix multiplication
    double[] matrixC = multiply(m_matrixA, m_matrixB, m_N, m_M, m_L);

    // debug output
    if ((m_N < 11) && (m_L < 11)) {
      System.out.println("MatrixA:");
      printMatrix(m_matrixA, m_N, m_M);
      System.out.println("TransposedMatrixA:");
      printMatrix(m_transposedMatrixAgpu, m_M, m_N);
      System.out.println("MatrixB:");
      printMatrix(m_matrixB, m_M, m_L);
      System.out.println("Reference result:");
      printMatrix(matrixC, m_N, m_L);
      System.out.println("CPU result:");
      printMatrix(rhs.m_matrixC, m_N, m_L);
      System.out.println("GPU result:");
      printMatrix(lhs.m_matrixC, m_N, m_L);
    }

    // verify results
    System.out.println("Verify CPU result...");
    if (verify(rhs.m_matrixC, matrixC, m_N, m_L) == false) {
      return false;
    }
    System.out.println("Verify GPU result...");
    if (verify(lhs.m_matrixC, matrixC, m_N, m_L) == false) {
      return false;
    }
    return true;
  }
}
