/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.superml.linear_model;

import org.superml.core.BaseEstimator;
import org.superml.core.Regressor;
import java.util.Arrays;

/**
 * Linear Regression using Ordinary Least Squares.
 * Similar to sklearn.linear_model.LinearRegression
 */
public class LinearRegression extends BaseEstimator implements Regressor {
    
    private double[] coefficients;
    private double intercept;
    private boolean fitted = false;
    private boolean fitIntercept = true;
    
    public LinearRegression() {
        params.put("fit_intercept", fitIntercept);
    }
    
    public LinearRegression(boolean fitIntercept) {
        this.fitIntercept = fitIntercept;
        params.put("fit_intercept", fitIntercept);
    }
    
    @Override
    public LinearRegression fit(double[][] X, double[] y) {
        int nSamples = X.length;
        int nFeatures = X[0].length;
        
        if (fitIntercept) {
            // Add bias column
            double[][] XWithBias = new double[nSamples][nFeatures + 1];
            for (int i = 0; i < nSamples; i++) {
                XWithBias[i][0] = 1.0;
                System.arraycopy(X[i], 0, XWithBias[i], 1, nFeatures);
            }
            
            // Solve using normal equation: (X^T * X)^-1 * X^T * y
            double[] weights = solveNormalEquation(XWithBias, y);
            intercept = weights[0];
            coefficients = Arrays.copyOfRange(weights, 1, weights.length);
        } else {
            coefficients = solveNormalEquation(X, y);
            intercept = 0.0;
        }
        
        fitted = true;
        return this;
    }
    
    @Override
    public double[] predict(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before making predictions");
        }
        
        double[] predictions = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = intercept;
            for (int j = 0; j < coefficients.length; j++) {
                predictions[i] += coefficients[j] * X[i][j];
            }
        }
        return predictions;
    }
    
    @Override
    public double score(double[][] X, double[] y) {
        double[] predictions = predict(X);
        return r2Score(y, predictions);
    }
    
    private double[] solveNormalEquation(double[][] X, double[] y) {
        // X^T * X
        double[][] XTX = matrixMultiply(transpose(X), X);
        
        // Add small regularization to diagonal to handle numerical instability
        int n = XTX.length;
        for (int i = 0; i < n; i++) {
            XTX[i][i] += 1e-8; // Small regularization term
        }
        
        // (X^T * X)^-1
        double[][] XTXInv = invertMatrix(XTX);
        
        // X^T * y
        double[] XTy = matrixVectorMultiply(transpose(X), y);
        
        // (X^T * X)^-1 * X^T * y
        return matrixVectorMultiply(XTXInv, XTy);
    }
    
    private double[][] transpose(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] transposed = new double[cols][rows];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed[j][i] = matrix[i][j];
            }
        }
        return transposed;
    }
    
    private double[][] matrixMultiply(double[][] A, double[][] B) {
        int rowsA = A.length;
        int colsA = A[0].length;
        int colsB = B[0].length;
        double[][] result = new double[rowsA][colsB];
        
        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                for (int k = 0; k < colsA; k++) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return result;
    }
    
    private double[] matrixVectorMultiply(double[][] matrix, double[] vector) {
        int rows = matrix.length;
        double[] result = new double[rows];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < vector.length; j++) {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        return result;
    }
    
    private double[][] invertMatrix(double[][] matrix) {
        int n = matrix.length;
        double[][] augmented = new double[n][2 * n];
        
        // Create augmented matrix [A|I]
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                augmented[i][j] = matrix[i][j];
                augmented[i][j + n] = (i == j) ? 1.0 : 0.0;
            }
        }
        
        // Gaussian elimination with partial pivoting
        for (int i = 0; i < n; i++) {
            // Find pivot
            int maxRow = i;
            for (int k = i + 1; k < n; k++) {
                if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
                    maxRow = k;
                }
            }
            
            // Swap rows
            double[] temp = augmented[i];
            augmented[i] = augmented[maxRow];
            augmented[maxRow] = temp;
            
            // Make diagonal element 1
            double pivot = augmented[i][i];
            if (Math.abs(pivot) < 1e-10) {
                throw new RuntimeException("Matrix is singular or near-singular, cannot invert");
            }
            for (int j = 0; j < 2 * n; j++) {
                augmented[i][j] /= pivot;
            }
            
            // Eliminate column
            for (int k = 0; k < n; k++) {
                if (k != i) {
                    double factor = augmented[k][i];
                    for (int j = 0; j < 2 * n; j++) {
                        augmented[k][j] -= factor * augmented[i][j];
                    }
                }
            }
        }
        
        // Extract inverse matrix
        double[][] inverse = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                inverse[i][j] = augmented[i][j + n];
            }
        }
        
        return inverse;
    }
    
    private double r2Score(double[] yTrue, double[] yPred) {
        double totalSumSquares = 0.0;
        double residualSumSquares = 0.0;
        double mean = Arrays.stream(yTrue).average().orElse(0.0);
        
        for (int i = 0; i < yTrue.length; i++) {
            totalSumSquares += Math.pow(yTrue[i] - mean, 2);
            residualSumSquares += Math.pow(yTrue[i] - yPred[i], 2);
        }
        
        return 1.0 - (residualSumSquares / totalSumSquares);
    }
    
    // Getters
    public double[] getCoefficients() {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before accessing coefficients");
        }
        return Arrays.copyOf(coefficients, coefficients.length);
    }
    
    public double getIntercept() {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before accessing intercept");
        }
        return intercept;
    }
    
    public boolean isFitIntercept() {
        return fitIntercept;
    }
    
    public LinearRegression setFitIntercept(boolean fitIntercept) {
        this.fitIntercept = fitIntercept;
        params.put("fit_intercept", fitIntercept);
        return this;
    }
}
