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
import org.superml.core.Classifier;

import java.util.*;

/**
 * Multinomial Logistic Regression (Softmax Regression) for multiclass classification.
 * 
 * This implementation uses the softmax function to model the probability distribution
 * over multiple classes. It optimizes the cross-entropy loss using gradient descent.
 * 
 * The softmax function ensures that the predicted probabilities sum to 1 across all classes,
 * making it suitable for multiclass classification problems.
 * 
 * Similar to sklearn.linear_model.LogisticRegression with multi_class='multinomial'
 */
public class SoftmaxRegression extends BaseEstimator implements Classifier {
    
    private double[][] weights; // weights[class][feature]
    private double[] classes;
    private boolean fitted = false;
    
    // Hyperparameters
    private double learningRate = 0.01;
    private int maxIter = 1000;
    private double tolerance = 1e-6;
    private double C = 1.0; // Inverse of regularization strength
    
    public SoftmaxRegression() {
        params.put("learning_rate", learningRate);
        params.put("max_iter", maxIter);
        params.put("tol", tolerance);
        params.put("C", C);
    }
    
    public SoftmaxRegression(double learningRate, int maxIter) {
        this();
        this.learningRate = learningRate;
        this.maxIter = maxIter;
        params.put("learning_rate", learningRate);
        params.put("max_iter", maxIter);
    }
    
    @Override
    public SoftmaxRegression fit(double[][] X, double[] y) {
        // Get unique classes
        classes = Arrays.stream(y).distinct().sorted().toArray();
        int nClasses = classes.length;
        int nFeatures = X[0].length;
        
        if (nClasses < 2) {
            throw new IllegalArgumentException("Number of classes must be at least 2");
        }
        
        // For binary classification, fall back to standard logistic regression
        if (nClasses == 2) {
            return fitBinary(X, y);
        }
        
        // Initialize weights: [nClasses x (nFeatures + 1)] for bias
        weights = new double[nClasses][nFeatures + 1];
        for (int i = 0; i < nClasses; i++) {
            // Initialize with small random values
            for (int j = 0; j < nFeatures + 1; j++) {
                weights[i][j] = (Math.random() - 0.5) * 0.01;
            }
        }
        
        // Add bias column to X
        double[][] XWithBias = addBiasColumn(X);
        
        // Create one-hot encoded labels
        double[][] yOneHot = createOneHotLabels(y);
        
        // Gradient descent
        for (int iter = 0; iter < maxIter; iter++) {
            // Forward pass: compute probabilities
            double[][] probabilities = predictProbabilities(XWithBias);
            
            // Compute gradients
            double[][] gradients = computeGradients(XWithBias, yOneHot, probabilities);
            
            // Apply L2 regularization (skip bias terms at index 0)
            for (int i = 0; i < nClasses; i++) {
                for (int j = 1; j < weights[i].length; j++) {
                    gradients[i][j] += (1.0 / C) * weights[i][j];
                }
            }
            
            // Update weights
            double maxGradient = 0.0;
            for (int i = 0; i < nClasses; i++) {
                for (int j = 0; j < weights[i].length; j++) {
                    weights[i][j] -= learningRate * gradients[i][j];
                    maxGradient = Math.max(maxGradient, Math.abs(gradients[i][j]));
                }
            }
            
            // Check convergence
            if (maxGradient < tolerance) {
                break;
            }
        }
        
        fitted = true;
        return this;
    }
    
    /**
     * Fallback to binary classification for 2 classes.
     */
    private SoftmaxRegression fitBinary(double[][] X, double[] y) {
        int nFeatures = X[0].length;
        
        // Initialize weights for binary case: just one weight vector
        weights = new double[1][nFeatures + 1];
        
        // Add bias column to X
        double[][] XWithBias = addBiasColumn(X);
        
        // Convert labels to 0/1
        double[] binaryY = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            binaryY[i] = (y[i] == classes[1]) ? 1.0 : 0.0;
        }
        
        // Standard logistic regression training
        for (int iter = 0; iter < maxIter; iter++) {
            double[] predictions = sigmoid(matrixVectorProduct(XWithBias, weights[0]));
            double[] gradients = computeBinaryGradients(XWithBias, binaryY, predictions);
            
            // Apply L2 regularization (skip bias term at index 0)
            for (int j = 1; j < weights[0].length; j++) {
                gradients[j] += (1.0 / C) * weights[0][j];
            }
            
            // Update weights
            double maxGradient = 0.0;
            for (int j = 0; j < weights[0].length; j++) {
                weights[0][j] -= learningRate * gradients[j];
                maxGradient = Math.max(maxGradient, Math.abs(gradients[j]));
            }
            
            // Check convergence
            if (maxGradient < tolerance) {
                break;
            }
        }
        
        fitted = true;
        return this;
    }
    
    @Override
    public double[] predict(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before making predictions");
        }
        
        double[][] probabilities = predictProba(X);
        double[] predictions = new double[X.length];
        
        for (int i = 0; i < X.length; i++) {
            // Find class with highest probability
            int maxIndex = 0;
            double maxProb = probabilities[i][0];
            
            for (int j = 1; j < probabilities[i].length; j++) {
                if (probabilities[i][j] > maxProb) {
                    maxProb = probabilities[i][j];
                    maxIndex = j;
                }
            }
            
            predictions[i] = classes[maxIndex];
        }
        
        return predictions;
    }
    
    @Override
    public double[][] predictProba(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before making predictions");
        }
        
        double[][] XWithBias = addBiasColumn(X);
        
        if (classes.length == 2) {
            // Binary case
            double[] probPos = sigmoid(matrixVectorProduct(XWithBias, weights[0]));
            double[][] result = new double[X.length][2];
            
            for (int i = 0; i < X.length; i++) {
                result[i][0] = 1 - probPos[i]; // Probability of class 0
                result[i][1] = probPos[i];     // Probability of class 1
            }
            
            return result;
        } else {
            // Multiclass case
            return predictProbabilities(XWithBias);
        }
    }
    
    @Override
    public double[][] predictLogProba(double[][] X) {
        double[][] probabilities = predictProba(X);
        double[][] logProbabilities = new double[probabilities.length][probabilities[0].length];
        
        for (int i = 0; i < probabilities.length; i++) {
            for (int j = 0; j < probabilities[i].length; j++) {
                logProbabilities[i][j] = Math.log(Math.max(probabilities[i][j], 1e-15)); // Avoid log(0)
            }
        }
        
        return logProbabilities;
    }
    
    @Override
    public double[] getClasses() {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before accessing classes");
        }
        return Arrays.copyOf(classes, classes.length);
    }
    
    @Override
    public double score(double[][] X, double[] y) {
        double[] predictions = predict(X);
        int correct = 0;
        
        for (int i = 0; i < y.length; i++) {
            if (predictions[i] == y[i]) {
                correct++;
            }
        }
        
        return (double) correct / y.length;
    }
    
    // Helper methods
    
    private double[][] addBiasColumn(double[][] X) {
        int nSamples = X.length;
        int nFeatures = X[0].length;
        double[][] XWithBias = new double[nSamples][nFeatures + 1];
        
        for (int i = 0; i < nSamples; i++) {
            XWithBias[i][0] = 1.0; // Bias term
            System.arraycopy(X[i], 0, XWithBias[i], 1, nFeatures);
        }
        
        return XWithBias;
    }
    
    private double[][] createOneHotLabels(double[] y) {
        int nSamples = y.length;
        int nClasses = classes.length;
        double[][] oneHot = new double[nSamples][nClasses];
        
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nClasses; j++) {
                oneHot[i][j] = (y[i] == classes[j]) ? 1.0 : 0.0;
            }
        }
        
        return oneHot;
    }
    
    private double[][] predictProbabilities(double[][] XWithBias) {
        int nSamples = XWithBias.length;
        int nClasses = weights.length;
        double[][] logits = new double[nSamples][nClasses];
        
        // Compute logits: X @ W^T
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nClasses; j++) {
                logits[i][j] = dotProduct(XWithBias[i], weights[j]);
            }
        }
        
        // Apply softmax
        return softmax(logits);
    }
    
    private double[][] softmax(double[][] logits) {
        int nSamples = logits.length;
        int nClasses = logits[0].length;
        double[][] probabilities = new double[nSamples][nClasses];
        
        for (int i = 0; i < nSamples; i++) {
            // Find max for numerical stability
            double max = Arrays.stream(logits[i]).max().orElse(0.0);
            
            // Compute exp(logits - max)
            double sum = 0.0;
            for (int j = 0; j < nClasses; j++) {
                probabilities[i][j] = Math.exp(logits[i][j] - max);
                sum += probabilities[i][j];
            }
            
            // Normalize
            for (int j = 0; j < nClasses; j++) {
                probabilities[i][j] /= sum;
            }
        }
        
        return probabilities;
    }
    
    private double[][] computeGradients(double[][] XWithBias, double[][] yOneHot, double[][] probabilities) {
        int nSamples = XWithBias.length;
        int nClasses = weights.length;
        int nFeatures = weights[0].length;
        double[][] gradients = new double[nClasses][nFeatures];
        
        // Compute gradients: X^T @ (probabilities - y_true) / n_samples
        for (int i = 0; i < nClasses; i++) {
            for (int j = 0; j < nFeatures; j++) {
                double gradient = 0.0;
                for (int k = 0; k < nSamples; k++) {
                    gradient += (probabilities[k][i] - yOneHot[k][i]) * XWithBias[k][j];
                }
                gradients[i][j] = gradient / nSamples;
            }
        }
        
        return gradients;
    }
    
    private double[] computeBinaryGradients(double[][] XWithBias, double[] y, double[] predictions) {
        int nSamples = XWithBias.length;
        int nFeatures = XWithBias[0].length;
        double[] gradients = new double[nFeatures];
        
        for (int j = 0; j < nFeatures; j++) {
            double gradient = 0.0;
            for (int i = 0; i < nSamples; i++) {
                gradient += (predictions[i] - y[i]) * XWithBias[i][j];
            }
            gradients[j] = gradient / nSamples;
        }
        
        return gradients;
    }
    
    private double dotProduct(double[] x, double[] w) {
        double result = 0.0;
        for (int i = 0; i < x.length; i++) {
            result += x[i] * w[i];
        }
        return result;
    }
    
    private double[] matrixVectorProduct(double[][] X, double[] w) {
        double[] result = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            result[i] = dotProduct(X[i], w);
        }
        return result;
    }
    
    private double[] sigmoid(double[] z) {
        return Arrays.stream(z).map(v -> 1.0 / (1.0 + Math.exp(-v))).toArray();
    }
    
    // Getters and setters for hyperparameters
    public double getLearningRate() { return learningRate; }
    public SoftmaxRegression setLearningRate(double learningRate) {
        this.learningRate = learningRate;
        params.put("learning_rate", learningRate);
        return this;
    }
    
    public int getMaxIter() { return maxIter; }
    public SoftmaxRegression setMaxIter(int maxIter) {
        this.maxIter = maxIter;
        params.put("max_iter", maxIter);
        return this;
    }
    
    public double getTolerance() { return tolerance; }
    public SoftmaxRegression setTolerance(double tolerance) {
        this.tolerance = tolerance;
        params.put("tol", tolerance);
        return this;
    }
    
    public double getC() { return C; }
    public SoftmaxRegression setC(double C) {
        this.C = C;
        params.put("C", C);
        return this;
    }
    
    /**
     * Get the weight matrix.
     * @return weight matrix [nClasses x nFeatures+1]
     */
    public double[][] getWeights() {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before accessing weights");
        }
        return Arrays.stream(weights).map(double[]::clone).toArray(double[][]::new);
    }
}
