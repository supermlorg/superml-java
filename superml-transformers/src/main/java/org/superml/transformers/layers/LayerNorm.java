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

package org.superml.transformers.layers;

import org.superml.core.BaseEstimator;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

/**
 * Layer Normalization for Transformer models.
 * 
 * Normalizes the inputs across the features dimension, helping to stabilize
 * training and improve convergence. Unlike batch normalization, layer normalization
 * normalizes across the feature dimension for each sample independently.
 * 
 * Formula: LayerNorm(x) = γ * (x - μ) / σ + β
 * Where:
 * - μ is the mean across features
 * - σ is the standard deviation across features  
 * - γ (gamma) is a learnable scale parameter
 * - β (beta) is a learnable shift parameter
 * 
 * @author SuperML Team
 * @version 2.1.0
 */
public class LayerNorm extends BaseEstimator {
    
    private int normalizedShape;    // Number of features to normalize over
    private double eps;             // Small constant for numerical stability
    private boolean elementwiseAffine; // Whether to use learnable parameters
    
    // Learnable parameters
    private double[] gamma;         // Scale parameters
    private double[] beta;          // Shift parameters
    
    private boolean isTrained = false;
    
    /**
     * Constructor for LayerNorm.
     * 
     * @param normalizedShape Number of features in the input
     */
    public LayerNorm(int normalizedShape) {
        this(normalizedShape, 1e-5, true);
    }
    
    /**
     * Full constructor for LayerNorm.
     * 
     * @param normalizedShape Number of features in the input
     * @param eps Small constant for numerical stability
     * @param elementwiseAffine Whether to use learnable affine parameters
     */
    public LayerNorm(int normalizedShape, double eps, boolean elementwiseAffine) {
        this.normalizedShape = normalizedShape;
        this.eps = eps;
        this.elementwiseAffine = elementwiseAffine;
        
        initializeParameters();
    }
    
    /**
     * Initialize learnable parameters.
     */
    private void initializeParameters() {
        if (elementwiseAffine) {
            // Initialize gamma to 1.0 (no scaling initially)
            gamma = new double[normalizedShape];
            for (int i = 0; i < normalizedShape; i++) {
                gamma[i] = 1.0;
            }
            
            // Initialize beta to 0.0 (no shift initially)
            beta = new double[normalizedShape];
            // beta is already initialized to zeros by default
        }
        
        this.isTrained = true;
    }
    
    /**
     * Apply layer normalization to input matrix.
     * 
     * @param input Input matrix (batch_size * seq_len, features)
     * @return Normalized output matrix (batch_size * seq_len, features)
     */
    public RealMatrix forward(RealMatrix input) {
        if (!isTrained) {
            throw new IllegalStateException("LayerNorm must be initialized before forward pass");
        }
        
        int numSamples = input.getRowDimension();
        int numFeatures = input.getColumnDimension();
        
        if (numFeatures != normalizedShape) {
            throw new IllegalArgumentException(
                String.format("Input features (%d) must match normalized shape (%d)", 
                            numFeatures, normalizedShape));
        }
        
        double[][] inputData = input.getData();
        double[][] outputData = new double[numSamples][numFeatures];
        
        // Normalize each sample independently
        for (int sample = 0; sample < numSamples; sample++) {
            // Calculate mean across features for this sample
            double mean = 0.0;
            for (int feature = 0; feature < numFeatures; feature++) {
                mean += inputData[sample][feature];
            }
            mean /= numFeatures;
            
            // Calculate variance across features for this sample
            double variance = 0.0;
            for (int feature = 0; feature < numFeatures; feature++) {
                double diff = inputData[sample][feature] - mean;
                variance += diff * diff;
            }
            variance /= numFeatures;
            
            // Normalize and apply affine transformation
            double std = Math.sqrt(variance + eps);
            for (int feature = 0; feature < numFeatures; feature++) {
                double normalized = (inputData[sample][feature] - mean) / std;
                
                if (elementwiseAffine) {
                    outputData[sample][feature] = gamma[feature] * normalized + beta[feature];
                } else {
                    outputData[sample][feature] = normalized;
                }
            }
        }
        
        return new Array2DRowRealMatrix(outputData);
    }
    
    /**
     * Normalize a single vector (useful for inference).
     * 
     * @param vector Input vector to normalize
     * @return Normalized vector
     */
    public double[] normalize(double[] vector) {
        if (vector.length != normalizedShape) {
            throw new IllegalArgumentException(
                String.format("Vector length (%d) must match normalized shape (%d)", 
                            vector.length, normalizedShape));
        }
        
        // Calculate mean
        double mean = 0.0;
        for (double value : vector) {
            mean += value;
        }
        mean /= vector.length;
        
        // Calculate variance
        double variance = 0.0;
        for (double value : vector) {
            double diff = value - mean;
            variance += diff * diff;
        }
        variance /= vector.length;
        
        // Normalize
        double std = Math.sqrt(variance + eps);
        double[] normalized = new double[vector.length];
        
        for (int i = 0; i < vector.length; i++) {
            normalized[i] = (vector[i] - mean) / std;
            
            if (elementwiseAffine) {
                normalized[i] = gamma[i] * normalized[i] + beta[i];
            }
        }
        
        return normalized;
    }
    
    /**
     * Update learnable parameters (for training).
     * 
     * @param newGamma New gamma (scale) parameters
     * @param newBeta New beta (shift) parameters
     */
    public void updateParameters(double[] newGamma, double[] newBeta) {
        if (!elementwiseAffine) {
            throw new IllegalStateException("Cannot update parameters when elementwise affine is disabled");
        }
        
        if (newGamma.length != normalizedShape || newBeta.length != normalizedShape) {
            throw new IllegalArgumentException("Parameter arrays must match normalized shape");
        }
        
        System.arraycopy(newGamma, 0, gamma, 0, normalizedShape);
        System.arraycopy(newBeta, 0, beta, 0, normalizedShape);
    }
    
    // Fluent configuration methods
    
    public LayerNorm setEps(double eps) {
        this.eps = eps;
        return this;
    }
    
    public LayerNorm setElementwiseAffine(boolean elementwiseAffine) {
        this.elementwiseAffine = elementwiseAffine;
        initializeParameters();
        return this;
    }
    
    // Getters
    public int getNormalizedShape() { return normalizedShape; }
    public double getEps() { return eps; }
    public boolean isElementwiseAffine() { return elementwiseAffine; }
    public double[] getGamma() { return gamma != null ? gamma.clone() : null; }
    public double[] getBeta() { return beta != null ? beta.clone() : null; }
    
    @Override
    public String toString() {
        return String.format("LayerNorm(normalized_shape=%d, eps=%.2e, elementwise_affine=%s)",
                normalizedShape, eps, elementwiseAffine);
    }
}
