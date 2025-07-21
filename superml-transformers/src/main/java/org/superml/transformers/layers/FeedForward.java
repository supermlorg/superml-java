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
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.Well19937c;

/**
 * Feed Forward Network for Transformer models.
 * 
 * A two-layer fully connected network with ReLU activation in between.
 * This is applied to each position separately and identically.
 * 
 * Architecture:
 * input -> Linear(model_dim, ff_dim) -> ReLU -> Linear(ff_dim, model_dim) -> output
 * 
 * Typically ff_dim = 4 * model_dim as suggested in the original Transformer paper.
 * 
 * @author SuperML Team
 * @version 2.1.0
 */
public class FeedForward extends BaseEstimator {
    
    private int modelDim;           // Input/output dimension
    private int ffDim;              // Hidden dimension (usually 4 * model_dim)
    private String activation;      // Activation function ("relu", "gelu")
    private double dropout;         // Dropout rate
    private boolean useBias;        // Whether to use bias in linear layers
    
    // Weight matrices and biases
    private RealMatrix W1, W2;      // Linear transformation weights
    private double[] bias1, bias2;   // Bias vectors
    
    // Random generator for initialization
    private RandomGenerator random;
    private boolean isTrained = false;
    
    /**
     * Constructor with default settings.
     * 
     * @param modelDim Input/output dimension
     * @param ffDim Hidden dimension
     */
    public FeedForward(int modelDim, int ffDim) {
        this(modelDim, ffDim, "relu", 0.1, true);
    }
    
    /**
     * Full constructor for FeedForward.
     * 
     * @param modelDim Input/output dimension
     * @param ffDim Hidden dimension
     * @param activation Activation function ("relu" or "gelu")
     * @param dropout Dropout rate
     * @param useBias Whether to use bias terms
     */
    public FeedForward(int modelDim, int ffDim, String activation, double dropout, boolean useBias) {
        this.modelDim = modelDim;
        this.ffDim = ffDim;
        this.activation = activation.toLowerCase();
        this.dropout = dropout;
        this.useBias = useBias;
        this.random = new Well19937c();
        
        if (!this.activation.equals("relu") && !this.activation.equals("gelu")) {
            throw new IllegalArgumentException("Activation must be 'relu' or 'gelu'");
        }
        
        initializeWeights();
    }
    
    /**
     * Initialize weight matrices and biases using Xavier initialization.
     */
    private void initializeWeights() {
        // Xavier initialization scales
        double scale1 = Math.sqrt(1.0 / modelDim);  // For first layer
        double scale2 = Math.sqrt(1.0 / ffDim);     // For second layer
        
        // Initialize weight matrices
        W1 = createRandomMatrix(modelDim, ffDim, scale1);
        W2 = createRandomMatrix(ffDim, modelDim, scale2);
        
        // Initialize biases if enabled
        if (useBias) {
            bias1 = new double[ffDim];
            bias2 = new double[modelDim];
            // Biases are initialized to zero (default)
        }
        
        this.isTrained = true;
    }
    
    /**
     * Create a random matrix with specified initialization scale.
     */
    private RealMatrix createRandomMatrix(int rows, int cols, double scale) {
        double[][] data = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = random.nextGaussian() * scale;
            }
        }
        return new Array2DRowRealMatrix(data);
    }
    
    /**
     * Forward pass through the feed forward network.
     * 
     * @param input Input matrix (batch_size * seq_len, model_dim)
     * @return Output matrix (batch_size * seq_len, model_dim)
     */
    public RealMatrix forward(RealMatrix input) {
        if (!isTrained) {
            throw new IllegalStateException("FeedForward must be initialized before forward pass");
        }
        
        if (input.getColumnDimension() != modelDim) {
            throw new IllegalArgumentException(
                String.format("Input dimension (%d) must match model dimension (%d)", 
                            input.getColumnDimension(), modelDim));
        }
        
        // First linear transformation: input * W1 + bias1
        RealMatrix hidden = input.multiply(W1);
        if (useBias) {
            hidden = addBias(hidden, bias1);
        }
        
        // Apply activation function
        hidden = applyActivation(hidden);
        
        // Apply dropout (simplified - in practice this would be conditional on training mode)
        // For now, we skip dropout to keep the implementation simple
        
        // Second linear transformation: hidden * W2 + bias2
        RealMatrix output = hidden.multiply(W2);
        if (useBias) {
            output = addBias(output, bias2);
        }
        
        return output;
    }
    
    /**
     * Apply activation function to the matrix.
     */
    private RealMatrix applyActivation(RealMatrix matrix) {
        double[][] data = matrix.getData();
        double[][] activatedData = new double[data.length][data[0].length];
        
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                switch (activation) {
                    case "relu":
                        activatedData[i][j] = relu(data[i][j]);
                        break;
                    case "gelu":
                        activatedData[i][j] = gelu(data[i][j]);
                        break;
                    default:
                        throw new IllegalStateException("Unknown activation: " + activation);
                }
            }
        }
        
        return new Array2DRowRealMatrix(activatedData);
    }
    
    /**
     * ReLU activation function.
     */
    private double relu(double x) {
        return Math.max(0.0, x);
    }
    
    /**
     * GELU activation function (Gaussian Error Linear Unit).
     * GELU(x) = x * Φ(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
     */
    private double gelu(double x) {
        double coefficient = Math.sqrt(2.0 / Math.PI);
        double inner = coefficient * (x + 0.044715 * x * x * x);
        return 0.5 * x * (1.0 + Math.tanh(inner));
    }
    
    /**
     * Add bias vector to each row of the matrix.
     */
    private RealMatrix addBias(RealMatrix matrix, double[] bias) {
        double[][] data = matrix.getData();
        double[][] biasedData = new double[data.length][data[0].length];
        
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                biasedData[i][j] = data[i][j] + bias[j];
            }
        }
        
        return new Array2DRowRealMatrix(biasedData);
    }
    
    // Fluent configuration methods
    
    public FeedForward setActivation(String activation) {
        activation = activation.toLowerCase();
        if (!activation.equals("relu") && !activation.equals("gelu")) {
            throw new IllegalArgumentException("Activation must be 'relu' or 'gelu'");
        }
        this.activation = activation;
        return this;
    }
    
    public FeedForward setDropout(double dropout) {
        this.dropout = dropout;
        return this;
    }
    
    public FeedForward setUseBias(boolean useBias) {
        this.useBias = useBias;
        initializeWeights();  // Re-initialize with new bias setting
        return this;
    }
    
    // Getters
    public int getModelDim() { return modelDim; }
    public int getFfDim() { return ffDim; }
    public String getActivation() { return activation; }
    public double getDropout() { return dropout; }
    public boolean getUseBias() { return useBias; }
    public RealMatrix getW1() { return W1.copy(); }
    public RealMatrix getW2() { return W2.copy(); }
    
    @Override
    public String toString() {
        return String.format("FeedForward(model_dim=%d, ff_dim=%d, activation=%s, dropout=%.2f, use_bias=%s)",
                modelDim, ffDim, activation, dropout, useBias);
    }
}
