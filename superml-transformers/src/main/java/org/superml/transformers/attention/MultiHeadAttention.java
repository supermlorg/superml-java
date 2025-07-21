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

package org.superml.transformers.attention;

import org.superml.core.BaseEstimator;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.Well19937c;
import org.superml.core.UnsupervisedLearner;

/**
 * Multi-Head Attention mechanism for Transformer models.
 * 
 * This implementation follows the "Attention Is All You Need" paper
 * and provides the core attention mechanism used in transformer architectures.
 * 
 * Features:
 * - Configurable number of attention heads
 * - Scaled dot-product attention
 * - Linear transformations for Q, K, V projections
 * - Optional attention masking
 * - Integration with SuperML BaseEstimator pattern
 * 
 * @author SuperML Team
 * @version 2.1.0
 */
public class MultiHeadAttention extends BaseEstimator implements UnsupervisedLearner {

    // Architecture parameters
    private int modelDim;           // Model dimension (d_model)
    private int numHeads;           // Number of attention heads
    private int headDim;            // Dimension per head (d_model / num_heads)
    private boolean useBias;        // Whether to use bias in projections
    private double dropout;         // Dropout rate
    
    // Weight matrices for Q, K, V projections and output
    private RealMatrix WQ, WK, WV, WO;
    private double[] biasQ, biasK, biasV, biasO;
    
    // Random generator for initialization
    private RandomGenerator random;
    
    // Training state
    private boolean isTrained = false;
    
    /**
     * Constructor for MultiHeadAttention.
     * 
     * @param modelDim The model dimension (must be divisible by numHeads)
     * @param numHeads The number of attention heads
     */
    public MultiHeadAttention(int modelDim, int numHeads) {
        if (modelDim % numHeads != 0) {
            throw new IllegalArgumentException("Model dimension must be divisible by number of heads");
        }
        
        this.modelDim = modelDim;
        this.numHeads = numHeads;
        this.headDim = modelDim / numHeads;
        this.useBias = true;
        this.dropout = 0.1;
        this.random = new Well19937c();
        
        initializeWeights();
    }
    
    /**
     * Initialize weight matrices using Xavier/Glorot initialization.
     */
    private void initializeWeights() {
        double scale = Math.sqrt(1.0 / modelDim);
        
        // Initialize Q, K, V, and output projection matrices
        WQ = createRandomMatrix(modelDim, modelDim, scale);
        WK = createRandomMatrix(modelDim, modelDim, scale);
        WV = createRandomMatrix(modelDim, modelDim, scale);
        WO = createRandomMatrix(modelDim, modelDim, scale);
        
        // Initialize bias vectors if enabled
        if (useBias) {
            biasQ = new double[modelDim];
            biasK = new double[modelDim];
            biasV = new double[modelDim];
            biasO = new double[modelDim];
            // Bias vectors are initialized to zero (default)
        }
        
        this.isTrained = true;
    }
    
    /**
     * Create a random matrix with Xavier initialization.
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
     * Forward pass of multi-head attention.
     * 
     * @param input Input matrix (batch_size * seq_len, model_dim)
     * @param mask Optional attention mask (batch_size * seq_len, seq_len)
     * @return Output matrix (batch_size * seq_len, model_dim)
     */
    public RealMatrix forward(RealMatrix input, RealMatrix mask) {
        if (!isTrained) {
            throw new IllegalStateException("MultiHeadAttention must be initialized before forward pass");
        }
        
        int seqLen = input.getRowDimension();
        
        // Linear projections: Q = input * WQ, K = input * WK, V = input * WV
        RealMatrix Q = input.multiply(WQ);
        RealMatrix K = input.multiply(WK);
        RealMatrix V = input.multiply(WV);
        
        // Add bias if enabled
        if (useBias) {
            Q = addBias(Q, biasQ);
            K = addBias(K, biasK);
            V = addBias(V, biasV);
        }
        
        // Reshape for multi-head attention: (seq_len, model_dim) -> (seq_len, num_heads, head_dim)
        // For simplicity, we'll process each head separately
        RealMatrix[] attentionOutputs = new RealMatrix[numHeads];
        
        for (int head = 0; head < numHeads; head++) {
            // Extract head-specific Q, K, V
            RealMatrix Qh = extractHead(Q, head);
            RealMatrix Kh = extractHead(K, head);
            RealMatrix Vh = extractHead(V, head);
            
            // Compute scaled dot-product attention for this head
            attentionOutputs[head] = scaledDotProductAttention(Qh, Kh, Vh, mask);
        }
        
        // Concatenate all heads
        RealMatrix concatenated = concatenateHeads(attentionOutputs);
        
        // Final linear projection
        RealMatrix output = concatenated.multiply(WO);
        if (useBias) {
            output = addBias(output, biasO);
        }
        
        return output;
    }
    
    /**
     * Extract queries, keys, or values for a specific attention head.
     */
    private RealMatrix extractHead(RealMatrix matrix, int headIndex) {
        int seqLen = matrix.getRowDimension();
        double[][] headData = new double[seqLen][headDim];
        
        int startCol = headIndex * headDim;
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < headDim; j++) {
                headData[i][j] = matrix.getEntry(i, startCol + j);
            }
        }
        
        return new Array2DRowRealMatrix(headData);
    }
    
    /**
     * Concatenate outputs from all attention heads.
     */
    private RealMatrix concatenateHeads(RealMatrix[] heads) {
        int seqLen = heads[0].getRowDimension();
        double[][] concatData = new double[seqLen][modelDim];
        
        for (int head = 0; head < numHeads; head++) {
            int startCol = head * headDim;
            for (int i = 0; i < seqLen; i++) {
                for (int j = 0; j < headDim; j++) {
                    concatData[i][startCol + j] = heads[head].getEntry(i, j);
                }
            }
        }
        
        return new Array2DRowRealMatrix(concatData);
    }
    
    /**
     * Scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V
     */
    private RealMatrix scaledDotProductAttention(RealMatrix Q, RealMatrix K, RealMatrix V, RealMatrix mask) {
        // Compute attention scores: QK^T
        RealMatrix scores = Q.multiply(K.transpose());
        
        // Scale by sqrt(head_dim)
        scores = scores.scalarMultiply(1.0 / Math.sqrt(headDim));
        
        // Apply mask (set masked positions to large negative value)
        if (mask != null) {
            scores = applyMask(scores, mask);
        }
        
        // Apply softmax to get attention weights
        RealMatrix attentionWeights = softmax(scores);
        
        // Apply attention weights to values: attention_weights * V
        return attentionWeights.multiply(V);
    }
    
    /**
     * Apply attention mask by setting masked positions to large negative values.
     */
    private RealMatrix applyMask(RealMatrix scores, RealMatrix mask) {
        double[][] maskedScores = scores.getData();
        double[][] maskData = mask.getData();
        
        for (int i = 0; i < maskedScores.length; i++) {
            for (int j = 0; j < maskedScores[i].length; j++) {
                if (maskData[i][j] == 0) {  // 0 means masked position
                    maskedScores[i][j] = -1e9;
                }
            }
        }
        
        return new Array2DRowRealMatrix(maskedScores);
    }
    
    /**
     * Apply softmax activation function row-wise.
     */
    private RealMatrix softmax(RealMatrix matrix) {
        double[][] data = matrix.getData();
        double[][] softmaxData = new double[data.length][data[0].length];
        
        for (int i = 0; i < data.length; i++) {
            // Find max for numerical stability
            double maxVal = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < data[i].length; j++) {
                maxVal = Math.max(maxVal, data[i][j]);
            }
            
            // Compute exponentials and sum
            double sum = 0.0;
            for (int j = 0; j < data[i].length; j++) {
                softmaxData[i][j] = Math.exp(data[i][j] - maxVal);
                sum += softmaxData[i][j];
            }
            
            // Normalize
            for (int j = 0; j < data[i].length; j++) {
                softmaxData[i][j] /= sum;
            }
        }
        
        return new Array2DRowRealMatrix(softmaxData);
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
    
    public MultiHeadAttention setUseBias(boolean useBias) {
        this.useBias = useBias;
        initializeWeights();  // Re-initialize with new bias setting
        return this;
    }
    
    public MultiHeadAttention setDropout(double dropout) {
        this.dropout = dropout;
        return this;
    }
    
    public MultiHeadAttention setNumHeads(int numHeads) {
        if (modelDim % numHeads != 0) {
            throw new IllegalArgumentException("Model dimension must be divisible by number of heads");
        }
        this.numHeads = numHeads;
        this.headDim = modelDim / numHeads;
        initializeWeights();  // Re-initialize with new head configuration
        return this;
    }
    
    // Getters
    public int getModelDim() { return modelDim; }
    public int getNumHeads() { return numHeads; }
    public int getHeadDim() { return headDim; }
    public boolean getUseBias() { return useBias; }
    public double getDropout() { return dropout; }
    
    @Override
    public String toString() {
        return String.format("MultiHeadAttention(model_dim=%d, num_heads=%d, head_dim=%d, use_bias=%s, dropout=%.2f)",
                modelDim, numHeads, headDim, useBias, dropout);
    }

    @Override
    public double[][] transform(double[][] X) {
        // Convert to RealMatrix and perform self-attention
        RealMatrix input = new Array2DRowRealMatrix(X);
        RealMatrix output = forward(input, null);
        return output.getData();
    }

    @Override
    public MultiHeadAttention fit(double[][] X) {
        // Initialize weights if not already done
        if (!isTrained) {
            initializeWeights();
        }
        return this;
    }
}
