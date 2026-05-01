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
import org.superml.transformers.attention.MultiHeadAttention;
import org.apache.commons.math3.linear.RealMatrix;

/**
 * Transformer Block (Encoder Layer) combining multi-head attention and feed forward network.
 * 
 * Architecture:
 * 1. Multi-Head Self-Attention with residual connection and layer norm
 * 2. Feed Forward Network with residual connection and layer norm
 * 
 * This follows the "Post-LN" variant where layer normalization is applied after the residual connection:
 * output1 = LayerNorm(input + MultiHeadAttention(input))
 * output2 = LayerNorm(output1 + FeedForward(output1))
 * 
 * @author SuperML Team
 * @version 2.1.0
 */
public class TransformerBlock extends BaseEstimator {
    
    private int modelDim;                    // Model dimension
    private int numHeads;                    // Number of attention heads
    private int ffDim;                       // Feed forward hidden dimension
    
    // Components
    private MultiHeadAttention attention;    // Multi-head attention layer
    private FeedForward feedForward;         // Feed forward network
    private LayerNorm layerNorm1;           // Layer norm after attention
    private LayerNorm layerNorm2;           // Layer norm after feed forward
    
    private boolean isTrained = false;
    
    /**
     * Constructor for TransformerBlock.
     * 
     * @param modelDim Model dimension
     * @param numHeads Number of attention heads
     * @param ffDim Feed forward hidden dimension (typically 4 * modelDim)
     */
    public TransformerBlock(int modelDim, int numHeads, int ffDim) {
        this.modelDim = modelDim;
        this.numHeads = numHeads;
        this.ffDim = ffDim;
        
        initializeLayers();
    }
    
    /**
     * Constructor with default feed forward dimension.
     * 
     * @param modelDim Model dimension
     * @param numHeads Number of attention heads
     */
    public TransformerBlock(int modelDim, int numHeads) {
        this(modelDim, numHeads, 4 * modelDim);  // Default ff_dim = 4 * model_dim
    }
    
    /**
     * Initialize all sub-layers.
     */
    private void initializeLayers() {
        // Initialize multi-head attention
        attention = new MultiHeadAttention(modelDim, numHeads);
        
        // Initialize feed forward network
        feedForward = new FeedForward(modelDim, ffDim);
        
        // Initialize layer normalization layers
        layerNorm1 = new LayerNorm(modelDim);
        layerNorm2 = new LayerNorm(modelDim);
        
        this.isTrained = true;
    }
    
    /**
     * Forward pass through the transformer block.
     * 
     * @param input Input matrix (batch_size * seq_len, model_dim)
     * @param mask Optional attention mask
     * @return Output matrix (batch_size * seq_len, model_dim)
     */
    public RealMatrix forward(RealMatrix input, RealMatrix mask) {
        if (!isTrained) {
            throw new IllegalStateException("TransformerBlock must be initialized before forward pass");
        }
        
        // 1. Multi-Head Self-Attention with residual connection and layer norm
        RealMatrix attentionOutput = attention.forward(input, mask);
        RealMatrix residual1 = input.add(attentionOutput);  // Residual connection
        RealMatrix normed1 = layerNorm1.forward(residual1); // Layer normalization
        
        // 2. Feed Forward Network with residual connection and layer norm
        RealMatrix ffOutput = feedForward.forward(normed1);
        RealMatrix residual2 = normed1.add(ffOutput);       // Residual connection
        RealMatrix normed2 = layerNorm2.forward(residual2); // Layer normalization
        
        return normed2;
    }
    
    /**
     * Forward pass without attention mask.
     * 
     * @param input Input matrix (batch_size * seq_len, model_dim)
     * @return Output matrix (batch_size * seq_len, model_dim)
     */
    public RealMatrix forward(RealMatrix input) {
        return forward(input, null);
    }
    
    /**
     * Get the attention weights from the last forward pass (for visualization).
     * Note: This is a simplified version - in a full implementation, we would
     * need to modify MultiHeadAttention to store and return attention weights.
     * 
     * @return Attention weights (if available)
     */
    public RealMatrix getAttentionWeights() {
        // TODO: Implement attention weight extraction from MultiHeadAttention
        // For now, return null as a placeholder
        return null;
    }
    
    // Fluent configuration methods
    
    public TransformerBlock setAttentionDropout(double dropout) {
        attention.setDropout(dropout);
        return this;
    }
    
    public TransformerBlock setFeedForwardDropout(double dropout) {
        feedForward.setDropout(dropout);
        return this;
    }
    
    public TransformerBlock setFeedForwardActivation(String activation) {
        feedForward.setActivation(activation);
        return this;
    }
    
    public TransformerBlock setLayerNormEps(double eps) {
        layerNorm1.setEps(eps);
        layerNorm2.setEps(eps);
        return this;
    }
    
    // Getters
    public int getModelDim() { return modelDim; }
    public int getNumHeads() { return numHeads; }
    public int getFfDim() { return ffDim; }
    public MultiHeadAttention getAttention() { return attention; }
    public FeedForward getFeedForward() { return feedForward; }
    public LayerNorm getLayerNorm1() { return layerNorm1; }
    public LayerNorm getLayerNorm2() { return layerNorm2; }
    
    @Override
    public String toString() {
        return String.format("TransformerBlock(model_dim=%d, num_heads=%d, ff_dim=%d)",
                modelDim, numHeads, ffDim);
    }
}
