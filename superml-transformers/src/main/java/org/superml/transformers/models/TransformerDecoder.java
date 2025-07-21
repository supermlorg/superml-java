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

package org.superml.transformers.models;

import org.superml.core.BaseEstimator;
import org.superml.transformers.layers.TransformerBlock;
import org.superml.transformers.layers.PositionalEncoding;
import org.superml.transformers.attention.MultiHeadAttention;
import org.superml.transformers.layers.LayerNorm;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;

import java.util.ArrayList;
import java.util.List;

/**
 * Transformer Decoder - Stack of decoder blocks for generating output sequences.
 * 
 * The decoder consists of:
 * 1. Output embedding + positional encoding
 * 2. Stack of N identical decoder blocks, each containing:
 *    - Masked self-attention
 *    - Cross-attention to encoder outputs
 *    - Feed forward network
 * 3. Final linear projection to vocabulary
 * 
 * This follows the architecture from "Attention Is All You Need" paper.
 * Commonly used for tasks like:
 * - Machine translation
 * - Text generation
 * - Sequence-to-sequence modeling
 * 
 * @author SuperML Team
 * @version 2.1.0
 */
public class TransformerDecoder extends BaseEstimator {
    
    private int numLayers;              // Number of decoder blocks
    private int modelDim;               // Model dimension
    private int numHeads;               // Number of attention heads per block
    private int ffDim;                  // Feed forward hidden dimension
    private int maxSeqLen;              // Maximum sequence length
    private double dropout;             // Dropout rate
    
    // Components
    private PositionalEncoding posEncoding;         // Positional encoding
    private List<DecoderBlock> layers;              // Stack of decoder blocks
    private boolean isTrained = false;
    
    /**
     * Constructor with default settings.
     * 
     * @param numLayers Number of decoder blocks
     * @param modelDim Model dimension
     * @param numHeads Number of attention heads
     */
    public TransformerDecoder(int numLayers, int modelDim, int numHeads) {
        this(numLayers, modelDim, numHeads, 4 * modelDim, 512, 0.1);
    }
    
    /**
     * Full constructor for TransformerDecoder.
     * 
     * @param numLayers Number of decoder blocks
     * @param modelDim Model dimension
     * @param numHeads Number of attention heads per block
     * @param ffDim Feed forward hidden dimension
     * @param maxSeqLen Maximum sequence length
     * @param dropout Dropout rate
     */
    public TransformerDecoder(int numLayers, int modelDim, int numHeads, int ffDim, 
                             int maxSeqLen, double dropout) {
        if (numLayers < 1) {
            throw new IllegalArgumentException("Number of layers must be at least 1");
        }
        if (modelDim % numHeads != 0) {
            throw new IllegalArgumentException("Model dimension must be divisible by number of heads");
        }
        
        this.numLayers = numLayers;
        this.modelDim = modelDim;
        this.numHeads = numHeads;
        this.ffDim = ffDim;
        this.maxSeqLen = maxSeqLen;
        this.dropout = dropout;
        
        initializeLayers();
    }
    
    /**
     * Initialize all decoder layers and components.
     */
    private void initializeLayers() {
        // Initialize positional encoding
        posEncoding = new PositionalEncoding(modelDim, maxSeqLen);
        
        // Initialize decoder blocks
        layers = new ArrayList<>();
        for (int i = 0; i < numLayers; i++) {
            DecoderBlock block = new DecoderBlock(modelDim, numHeads, ffDim, dropout);
            layers.add(block);
        }
        
        this.isTrained = true;
    }
    
    /**
     * Decode target sequence with encoder outputs.
     * 
     * @param targetEmbeddings Target embeddings (tgt_len, model_dim)
     * @param encoderOutputs Encoder outputs (src_len, model_dim)
     * @param targetMask Causal mask for target sequence (tgt_len, tgt_len)
     * @param srcMask Optional source mask (tgt_len, src_len)
     * @return Decoded representations (tgt_len, model_dim)
     */
    public RealMatrix decode(RealMatrix targetEmbeddings, RealMatrix encoderOutputs, 
                           RealMatrix targetMask, RealMatrix srcMask) {
        if (!isTrained) {
            throw new IllegalStateException("TransformerDecoder must be initialized before decoding");
        }
        
        int tgtLen = targetEmbeddings.getRowDimension();
        int embDim = targetEmbeddings.getColumnDimension();
        
        if (embDim != modelDim) {
            throw new IllegalArgumentException(
                String.format("Embedding dimension (%d) must match model dimension (%d)", 
                            embDim, modelDim));
        }
        
        if (tgtLen > maxSeqLen) {
            throw new IllegalArgumentException(
                String.format("Target length (%d) exceeds maximum (%d)", tgtLen, maxSeqLen));
        }
        
        // Add positional encodings to target embeddings
        RealMatrix x = posEncoding.addPositionalEncoding(targetEmbeddings);
        
        // Pass through each decoder block
        for (DecoderBlock layer : layers) {
            x = layer.forward(x, encoderOutputs, targetMask, srcMask);
        }
        
        return x;
    }
    
    /**
     * Create causal mask for autoregressive generation.
     * Prevents positions from attending to future positions.
     * 
     * @param seqLen Sequence length
     * @return Causal mask (seqLen, seqLen)
     */
    public static RealMatrix createCausalMask(int seqLen) {
        double[][] maskData = new double[seqLen][seqLen];
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j <= i; j++) {
                maskData[i][j] = 1.0; // Can attend to current and previous positions
            }
            // Future positions remain 0 (masked)
        }
        return new Array2DRowRealMatrix(maskData);
    }
    
    /**
     * Generate next token probabilities for autoregressive generation.
     * 
     * @param targetEmbeddings Target embeddings so far
     * @param encoderOutputs Encoder outputs
     * @param vocabSize Vocabulary size for output projection
     * @return Next token logits (1, vocab_size)
     */
    public RealMatrix generateNextToken(RealMatrix targetEmbeddings, RealMatrix encoderOutputs, int vocabSize) {
        int tgtLen = targetEmbeddings.getRowDimension();
        
        // Create causal mask
        RealMatrix causalMask = createCausalMask(tgtLen);
        
        // Decode
        RealMatrix decoded = decode(targetEmbeddings, encoderOutputs, causalMask, null);
        
        // Get representation of last position (next token prediction)
        RealMatrix lastTokenRepr = decoded.getSubMatrix(tgtLen - 1, tgtLen - 1, 0, modelDim - 1);
        
        // Project to vocabulary size (simplified - in practice this would be a learned linear layer)
        // For now, return a random projection as placeholder
        double[][] logitsData = new double[1][vocabSize];
        for (int v = 0; v < vocabSize; v++) {
            double sum = 0.0;
            for (int d = 0; d < modelDim; d++) {
                sum += lastTokenRepr.getEntry(0, d) * Math.sin(d * 0.1 + v * 0.01); // Dummy projection
            }
            logitsData[0][v] = sum;
        }
        
        return new Array2DRowRealMatrix(logitsData);
    }
    
    // Fluent configuration methods
    
    public TransformerDecoder setDropout(double dropout) {
        this.dropout = dropout;
        // Update all layers
        for (DecoderBlock layer : layers) {
            layer.setDropout(dropout);
        }
        return this;
    }
    
    public TransformerDecoder setFeedForwardActivation(String activation) {
        for (DecoderBlock layer : layers) {
            layer.setFeedForwardActivation(activation);
        }
        return this;
    }
    
    // Getters
    public int getNumLayers() { return numLayers; }
    public int getModelDim() { return modelDim; }
    public int getNumHeads() { return numHeads; }
    public int getFfDim() { return ffDim; }
    public int getMaxSeqLen() { return maxSeqLen; }
    public double getDropout() { return dropout; }
    public List<DecoderBlock> getLayers() { return new ArrayList<>(layers); }
    public PositionalEncoding getPositionalEncoding() { return posEncoding; }
    
    @Override
    public String toString() {
        return String.format("TransformerDecoder(layers=%d, model_dim=%d, heads=%d, ff_dim=%d, max_seq_len=%d, dropout=%.2f)",
                numLayers, modelDim, numHeads, ffDim, maxSeqLen, dropout);
    }
    
    /**
     * Decoder Block - Single decoder layer with masked self-attention and cross-attention.
     */
    public static class DecoderBlock extends BaseEstimator {
        
        private int modelDim;
        private int numHeads;
        private int ffDim;
        private double dropout;
        
        // Components
        private MultiHeadAttention selfAttention;    // Masked self-attention
        private MultiHeadAttention crossAttention;   // Cross-attention to encoder
        private org.superml.transformers.layers.FeedForward feedForward;  // Feed forward network
        private LayerNorm layerNorm1;                // After self-attention
        private LayerNorm layerNorm2;                // After cross-attention
        private LayerNorm layerNorm3;                // After feed forward
        
        public DecoderBlock(int modelDim, int numHeads, int ffDim, double dropout) {
            this.modelDim = modelDim;
            this.numHeads = numHeads;
            this.ffDim = ffDim;
            this.dropout = dropout;
            
            initializeComponents();
        }
        
        private void initializeComponents() {
            selfAttention = new MultiHeadAttention(modelDim, numHeads).setDropout(dropout);
            crossAttention = new MultiHeadAttention(modelDim, numHeads).setDropout(dropout);
            feedForward = new org.superml.transformers.layers.FeedForward(modelDim, ffDim, "relu", dropout, true);
            layerNorm1 = new LayerNorm(modelDim);
            layerNorm2 = new LayerNorm(modelDim);
            layerNorm3 = new LayerNorm(modelDim);
        }
        
        /**
         * Forward pass through decoder block.
         * 
         * @param target Target sequence representations
         * @param encoderOutput Encoder output representations
         * @param targetMask Causal mask for target
         * @param srcMask Source mask (optional)
         * @return Processed representations
         */
        public RealMatrix forward(RealMatrix target, RealMatrix encoderOutput, 
                                RealMatrix targetMask, RealMatrix srcMask) {
            // 1. Masked self-attention
            RealMatrix selfAttnOut = selfAttention.forward(target, targetMask);
            RealMatrix residual1 = target.add(selfAttnOut);
            RealMatrix normed1 = layerNorm1.forward(residual1);
            
            // 2. Cross-attention (query from decoder, key/value from encoder)
            RealMatrix crossAttnOut = computeCrossAttention(normed1, encoderOutput, srcMask);
            RealMatrix residual2 = normed1.add(crossAttnOut);
            RealMatrix normed2 = layerNorm2.forward(residual2);
            
            // 3. Feed forward
            RealMatrix ffOut = feedForward.forward(normed2);
            RealMatrix residual3 = normed2.add(ffOut);
            RealMatrix normed3 = layerNorm3.forward(residual3);
            
            return normed3;
        }
        
        /**
         * Compute cross-attention between decoder and encoder.
         * This is a simplified implementation.
         */
        private RealMatrix computeCrossAttention(RealMatrix query, RealMatrix keyValue, RealMatrix mask) {
            // For cross-attention, we need to modify MultiHeadAttention to accept separate K,V
            // For now, use self-attention on concatenated representations as approximation
            
            int queryLen = query.getRowDimension();
            int kvLen = keyValue.getRowDimension();
            
            // Create combined input for cross-attention approximation
            double[][] combinedData = new double[queryLen + kvLen][modelDim];
            
            // Copy query
            for (int i = 0; i < queryLen; i++) {
                for (int j = 0; j < modelDim; j++) {
                    combinedData[i][j] = query.getEntry(i, j);
                }
            }
            
            // Copy key/value
            for (int i = 0; i < kvLen; i++) {
                for (int j = 0; j < modelDim; j++) {
                    combinedData[queryLen + i][j] = keyValue.getEntry(i, j);
                }
            }
            
            RealMatrix combined = new Array2DRowRealMatrix(combinedData);
            RealMatrix crossAttnResult = crossAttention.forward(combined, null);
            
            // Extract query part (first queryLen rows)
            return crossAttnResult.getSubMatrix(0, queryLen - 1, 0, modelDim - 1);
        }
        
        // Configuration methods
        public DecoderBlock setDropout(double dropout) {
            this.dropout = dropout;
            selfAttention.setDropout(dropout);
            crossAttention.setDropout(dropout);
            feedForward.setDropout(dropout);
            return this;
        }
        
        public DecoderBlock setFeedForwardActivation(String activation) {
            feedForward.setActivation(activation);
            return this;
        }
    }
}
