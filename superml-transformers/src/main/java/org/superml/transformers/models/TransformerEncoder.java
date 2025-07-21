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
import org.apache.commons.math3.linear.RealMatrix;

import java.util.ArrayList;
import java.util.List;

/**
 * Transformer Encoder - Stack of transformer blocks for encoding input sequences.
 * 
 * The encoder consists of:
 * 1. Input embedding + positional encoding
 * 2. Stack of N identical transformer blocks
 * 3. Optional final layer normalization
 * 
 * This follows the architecture from "Attention Is All You Need" paper.
 * Commonly used for tasks like:
 * - Text classification (BERT-style)
 * - Feature extraction
 * - Representation learning
 * 
 * @author SuperML Team
 * @version 2.1.0
 */
public class TransformerEncoder extends BaseEstimator {
    
    private int numLayers;              // Number of transformer blocks
    private int modelDim;               // Model dimension
    private int numHeads;               // Number of attention heads per block
    private int ffDim;                  // Feed forward hidden dimension
    private int maxSeqLen;              // Maximum sequence length
    private double dropout;             // Dropout rate
    private boolean useLayerNorm;       // Whether to apply final layer norm
    
    // Components
    private PositionalEncoding posEncoding;    // Positional encoding
    private List<TransformerBlock> layers;     // Stack of transformer blocks
    private boolean isTrained = false;
    
    /**
     * Constructor with default settings.
     * 
     * @param numLayers Number of transformer blocks
     * @param modelDim Model dimension
     * @param numHeads Number of attention heads
     */
    public TransformerEncoder(int numLayers, int modelDim, int numHeads) {
        this(numLayers, modelDim, numHeads, 4 * modelDim, 512, 0.1, true);
    }
    
    /**
     * Full constructor for TransformerEncoder.
     * 
     * @param numLayers Number of transformer blocks
     * @param modelDim Model dimension
     * @param numHeads Number of attention heads per block
     * @param ffDim Feed forward hidden dimension
     * @param maxSeqLen Maximum sequence length
     * @param dropout Dropout rate
     * @param useLayerNorm Whether to apply final layer normalization
     */
    public TransformerEncoder(int numLayers, int modelDim, int numHeads, int ffDim, 
                             int maxSeqLen, double dropout, boolean useLayerNorm) {
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
        this.useLayerNorm = useLayerNorm;
        
        initializeLayers();
    }
    
    /**
     * Initialize all encoder layers and components.
     */
    private void initializeLayers() {
        // Initialize positional encoding
        posEncoding = new PositionalEncoding(modelDim, maxSeqLen);
        
        // Initialize transformer blocks
        layers = new ArrayList<>();
        for (int i = 0; i < numLayers; i++) {
            TransformerBlock block = new TransformerBlock(modelDim, numHeads, ffDim)
                    .setAttentionDropout(dropout)
                    .setFeedForwardDropout(dropout);
            layers.add(block);
        }
        
        this.isTrained = true;
    }
    
    /**
     * Encode input sequence through the transformer encoder.
     * 
     * @param embeddings Input embeddings (seq_len, model_dim)
     * @param mask Optional attention mask (seq_len, seq_len)
     * @return Encoded representations (seq_len, model_dim)
     */
    public RealMatrix encode(RealMatrix embeddings, RealMatrix mask) {
        if (!isTrained) {
            throw new IllegalStateException("TransformerEncoder must be initialized before encoding");
        }
        
        int seqLen = embeddings.getRowDimension();
        int embDim = embeddings.getColumnDimension();
        
        if (embDim != modelDim) {
            throw new IllegalArgumentException(
                String.format("Embedding dimension (%d) must match model dimension (%d)", 
                            embDim, modelDim));
        }
        
        if (seqLen > maxSeqLen) {
            throw new IllegalArgumentException(
                String.format("Sequence length (%d) exceeds maximum (%d)", seqLen, maxSeqLen));
        }
        
        // Add positional encodings to embeddings
        RealMatrix x = posEncoding.addPositionalEncoding(embeddings);
        
        // Pass through each transformer block
        for (TransformerBlock layer : layers) {
            x = layer.forward(x, mask);
        }
        
        return x;
    }
    
    /**
     * Encode input sequence without attention mask.
     * 
     * @param embeddings Input embeddings (seq_len, model_dim)
     * @return Encoded representations (seq_len, model_dim)
     */
    public RealMatrix encode(RealMatrix embeddings) {
        return encode(embeddings, null);
    }
    
    /**
     * Get encoded representation for the [CLS] token (first token).
     * Commonly used for classification tasks.
     * 
     * @param embeddings Input embeddings with [CLS] token as first row
     * @param mask Optional attention mask
     * @return [CLS] token representation (1, model_dim)
     */
    public RealMatrix getClassificationRepresentation(RealMatrix embeddings, RealMatrix mask) {
        RealMatrix encoded = encode(embeddings, mask);
        // Return first row (CLS token representation)
        return encoded.getSubMatrix(0, 0, 0, modelDim - 1);
    }
    
    /**
     * Get pooled representation by averaging over sequence length.
     * Alternative to CLS token for classification.
     * 
     * @param embeddings Input embeddings
     * @param mask Optional attention mask (can be used to exclude padding)
     * @return Pooled representation (1, model_dim)
     */
    public RealMatrix getPooledRepresentation(RealMatrix embeddings, RealMatrix mask) {
        RealMatrix encoded = encode(embeddings, mask);
        int seqLen = encoded.getRowDimension();
        
        // Simple average pooling
        double[] pooled = new double[modelDim];
        for (int dim = 0; dim < modelDim; dim++) {
            double sum = 0.0;
            for (int pos = 0; pos < seqLen; pos++) {
                sum += encoded.getEntry(pos, dim);
            }
            pooled[dim] = sum / seqLen;
        }
        
        // Convert to matrix
        double[][] pooledMatrix = new double[1][modelDim];
        pooledMatrix[0] = pooled;
        return new org.apache.commons.math3.linear.Array2DRowRealMatrix(pooledMatrix);
    }
    
    /**
     * Encode batch of sequences.
     * 
     * @param batchEmbeddings Batch embeddings (batch_size * seq_len, model_dim)
     * @param batchSize Number of sequences in batch
     * @param seqLen Length of each sequence
     * @param mask Optional batch attention mask
     * @return Batch encoded representations (batch_size * seq_len, model_dim)
     */
    public RealMatrix encodeBatch(RealMatrix batchEmbeddings, int batchSize, int seqLen, RealMatrix mask) {
        if (batchEmbeddings.getRowDimension() != batchSize * seqLen) {
            throw new IllegalArgumentException("Batch embeddings size mismatch");
        }
        
        // For simplicity, process each sequence in the batch separately
        // In a full implementation, we would vectorize this operation
        double[][] resultData = new double[batchSize * seqLen][modelDim];
        
        for (int batch = 0; batch < batchSize; batch++) {
            // Extract sequence for this batch
            int startRow = batch * seqLen;
            RealMatrix seqEmbeddings = batchEmbeddings.getSubMatrix(
                startRow, startRow + seqLen - 1, 0, modelDim - 1);
            
            // Extract mask for this sequence if provided
            RealMatrix seqMask = null;
            if (mask != null) {
                int maskStartRow = batch * seqLen;
                seqMask = mask.getSubMatrix(maskStartRow, maskStartRow + seqLen - 1, 0, seqLen - 1);
            }
            
            // Encode this sequence
            RealMatrix encoded = encode(seqEmbeddings, seqMask);
            
            // Copy back to result
            for (int pos = 0; pos < seqLen; pos++) {
                for (int dim = 0; dim < modelDim; dim++) {
                    resultData[startRow + pos][dim] = encoded.getEntry(pos, dim);
                }
            }
        }
        
        return new org.apache.commons.math3.linear.Array2DRowRealMatrix(resultData);
    }
    
    // Fluent configuration methods
    
    public TransformerEncoder setDropout(double dropout) {
        this.dropout = dropout;
        // Update all layers
        for (TransformerBlock layer : layers) {
            layer.setAttentionDropout(dropout)
                 .setFeedForwardDropout(dropout);
        }
        return this;
    }
    
    public TransformerEncoder setFeedForwardActivation(String activation) {
        for (TransformerBlock layer : layers) {
            layer.setFeedForwardActivation(activation);
        }
        return this;
    }
    
    public TransformerEncoder setLayerNormEps(double eps) {
        for (TransformerBlock layer : layers) {
            layer.setLayerNormEps(eps);
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
    public boolean getUseLayerNorm() { return useLayerNorm; }
    public List<TransformerBlock> getLayers() { return new ArrayList<>(layers); }
    public PositionalEncoding getPositionalEncoding() { return posEncoding; }
    
    @Override
    public String toString() {
        return String.format("TransformerEncoder(layers=%d, model_dim=%d, heads=%d, ff_dim=%d, max_seq_len=%d, dropout=%.2f)",
                numLayers, modelDim, numHeads, ffDim, maxSeqLen, dropout);
    }
}
