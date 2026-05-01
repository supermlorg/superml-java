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

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

/**
 * Positional Encoding for Transformer models.
 * 
 * Adds positional information to input embeddings using sinusoidal functions
 * as described in "Attention Is All You Need". This allows the model to 
 * understand the relative and absolute positions of tokens in a sequence.
 * 
 * The encoding uses sine and cosine functions of different frequencies:
 * - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
 * - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
 * 
 * @author SuperML Team  
 * @version 2.1.0
 */
public class PositionalEncoding {
    
    private int maxSeqLen;      // Maximum sequence length
    private int modelDim;       // Model dimension
    private RealMatrix encoding; // Pre-computed positional encodings
    
    /**
     * Constructor for PositionalEncoding.
     * 
     * @param modelDim The model dimension
     * @param maxSeqLen Maximum sequence length to pre-compute encodings for
     */
    public PositionalEncoding(int modelDim, int maxSeqLen) {
        this.modelDim = modelDim;
        this.maxSeqLen = maxSeqLen;
        
        precomputeEncodings();
    }
    
    /**
     * Pre-compute positional encodings for efficiency.
     */
    private void precomputeEncodings() {
        double[][] encodingData = new double[maxSeqLen][modelDim];
        
        for (int pos = 0; pos < maxSeqLen; pos++) {
            for (int i = 0; i < modelDim; i++) {
                double angle = pos / Math.pow(10000.0, (2.0 * (i / 2)) / modelDim);
                
                if (i % 2 == 0) {
                    // Even indices: sine
                    encodingData[pos][i] = Math.sin(angle);
                } else {
                    // Odd indices: cosine  
                    encodingData[pos][i] = Math.cos(angle);
                }
            }
        }
        
        this.encoding = new Array2DRowRealMatrix(encodingData);
    }
    
    /**
     * Add positional encodings to input embeddings.
     * 
     * @param embeddings Input embeddings matrix (seq_len, model_dim)
     * @return Embeddings with positional encodings added (seq_len, model_dim)
     */
    public RealMatrix addPositionalEncoding(RealMatrix embeddings) {
        int seqLen = embeddings.getRowDimension();
        int embeddingDim = embeddings.getColumnDimension();
        
        if (embeddingDim != modelDim) {
            throw new IllegalArgumentException(
                String.format("Embedding dimension (%d) must match model dimension (%d)", 
                            embeddingDim, modelDim));
        }
        
        if (seqLen > maxSeqLen) {
            throw new IllegalArgumentException(
                String.format("Sequence length (%d) exceeds maximum (%d)", seqLen, maxSeqLen));
        }
        
        // Extract relevant positional encodings
        RealMatrix posEncoding = encoding.getSubMatrix(0, seqLen - 1, 0, modelDim - 1);
        
        // Add positional encodings to embeddings
        return embeddings.add(posEncoding);
    }
    
    /**
     * Get positional encodings for a specific sequence length.
     * 
     * @param seqLen Sequence length
     * @return Positional encodings matrix (seq_len, model_dim)
     */
    public RealMatrix getEncodings(int seqLen) {
        if (seqLen > maxSeqLen) {
            throw new IllegalArgumentException(
                String.format("Sequence length (%d) exceeds maximum (%d)", seqLen, maxSeqLen));
        }
        
        return encoding.getSubMatrix(0, seqLen - 1, 0, modelDim - 1);
    }
    
    /**
     * Create positional encodings for a batch of sequences.
     * 
     * @param batchSize Number of sequences in the batch
     * @param seqLen Sequence length
     * @return Positional encodings matrix (batch_size * seq_len, model_dim)
     */
    public RealMatrix createBatchEncodings(int batchSize, int seqLen) {
        if (seqLen > maxSeqLen) {
            throw new IllegalArgumentException(
                String.format("Sequence length (%d) exceeds maximum (%d)", seqLen, maxSeqLen));
        }
        
        double[][] batchEncodingData = new double[batchSize * seqLen][modelDim];
        RealMatrix seqEncoding = getEncodings(seqLen);
        
        for (int batch = 0; batch < batchSize; batch++) {
            int startRow = batch * seqLen;
            for (int pos = 0; pos < seqLen; pos++) {
                for (int dim = 0; dim < modelDim; dim++) {
                    batchEncodingData[startRow + pos][dim] = seqEncoding.getEntry(pos, dim);
                }
            }
        }
        
        return new Array2DRowRealMatrix(batchEncodingData);
    }
    
    // Getters
    public int getMaxSeqLen() { return maxSeqLen; }
    public int getModelDim() { return modelDim; }
    
    @Override
    public String toString() {
        return String.format("PositionalEncoding(model_dim=%d, max_seq_len=%d)", modelDim, maxSeqLen);
    }
}
