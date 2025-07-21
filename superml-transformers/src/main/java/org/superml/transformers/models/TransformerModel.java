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
import org.superml.core.Classifier;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;

/**
 * Complete Transformer Model - Full encoder-decoder architecture.
 * 
 * This is the complete transformer model from "Attention Is All You Need".
 * Can be configured for different tasks:
 * - Encoder-only (BERT-style): Text classification, feature extraction
 * - Decoder-only (GPT-style): Text generation, language modeling
 * - Encoder-decoder: Translation, summarization, seq2seq tasks
 * 
 * Features:
 * - Configurable encoder/decoder layers
 * - Multiple attention heads
 * - Positional encoding
 * - Support for various NLP tasks
 * - Integration with SuperML ecosystem
 * 
 * @author SuperML Team
 * @version 2.1.0
 */
public class TransformerModel extends BaseEstimator implements Classifier {
    
    public enum Architecture {
        ENCODER_ONLY,       // BERT-style (classification, feature extraction)
        DECODER_ONLY,       // GPT-style (generation, language modeling)  
        ENCODER_DECODER     // Full transformer (translation, seq2seq)
    }
    
    private Architecture architecture;      // Model architecture type
    private int modelDim;                  // Model dimension
    private int numHeads;                  // Number of attention heads
    private int numEncoderLayers;          // Number of encoder layers
    private int numDecoderLayers;          // Number of decoder layers
    private int ffDim;                     // Feed forward dimension
    private int maxSeqLen;                 // Maximum sequence length
    private int vocabSize;                 // Vocabulary size
    private double dropout;                // Dropout rate
    
    // Components
    private TransformerEncoder encoder;     // Encoder component (optional)
    private TransformerDecoder decoder;     // Decoder component (optional)
    private RealMatrix outputProjection;    // Output projection to vocabulary
    private boolean isTrained = false;
    
    // Classification head (for encoder-only models)
    private RealMatrix classificationHead; // Linear layer for classification
    private int numClasses;                // Number of output classes
    
    /**
     * Create encoder-only transformer (BERT-style).
     * 
     * @param numLayers Number of encoder layers
     * @param modelDim Model dimension
     * @param numHeads Number of attention heads
     * @param numClasses Number of output classes for classification
     * @return Configured transformer model
     */
    public static TransformerModel createEncoderOnly(int numLayers, int modelDim, 
                                                    int numHeads, int numClasses) {
        return new TransformerModel(Architecture.ENCODER_ONLY, modelDim, numHeads,
                                   numLayers, 0, 4 * modelDim, 512, 0, 0.1, numClasses);
    }
    
    /**
     * Create decoder-only transformer (GPT-style).
     * 
     * @param numLayers Number of decoder layers
     * @param modelDim Model dimension
     * @param numHeads Number of attention heads
     * @param vocabSize Vocabulary size
     * @return Configured transformer model
     */
    public static TransformerModel createDecoderOnly(int numLayers, int modelDim, 
                                                    int numHeads, int vocabSize) {
        return new TransformerModel(Architecture.DECODER_ONLY, modelDim, numHeads,
                                   0, numLayers, 4 * modelDim, 512, vocabSize, 0.1, 0);
    }
    
    /**
     * Create full encoder-decoder transformer.
     * 
     * @param numEncoderLayers Number of encoder layers
     * @param numDecoderLayers Number of decoder layers
     * @param modelDim Model dimension
     * @param numHeads Number of attention heads
     * @param vocabSize Vocabulary size
     * @return Configured transformer model
     */
    public static TransformerModel createEncoderDecoder(int numEncoderLayers, int numDecoderLayers,
                                                       int modelDim, int numHeads, int vocabSize) {
        return new TransformerModel(Architecture.ENCODER_DECODER, modelDim, numHeads,
                                   numEncoderLayers, numDecoderLayers, 4 * modelDim, 512, 
                                   vocabSize, 0.1, 0);
    }
    
    /**
     * Full constructor for TransformerModel.
     */
    private TransformerModel(Architecture architecture, int modelDim, int numHeads,
                            int numEncoderLayers, int numDecoderLayers, int ffDim,
                            int maxSeqLen, int vocabSize, double dropout, int numClasses) {
        this.architecture = architecture;
        this.modelDim = modelDim;
        this.numHeads = numHeads;
        this.numEncoderLayers = numEncoderLayers;
        this.numDecoderLayers = numDecoderLayers;
        this.ffDim = ffDim;
        this.maxSeqLen = maxSeqLen;
        this.vocabSize = vocabSize;
        this.dropout = dropout;
        this.numClasses = numClasses;
        
        initializeComponents();
    }
    
    /**
     * Initialize model components based on architecture.
     */
    private void initializeComponents() {
        // Initialize encoder if needed
        if (architecture == Architecture.ENCODER_ONLY || architecture == Architecture.ENCODER_DECODER) {
            encoder = new TransformerEncoder(numEncoderLayers, modelDim, numHeads, ffDim, 
                                           maxSeqLen, dropout, true);
        }
        
        // Initialize decoder if needed
        if (architecture == Architecture.DECODER_ONLY || architecture == Architecture.ENCODER_DECODER) {
            decoder = new TransformerDecoder(numDecoderLayers, modelDim, numHeads, ffDim, 
                                           maxSeqLen, dropout);
        }
        
        // Initialize output projections
        if (architecture == Architecture.ENCODER_ONLY && numClasses > 0) {
            // Classification head
            classificationHead = createRandomMatrix(modelDim, numClasses, Math.sqrt(1.0 / modelDim));
        } else if (vocabSize > 0) {
            // Output vocabulary projection
            outputProjection = createRandomMatrix(modelDim, vocabSize, Math.sqrt(1.0 / modelDim));
        }
        
        this.isTrained = true;
    }
    
    /**
     * Create random matrix for weight initialization.
     */
    private RealMatrix createRandomMatrix(int rows, int cols, double scale) {
        double[][] data = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = (Math.random() - 0.5) * 2 * scale;
            }
        }
        return new Array2DRowRealMatrix(data);
    }
    
    /**
     * Forward pass for encoder-only architecture (classification).
     * 
     * @param embeddings Input embeddings (seq_len, model_dim)
     * @param mask Optional attention mask
     * @return Classification logits (1, num_classes)
     */
    public RealMatrix forwardEncoderOnly(RealMatrix embeddings, RealMatrix mask) {
        if (architecture != Architecture.ENCODER_ONLY) {
            throw new IllegalStateException("Model is not configured for encoder-only architecture");
        }
        
        // Encode input
        RealMatrix encoded = encoder.encode(embeddings, mask);
        
        // Get [CLS] token representation (first token)
        RealMatrix clsRepr = encoded.getSubMatrix(0, 0, 0, modelDim - 1);
        
        // Apply classification head
        RealMatrix logits = clsRepr.multiply(classificationHead);
        
        return logits;
    }
    
    /**
     * Forward pass for decoder-only architecture (generation).
     * 
     * @param embeddings Input embeddings (seq_len, model_dim)
     * @return Next token logits (1, vocab_size)
     */
    public RealMatrix forwardDecoderOnly(RealMatrix embeddings) {
        if (architecture != Architecture.DECODER_ONLY) {
            throw new IllegalStateException("Model is not configured for decoder-only architecture");
        }
        
        int seqLen = embeddings.getRowDimension();
        RealMatrix causalMask = TransformerDecoder.createCausalMask(seqLen);
        
        // Decode with causal mask
        RealMatrix decoded = decoder.decode(embeddings, null, causalMask, null);
        
        // Get last token representation for next token prediction
        RealMatrix lastToken = decoded.getSubMatrix(seqLen - 1, seqLen - 1, 0, modelDim - 1);
        
        // Project to vocabulary
        RealMatrix logits = lastToken.multiply(outputProjection);
        
        return logits;
    }
    
    /**
     * Forward pass for encoder-decoder architecture (seq2seq).
     * 
     * @param srcEmbeddings Source embeddings (src_len, model_dim)
     * @param tgtEmbeddings Target embeddings (tgt_len, model_dim)
     * @param srcMask Source attention mask (optional)
     * @param tgtMask Target causal mask (optional)
     * @return Output logits (tgt_len, vocab_size)
     */
    public RealMatrix forwardEncoderDecoder(RealMatrix srcEmbeddings, RealMatrix tgtEmbeddings,
                                          RealMatrix srcMask, RealMatrix tgtMask) {
        if (architecture != Architecture.ENCODER_DECODER) {
            throw new IllegalStateException("Model is not configured for encoder-decoder architecture");
        }
        
        // Encode source
        RealMatrix encoderOutput = encoder.encode(srcEmbeddings, srcMask);
        
        // Decode target with cross-attention to encoder output
        RealMatrix decoded = decoder.decode(tgtEmbeddings, encoderOutput, tgtMask, srcMask);
        
        // Project all positions to vocabulary
        RealMatrix logits = decoded.multiply(outputProjection);
        
        return logits;
    }
    
    /**
     * Implementation of Classifier interface for encoder-only models.
     */
    @Override
    public TransformerModel fit(double[][] X, double[] y) {
        if (architecture != Architecture.ENCODER_ONLY) {
            throw new UnsupportedOperationException("fit() only supported for encoder-only models");
        }
        
        System.out.println("🚀 Training TransformerModel for classification...");
        System.out.println("   Architecture: " + architecture);
        System.out.println("   Input samples: " + X.length);
        System.out.println("   Classes: " + numClasses);
        
        // In a real implementation, this would perform actual training
        // For now, we just validate the input and mark as trained
        
        // Validate input dimensions
        if (X.length == 0 || X[0].length != modelDim) {
            throw new IllegalArgumentException("Input dimension must match model dimension");
        }
        
        // The model is already initialized, so we're "trained"
        System.out.println("✅ Model training completed (placeholder implementation)");
        
        return this;
    }
    
    @Override
    public double[] predict(double[][] X) {
        if (architecture != Architecture.ENCODER_ONLY) {
            throw new UnsupportedOperationException("predict() only supported for encoder-only models");
        }
        
        if (!isTrained) {
            throw new IllegalStateException("Model must be trained before prediction");
        }
        
        double[] predictions = new double[X.length];
        
        for (int i = 0; i < X.length; i++) {
            // Convert input to matrix format
            RealMatrix embeddings = new Array2DRowRealMatrix(new double[][]{X[i]});
            
            // Forward pass
            RealMatrix logits = forwardEncoderOnly(embeddings, null);
            
            // Get predicted class (argmax)
            double maxLogit = Double.NEGATIVE_INFINITY;
            int maxClass = 0;
            for (int c = 0; c < numClasses; c++) {
                double logit = logits.getEntry(0, c);
                if (logit > maxLogit) {
                    maxLogit = logit;
                    maxClass = c;
                }
            }
            
            predictions[i] = maxClass;
        }
        
        return predictions;
    }
    
    @Override
    public double[][] predictProba(double[][] X) {
        if (architecture != Architecture.ENCODER_ONLY) {
            throw new UnsupportedOperationException("predictProba() only supported for encoder-only models");
        }
        
        if (!isTrained) {
            throw new IllegalStateException("Model must be trained before prediction");
        }
        
        double[][] probabilities = new double[X.length][numClasses];
        
        for (int i = 0; i < X.length; i++) {
            // Convert input to matrix format
            RealMatrix embeddings = new Array2DRowRealMatrix(new double[][]{X[i]});
            
            // Forward pass
            RealMatrix logits = forwardEncoderOnly(embeddings, null);
            
            // Apply softmax to get probabilities
            double[] logitsArray = new double[numClasses];
            for (int c = 0; c < numClasses; c++) {
                logitsArray[c] = logits.getEntry(0, c);
            }
            
            probabilities[i] = softmax(logitsArray);
        }
        
        return probabilities;
    }
    
    @Override
    public double[][] predictLogProba(double[][] X) {
        if (architecture != Architecture.ENCODER_ONLY) {
            throw new UnsupportedOperationException("predictLogProba() only supported for encoder-only models");
        }
        
        double[][] probabilities = predictProba(X);
        double[][] logProbabilities = new double[probabilities.length][probabilities[0].length];
        
        for (int i = 0; i < probabilities.length; i++) {
            for (int j = 0; j < probabilities[i].length; j++) {
                logProbabilities[i][j] = Math.log(probabilities[i][j] + 1e-15); // Add small epsilon for numerical stability
            }
        }
        
        return logProbabilities;
    }
    
    @Override
    public double[] getClasses() {
        if (architecture != Architecture.ENCODER_ONLY) {
            throw new UnsupportedOperationException("getClasses() only supported for encoder-only models");
        }
        
        double[] classes = new double[numClasses];
        for (int i = 0; i < numClasses; i++) {
            classes[i] = i; // Simple class labels 0, 1, 2, ...
        }
        return classes;
    }
    
    @Override
    public double score(double[][] X, double[] y) {
        if (architecture != Architecture.ENCODER_ONLY) {
            throw new UnsupportedOperationException("score() only supported for encoder-only models");
        }
        
        if (!isTrained) {
            throw new IllegalStateException("Model must be trained before scoring");
        }
        
        // Make predictions
        double[] predictions = predict(X);
        
        // Calculate accuracy
        int correct = 0;
        for (int i = 0; i < predictions.length; i++) {
            if (Math.abs(predictions[i] - y[i]) < 1e-6) {
                correct++;
            }
        }
        
        return (double) correct / predictions.length;
    }
    
    /**
     * Apply softmax activation to convert logits to probabilities.
     */
    private double[] softmax(double[] logits) {
        // Find max for numerical stability
        double max = Double.NEGATIVE_INFINITY;
        for (double logit : logits) {
            max = Math.max(max, logit);
        }
        
        // Compute exponentials and sum
        double[] exp = new double[logits.length];
        double sum = 0.0;
        for (int i = 0; i < logits.length; i++) {
            exp[i] = Math.exp(logits[i] - max);
            sum += exp[i];
        }
        
        // Normalize
        double[] probabilities = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probabilities[i] = exp[i] / sum;
        }
        
        return probabilities;
    }
    
    /**
     * Generate text using decoder-only model.
     * 
     * @param seedEmbeddings Initial embeddings to start generation
     * @param maxLength Maximum length to generate
     * @param temperature Sampling temperature (not implemented yet)
     * @return Generated token indices
     */
    public int[] generateText(RealMatrix seedEmbeddings, int maxLength, double temperature) {
        if (architecture != Architecture.DECODER_ONLY) {
            throw new UnsupportedOperationException("Text generation only supported for decoder-only models");
        }
        
        int seedLen = seedEmbeddings.getRowDimension();
        int[] generated = new int[maxLength];
        
        // Current sequence (will be extended during generation)
        RealMatrix currentSequence = seedEmbeddings.copy();
        
        for (int step = 0; step < maxLength - seedLen; step++) {
            // Forward pass to get next token logits
            RealMatrix logits = forwardDecoderOnly(currentSequence);
            
            // Sample next token (greedy for now)
            double maxLogit = Double.NEGATIVE_INFINITY;
            int nextToken = 0;
            for (int v = 0; v < vocabSize; v++) {
                if (logits.getEntry(0, v) > maxLogit) {
                    maxLogit = logits.getEntry(0, v);
                    nextToken = v;
                }
            }
            
            generated[seedLen + step] = nextToken;
            
            // Extend sequence (simplified - in practice we'd use token embeddings)
            // For now, create dummy embedding for the next token
            double[][] nextTokenEmb = new double[1][modelDim];
            for (int d = 0; d < modelDim; d++) {
                nextTokenEmb[0][d] = Math.sin(nextToken * 0.1 + d * 0.01); // Dummy embedding
            }
            
            // Concatenate to current sequence
            int currentLen = currentSequence.getRowDimension();
            double[][] extendedData = new double[currentLen + 1][modelDim];
            
            // Copy existing sequence
            for (int pos = 0; pos < currentLen; pos++) {
                for (int dim = 0; dim < modelDim; dim++) {
                    extendedData[pos][dim] = currentSequence.getEntry(pos, dim);
                }
            }
            
            // Add new token
            extendedData[currentLen] = nextTokenEmb[0];
            
            currentSequence = new Array2DRowRealMatrix(extendedData);
        }
        
        return generated;
    }
    
    // Getters
    public Architecture getArchitecture() { return architecture; }
    public int getModelDim() { return modelDim; }
    public int getNumHeads() { return numHeads; }
    public int getNumEncoderLayers() { return numEncoderLayers; }
    public int getNumDecoderLayers() { return numDecoderLayers; }
    public int getFfDim() { return ffDim; }
    public int getMaxSeqLen() { return maxSeqLen; }
    public int getVocabSize() { return vocabSize; }
    public double getDropout() { return dropout; }
    public int getNumClasses() { return numClasses; }
    public TransformerEncoder getEncoder() { return encoder; }
    public TransformerDecoder getDecoder() { return decoder; }
    
    @Override
    public String toString() {
        return String.format("TransformerModel(arch=%s, model_dim=%d, heads=%d, enc_layers=%d, dec_layers=%d, max_seq_len=%d)",
                architecture, modelDim, numHeads, numEncoderLayers, numDecoderLayers, maxSeqLen);
    }
}
