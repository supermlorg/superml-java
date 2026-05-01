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

package org.superml.examples.transformers;

import org.superml.transformers.attention.MultiHeadAttention;
import org.superml.transformers.layers.*;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

/**
 * Comprehensive example demonstrating Transformer components.
 * 
 * This example shows:
 * 1. Individual component usage
 * 2. Complete transformer block processing
 * 3. Performance timing and memory usage
 * 4. Different configurations and use cases
 * 
 * @author SuperML Team
 * @version 2.1.0
 */
public class TransformerComponentsExample {
    
    public static void main(String[] args) {
        System.out.println("🚀 SuperML Transformer Implementation Demo");
        System.out.println("=========================================");
        
        // Demo configurations
        demoMultiHeadAttention();
        demoPositionalEncoding();
        demoLayerNormalization();
        demoFeedForward();
        demoTransformerBlock();
        demoPerformanceBenchmark();
        
        System.out.println("\n✅ All Transformer components working successfully!");
    }
    
    /**
     * Demonstrate Multi-Head Attention mechanism.
     */
    private static void demoMultiHeadAttention() {
        System.out.println("\n🧠 Multi-Head Attention Demo");
        System.out.println("-----------------------------");
        
        // Create attention with different configurations
        MultiHeadAttention attention8 = new MultiHeadAttention(512, 8);
        MultiHeadAttention attention16 = new MultiHeadAttention(512, 16);
        
        // Generate sample input (simulating embeddings for a sentence)
        double[][] inputData = generateRandomMatrix(10, 512); // 10 tokens, 512 dimensions
        RealMatrix input = new Array2DRowRealMatrix(inputData);
        
        System.out.println("Input shape: " + input.getRowDimension() + " x " + input.getColumnDimension());
        
        // Forward pass
        long startTime = System.nanoTime();
        RealMatrix output8 = attention8.forward(input, null);
        long endTime = System.nanoTime();
        
        System.out.println("✅ 8-head attention output shape: " + output8.getRowDimension() + " x " + output8.getColumnDimension());
        System.out.println("   Processing time: " + (endTime - startTime) / 1_000_000.0 + " ms");
        
        // Test with attention mask (causal mask for autoregressive generation)
        RealMatrix causalMask = createCausalMask(10);
        startTime = System.nanoTime();
        RealMatrix maskedOutput = attention8.forward(input, causalMask);
        endTime = System.nanoTime();
        
        System.out.println("✅ Masked attention output shape: " + maskedOutput.getRowDimension() + " x " + maskedOutput.getColumnDimension());
        System.out.println("   Processing time with mask: " + (endTime - startTime) / 1_000_000.0 + " ms");
        
        // Compare different head configurations
        RealMatrix output16 = attention16.forward(input, null);
        System.out.println("✅ 16-head attention works with same input");
        
        System.out.println("📊 Attention Configurations:");
        System.out.println("   " + attention8);
        System.out.println("   " + attention16);
    }
    
    /**
     * Demonstrate Positional Encoding.
     */
    private static void demoPositionalEncoding() {
        System.out.println("\n📍 Positional Encoding Demo");
        System.out.println("----------------------------");
        
        PositionalEncoding posEnc = new PositionalEncoding(512, 1000);
        
        // Generate embeddings
        double[][] embeddingData = generateRandomMatrix(20, 512); // 20 tokens
        RealMatrix embeddings = new Array2DRowRealMatrix(embeddingData);
        
        System.out.println("Original embeddings shape: " + embeddings.getRowDimension() + " x " + embeddings.getColumnDimension());
        
        // Add positional encodings
        RealMatrix positionedEmbeddings = posEnc.addPositionalEncoding(embeddings);
        
        System.out.println("✅ Positioned embeddings shape: " + positionedEmbeddings.getRowDimension() + " x " + positionedEmbeddings.getColumnDimension());
        
        // Show that positional encodings are deterministic
        RealMatrix posEncodings1 = posEnc.getEncodings(5);
        RealMatrix posEncodings2 = posEnc.getEncodings(5);
        
        boolean identical = true;
        for (int i = 0; i < 5 && identical; i++) {
            for (int j = 0; j < 512 && identical; j++) {
                if (Math.abs(posEncodings1.getEntry(i, j) - posEncodings2.getEntry(i, j)) > 1e-10) {
                    identical = false;
                }
            }
        }
        System.out.println("✅ Positional encodings are deterministic: " + identical);
        
        // Test batch encodings
        RealMatrix batchEncodings = posEnc.createBatchEncodings(4, 10); // 4 sequences of length 10
        System.out.println("✅ Batch encodings shape: " + batchEncodings.getRowDimension() + " x " + batchEncodings.getColumnDimension());
        
        System.out.println("📊 " + posEnc);
    }
    
    /**
     * Demonstrate Layer Normalization.
     */
    private static void demoLayerNormalization() {
        System.out.println("\n🔧 Layer Normalization Demo");
        System.out.println("----------------------------");
        
        LayerNorm layerNorm = new LayerNorm(512);
        
        // Create input with known statistics
        double[][] inputData = new double[5][512];
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 512; j++) {
                inputData[i][j] = Math.random() * 10 + 5; // Mean around 5, varied scale
            }
        }
        
        RealMatrix input = new Array2DRowRealMatrix(inputData);
        RealMatrix normalized = layerNorm.forward(input);
        
        System.out.println("Input shape: " + input.getRowDimension() + " x " + input.getColumnDimension());
        System.out.println("✅ Normalized shape: " + normalized.getRowDimension() + " x " + normalized.getColumnDimension());
        
        // Check normalization properties (mean ≈ 0, std ≈ 1)
        for (int sample = 0; sample < 2; sample++) { // Check first 2 samples
            double sum = 0, sumSquares = 0;
            for (int feature = 0; feature < 512; feature++) {
                double val = normalized.getEntry(sample, feature);
                sum += val;
                sumSquares += val * val;
            }
            double mean = sum / 512;
            double std = Math.sqrt(sumSquares / 512 - mean * mean);
            
            System.out.println("   Sample " + sample + " - Mean: " + String.format("%.6f", mean) + 
                             ", Std: " + String.format("%.6f", std));
        }
        
        // Test single vector normalization
        double[] vector = new double[512];
        for (int i = 0; i < 512; i++) {
            vector[i] = Math.random() * 20 - 10;
        }
        double[] normalizedVector = layerNorm.normalize(vector);
        System.out.println("✅ Single vector normalization works");
        
        System.out.println("📊 " + layerNorm);
    }
    
    /**
     * Demonstrate Feed Forward Network.
     */
    private static void demoFeedForward() {
        System.out.println("\n🔀 Feed Forward Network Demo");
        System.out.println("-----------------------------");
        
        FeedForward ffRelu = new FeedForward(256, 1024, "relu", 0.1, true);
        FeedForward ffGelu = new FeedForward(256, 1024, "gelu", 0.1, true);
        
        double[][] inputData = generateRandomMatrix(15, 256); // 15 positions, 256 dimensions
        RealMatrix input = new Array2DRowRealMatrix(inputData);
        
        System.out.println("Input shape: " + input.getRowDimension() + " x " + input.getColumnDimension());
        
        // Test ReLU activation
        long startTime = System.nanoTime();
        RealMatrix outputRelu = ffRelu.forward(input);
        long endTime = System.nanoTime();
        
        System.out.println("✅ ReLU FF output shape: " + outputRelu.getRowDimension() + " x " + outputRelu.getColumnDimension());
        System.out.println("   Processing time: " + (endTime - startTime) / 1_000_000.0 + " ms");
        
        // Test GELU activation
        startTime = System.nanoTime();
        RealMatrix outputGelu = ffGelu.forward(input);
        endTime = System.nanoTime();
        
        System.out.println("✅ GELU FF output shape: " + outputGelu.getRowDimension() + " x " + outputGelu.getColumnDimension());
        System.out.println("   Processing time: " + (endTime - startTime) / 1_000_000.0 + " ms");
        
        // Verify outputs are different (due to different activations)
        boolean different = false;
        for (int i = 0; i < 5 && !different; i++) {
            for (int j = 0; j < 10 && !different; j++) {
                if (Math.abs(outputRelu.getEntry(i, j) - outputGelu.getEntry(i, j)) > 1e-6) {
                    different = true;
                }
            }
        }
        System.out.println("✅ ReLU and GELU produce different outputs: " + different);
        
        System.out.println("📊 Feed Forward Configurations:");
        System.out.println("   " + ffRelu);
        System.out.println("   " + ffGelu);
    }
    
    /**
     * Demonstrate complete Transformer Block.
     */
    private static void demoTransformerBlock() {
        System.out.println("\n🧩 Complete Transformer Block Demo");
        System.out.println("-----------------------------------");
        
        TransformerBlock transformer = new TransformerBlock(256, 8, 1024);
        
        double[][] inputData = generateRandomMatrix(12, 256); // 12 tokens, 256 dimensions
        RealMatrix input = new Array2DRowRealMatrix(inputData);
        
        System.out.println("Input shape: " + input.getRowDimension() + " x " + input.getColumnDimension());
        
        // Forward pass
        long startTime = System.nanoTime();
        RealMatrix output = transformer.forward(input);
        long endTime = System.nanoTime();
        
        System.out.println("✅ Transformer output shape: " + output.getRowDimension() + " x " + output.getColumnDimension());
        System.out.println("   Total processing time: " + (endTime - startTime) / 1_000_000.0 + " ms");
        
        // Test with causal mask
        RealMatrix causalMask = createCausalMask(12);
        startTime = System.nanoTime();
        RealMatrix maskedOutput = transformer.forward(input, causalMask);
        endTime = System.nanoTime();
        
        System.out.println("✅ Masked transformer output shape: " + maskedOutput.getRowDimension() + " x " + maskedOutput.getColumnDimension());
        System.out.println("   Masked processing time: " + (endTime - startTime) / 1_000_000.0 + " ms");
        
        // Test stacked transformer blocks (simulating a deeper model)
        TransformerBlock transformer2 = new TransformerBlock(256, 8, 1024);
        RealMatrix layer2Output = transformer2.forward(output);
        System.out.println("✅ Two-layer transformer works: " + layer2Output.getRowDimension() + " x " + layer2Output.getColumnDimension());
        
        System.out.println("📊 " + transformer);
    }
    
    /**
     * Benchmark performance with different configurations.
     */
    private static void demoPerformanceBenchmark() {
        System.out.println("\n⚡ Performance Benchmark");
        System.out.println("------------------------");
        
        int[] sequenceLengths = {8, 16, 32, 64};
        int[] modelDims = {128, 256, 512};
        
        System.out.println("Testing Transformer performance across configurations:");
        System.out.println("Seq Len | Model Dim | Heads | Time (ms) | Memory (approx)");
        System.out.println("---------|-----------|--------|-----------|----------------");
        
        for (int seqLen : sequenceLengths) {
            for (int modelDim : modelDims) {
                int numHeads = Math.min(modelDim / 64, 16); // Reasonable head count
                if (modelDim % numHeads != 0) numHeads = 8; // Fallback
                
                TransformerBlock transformer = new TransformerBlock(modelDim, numHeads);
                double[][] inputData = generateRandomMatrix(seqLen, modelDim);
                RealMatrix input = new Array2DRowRealMatrix(inputData);
                
                // Warm up
                transformer.forward(input);
                
                // Benchmark
                long startTime = System.nanoTime();
                RealMatrix output = transformer.forward(input);
                long endTime = System.nanoTime();
                
                double timeMs = (endTime - startTime) / 1_000_000.0;
                long approxMemory = (long) seqLen * modelDim * 8 * 4; // Rough estimate in bytes
                
                System.out.printf("%8d | %9d | %6d | %9.2f | %10s%n", 
                                seqLen, modelDim, numHeads, timeMs, 
                                formatBytes(approxMemory));
            }
        }
        
        System.out.println("\n📈 Performance Summary:");
        System.out.println("   • Larger sequences take proportionally more time");
        System.out.println("   • Higher model dimensions increase computation");
        System.out.println("   • Memory usage scales with sequence length × model dimension");
        System.out.println("   • All configurations complete successfully");
    }
    
    // Helper methods
    
    private static double[][] generateRandomMatrix(int rows, int cols) {
        double[][] matrix = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = (Math.random() - 0.5) * 2; // Range [-1, 1]
            }
        }
        return matrix;
    }
    
    private static RealMatrix createCausalMask(int seqLen) {
        double[][] maskData = new double[seqLen][seqLen];
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j <= i; j++) {
                maskData[i][j] = 1.0; // Allow attention to current and previous positions
            }
        }
        return new Array2DRowRealMatrix(maskData);
    }
    
    private static String formatBytes(long bytes) {
        if (bytes < 1024) return bytes + "B";
        if (bytes < 1024 * 1024) return String.format("%.1fKB", bytes / 1024.0);
        return String.format("%.1fMB", bytes / (1024.0 * 1024.0));
    }
}
