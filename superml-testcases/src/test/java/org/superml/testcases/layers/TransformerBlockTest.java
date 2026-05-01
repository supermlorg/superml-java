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

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test class for TransformerBlock integration.
 */
public class TransformerBlockTest {
    
    private TransformerBlock transformerBlock;
    
    @BeforeEach
    void setUp() {
        // Create transformer block: model_dim=256, num_heads=8, ff_dim=1024
        transformerBlock = new TransformerBlock(256, 8, 1024);
    }
    
    @Test
    void testConstructorValid() {
        assertNotNull(transformerBlock);
        assertEquals(256, transformerBlock.getModelDim());
        assertEquals(8, transformerBlock.getNumHeads());
        assertEquals(1024, transformerBlock.getFfDim());
    }
    
    @Test
    void testConstructorWithDefaultFFDim() {
        TransformerBlock block = new TransformerBlock(128, 4);
        assertEquals(128, block.getModelDim());
        assertEquals(4, block.getNumHeads());
        assertEquals(512, block.getFfDim()); // 4 * 128 = 512
    }
    
    @Test
    void testForwardPass() {
        // Create input: seq_len=10, model_dim=256
        double[][] inputData = new double[10][256];
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 256; j++) {
                inputData[i][j] = Math.random() - 0.5;
            }
        }
        
        RealMatrix input = new Array2DRowRealMatrix(inputData);
        RealMatrix output = transformerBlock.forward(input);
        
        assertNotNull(output);
        assertEquals(input.getRowDimension(), output.getRowDimension());
        assertEquals(input.getColumnDimension(), output.getColumnDimension());
        
        // Output should be different from input (transformation occurred)
        boolean different = false;
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 256; j++) {
                if (Math.abs(input.getEntry(i, j) - output.getEntry(i, j)) > 1e-6) {
                    different = true;
                    break;
                }
            }
            if (different) break;
        }
        assertTrue(different, "Output should be different from input");
    }
    
    @Test
    void testForwardPassWithMask() {
        // Create input
        double[][] inputData = new double[8][256];
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 256; j++) {
                inputData[i][j] = Math.random();
            }
        }
        RealMatrix input = new Array2DRowRealMatrix(inputData);
        
        // Create causal mask (lower triangular)
        double[][] maskData = new double[8][8];
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j <= i; j++) {
                maskData[i][j] = 1.0;
            }
        }
        RealMatrix mask = new Array2DRowRealMatrix(maskData);
        
        RealMatrix output = transformerBlock.forward(input, mask);
        
        assertNotNull(output);
        assertEquals(8, output.getRowDimension());
        assertEquals(256, output.getColumnDimension());
    }
    
    @Test
    void testMultipleForwardPasses() {
        // Test that multiple forward passes produce consistent results
        double[][] inputData = new double[5][256];
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 256; j++) {
                inputData[i][j] = Math.random();
            }
        }
        RealMatrix input = new Array2DRowRealMatrix(inputData);
        
        RealMatrix output1 = transformerBlock.forward(input);
        RealMatrix output2 = transformerBlock.forward(input);
        
        // Same input should produce same output (deterministic)
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 256; j++) {
                assertEquals(output1.getEntry(i, j), output2.getEntry(i, j), 1e-10,
                            "Multiple passes should be deterministic");
            }
        }
    }
    
    @Test
    void testFluentConfiguration() {
        TransformerBlock configured = new TransformerBlock(128, 4, 512)
                .setAttentionDropout(0.1)
                .setFeedForwardDropout(0.2)
                .setFeedForwardActivation("gelu")
                .setLayerNormEps(1e-6);
        
        assertEquals(128, configured.getModelDim());
        assertEquals(4, configured.getNumHeads());
        assertEquals(512, configured.getFfDim());
    }
    
    @Test
    void testComponentAccess() {
        assertNotNull(transformerBlock.getAttention());
        assertNotNull(transformerBlock.getFeedForward());
        assertNotNull(transformerBlock.getLayerNorm1());
        assertNotNull(transformerBlock.getLayerNorm2());
    }
    
    @Test
    void testToString() {
        String description = transformerBlock.toString();
        assertTrue(description.contains("TransformerBlock"));
        assertTrue(description.contains("256"));
        assertTrue(description.contains("8"));
        assertTrue(description.contains("1024"));
    }
    
    @Test
    void testDifferentSequenceLengths() {
        // Test various sequence lengths
        int[] sequenceLengths = {1, 3, 7, 16, 32};
        
        for (int seqLen : sequenceLengths) {
            double[][] inputData = new double[seqLen][256];
            for (int i = 0; i < seqLen; i++) {
                for (int j = 0; j < 256; j++) {
                    inputData[i][j] = Math.random();
                }
            }
            
            RealMatrix input = new Array2DRowRealMatrix(inputData);
            RealMatrix output = transformerBlock.forward(input);
            
            assertEquals(seqLen, output.getRowDimension(),
                        "Sequence length should be preserved for seqLen=" + seqLen);
            assertEquals(256, output.getColumnDimension(),
                        "Model dimension should be preserved for seqLen=" + seqLen);
        }
    }
}
