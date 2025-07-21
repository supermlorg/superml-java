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

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test class for MultiHeadAttention.
 */
public class MultiHeadAttentionTest {
    
    private MultiHeadAttention attention;
    
    @BeforeEach
    void setUp() {
        // Create a multi-head attention with 8 heads and model dimension of 512
        attention = new MultiHeadAttention(512, 8);
    }
    
    @Test
    void testConstructorValid() {
        assertNotNull(attention);
        assertEquals(512, attention.getModelDim());
        assertEquals(8, attention.getNumHeads());
        assertEquals(64, attention.getHeadDim()); // 512 / 8 = 64
    }
    
    @Test
    void testConstructorInvalidDimensions() {
        // Model dimension not divisible by number of heads
        assertThrows(IllegalArgumentException.class, () -> {
            new MultiHeadAttention(100, 3);
        });
    }
    
    @Test
    void testForwardPass() {
        // Create dummy input: batch_size=2, seq_len=10, model_dim=512
        // Input matrix: (batch_size * seq_len, model_dim) = (20, 512)
        double[][] inputData = new double[20][512];
        
        // Fill with random values
        for (int i = 0; i < 20; i++) {
            for (int j = 0; j < 512; j++) {
                inputData[i][j] = Math.random() - 0.5; // Random values between -0.5 and 0.5
            }
        }
        
        RealMatrix input = new Array2DRowRealMatrix(inputData);
        RealMatrix output = attention.forward(input, null);
        
        assertNotNull(output);
        assertEquals(input.getRowDimension(), output.getRowDimension()); // Same number of rows
        assertEquals(input.getColumnDimension(), output.getColumnDimension()); // Same number of columns
    }
    
    @Test
    void testForwardPassWithMask() {
        // Create dummy input
        double[][] inputData = new double[10][512];
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 512; j++) {
                inputData[i][j] = Math.random() - 0.5;
            }
        }
        RealMatrix input = new Array2DRowRealMatrix(inputData);
        
        // Create attention mask (1s for valid positions, 0s for masked)
        double[][] maskData = new double[10][10];
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j <= i; j++) { // Lower triangular mask
                maskData[i][j] = 1.0;
            }
        }
        RealMatrix mask = new Array2DRowRealMatrix(maskData);
        
        RealMatrix output = attention.forward(input, mask);
        
        assertNotNull(output);
        assertEquals(input.getRowDimension(), output.getRowDimension());
        assertEquals(input.getColumnDimension(), output.getColumnDimension());
    }
    
    @Test
    void testFluentConfiguration() {
        MultiHeadAttention configured = new MultiHeadAttention(256, 4)
                .setUseBias(false)
                .setDropout(0.2);
        
        assertEquals(256, configured.getModelDim());
        assertEquals(4, configured.getNumHeads());
        assertEquals(64, configured.getHeadDim());
        assertFalse(configured.getUseBias());
        assertEquals(0.2, configured.getDropout(), 1e-6);
    }
    
    @Test
    void testSetNumHeadsReconfiguration() {
        // Change number of heads
        attention.setNumHeads(16);
        
        assertEquals(16, attention.getNumHeads());
        assertEquals(32, attention.getHeadDim()); // 512 / 16 = 32
        
        // Test that it still works after reconfiguration
        double[][] inputData = new double[5][512];
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 512; j++) {
                inputData[i][j] = Math.random();
            }
        }
        
        RealMatrix input = new Array2DRowRealMatrix(inputData);
        RealMatrix output = attention.forward(input, null);
        
        assertNotNull(output);
        assertEquals(5, output.getRowDimension());
        assertEquals(512, output.getColumnDimension());
    }
    
    @Test
    void testToString() {
        String description = attention.toString();
        assertTrue(description.contains("MultiHeadAttention"));
        assertTrue(description.contains("512"));
        assertTrue(description.contains("8"));
    }
    
    @Test
    void testAttentionOutputShape() {
        // Test different sequence lengths
        int[] sequenceLengths = {1, 5, 10, 32};
        
        for (int seqLen : sequenceLengths) {
            double[][] inputData = new double[seqLen][512];
            for (int i = 0; i < seqLen; i++) {
                for (int j = 0; j < 512; j++) {
                    inputData[i][j] = Math.random();
                }
            }
            
            RealMatrix input = new Array2DRowRealMatrix(inputData);
            RealMatrix output = attention.forward(input, null);
            
            assertEquals(seqLen, output.getRowDimension(), 
                        "Output sequence length should match input for seqLen=" + seqLen);
            assertEquals(512, output.getColumnDimension(),
                        "Output model dimension should be preserved for seqLen=" + seqLen);
        }
    }
}
