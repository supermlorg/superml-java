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

package org.superml.examples;

import org.superml.inference.InferenceEngine;
import org.superml.inference.BatchInferenceProcessor;
import org.superml.linear_model.LogisticRegression;

/**
 * Inference example
 * Demonstrates model inference capabilities
 */
public class InferenceExample {
    
    public static void main(String[] args) {
        System.out.println("=== SuperML 2.0.0 - Inference Example ===\n");
        
        try {
            // Generate training data
            double[][] XTrain = generateInferenceData(100, 4);
            double[] yTrain = generateInferenceLabels(XTrain);
            
            System.out.println("Generated training data: " + XTrain.length + " samples");
            
            // Train model
            LogisticRegression model = new LogisticRegression();
            model.fit(XTrain, yTrain);
            System.out.println("Model trained successfully");
            
            // Create inference engine
            System.out.println("\n🔄 Setting up Inference Engine...");
            InferenceEngine inferenceEngine = new InferenceEngine();
            
            // Single prediction inference
            System.out.println("\n📊 Single Prediction Inference:");
            double[][] singleSample = generateInferenceData(1, 4);
            
            long startTime = System.nanoTime();
            double[] singlePrediction = model.predict(singleSample);
            long singleInferenceTime = System.nanoTime() - startTime;
            
            System.out.println("Single prediction: " + Math.round(singlePrediction[0]));
            System.out.println("Inference time: " + (singleInferenceTime / 1_000_000.0) + " ms");
            
            // Batch inference
            System.out.println("\n📈 Batch Inference:");
            double[][] batchData = generateInferenceData(50, 4);
            
            startTime = System.nanoTime();
            double[] batchPredictions = model.predict(batchData);
            long batchInferenceTime = System.nanoTime() - startTime;
            
            System.out.println("Batch size: " + batchData.length);
            System.out.println("Batch inference time: " + (batchInferenceTime / 1_000_000.0) + " ms");
            System.out.println("Average time per sample: " + 
                (batchInferenceTime / 1_000_000.0 / batchData.length) + " ms");
            
            // Performance comparison
            System.out.println("\n⚡ Performance Analysis:");
            double speedup = (singleInferenceTime * batchData.length) / (double) batchInferenceTime;
            System.out.println("Batch processing speedup: " + String.format("%.2fx", speedup));
            
            // Stream inference simulation
            System.out.println("\n🌊 Stream Inference Simulation:");
            BatchInferenceProcessor batchProcessor = new BatchInferenceProcessor(inferenceEngine);
            
            int streamSize = 20;
            int processed = 0;
            
            for (int i = 0; i < streamSize; i += 5) {
                double[][] streamBatch = generateInferenceData(Math.min(5, streamSize - i), 4);
                
                startTime = System.nanoTime();
                double[] streamPredictions = model.predict(streamBatch);
                long streamTime = System.nanoTime() - startTime;
                
                processed += streamBatch.length;
                System.out.println("Processed batch " + ((i/5) + 1) + 
                    " (" + streamBatch.length + " samples) in " + 
                    (streamTime / 1_000_000.0) + " ms");
            }
            
            System.out.println("Total stream samples processed: " + processed);
            
            // Memory usage simulation
            System.out.println("\n💾 Memory Usage Analysis:");
            Runtime runtime = Runtime.getRuntime();
            long memoryBefore = runtime.totalMemory() - runtime.freeMemory();
            
            // Large batch processing
            double[][] largeBatch = generateInferenceData(500, 4);
            double[] largePredictions = model.predict(largeBatch);
            
            long memoryAfter = runtime.totalMemory() - runtime.freeMemory();
            long memoryUsed = memoryAfter - memoryBefore;
            
            System.out.println("Large batch size: " + largeBatch.length);
            System.out.println("Memory used: " + (memoryUsed / 1024.0) + " KB");
            System.out.println("Memory per sample: " + (memoryUsed / (double) largeBatch.length) + " bytes");
            
            // Inference statistics
            System.out.println("\n📊 Inference Statistics:");
            int positiveCount = 0;
            for (double pred : largePredictions) {
                if (Math.round(pred) == 1) positiveCount++;
            }
            
            System.out.println("Total predictions: " + largePredictions.length);
            System.out.println("Positive predictions: " + positiveCount);
            System.out.println("Negative predictions: " + (largePredictions.length - positiveCount));
            System.out.println("Positive rate: " + 
                String.format("%.3f", (double) positiveCount / largePredictions.length));
            
            System.out.println("\n✅ Inference example completed successfully!");
            
        } catch (Exception e) {
            System.err.println("❌ Error running inference example: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Generate synthetic data for inference
     */
    private static double[][] generateInferenceData(int samples, int features) {
        double[][] data = new double[samples][features];
        java.util.Random random = new java.util.Random(42);
        
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < features; j++) {
                data[i][j] = random.nextGaussian() * 2.0 + j * 0.5;
            }
        }
        return data;
    }
    
    /**
     * Generate labels for inference data
     */
    private static double[] generateInferenceLabels(double[][] X) {
        double[] labels = new double[X.length];
        
        for (int i = 0; i < X.length; i++) {
            // Simple decision rule
            double sum = 0.0;
            for (int j = 0; j < X[i].length; j++) {
                sum += X[i][j];
            }
            labels[i] = sum > 3.0 ? 1.0 : 0.0;
        }
        return labels;
    }
}
