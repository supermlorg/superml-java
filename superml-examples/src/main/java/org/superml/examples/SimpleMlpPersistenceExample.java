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

import org.superml.neural.MLPClassifier;
import org.superml.persistence.ModelPersistence;
import org.superml.inference.InferenceEngine;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

/**
 * Simple example demonstrating MLP model persistence and inference workflow.
 * This example shows the complete cycle from training to deployment.
 * 
 * @author SuperML Java Team
 */
public class SimpleMlpPersistenceExample {
    
    private static final Logger logger = LoggerFactory.getLogger(SimpleMlpPersistenceExample.class);
    private static final Random random = new Random(42);
    
    public static void main(String[] args) {
        System.out.println("=== Simple MLP Persistence & Inference Example ===\n");
        
        try {
            // Phase 1: Train and save model
            System.out.println("Phase 1: Training and Persisting MLP Model");
            System.out.println("=========================================");
            String modelPath = trainAndSaveModel();
            
            // Phase 2: Load and infer with model
            System.out.println("\nPhase 2: Loading and Inference");
            System.out.println("=============================");
            testModelInference(modelPath);
            
            System.out.println("\n✅ Example completed successfully!");
            
        } catch (Exception e) {
            logger.error("Example execution failed", e);
            e.printStackTrace();
        }
    }
    
    /**
     * Train an MLP classifier and save it
     */
    private static String trainAndSaveModel() {
        // Generate simple binary classification dataset
        double[][] X = generateData(500, 10);
        double[] y = generateLabels(500);
        
        // Create and configure MLP
        MLPClassifier mlp = new MLPClassifier()
            .setHiddenLayerSizes(32, 16)
            .setActivation("relu")
            .setLearningRate(0.01)
            .setMaxIter(100)
            .setBatchSize(32);
        
        // Train the model
        System.out.println("Training MLP with " + X.length + " samples, " + X[0].length + " features...");
        mlp.fit(X, y);
        
        // Evaluate performance
        double[] predictions = mlp.predict(X);
        double accuracy = calculateAccuracy(y, predictions);
        System.out.printf("Training accuracy: %.4f%n", accuracy);
        
        // Save model
        String modelPath = "models/simple_mlp.superml";
        String description = "Simple MLP classifier for demonstration";
        
        ModelPersistence.save(mlp, modelPath, description, null);
        System.out.println("✓ Model saved to: " + modelPath);
        
        return modelPath;
    }
    
    /**
     * Load the saved model and perform inference
     */
    private static void testModelInference(String modelPath) {
        // Create inference engine
        InferenceEngine engine = new InferenceEngine();
        
        // Load model
        System.out.println("Loading model from: " + modelPath);
        var modelInfo = engine.loadModel("simple_mlp", modelPath);
        System.out.println("✓ Model loaded: " + modelInfo.modelClass);
        
        // Generate test data
        double[] testSample = generateSample(10);
        System.out.println("Test sample: " + java.util.Arrays.toString(testSample));
        
        // Make prediction
        double prediction = engine.predict("simple_mlp", testSample);
        System.out.printf("Prediction: %.4f%n", prediction);
        
        // Test batch prediction
        double[][] batchSamples = {
            generateSample(10),
            generateSample(10),
            generateSample(10)
        };
        double[] batchPredictions = engine.predict("simple_mlp", batchSamples);
        System.out.println("Batch predictions: " + java.util.Arrays.toString(batchPredictions));
        
        // Show model metadata
        try {
            var metadata = ModelPersistence.getMetadata(modelPath);
            System.out.println("\nModel Metadata:");
            System.out.println("- Class: " + metadata.modelClass);
            System.out.println("- Saved: " + metadata.savedAt);
            System.out.println("- Description: " + metadata.description);
        } catch (Exception e) {
            logger.warn("Could not retrieve metadata", e);
        }
    }
    
    /**
     * Generate synthetic classification data
     */
    private static double[][] generateData(int samples, int features) {
        double[][] data = new double[samples][features];
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < features; j++) {
                data[i][j] = random.nextGaussian();
            }
        }
        return data;
    }
    
    /**
     * Generate binary labels
     */
    private static double[] generateLabels(int samples) {
        double[] labels = new double[samples];
        for (int i = 0; i < samples; i++) {
            labels[i] = random.nextDouble() > 0.5 ? 1.0 : 0.0;
        }
        return labels;
    }
    
    /**
     * Generate a single test sample
     */
    private static double[] generateSample(int features) {
        double[] sample = new double[features];
        for (int i = 0; i < features; i++) {
            sample[i] = random.nextGaussian();
        }
        return sample;
    }
    
    /**
     * Calculate classification accuracy
     */
    private static double calculateAccuracy(double[] actual, double[] predicted) {
        int correct = 0;
        for (int i = 0; i < actual.length; i++) {
            if (Math.round(predicted[i]) == Math.round(actual[i])) {
                correct++;
            }
        }
        return (double) correct / actual.length;
    }
}
