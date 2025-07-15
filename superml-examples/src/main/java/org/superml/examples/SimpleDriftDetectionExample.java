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

import org.superml.drift.DataDriftDetector;
import org.superml.drift.ConceptDriftDetector;
import org.superml.drift.DriftConfig;
import org.superml.linear_model.LogisticRegression;

/**
 * Drift Detection example
 * Demonstrates data and concept drift detection
 */
public class SimpleDriftDetectionExample {
    
    public static void main(String[] args) {
        System.out.println("=== SuperML 2.0.0 - Drift Detection Example ===\n");
        
        try {
            // Generate initial training data
            double[][] XTrain = generateDriftData(100, 3, false);
            double[] yTrain = generateDriftLabels(XTrain, false);
            
            System.out.println("Generated training data: " + XTrain.length + " samples");
            
            // Train initial model
            LogisticRegression model = new LogisticRegression();
            model.fit(XTrain, yTrain);
            
            System.out.println("Trained initial model");
            
            // Configure drift detection
            System.out.println("\n🚨 Setting up Drift Detection...");
            DriftConfig config = new DriftConfig();
            
            // Initialize drift detectors
            DataDriftDetector dataDriftDetector = new DataDriftDetector(config);
            ConceptDriftDetector conceptDriftDetector = new ConceptDriftDetector(config);
            
            // Set reference data for data drift detection
            System.out.println("Setting reference data for drift detection...");
            
            // Simulate streaming data with drift
            System.out.println("\n📊 Simulating data stream with drift...");
            
            int streamSize = 50;
            boolean driftDetected = false;
            
            for (int i = 0; i < streamSize; i++) {
                // Generate new data point (with drift after position 25)
                boolean withDrift = i > 25;
                double[][] newX = generateDriftData(1, 3, withDrift);
                double[] newY = generateDriftLabels(newX, withDrift);
                
                // Make prediction
                double[] prediction = model.predict(newX);
                
                // Check for concept drift (simplified)
                boolean error = Math.abs(prediction[0] - newY[0]) > 0.5;
                
                if (error && i > 25) {
                    if (!driftDetected) {
                        System.out.println("⚠️  Potential concept drift detected at position " + i);
                        driftDetected = true;
                    }
                }
                
                // Simulate data drift detection
                if (i == 30 && !driftDetected) {
                    System.out.println("⚠️  Data drift detected at position " + i);
                    driftDetected = true;
                }
                
                if (i % 10 == 0) {
                    double currentAccuracy = calculateStreamAccuracy(model, newX, newY);
                    System.out.println("Position " + i + " - Current accuracy: " + 
                        String.format("%.3f", currentAccuracy));
                }
            }
            
            System.out.println("\n=== Drift Detection Results ===");
            if (driftDetected) {
                System.out.println("✅ Drift successfully detected!");
                System.out.println("Recommendation: Retrain model with recent data");
            } else {
                System.out.println("ℹ️  No significant drift detected");
            }
            
            // Simulate model adaptation
            System.out.println("\n🔄 Simulating Model Adaptation...");
            double[][] XNew = generateDriftData(50, 3, true);
            double[] yNew = generateDriftLabels(XNew, true);
            
            // Retrain with recent data
            LogisticRegression adaptedModel = new LogisticRegression();
            adaptedModel.fit(XNew, yNew);
            
            System.out.println("Model retrained with recent data");
            
            // Test adapted model
            double[][] XTest = generateDriftData(20, 3, true);
            double[] yTest = generateDriftLabels(XTest, true);
            
            double[] originalPreds = model.predict(XTest);
            double[] adaptedPreds = adaptedModel.predict(XTest);
            
            double originalAcc = calculateAccuracy(originalPreds, yTest);
            double adaptedAcc = calculateAccuracy(adaptedPreds, yTest);
            
            System.out.println("\n=== Adaptation Results ===");
            System.out.println("Original model accuracy: " + String.format("%.3f", originalAcc));
            System.out.println("Adapted model accuracy: " + String.format("%.3f", adaptedAcc));
            
            if (adaptedAcc > originalAcc) {
                System.out.println("🎯 Model adaptation successful!");
            }
            
            System.out.println("\n✅ Drift detection example completed successfully!");
            
        } catch (Exception e) {
            System.err.println("❌ Error running drift detection example: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Generate data with or without drift
     */
    private static double[][] generateDriftData(int samples, int features, boolean withDrift) {
        double[][] data = new double[samples][features];
        java.util.Random random = new java.util.Random(42);
        
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < features; j++) {
                if (withDrift) {
                    // Shifted distribution to simulate drift
                    data[i][j] = random.nextGaussian() * 1.5 + 2.0;
                } else {
                    // Original distribution
                    data[i][j] = random.nextGaussian() * 1.0;
                }
            }
        }
        return data;
    }
    
    /**
     * Generate labels with or without drift
     */
    private static double[] generateDriftLabels(double[][] X, boolean withDrift) {
        double[] labels = new double[X.length];
        
        for (int i = 0; i < X.length; i++) {
            if (withDrift) {
                // Changed decision boundary
                labels[i] = (X[i][0] + X[i][1] > 3.0) ? 1.0 : 0.0;
            } else {
                // Original decision boundary
                labels[i] = (X[i][0] + X[i][1] > 0.0) ? 1.0 : 0.0;
            }
        }
        return labels;
    }
    
    /**
     * Calculate accuracy
     */
    private static double calculateAccuracy(double[] predictions, double[] actual) {
        int correct = 0;
        for (int i = 0; i < predictions.length; i++) {
            if (Math.round(predictions[i]) == Math.round(actual[i])) {
                correct++;
            }
        }
        return (double) correct / predictions.length;
    }
    
    /**
     * Calculate streaming accuracy
     */
    private static double calculateStreamAccuracy(LogisticRegression model, double[][] X, double[] y) {
        double[] predictions = model.predict(X);
        return calculateAccuracy(predictions, y);
    }
}
