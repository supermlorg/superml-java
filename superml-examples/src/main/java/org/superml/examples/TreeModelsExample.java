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

import org.superml.tree.DecisionTree;
import org.superml.tree.RandomForest;

/**
 * Tree-based algorithms example
 * Demonstrates decision trees and random forests
 */
public class TreeModelsExample {
    
    public static void main(String[] args) {
        System.out.println("=== SuperML 2.0.0 - Tree Models Example ===\n");
        
        try {
            // Generate synthetic data for classification
            double[][] X = generateTreeData(200, 5);
            double[] y = generateTreeLabels(X);
            
            System.out.println("Generated " + X.length + " samples with " + X[0].length + " features");
            
            // Split data
            int trainSize = (int)(X.length * 0.8);
            double[][] XTrain = new double[trainSize][];
            double[][] XTest = new double[X.length - trainSize][];
            double[] yTrain = new double[trainSize];
            double[] yTest = new double[X.length - trainSize];
            
            System.arraycopy(X, 0, XTrain, 0, trainSize);
            System.arraycopy(X, trainSize, XTest, 0, X.length - trainSize);
            System.arraycopy(y, 0, yTrain, 0, trainSize);
            System.arraycopy(y, trainSize, yTest, 0, X.length - trainSize);
            
            // 1. Decision Tree Example
            System.out.println("\n🌳 Training Decision Tree...");
            DecisionTree dt = new DecisionTree();
            dt.fit(XTrain, yTrain);
            
            double[] dtPredictions = dt.predict(XTest);
            double dtAccuracy = calculateAccuracy(dtPredictions, yTest);
            
            System.out.println("Decision Tree Accuracy: " + String.format("%.3f", dtAccuracy));
            
            // 2. Random Forest Example
            System.out.println("\n🌲🌲🌲 Training Random Forest...");
            RandomForest rf = new RandomForest();
            rf.fit(XTrain, yTrain);
            
            double[] rfPredictions = rf.predict(XTest);
            double rfAccuracy = calculateAccuracy(rfPredictions, yTest);
            
            System.out.println("Random Forest Accuracy: " + String.format("%.3f", rfAccuracy));
            
            // Compare models
            System.out.println("\n=== Model Comparison ===");
            System.out.println("Decision Tree: " + String.format("%.3f", dtAccuracy));
            System.out.println("Random Forest: " + String.format("%.3f", rfAccuracy));
            
            if (rfAccuracy > dtAccuracy) {
                System.out.println("🏆 Random Forest performs better!");
            } else {
                System.out.println("🏆 Decision Tree performs better!");
            }
            
            // Show some predictions
            System.out.println("\n=== Sample Predictions ===");
            for (int i = 0; i < Math.min(5, dtPredictions.length); i++) {
                System.out.println("Sample " + (i+1) + 
                    " - DT: " + Math.round(dtPredictions[i]) + 
                    ", RF: " + Math.round(rfPredictions[i]) + 
                    ", Actual: " + Math.round(yTest[i]));
            }
            
            System.out.println("\n✅ Tree models example completed successfully!");
            
        } catch (Exception e) {
            System.err.println("❌ Error running tree models example: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Generate synthetic data with tree-friendly features
     */
    private static double[][] generateTreeData(int samples, int features) {
        double[][] data = new double[samples][features];
        java.util.Random random = new java.util.Random(42);
        
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < features; j++) {
                // Mix of continuous and discrete-like features
                if (j % 2 == 0) {
                    data[i][j] = random.nextGaussian() * 2.0;
                } else {
                    data[i][j] = random.nextInt(10); // More discrete
                }
            }
        }
        return data;
    }
    
    /**
     * Generate labels based on tree-like rules
     */
    private static double[] generateTreeLabels(double[][] X) {
        double[] labels = new double[X.length];
        
        for (int i = 0; i < X.length; i++) {
            // Simple rule-based labeling (tree-friendly)
            if (X[i][0] > 0 && X[i][1] > 5) {
                labels[i] = 1.0;
            } else if (X[i][2] < -1 || X[i][3] > 8) {
                labels[i] = 2.0;
            } else {
                labels[i] = 0.0;
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
}
