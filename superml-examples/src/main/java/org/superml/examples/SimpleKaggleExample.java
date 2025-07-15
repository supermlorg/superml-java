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

import org.superml.linear_model.LogisticRegression;

/**
 * Simple Kaggle-style competition example
 * Demonstrates basic ML workflow for competition submissions.
 */
public class SimpleKaggleExample {
    
    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("     SuperML Java - Kaggle-style Competition Example");
        System.out.println("=".repeat(60));
        
        try {
            // 1. Simulate competition dataset
            System.out.println("Loading competition dataset (synthetic)...");
            double[][] X = generateCompetitionData(150, 4);
            int[] yInt = generateCompetitionLabels(150);
            double[] y = toDoubleArray(yInt);
            
            System.out.printf("Competition data: %d samples, %d features\n", X.length, X[0].length);
            
            // 2. Split into train and validation
            int trainSize = (int)(X.length * 0.8);
            double[][] XTrain = new double[trainSize][];
            double[][] XVal = new double[X.length - trainSize][];
            double[] yTrain = new double[trainSize];
            double[] yVal = new double[X.length - trainSize];
            
            System.arraycopy(X, 0, XTrain, 0, trainSize);
            System.arraycopy(X, trainSize, XVal, 0, X.length - trainSize);
            System.arraycopy(y, 0, yTrain, 0, trainSize);
            System.arraycopy(y, trainSize, yVal, 0, X.length - trainSize);
            
            System.out.println("Training samples: " + XTrain.length);
            System.out.println("Validation samples: " + XVal.length);
            
            // 3. Train competition model
            System.out.println("\nTraining competition model...");
            LogisticRegression model = new LogisticRegression();
            model.fit(XTrain, yTrain);
            
            // 4. Validate model
            double[] predictions = model.predict(XVal);
            
            int correct = 0;
            for (int i = 0; i < predictions.length; i++) {
                if (Math.round(predictions[i]) == Math.round(yVal[i])) {
                    correct++;
                }
            }
            double accuracy = (double) correct / predictions.length;
            
            System.out.println("\n=== Competition Results ===");
            System.out.println("Validation Accuracy: " + String.format("%.3f", accuracy));
            System.out.println("Score for leaderboard: " + String.format("%.6f", accuracy));
            
            // 5. Generate submission format
            System.out.println("\n=== Generating Submission ===");
            System.out.println("id,prediction");
            for (int i = 0; i < Math.min(10, predictions.length); i++) {
                System.out.println((i + 1) + "," + Math.round(predictions[i]));
            }
            System.out.println("... (remaining " + (predictions.length - 10) + " predictions)");
            
            System.out.println("\n✅ Competition example completed successfully!");
            System.out.println("Ready for Kaggle submission!");
            
        } catch (Exception e) {
            System.err.println("❌ Error in competition workflow: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Convert int array to double array
     */
    private static double[] toDoubleArray(int[] intArray) {
        double[] doubleArray = new double[intArray.length];
        for (int i = 0; i < intArray.length; i++) {
            doubleArray[i] = intArray[i];
        }
        return doubleArray;
    }
    
    /**
     * Generate synthetic competition data
     */
    private static double[][] generateCompetitionData(int samples, int features) {
        double[][] data = new double[samples][features];
        java.util.Random random = new java.util.Random(42);
        
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < features; j++) {
                data[i][j] = random.nextGaussian() * 2.0 + j; // Some pattern
            }
        }
        return data;
    }
    
    /**
     * Generate synthetic competition labels (multi-class)
     */
    private static int[] generateCompetitionLabels(int samples) {
        int[] labels = new int[samples];
        java.util.Random random = new java.util.Random(42);
        
        for (int i = 0; i < samples; i++) {
            labels[i] = random.nextInt(3); // 3 classes: 0, 1, 2
        }
        return labels;
    }
}
