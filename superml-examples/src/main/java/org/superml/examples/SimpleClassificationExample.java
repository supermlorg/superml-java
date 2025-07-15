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

import org.superml.core.Estimator;
import org.superml.linear_model.LogisticRegression;
import org.superml.datasets.Datasets;
import org.superml.utils.DataUtils;
import org.superml.metrics.Metrics;

/**
 * Simple classification example using basic SuperML classes
 */
public class SimpleClassificationExample {
    
    public static void main(String[] args) {
        System.out.println("=== SuperML 2.0.0 - Simple Classification Example ===\n");
        
        try {
            // Generate synthetic data
            double[][] X = generateSyntheticData(100, 4);
            int[] yInt = generateSyntheticLabels(100);
            double[] y = toDoubleArray(yInt);
            
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
            
            System.out.println("Training samples: " + XTrain.length);
            System.out.println("Test samples: " + XTest.length);
            
            // Create and train model
            LogisticRegression model = new LogisticRegression();
            System.out.println("\nTraining Logistic Regression model...");
            model.fit(XTrain, yTrain);
            
            // Make predictions
            double[] predictions = model.predict(XTest);
            
            // Calculate basic accuracy
            int correct = 0;
            for (int i = 0; i < predictions.length; i++) {
                if (Math.round(predictions[i]) == Math.round(yTest[i])) {
                    correct++;
                }
            }
            double accuracy = (double) correct / predictions.length;
            
            System.out.println("\n=== Results ===");
            System.out.println("Accuracy: " + String.format("%.3f", accuracy));
            System.out.println("Correct predictions: " + correct + "/" + predictions.length);
            
            // Show some predictions
            System.out.println("\n=== Sample Predictions ===");
            for (int i = 0; i < Math.min(5, predictions.length); i++) {
                System.out.println("Sample " + (i+1) + " - Predicted: " + Math.round(predictions[i]) + ", Actual: " + Math.round(yTest[i]));
            }
            
            System.out.println("\n✅ Classification example completed successfully!");
            
        } catch (Exception e) {
            System.err.println("❌ Error running classification example: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Generate synthetic feature data
     */
    private static double[][] generateSyntheticData(int samples, int features) {
        double[][] data = new double[samples][features];
        java.util.Random random = new java.util.Random(42);
        
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < features; j++) {
                data[i][j] = random.nextGaussian();
            }
        }
        return data;
    }
    
    /**
     * Generate synthetic binary labels
     */
    private static int[] generateSyntheticLabels(int samples) {
        int[] labels = new int[samples];
        java.util.Random random = new java.util.Random(42);
        
        for (int i = 0; i < samples; i++) {
            labels[i] = random.nextBoolean() ? 1 : 0;
        }
        return labels;
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
}
