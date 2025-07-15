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

// Import visualization components
import org.superml.visualization.VisualizationFactory;
import org.superml.visualization.classification.ConfusionMatrix;

/**
 * SuperML Java 2.0.0 - Confusion Matrix Example
 * 
 * This example demonstrates:
 * 1. Multi-class classification with confusion matrix analysis
 * 2. Custom confusion matrix implementation
 * 3. Precision, Recall, and F1-Score calculations per class
 * 4. Classification metrics visualization
 */
public class ConfusionMatrixExample {
    
    public static void main(String[] args) {
        System.out.println("=== SuperML 2.0.0 - Confusion Matrix Example ===\n");
        
        try {
            // Generate multi-class dataset (3 classes)
            System.out.println("Generated 300 samples with 4 features for 3-class classification");
            double[][] X = generateSyntheticData(300, 4);
            int[] y = generateMultiClassLabels(300, 3);
            
            // Split data into training and testing
            int trainSize = (int)(X.length * 0.8);
            double[][] XTrain = new double[trainSize][];
            double[][] XTest = new double[X.length - trainSize][];
            int[] yTrain = new int[trainSize];
            int[] yTest = new int[X.length - trainSize];
            
            System.arraycopy(X, 0, XTrain, 0, trainSize);
            System.arraycopy(X, trainSize, XTest, 0, X.length - trainSize);
            System.arraycopy(y, 0, yTrain, 0, trainSize);
            System.arraycopy(y, trainSize, yTest, 0, X.length - trainSize);
            
            System.out.println("Training samples: " + XTrain.length);
            System.out.println("Test samples: " + XTest.length);
            
            // Train logistic regression model
            System.out.println("\nTraining Multi-class Logistic Regression...");
            LogisticRegression model = new LogisticRegression();
            model.fit(XTrain, convertToDouble(yTrain));
            
            // Make predictions
            double[] predictions = model.predict(XTest);
            int[] predInt = convertToInt(predictions);
            
            // Calculate basic accuracy
            double accuracy = calculateAccuracy(yTest, predInt);
            System.out.println("Overall Accuracy: " + String.format("%.3f", accuracy));
            
            // Create enhanced confusion matrix using visualization module
            System.out.println("\n📊 Enhanced Confusion Matrix Analysis");
            System.out.println("=====================================");
            
            String[] classNames = {"Class A", "Class B", "Class C"};
            ConfusionMatrix confMatrix = VisualizationFactory.createConfusionMatrix(yTest, predInt, classNames);
            confMatrix.setTitle("Multi-class Classification Results");
            confMatrix.display();
            
            // Show sample predictions with confidence
            System.out.println("=== Sample Predictions ===");
            for (int i = 0; i < Math.min(10, XTest.length); i++) {
                int pred = predInt[i];
                int act = yTest[i];
                String status = (pred == act) ? "✅" : "❌";
                System.out.println(String.format("Sample %d - Predicted: %s, Actual: %s %s", 
                    i + 1, classNames[pred], classNames[act], status));
            }
            
            System.out.println("\n✅ Enhanced confusion matrix example completed successfully!");
            System.out.println("🎉 Using SuperML Visualization Module for professional output!");
            
        } catch (Exception e) {
            System.err.println("❌ Error in confusion matrix example: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    // Helper methods
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
    
    private static int[] generateMultiClassLabels(int samples, int numClasses) {
        int[] labels = new int[samples];
        java.util.Random random = new java.util.Random(42);
        
        for (int i = 0; i < samples; i++) {
            // Generate labels based on some pattern to make them learnable
            double sum = 0;
            for (int j = 0; j < 4; j++) { // Assume 4 features
                sum += random.nextGaussian();
            }
            if (sum < -0.5) {
                labels[i] = 0;
            } else if (sum > 0.5) {
                labels[i] = 2;
            } else {
                labels[i] = 1;
            }
        }
        return labels;
    }
    
    private static double[] convertToDouble(int[] intArray) {
        double[] doubleArray = new double[intArray.length];
        for (int i = 0; i < intArray.length; i++) {
            doubleArray[i] = (double) intArray[i];
        }
        return doubleArray;
    }
    
    private static int[] convertToInt(double[] doubleArray) {
        int[] intArray = new int[doubleArray.length];
        for (int i = 0; i < doubleArray.length; i++) {
            intArray[i] = (int) Math.round(doubleArray[i]);
        }
        return intArray;
    }
    
    private static double calculateAccuracy(int[] actual, int[] predicted) {
        int correct = 0;
        for (int i = 0; i < actual.length; i++) {
            if (actual[i] == predicted[i]) {
                correct++;
            }
        }
        return (double) correct / actual.length;
    }
}
