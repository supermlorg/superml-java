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

import org.superml.linear_model.LinearRegression;
import org.superml.utils.DataUtils;

/**
 * Simple regression example using basic SuperML classes
 */
public class SimpleRegressionExample {
    
    public static void main(String[] args) {
        System.out.println("=== SuperML 2.0.0 - Simple Regression Example ===\n");
        
        try {
            // Generate synthetic regression data
            double[][] X = generateSyntheticFeatures(100, 3);
            double[] y = generateSyntheticTarget(X);
            
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
            LinearRegression model = new LinearRegression();
            System.out.println("\nTraining Linear Regression model...");
            model.fit(XTrain, yTrain);
            
            // Make predictions
            double[] predictions = model.predict(XTest);
            
            // Calculate basic MSE
            double mse = 0.0;
            for (int i = 0; i < predictions.length; i++) {
                double error = predictions[i] - yTest[i];
                mse += error * error;
            }
            mse /= predictions.length;
            
            System.out.println("\n=== Results ===");
            System.out.println("Mean Squared Error: " + String.format("%.6f", mse));
            System.out.println("Root Mean Squared Error: " + String.format("%.6f", Math.sqrt(mse)));
            
            // Show some predictions
            System.out.println("\n=== Sample Predictions ===");
            for (int i = 0; i < Math.min(5, predictions.length); i++) {
                System.out.println("Sample " + (i+1) + 
                    " - Predicted: " + String.format("%.3f", predictions[i]) + 
                    ", Actual: " + String.format("%.3f", yTest[i]));
            }
            
            System.out.println("\n✅ Regression example completed successfully!");
            
        } catch (Exception e) {
            System.err.println("❌ Error running regression example: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Generate synthetic feature data
     */
    private static double[][] generateSyntheticFeatures(int samples, int features) {
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
     * Generate synthetic target values with some linear relationship
     */
    private static double[] generateSyntheticTarget(double[][] X) {
        double[] y = new double[X.length];
        java.util.Random random = new java.util.Random(42);
        double[] coefficients = {1.5, -2.0, 0.8}; // True coefficients
        
        for (int i = 0; i < X.length; i++) {
            y[i] = 0.0;
            for (int j = 0; j < X[i].length; j++) {
                y[i] += coefficients[j] * X[i][j];
            }
            y[i] += random.nextGaussian() * 0.1; // Add noise
        }
        return y;
    }
}
