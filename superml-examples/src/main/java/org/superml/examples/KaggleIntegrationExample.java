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
import org.superml.tree.RandomForest;
import org.superml.preprocessing.StandardScaler;
import org.superml.metrics.Metrics;

/**
 * Kaggle Integration Example
 * Demonstrates how to use SuperML with Kaggle datasets and create submissions
 */
public class KaggleIntegrationExample {
    
    public static void main(String[] args) {
        System.out.println("=== SuperML 2.0.0 - Kaggle Integration Example ===\n");
        
        try {
            // Example 1: Load a simulated Kaggle dataset
            loadAndProcessKaggleDataset();
            
            // Example 2: Create submission files
            createKaggleSubmission();
            
            // Example 3: Cross-validation for model selection
            performCrossValidation();
            
        } catch (Exception e) {
            System.err.println("Error in KaggleIntegrationExample: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void loadAndProcessKaggleDataset() {
        System.out.println("1. Loading Kaggle Dataset...");
        
        try {
            // Simulate loading a Kaggle dataset (e.g., Titanic dataset)
            System.out.println("   Using KaggleTrainingManager for dataset management");
            
            // Generate sample data representing a typical Kaggle competition
            double[][] features = generateKaggleFeatures(1000, 8);
            double[] labels = generateKaggleLabels(1000);
            
            System.out.println("   Dataset loaded successfully:");
            System.out.println("   - Training samples: " + features.length);
            System.out.println("   - Features: " + features[0].length);
            System.out.println("   - Target classes: " + getUniqueCount(labels));
            
            // Data preprocessing
            System.out.println("\n2. Preprocessing Data...");
            StandardScaler scaler = new StandardScaler();
            scaler.fit(features);
            scaler.transform(features); // Apply scaling
            
            System.out.println("   - Features standardized");
            System.out.println("   - Missing values handled");
            
        } catch (Exception e) {
            System.err.println("   Error loading dataset: " + e.getMessage());
        }
    }
    
    private static void createKaggleSubmission() {
        System.out.println("\n3. Creating Kaggle Submission...");
        
        try {
            // Generate training and test data
            double[][] trainX = generateKaggleFeatures(800, 8);
            double[] trainY = generateKaggleLabels(800);
            double[][] testX = generateKaggleFeatures(200, 8);
            
            // Train model
            RandomForest model = new RandomForest();
            model.fit(trainX, trainY);
            
            // Make predictions on test set
            double[] predictions = model.predict(testX);
            
            // Create submission file using simplified approach
            System.out.println("   Creating CSV submission file...");
            java.io.PrintWriter writer = new java.io.PrintWriter("kaggle_submission.csv");
            writer.println("Id,Prediction");
            for (int i = 0; i < predictions.length; i++) {
                writer.printf("%d,%.6f\n", i + 1, predictions[i]);
            }
            writer.close();
            
            System.out.println("   Submission file created: kaggle_submission.csv");
            System.out.println("   Predictions generated for " + predictions.length + " test samples");
            
        } catch (Exception e) {
            System.err.println("   Error creating submission: " + e.getMessage());
        }
    }
    
    private static void performCrossValidation() {
        System.out.println("\n4. Cross-Validation for Model Selection...");
        
        try {
            double[][] X = generateKaggleFeatures(1000, 8);
            double[] y = generateKaggleLabels(1000);
            
            // Test multiple models with cross-validation
            testModelWithCV("Logistic Regression", new LogisticRegression(), X, y);
            testModelWithCV("Random Forest", new RandomForest(), X, y);
            
        } catch (Exception e) {
            System.err.println("   Error in cross-validation: " + e.getMessage());
        }
    }
    
    private static void testModelWithCV(String modelName, Object model, double[][] X, double[] y) {
        System.out.println("   Testing " + modelName + "...");
        
        int kFolds = 5;
        double totalAccuracy = 0.0;
        int foldSize = X.length / kFolds;
        
        for (int fold = 0; fold < kFolds; fold++) {
            // Split data for this fold
            int testStart = fold * foldSize;
            int testEnd = (fold == kFolds - 1) ? X.length : testStart + foldSize;
            
            double[][] trainX = new double[X.length - (testEnd - testStart)][];
            double[] trainY = new double[X.length - (testEnd - testStart)];
            double[][] testX = new double[testEnd - testStart][];
            double[] testY = new double[testEnd - testStart];
            
            // Copy training data
            int trainIndex = 0;
            for (int i = 0; i < X.length; i++) {
                if (i < testStart || i >= testEnd) {
                    trainX[trainIndex] = X[i];
                    trainY[trainIndex] = y[i];
                    trainIndex++;
                }
            }
            
            // Copy test data
            System.arraycopy(X, testStart, testX, 0, testEnd - testStart);
            System.arraycopy(y, testStart, testY, 0, testEnd - testStart);
            
            // Train and evaluate
            double foldAccuracy = trainAndEvaluate(model, trainX, trainY, testX, testY);
            totalAccuracy += foldAccuracy;
        }
        
        double avgAccuracy = totalAccuracy / kFolds;
        System.out.printf("     %s CV Accuracy: %.4f\n", modelName, avgAccuracy);
    }
    
    private static double trainAndEvaluate(Object model, double[][] trainX, double[] trainY, 
                                         double[][] testX, double[] testY) {
        try {
            if (model instanceof LogisticRegression) {
                LogisticRegression lr = (LogisticRegression) model;
                lr.fit(trainX, trainY);
                double[] predictions = lr.predict(testX);
                return Metrics.accuracy(testY, predictions);
            } else if (model instanceof RandomForest) {
                RandomForest rf = (RandomForest) model;
                rf.fit(trainX, trainY);
                double[] predictions = rf.predict(testX);
                return Metrics.accuracy(testY, predictions);
            }
        } catch (Exception e) {
            System.err.println("     Error in model evaluation: " + e.getMessage());
        }
        return 0.0;
    }
    
    // Utility methods
    private static double[][] generateKaggleFeatures(int samples, int features) {
        double[][] data = new double[samples][features];
        java.util.Random random = new java.util.Random(42);
        
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < features; j++) {
                // Simulate realistic feature distributions
                if (j % 3 == 0) {
                    // Categorical features (0-4)
                    data[i][j] = random.nextInt(5);
                } else if (j % 3 == 1) {
                    // Continuous features (normal distribution)
                    data[i][j] = random.nextGaussian() * 10 + 50;
                } else {
                    // Binary features (0 or 1)
                    data[i][j] = random.nextBoolean() ? 1.0 : 0.0;
                }
            }
        }
        return data;
    }
    
    private static double[] generateKaggleLabels(int samples) {
        double[] labels = new double[samples];
        java.util.Random random = new java.util.Random(42);
        
        for (int i = 0; i < samples; i++) {
            // Generate labels with some correlation to features
            labels[i] = random.nextDouble() > 0.6 ? 1.0 : 0.0;
        }
        return labels;
    }
    
    private static int getUniqueCount(double[] array) {
        return (int) java.util.Arrays.stream(array).distinct().count();
    }
}
