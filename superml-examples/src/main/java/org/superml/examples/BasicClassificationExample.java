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
import org.superml.tree.DecisionTree;
import org.superml.tree.RandomForest;
import org.superml.preprocessing.StandardScaler;
import org.superml.metrics.Metrics;

/**
 * Basic Classification Example
 * Demonstrates various classification algorithms in SuperML Java 2.0.0
 */
public class BasicClassificationExample {
    
    public static void main(String[] args) {
        System.out.println("=== SuperML 2.0.0 - Basic Classification Example ===\n");
        
        try {
            // Generate synthetic classification dataset
            System.out.println("1. Loading Dataset...");
            double[][] X = generateClassificationData(200, 4);
            double[] y = generateClassificationLabels(200);
            
            System.out.println("   Features: " + X[0].length);
            System.out.println("   Samples: " + X.length);
            System.out.println("   Classes: " + getUniqueClasses(y).length);
            
            // Split data (80% train, 20% test)
            System.out.println("\n2. Splitting Data (80% train, 20% test)...");
            int trainSize = (int)(X.length * 0.8);
            double[][] XTrain = new double[trainSize][];
            double[][] XTest = new double[X.length - trainSize][];
            double[] yTrain = new double[trainSize];
            double[] yTest = new double[X.length - trainSize];
            
            System.arraycopy(X, 0, XTrain, 0, trainSize);
            System.arraycopy(X, trainSize, XTest, 0, X.length - trainSize);
            System.arraycopy(y, 0, yTrain, 0, trainSize);
            System.arraycopy(y, trainSize, yTest, 0, X.length - trainSize);
            
            // Preprocessing - Scale features
            System.out.println("\n3. Preprocessing - Standardizing Features...");
            StandardScaler scaler = new StandardScaler();
            scaler.fit(XTrain);
            double[][] XTrainScaled = scaler.transform(XTrain);
            double[][] XTestScaled = scaler.transform(XTest);
            
            // Test multiple classifiers
            testLogisticRegression(XTrainScaled, yTrain, XTestScaled, yTest);
            testDecisionTree(XTrain, yTrain, XTest, yTest); // Trees don't need scaling
            testRandomForest(XTrain, yTrain, XTest, yTest);
            
        } catch (Exception e) {
            System.err.println("Error in BasicClassificationExample: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void testLogisticRegression(double[][] XTrain, double[] yTrain, 
                                             double[][] XTest, double[] yTest) {
        System.out.println("\n4. Training Logistic Regression...");
        
        try {
            LogisticRegression lr = new LogisticRegression();
            lr.setMaxIter(1000);
            lr.setLearningRate(0.01);
            
            long startTime = System.currentTimeMillis();
            lr.fit(XTrain, yTrain);
            long trainTime = System.currentTimeMillis() - startTime;
            
            System.out.println("   Training completed in " + trainTime + "ms");
            
            // Make predictions
            double[] predictions = lr.predict(XTest);
            
            // Calculate metrics
            double accuracy = Metrics.accuracy(yTest, predictions);
            double precision = Metrics.precision(yTest, predictions);
            double recall = Metrics.recall(yTest, predictions);
            double f1 = Metrics.f1Score(yTest, predictions);
            
            System.out.println("   Logistic Regression Results:");
            System.out.printf("   Accuracy:  %.4f\n", accuracy);
            System.out.printf("   Precision: %.4f\n", precision);
            System.out.printf("   Recall:    %.4f\n", recall);
            System.out.printf("   F1-Score:  %.4f\n", f1);
            
        } catch (Exception e) {
            System.err.println("   Error in Logistic Regression: " + e.getMessage());
        }
    }
    
    private static void testDecisionTree(double[][] XTrain, double[] yTrain, 
                                       double[][] XTest, double[] yTest) {
        System.out.println("\n5. Training Decision Tree...");
        
        try {
            DecisionTree dt = new DecisionTree();
            
            long startTime = System.currentTimeMillis();
            dt.fit(XTrain, yTrain);
            long trainTime = System.currentTimeMillis() - startTime;
            
            System.out.println("   Training completed in " + trainTime + "ms");
            
            // Make predictions
            double[] predictions = dt.predict(XTest);
            
            // Calculate metrics
            double accuracy = Metrics.accuracy(yTest, predictions);
            
            System.out.println("   Decision Tree Results:");
            System.out.printf("   Accuracy:  %.4f\n", accuracy);
            
        } catch (Exception e) {
            System.err.println("   Error in Decision Tree: " + e.getMessage());
        }
    }
    
    private static void testRandomForest(double[][] XTrain, double[] yTrain, 
                                       double[][] XTest, double[] yTest) {
        System.out.println("\n6. Training Random Forest...");
        
        try {
            RandomForest rf = new RandomForest();
            
            long startTime = System.currentTimeMillis();
            rf.fit(XTrain, yTrain);
            long trainTime = System.currentTimeMillis() - startTime;
            
            System.out.println("   Training completed in " + trainTime + "ms");
            
            // Make predictions
            double[] predictions = rf.predict(XTest);
            
            // Calculate metrics
            double accuracy = Metrics.accuracy(yTest, predictions);
            
            System.out.println("   Random Forest Results:");
            System.out.printf("   Accuracy:  %.4f\n", accuracy);
            
        } catch (Exception e) {
            System.err.println("   Error in Random Forest: " + e.getMessage());
        }
    }
    
    // Utility methods for data generation
    private static double[][] generateClassificationData(int nSamples, int nFeatures) {
        double[][] data = new double[nSamples][nFeatures];
        java.util.Random random = new java.util.Random(42);
        
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                data[i][j] = random.nextGaussian() * 2.0 + (i % 2 == 0 ? 1.0 : -1.0);
            }
        }
        return data;
    }
    
    private static double[] generateClassificationLabels(int nSamples) {
        double[] labels = new double[nSamples];
        java.util.Random random = new java.util.Random(42);
        
        for (int i = 0; i < nSamples; i++) {
            labels[i] = random.nextBoolean() ? 1.0 : 0.0;
        }
        return labels;
    }
    
    private static double[] getUniqueClasses(double[] labels) {
        return java.util.Arrays.stream(labels).distinct().toArray();
    }
}
