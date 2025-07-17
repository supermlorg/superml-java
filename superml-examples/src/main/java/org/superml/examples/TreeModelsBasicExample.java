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

import org.superml.tree.*;
import java.util.Random;

/**
 * Comprehensive example demonstrating SuperML tree algorithms
 * 
 * Shows usage of all major tree-based models:
 * - DecisionTree for classification and regression
 * - RandomForest for ensemble learning
 * - GradientBoosting for advanced ensemble methods
 * - XGBoost for state-of-the-art performance
 * 
 * NOTE: Advanced AutoTrainer and Visualization features will be available when dependency issues are resolved
 * 
 * @author SuperML Team
 */
public class TreeModelsBasicExample {
    
    public static void main(String[] args) {
        System.out.println("🌳 SuperML Tree Models Basic Example");
        System.out.println("===================================\n");
        
        // Test classification
        System.out.println("1. TREE-BASED CLASSIFICATION");
        System.out.println("============================");
        testClassification();
        
        System.out.println();
        
        // Test regression
        System.out.println("2. TREE-BASED REGRESSION");
        System.out.println("========================");
        testRegression();
        
        System.out.println();
        System.out.println("Advanced AutoTrainer and Visualization coming in next release!");
        System.out.println("🎯 Tree Models Basic Example Complete!");
    }
    
    private static void testClassification() {
        // Generate classification data
        DataGenerator generator = new DataGenerator();
        DataGenerator.ClassificationData data = generator.generateClassificationData(1000, 10, 3);
        
        // Split data
        int trainSize = (int)(data.X.length * 0.8);
        double[][] XTrain = new double[trainSize][];
        double[] yTrain = new double[trainSize];
        double[][] XTest = new double[data.X.length - trainSize][];
        double[] yTest = new double[data.X.length - trainSize];
        
        System.arraycopy(data.X, 0, XTrain, 0, trainSize);
        System.arraycopy(data.y, 0, yTrain, 0, trainSize);
        System.arraycopy(data.X, trainSize, XTest, 0, data.X.length - trainSize);
        System.arraycopy(data.y, trainSize, yTest, 0, data.X.length - trainSize);
        
        System.out.println("Dataset: " + data.X.length + " samples, " + data.X[0].length + " features, 3 classes");
        System.out.println("Train/Test split: " + trainSize + "/" + (data.X.length - trainSize));
        System.out.println();
        
        // Test DecisionTree
        System.out.println("--- Decision Tree Classifier ---");
        DecisionTree dt = new DecisionTree()
            .setMaxDepth(10)
            .setMinSamplesSplit(5);
        dt.fit(XTrain, yTrain);
        double dtScore = dt.score(XTest, yTest);
        System.out.printf("  Accuracy: %.4f\n", dtScore);
        
        // Test RandomForest
        System.out.println("--- Random Forest Classifier ---");
        RandomForest rf = new RandomForest()
            .setNEstimators(100)
            .setMaxDepth(10)
            .setRandomState(42);
        rf.fit(XTrain, yTrain);
        double rfScore = rf.score(XTest, yTest);
        System.out.printf("  Accuracy: %.4f\n", rfScore);
        
        // Test GradientBoosting
        System.out.println("--- Gradient Boosting Classifier ---");
        GradientBoosting gb = new GradientBoosting()
            .setNEstimators(100)
            .setLearningRate(0.1)
            .setMaxDepth(6);
        gb.fit(XTrain, yTrain);
        double gbScore = gb.score(XTest, yTest);
        System.out.printf("  Accuracy: %.4f\n", gbScore);
        
        // Test XGBoost
        System.out.println("--- XGBoost Classifier ---");
        XGBoost xgb = new XGBoost()
            .setNEstimators(100)
            .setLearningRate(0.1)
            .setMaxDepth(6)
            .setRandomState(42);
        xgb.fit(XTrain, yTrain);
        double xgbScore = xgb.score(XTest, yTest);
        System.out.printf("  Accuracy: %.4f\n", xgbScore);
        
        System.out.println();
        System.out.println("Performance Summary (Classification):");
        System.out.printf("  Decision Tree:     %.4f\n", dtScore);
        System.out.printf("  Random Forest:     %.4f\n", rfScore);
        System.out.printf("  Gradient Boosting: %.4f\n", gbScore);
        System.out.printf("  XGBoost:           %.4f\n", xgbScore);
    }
    
    private static void testRegression() {
        // Generate regression data
        DataGenerator generator = new DataGenerator();
        DataGenerator.RegressionData data = generator.generateRegressionData(800, 8);
        
        // Split data
        int trainSize = (int)(data.X.length * 0.8);
        double[][] XTrain = new double[trainSize][];
        double[] yTrain = new double[trainSize];
        double[][] XTest = new double[data.X.length - trainSize][];
        double[] yTest = new double[data.X.length - trainSize];
        
        System.arraycopy(data.X, 0, XTrain, 0, trainSize);
        System.arraycopy(data.y, 0, yTrain, 0, trainSize);
        System.arraycopy(data.X, trainSize, XTest, 0, data.X.length - trainSize);
        System.arraycopy(data.y, trainSize, yTest, 0, data.X.length - trainSize);
        
        System.out.println("Dataset: " + data.X.length + " samples, " + data.X[0].length + " features");
        System.out.println("Train/Test split: " + trainSize + "/" + (data.X.length - trainSize));
        System.out.println();
        
        // Test DecisionTree Regression
        System.out.println("--- Decision Tree Regressor ---");
        DecisionTree dtReg = new DecisionTree()
            .setMaxDepth(10)
            .setMinSamplesSplit(5);
        dtReg.fit(XTrain, yTrain);
        double dtR2 = dtReg.score(XTest, yTest);
        System.out.printf("  R² Score: %.4f\n", dtR2);
        
        // Test RandomForest Regression
        System.out.println("--- Random Forest Regressor ---");
        RandomForest rfReg = new RandomForest()
            .setNEstimators(100)
            .setMaxDepth(10)
            .setRandomState(42);
        rfReg.fit(XTrain, yTrain);
        double rfR2 = rfReg.score(XTest, yTest);
        System.out.printf("  R² Score: %.4f\n", rfR2);
        
        // Test GradientBoosting Regression
        System.out.println("--- Gradient Boosting Regressor ---");
        GradientBoosting gbReg = new GradientBoosting()
            .setNEstimators(100)
            .setLearningRate(0.1)
            .setMaxDepth(6);
        gbReg.fit(XTrain, yTrain);
        double gbR2 = gbReg.score(XTest, yTest);
        System.out.printf("  R² Score: %.4f\n", gbR2);
        
        // Test XGBoost Regression
        System.out.println("--- XGBoost Regressor ---");
        XGBoost xgbReg = new XGBoost()
            .setNEstimators(100)
            .setLearningRate(0.1)
            .setMaxDepth(6)
            .setRandomState(42);
        xgbReg.fit(XTrain, yTrain);
        double xgbR2 = xgbReg.score(XTest, yTest);
        System.out.printf("  R² Score: %.4f\n", xgbR2);
        
        System.out.println();
        System.out.println("Performance Summary (Regression):");
        System.out.printf("  Decision Tree:     %.4f\n", dtR2);
        System.out.printf("  Random Forest:     %.4f\n", rfR2);
        System.out.printf("  Gradient Boosting: %.4f\n", gbR2);
        System.out.printf("  XGBoost:           %.4f\n", xgbR2);
    }
    
    /**
     * Utility class for generating synthetic datasets
     */
    public static class DataGenerator {
        private Random random = new Random(42);
        
        public ClassificationData generateClassificationData(int nSamples, int nFeatures, int nClasses) {
            double[][] X = new double[nSamples][nFeatures];
            double[] y = new double[nSamples];
            
            // Generate features
            for (int i = 0; i < nSamples; i++) {
                for (int j = 0; j < nFeatures; j++) {
                    X[i][j] = random.nextGaussian();
                }
                
                // Generate labels based on linear combination of features
                double score = 0.0;
                for (int j = 0; j < Math.min(5, nFeatures); j++) {
                    score += X[i][j] * (j + 1) * 0.5;
                }
                
                // Create multiple classes
                if (score < -0.5) {
                    y[i] = 0.0;
                } else if (score > 0.5) {
                    y[i] = 2.0;
                } else {
                    y[i] = 1.0;
                }
            }
            
            return new ClassificationData(X, y);
        }
        
        public RegressionData generateRegressionData(int nSamples, int nFeatures) {
            double[][] X = new double[nSamples][nFeatures];
            double[] y = new double[nSamples];
            
            // Generate features
            for (int i = 0; i < nSamples; i++) {
                for (int j = 0; j < nFeatures; j++) {
                    X[i][j] = random.nextGaussian();
                }
                
                // Generate target as nonlinear combination with noise
                double target = 0.0;
                for (int j = 0; j < nFeatures; j++) {
                    target += X[i][j] * (j + 1) * 0.3;
                    // Add some nonlinearity
                    if (j < nFeatures / 2) {
                        target += Math.sin(X[i][j]) * 0.2;
                    }
                }
                target += random.nextGaussian() * 0.1; // Add noise
                y[i] = target;
            }
            
            return new RegressionData(X, y);
        }
        
        public static class ClassificationData {
            public final double[][] X;
            public final double[] y;
            
            public ClassificationData(double[][] X, double[] y) {
                this.X = X;
                this.y = y;
            }
        }
        
        public static class RegressionData {
            public final double[][] X;
            public final double[] y;
            
            public RegressionData(double[][] X, double[] y) {
                this.X = X;
                this.y = y;
            }
        }
    }
}
