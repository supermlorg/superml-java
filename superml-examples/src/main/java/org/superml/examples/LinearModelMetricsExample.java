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

import org.superml.linear_model.*;

import java.util.Random;

/**
 * Comprehensive example demonstrating linear model evaluation
 * 
 * Shows basic evaluation for all linear models:
 * - LinearRegression, Ridge, Lasso, SGDRegressor evaluation
 * - LogisticRegression, SGDClassifier evaluation
 * - Basic metrics and comparisons
 * 
 * NOTE: Enhanced metrics with detailed evaluation will be available in the next release
 * 
 * @author SuperML Team
 */
public class LinearModelMetricsExample {
    
    public static void main(String[] args) {
        System.out.println("=== SuperML Linear Model Basic Evaluation Example ===");
        System.out.println();
        
        // Test regression models
        System.out.println("1. REGRESSION MODELS EVALUATION");
        System.out.println("================================");
        testRegressionModels();
        
        System.out.println();
        
        // Test classification models
        System.out.println("2. CLASSIFICATION MODELS EVALUATION");
        System.out.println("====================================");
        testClassificationModels();
        
        System.out.println();
        System.out.println("Enhanced metrics with detailed evaluation coming in next release!");
        System.out.println("=== Linear Model Basic Evaluation Example Complete ===");
    }
    
    private static void testRegressionModels() {
        // Generate regression data
        DataGenerator generator = new DataGenerator();
        DataGenerator.RegressionData data = generator.generateRegressionData(500, 10);
        
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
        
        // Test LinearRegression
        System.out.println("--- LinearRegression ---");
        LinearRegression lr = new LinearRegression();
        lr.fit(XTrain, yTrain);
        double lrScore = lr.score(XTest, yTest);
        System.out.printf("  R² Score: %.4f\n", lrScore);
        
        // Test Ridge
        System.out.println("--- Ridge Regression ---");
        Ridge ridge = new Ridge().setAlpha(1.0);
        ridge.fit(XTrain, yTrain);
        double ridgeScore = ridge.score(XTest, yTest);
        System.out.printf("  R² Score: %.4f\n", ridgeScore);
        
        // Test Lasso
        System.out.println("--- Lasso Regression ---");
        Lasso lasso = new Lasso().setAlpha(0.1);
        lasso.fit(XTrain, yTrain);
        double lassoScore = lasso.score(XTest, yTest);
        System.out.printf("  R² Score: %.4f\n", lassoScore);
        
        // Test SGDRegressor
        System.out.println("--- SGD Regressor ---");
        SGDRegressor sgdReg = new SGDRegressor()
            .setLoss("squared_loss")
            .setPenalty("l2")
            .setAlpha(0.001)
            .setMaxIter(1000);
        sgdReg.fit(XTrain, yTrain);
        double sgdScore = sgdReg.score(XTest, yTest);
        System.out.printf("  R² Score: %.4f\n", sgdScore);
    }
    
    private static void testClassificationModels() {
        // Generate classification data
        DataGenerator generator = new DataGenerator();
        DataGenerator.ClassificationData data = generator.generateClassificationData(500, 10, 2);
        
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
        
        // Test LogisticRegression
        System.out.println("--- Logistic Regression ---");
        LogisticRegression logReg = new LogisticRegression()
            .setLearningRate(0.01)
            .setMaxIter(1000);
        logReg.fit(XTrain, yTrain);
        double logScore = logReg.score(XTest, yTest);
        System.out.printf("  Accuracy: %.4f\n", logScore);
        
        // Test SGDClassifier
        System.out.println("--- SGD Classifier ---");
        SGDClassifier sgdClf = new SGDClassifier()
            .setLoss("log")
            .setPenalty("l2")
            .setAlpha(0.001)
            .setMaxIter(1000);
        sgdClf.fit(XTrain, yTrain);
        double sgdScore = sgdClf.score(XTest, yTest);
        System.out.printf("  Accuracy: %.4f\n", sgdScore);
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
                y[i] = score > 0 ? 1.0 : 0.0;
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
                
                // Generate target as linear combination with noise
                double target = 0.0;
                for (int j = 0; j < nFeatures; j++) {
                    target += X[i][j] * (j + 1) * 0.3;
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
