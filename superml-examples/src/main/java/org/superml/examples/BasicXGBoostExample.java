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

import org.superml.tree.XGBoost;
import org.superml.metrics.Metrics;

import java.util.*;

/**
 * Basic XGBoost Example - Quick start with XGBoost
 * 
 * This example demonstrates:
 * - Basic XGBoost model training
 * - Hyperparameter tuning
 * - Feature importance analysis
 * - Early stopping with validation
 * 
 * @author SuperML Team
 * @version 2.1.0
 */
public class BasicXGBoostExample {
    
    public static void main(String[] args) {
        System.out.println("🚀 SuperML Java 2.0.0 - XGBoost Quick Start");
        System.out.println("=" + "=".repeat(50));
        
        try {
            // Generate sample dataset
            System.out.println("\n📊 Generating sample dataset...");
            Dataset data = generateSampleDataset(1000, 10);
            
            // Split into train/test
            DataSplit split = splitData(data.X, data.y, 0.2);
            System.out.printf("Training samples: %d, Test samples: %d%n", 
                split.XTrain.length, split.XTest.length);
            System.out.printf("Features: %d%n", data.X[0].length);
            
            // 1. Basic XGBoost Training
            basicXGBoostExample(split);
            
            // 2. Advanced XGBoost with Regularization
            advancedXGBoostExample(split);
            
            // 3. Feature Importance
            featureImportanceExample(split);
            
            System.out.println("\n🎉 XGBoost quick start completed!");
            
        } catch (Exception e) {
            System.err.println("❌ Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Basic XGBoost training example
     */
    private static void basicXGBoostExample(DataSplit split) {
        System.out.println("\n" + "=".repeat(40));
        System.out.println("🌟 Basic XGBoost Training");
        System.out.println("=".repeat(40));
        
        // Create XGBoost model with basic parameters
        XGBoost xgb = new XGBoost()
            .setNEstimators(100)        // Number of trees
            .setLearningRate(0.1)       // Step size shrinkage
            .setMaxDepth(6)             // Maximum tree depth
            .setRandomState(42);        // For reproducibility
        
        // Train the model
        System.out.println("Training XGBoost model...");
        long startTime = System.currentTimeMillis();
        xgb.fit(split.XTrain, split.yTrain);
        long trainTime = System.currentTimeMillis() - startTime;
        
        // Make predictions
        double[] predictions = xgb.predict(split.XTest);
        
        // Calculate metrics
        double accuracy = Metrics.accuracy(split.yTest, predictions);
        double f1Score = Metrics.f1Score(split.yTest, predictions);
        
        // Display results
        System.out.println("\n📊 Results:");
        System.out.printf("├─ Training time: %d ms%n", trainTime);
        System.out.printf("├─ Test accuracy: %.4f%n", accuracy);
        System.out.printf("└─ F1-score: %.4f%n", f1Score);
    }
    
    /**
     * Advanced XGBoost with regularization and early stopping
     */
    private static void advancedXGBoostExample(DataSplit split) {
        System.out.println("\n" + "=".repeat(40));
        System.out.println("🔧 Advanced XGBoost Configuration");
        System.out.println("=".repeat(40));
        
        // Create validation set from training data
        int validSize = split.XTrain.length / 5;
        int newTrainSize = split.XTrain.length - validSize;
        
        double[][] XTrain = Arrays.copyOfRange(split.XTrain, 0, newTrainSize);
        double[] yTrain = Arrays.copyOfRange(split.yTrain, 0, newTrainSize);
        double[][] XValid = Arrays.copyOfRange(split.XTrain, newTrainSize, split.XTrain.length);
        double[] yValid = Arrays.copyOfRange(split.yTrain, newTrainSize, split.yTrain.length);
        
        // Advanced XGBoost with regularization
        XGBoost xgb = new XGBoost()
            .setNEstimators(300)            // More trees
            .setLearningRate(0.05)          // Lower learning rate
            .setMaxDepth(8)                 // Deeper trees
            .setGamma(0.1)                  // Minimum loss reduction for split
            .setLambda(1.0)                 // L2 regularization
            .setAlpha(0.1)                  // L1 regularization
            .setSubsample(0.8)              // Row sampling ratio
            .setColsampleBytree(0.8)        // Column sampling per tree
            .setMinChildWeight(3)           // Minimum child weight
            .setEarlyStoppingRounds(20)     // Early stopping patience
            .setValidationFraction(0.2)     // Validation fraction for early stopping
            .setRandomState(42)
            .setSilent(false);              // Show training progress
        
        // Train with validation data for early stopping
        System.out.println("Training advanced XGBoost with early stopping...");
        long startTime = System.currentTimeMillis();
        xgb.fit(XTrain, yTrain, XValid, yValid);
        long trainTime = System.currentTimeMillis() - startTime;
        
        // Make predictions
        double[] predictions = xgb.predict(split.XTest);
        
        // Calculate metrics
        double accuracy = Metrics.accuracy(split.yTest, predictions);
        double f1Score = Metrics.f1Score(split.yTest, predictions);
        
        // Display results
        System.out.println("\n📊 Advanced Results:");
        System.out.printf("├─ Training time: %d ms%n", trainTime);
        System.out.printf("├─ Test accuracy: %.4f%n", accuracy);
        System.out.printf("├─ F1-score: %.4f%n", f1Score);
        System.out.printf("└─ Trees built: %d%n", xgb.getNEstimators());
        
        // Show training progress
        Map<String, List<Double>> evalResults = xgb.getEvalResults();
        List<Double> validLoss = evalResults.get("valid-logloss");
        if (validLoss != null && validLoss.size() > 5) {
            System.out.println("\n📈 Training Progress (last 5 iterations):");
            int start = validLoss.size() - 5;
            for (int i = start; i < validLoss.size(); i++) {
                System.out.printf("   Iteration %d: %.6f%n", i + 1, validLoss.get(i));
            }
        }
        
        System.out.println("\n🔧 Key Hyperparameters:");
        System.out.printf("├─ Learning Rate: %.3f%n", xgb.getLearningRate());
        System.out.printf("├─ Max Depth: %d%n", xgb.getMaxDepth());
        System.out.printf("├─ L1 Regularization (Alpha): %.3f%n", xgb.getAlpha());
        System.out.printf("├─ L2 Regularization (Lambda): %.3f%n", xgb.getLambda());
        System.out.printf("└─ Subsample Ratio: %.3f%n", xgb.getSubsample());
    }
    
    /**
     * Feature importance analysis example
     */
    private static void featureImportanceExample(DataSplit split) {
        System.out.println("\n" + "=".repeat(40));
        System.out.println("📈 Feature Importance Analysis");
        System.out.println("=".repeat(40));
        
        // Train XGBoost for feature analysis
        XGBoost xgb = new XGBoost()
            .setNEstimators(150)
            .setLearningRate(0.1)
            .setMaxDepth(6)
            .setRandomState(42);
        
        System.out.println("Training XGBoost for feature analysis...");
        xgb.fit(split.XTrain, split.yTrain);
        
        // Get feature importance statistics
        Map<String, double[]> importanceStats = xgb.getFeatureImportanceStats();
        
        System.out.println("\n📊 Feature Importance (Top 8 features):");
        
        // Weight-based importance (how often features are used)
        double[] weightImportance = importanceStats.get("weight");
        System.out.println("\n🔸 Weight-based (Feature Usage Frequency):");
        displayTopFeatures(weightImportance, 8);
        
        // Gain-based importance (average information gain)
        double[] gainImportance = importanceStats.get("gain");
        System.out.println("\n🔸 Gain-based (Average Information Gain):");
        displayTopFeatures(gainImportance, 8);
        
        // Simple visualization
        System.out.println("\n📊 Feature Importance Visualization (Weight-based):");
        visualizeImportance(weightImportance, 10);
        
        // Feature ranking
        System.out.println("\n🏆 Top 5 Most Important Features:");
        List<FeatureRank> rankings = getRankedFeatures(gainImportance);
        for (int i = 0; i < Math.min(5, rankings.size()); i++) {
            FeatureRank rank = rankings.get(i);
            System.out.printf("   %d. Feature_%d (Gain: %.4f)%n", 
                i + 1, rank.index, rank.importance);
        }
    }
    
    // Helper methods
    
    private static void displayTopFeatures(double[] importance, int topK) {
        List<FeatureRank> features = getRankedFeatures(importance);
        
        for (int i = 0; i < Math.min(topK, features.size()); i++) {
            FeatureRank feature = features.get(i);
            if (feature.importance > 0) {
                System.out.printf("   %2d. Feature_%d: %.4f%n", 
                    i + 1, feature.index, feature.importance);
            }
        }
    }
    
    private static List<FeatureRank> getRankedFeatures(double[] importance) {
        List<FeatureRank> features = new ArrayList<>();
        for (int i = 0; i < importance.length; i++) {
            features.add(new FeatureRank(i, importance[i]));
        }
        features.sort((a, b) -> Double.compare(b.importance, a.importance));
        return features;
    }
    
    private static void visualizeImportance(double[] importance, int maxFeatures) {
        double maxImportance = Arrays.stream(importance).max().orElse(1.0);
        
        for (int i = 0; i < Math.min(maxFeatures, importance.length); i++) {
            if (importance[i] > 0) {
                int barLength = (int) ((importance[i] / maxImportance) * 20);
                String bar = "█".repeat(Math.max(1, barLength));
                System.out.printf("   Feature_%2d │%-20s│ %.3f%n", i, bar, importance[i]);
            }
        }
    }
    
    private static Dataset generateSampleDataset(int samples, int features) {
        Random random = new Random(42);
        double[][] X = new double[samples][features];
        double[] y = new double[samples];
        
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < features; j++) {
                X[i][j] = random.nextGaussian();
            }
            
            // Create decision rule with feature interactions
            double score = 0.0;
            score += 2.0 * X[i][0] + 1.5 * X[i][1]; // Linear terms
            score += X[i][2] * X[i][3]; // Interaction term
            score += Math.sin(X[i][4]); // Non-linear term
            
            // Add some noise
            score += 0.5 * random.nextGaussian();
            
            y[i] = score > 0 ? 1.0 : 0.0;
        }
        
        return new Dataset(X, y);
    }
    
    private static DataSplit splitData(double[][] X, double[] y, double testRatio) {
        int totalSamples = X.length;
        int testSize = (int) (totalSamples * testRatio);
        int trainSize = totalSamples - testSize;
        
        // Shuffle data
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < totalSamples; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, new Random(42));
        
        // Split data
        double[][] XTrain = new double[trainSize][];
        double[] yTrain = new double[trainSize];
        double[][] XTest = new double[testSize][];
        double[] yTest = new double[testSize];
        
        for (int i = 0; i < trainSize; i++) {
            int idx = indices.get(i);
            XTrain[i] = Arrays.copyOf(X[idx], X[idx].length);
            yTrain[i] = y[idx];
        }
        
        for (int i = 0; i < testSize; i++) {
            int idx = indices.get(trainSize + i);
            XTest[i] = Arrays.copyOf(X[idx], X[idx].length);
            yTest[i] = y[idx];
        }
        
        return new DataSplit(XTrain, yTrain, XTest, yTest);
    }
    
    // Data classes
    
    private static class Dataset {
        final double[][] X;
        final double[] y;
        
        Dataset(double[][] X, double[] y) {
            this.X = X;
            this.y = y;
        }
    }
    
    private static class DataSplit {
        final double[][] XTrain, XTest;
        final double[] yTrain, yTest;
        
        DataSplit(double[][] XTrain, double[] yTrain, double[][] XTest, double[] yTest) {
            this.XTrain = XTrain;
            this.yTrain = yTrain;
            this.XTest = XTest;
            this.yTest = yTest;
        }
    }
    
    private static class FeatureRank {
        final int index;
        final double importance;
        
        FeatureRank(int index, double importance) {
            this.index = index;
            this.importance = importance;
        }
    }
}
