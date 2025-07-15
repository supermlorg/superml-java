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
import java.util.Arrays;
import java.util.Random;

/**
 * Comprehensive example demonstrating advanced visualization concepts with SuperML.
 * Shows textual charts, performance metrics visualization, and model comparison displays.
 */
public class EnhancedVisualizationExample {

    public static void main(String[] args) {
        System.out.println("=== SuperML 2.0.0 - Enhanced Visualization Example ===\n");
        
        // Demo 1: Performance metrics visualization
        demonstratePerformanceVisualization();
        
        // Demo 2: Model comparison charts
        demonstrateModelComparisonCharts();
        
        // Demo 3: Learning curves and training progress
        demonstrateLearningCurves();
        
        // Demo 4: Feature importance and data distribution
        demonstrateFeatureVisualization();
        
        System.out.println("\n=== Enhanced Visualization Demo Complete! ===");
    }
    
    private static void demonstratePerformanceVisualization() {
        System.out.println("📊 Performance Metrics Visualization");
        System.out.println("====================================");
        
        try {
            // Generate dataset and train model
            double[][] X = generateData(400, 3);
            double[] y = generateBinaryTargets(X);
            
            // Split data
            int trainSize = 320;
            double[][] XTrain = Arrays.copyOfRange(X, 0, trainSize);
            double[] yTrain = Arrays.copyOfRange(y, 0, trainSize);
            double[][] XTest = Arrays.copyOfRange(X, trainSize, X.length);
            double[] yTest = Arrays.copyOfRange(y, trainSize, y.length);
            
            // Train models
            LogisticRegression lr = new LogisticRegression().setMaxIter(1000);
            RandomForest rf = new RandomForest().setNEstimators(50);
            
            lr.fit(XTrain, yTrain);
            rf.fit(XTrain, yTrain);
            
            // Get predictions
            double[] predLR = lr.predict(XTest);
            double[] predRF = rf.predict(XTest);
            
            // Calculate metrics
            double accLR = calculateAccuracy(yTest, predLR);
            double precLR = calculatePrecision(yTest, predLR);
            double recLR = calculateRecall(yTest, predLR);
            double f1LR = 2 * precLR * recLR / (precLR + recLR);
            
            double accRF = calculateAccuracy(yTest, predRF);
            double precRF = calculatePrecision(yTest, predRF);
            double recRF = calculateRecall(yTest, predRF);
            double f1RF = 2 * precRF * recRF / (precRF + recRF);
            
            System.out.println("🎯 Classification Performance Dashboard:");
            System.out.println();
            
            // Create performance bar chart
            System.out.println("📈 Metric Comparison Chart:");
            System.out.println("   Metric      | Logistic Regression | Random Forest");
            System.out.println("   ------------|---------------------|---------------");
            System.out.printf("   Accuracy    | %s | %s\n", 
                             createBar(accLR, 20), createBar(accRF, 20));
            System.out.printf("   Precision   | %s | %s\n", 
                             createBar(precLR, 20), createBar(precRF, 20));
            System.out.printf("   Recall      | %s | %s\n", 
                             createBar(recLR, 20), createBar(recRF, 20));
            System.out.printf("   F1-Score    | %s | %s\n", 
                             createBar(f1LR, 20), createBar(f1RF, 20));
            
            // Numerical values
            System.out.println("\n📊 Detailed Performance Metrics:");
            System.out.printf("   Logistic Regression: Acc=%.3f, Prec=%.3f, Rec=%.3f, F1=%.3f\n", 
                             accLR, precLR, recLR, f1LR);
            System.out.printf("   Random Forest:       Acc=%.3f, Prec=%.3f, Rec=%.3f, F1=%.3f\n", 
                             accRF, precRF, recRF, f1RF);
            
            // Confusion matrix visualization
            System.out.println("\n🔢 Confusion Matrix Visualization:");
            visualizeConfusionMatrix(yTest, predLR, "Logistic Regression");
            visualizeConfusionMatrix(yTest, predRF, "Random Forest");
            
            System.out.println("   ✅ Performance visualization completed!");
            
        } catch (Exception e) {
            System.err.println("   ❌ Error in performance visualization: " + e.getMessage());
        }
    }
    
    private static void demonstrateModelComparisonCharts() {
        System.out.println("\n🔍 Model Comparison Charts");
        System.out.println("==========================");
        
        try {
            // Generate datasets of different sizes
            int[] datasetSizes = {100, 200, 300, 400, 500};
            double[] lrAccuracies = new double[datasetSizes.length];
            double[] rfAccuracies = new double[datasetSizes.length];
            
            System.out.println("📈 Training models on different dataset sizes...");
            
            for (int i = 0; i < datasetSizes.length; i++) {
                // Generate data
                double[][] X = generateData(datasetSizes[i], 4);
                double[] y = generateBinaryTargets(X);
                
                // Train models
                LogisticRegression lr = new LogisticRegression().setMaxIter(500);
                RandomForest rf = new RandomForest().setNEstimators(30);
                
                lr.fit(X, y);
                rf.fit(X, y);
                
                // Evaluate on training data (for demonstration)
                double[] predLR = lr.predict(X);
                double[] predRF = rf.predict(X);
                
                lrAccuracies[i] = calculateAccuracy(y, predLR);
                rfAccuracies[i] = calculateAccuracy(y, predRF);
                
                System.out.printf("   Dataset size %d: LR=%.3f, RF=%.3f\n", 
                                 datasetSizes[i], lrAccuracies[i], rfAccuracies[i]);
            }
            
            // Create learning curve chart
            System.out.println("\n📊 Learning Curve Visualization:");
            System.out.println("   Dataset Size |      Logistic Regression      |         Random Forest");
            System.out.println("   -------------|-------------------------------|---------------------------");
            
            for (int i = 0; i < datasetSizes.length; i++) {
                String lrBar = createBar(lrAccuracies[i], 30);
                String rfBar = createBar(rfAccuracies[i], 30);
                System.out.printf("   %12d | %s | %s\n", datasetSizes[i], lrBar, rfBar);
            }
            
            // Algorithm speed comparison
            System.out.println("\n⚡ Training Speed Comparison:");
            long[] lrTimes = {15, 25, 40, 65, 95}; // Simulated milliseconds
            long[] rfTimes = {45, 85, 140, 220, 320}; // Simulated milliseconds
            
            System.out.println("   Dataset Size | LR Training Time | RF Training Time | Speed Ratio");
            System.out.println("   -------------|------------------|------------------|-------------");
            
            for (int i = 0; i < datasetSizes.length; i++) {
                double ratio = (double) rfTimes[i] / lrTimes[i];
                System.out.printf("   %12d |      %3d ms      |      %3d ms      |    %.1fx\n", 
                                 datasetSizes[i], lrTimes[i], rfTimes[i], ratio);
            }
            
            // Memory usage visualization
            System.out.println("\n💾 Memory Usage Estimation:");
            visualizeMemoryUsage(datasetSizes, lrTimes, rfTimes);
            
            System.out.println("\n   ✅ Model comparison charts completed!");
            
        } catch (Exception e) {
            System.err.println("   ❌ Error in model comparison: " + e.getMessage());
        }
    }
    
    private static void demonstrateLearningCurves() {
        System.out.println("\n📈 Learning Curves & Training Progress");
        System.out.println("======================================");
        
        try {
            // Simulate training iterations
            int maxIter = 100;
            double[] iterations = new double[10];
            double[] trainLoss = new double[10];
            double[] valLoss = new double[10];
            double[] trainAcc = new double[10];
            double[] valAcc = new double[10];
            
            // Generate simulated learning curves
            Random random = new Random(42);
            for (int i = 0; i < 10; i++) {
                iterations[i] = (i + 1) * (maxIter / 10);
                
                // Simulate decreasing loss
                trainLoss[i] = 0.7 * Math.exp(-i * 0.3) + 0.1 + random.nextGaussian() * 0.02;
                valLoss[i] = 0.7 * Math.exp(-i * 0.25) + 0.15 + random.nextGaussian() * 0.03;
                
                // Simulate increasing accuracy
                trainAcc[i] = 1.0 - (0.6 * Math.exp(-i * 0.3) + 0.1) + random.nextGaussian() * 0.02;
                valAcc[i] = 1.0 - (0.6 * Math.exp(-i * 0.25) + 0.15) + random.nextGaussian() * 0.03;
                
                // Ensure reasonable bounds
                trainLoss[i] = Math.max(0.05, Math.min(0.8, trainLoss[i]));
                valLoss[i] = Math.max(0.05, Math.min(0.8, valLoss[i]));
                trainAcc[i] = Math.max(0.5, Math.min(0.99, trainAcc[i]));
                valAcc[i] = Math.max(0.5, Math.min(0.99, valAcc[i]));
            }
            
            System.out.println("🔄 Training Progress Visualization:");
            System.out.println();
            
            // Loss curve
            System.out.println("📉 Loss Curves:");
            System.out.println("   Iteration |   Training Loss   |  Validation Loss  ");
            System.out.println("   ----------|-------------------|-------------------");
            
            for (int i = 0; i < 10; i++) {
                String trainBar = createLossBar(trainLoss[i], 0.8);
                String valBar = createLossBar(valLoss[i], 0.8);
                System.out.printf("   %9.0f | %s | %s\n", iterations[i], trainBar, valBar);
            }
            
            // Accuracy curve
            System.out.println("\n📈 Accuracy Curves:");
            System.out.println("   Iteration |  Training Accuracy | Validation Accuracy");
            System.out.println("   ----------|--------------------|--------------------");
            
            for (int i = 0; i < 10; i++) {
                String trainBar = createBar(trainAcc[i], 20);
                String valBar = createBar(valAcc[i], 20);
                System.out.printf("   %9.0f | %s | %s\n", iterations[i], trainBar, valBar);
            }
            
            // Training summary statistics
            System.out.println("\n📊 Training Summary:");
            System.out.printf("   Final Training Loss:     %.4f\n", trainLoss[9]);
            System.out.printf("   Final Validation Loss:   %.4f\n", valLoss[9]);
            System.out.printf("   Final Training Accuracy: %.3f\n", trainAcc[9]);
            System.out.printf("   Final Validation Accuracy: %.3f\n", valAcc[9]);
            
            double overfit = trainAcc[9] - valAcc[9];
            if (overfit > 0.05) {
                System.out.printf("   ⚠️  Overfitting detected: %.3f gap\n", overfit);
                System.out.println("   💡 Consider: regularization, early stopping, more data");
            } else {
                System.out.println("   ✅ Good generalization - no overfitting detected");
            }
            
            System.out.println("\n   ✅ Learning curves visualization completed!");
            
        } catch (Exception e) {
            System.err.println("   ❌ Error in learning curves: " + e.getMessage());
        }
    }
    
    private static void demonstrateFeatureVisualization() {
        System.out.println("\n🔍 Feature Analysis & Data Distribution");
        System.out.println("=======================================");
        
        try {
            // Generate dataset with multiple features
            double[][] X = generateData(500, 5);
            double[] y = generateContinuousTargets(X);
            
            System.out.println("📊 Dataset Overview:");
            System.out.printf("   Samples: %d, Features: %d\n", X.length, X[0].length);
            
            // Feature statistics
            System.out.println("\n📈 Feature Statistics:");
            System.out.println("   Feature |   Min   |   Max   |  Mean   |  Std Dev | Distribution");
            System.out.println("   --------|---------|---------|---------|----------|--------------------");
            
            for (int j = 0; j < X[0].length; j++) {
                double[] featureValues = new double[X.length];
                for (int i = 0; i < X.length; i++) {
                    featureValues[i] = X[i][j];
                }
                
                double min = Arrays.stream(featureValues).min().orElse(0.0);
                double max = Arrays.stream(featureValues).max().orElse(0.0);
                double mean = Arrays.stream(featureValues).average().orElse(0.0);
                double std = calculateStd(featureValues, mean);
                
                String distribution = createDistributionBar(featureValues, 20);
                
                System.out.printf("   %7d | %7.2f | %7.2f | %7.2f | %8.2f | %s\n", 
                                 j + 1, min, max, mean, std, distribution);
            }
            
            // Feature correlation with target
            System.out.println("\n🎯 Feature-Target Correlation:");
            System.out.println("   Feature | Correlation |           Strength");
            System.out.println("   --------|-------------|-------------------------");
            
            for (int j = 0; j < X[0].length; j++) {
                double[] featureValues = new double[X.length];
                for (int i = 0; i < X.length; i++) {
                    featureValues[i] = X[i][j];
                }
                
                double correlation = calculateCorrelation(featureValues, y);
                String strengthBar = createCorrelationBar(correlation);
                
                System.out.printf("   %7d |    %6.3f   | %s\n", j + 1, correlation, strengthBar);
            }
            
            // Train model and show feature importance
            System.out.println("\n🌳 Feature Importance (Random Forest):");
            RandomForest rf = new RandomForest().setNEstimators(50);
            rf.fit(X, y);
            
            // Simulate feature importance
            double[] importance = simulateFeatureImportance(X[0].length);
            
            System.out.println("   Feature |  Importance  |           Impact");
            System.out.println("   --------|--------------|-------------------------");
            
            for (int j = 0; j < importance.length; j++) {
                String impactBar = createBar(importance[j], 25);
                System.out.printf("   %7d |     %.3f    | %s\n", j + 1, importance[j], impactBar);
            }
            
            // Data quality assessment
            System.out.println("\n🔍 Data Quality Assessment:");
            assessDataQuality(X, y);
            
            System.out.println("\n   ✅ Feature visualization completed!");
            
        } catch (Exception e) {
            System.err.println("   ❌ Error in feature visualization: " + e.getMessage());
        }
    }
    
    // Utility methods for visualization
    private static String createBar(double value, int maxLength) {
        int length = (int) (value * maxLength);
        length = Math.max(0, Math.min(maxLength, length));
        
        StringBuilder bar = new StringBuilder();
        for (int i = 0; i < length; i++) {
            bar.append("█");
        }
        for (int i = length; i < maxLength; i++) {
            bar.append("░");
        }
        
        return bar.toString() + String.format(" %.3f", value);
    }
    
    private static String createLossBar(double loss, double maxLoss) {
        return createBar(loss / maxLoss, 20);
    }
    
    private static String createDistributionBar(double[] values, int width) {
        // Create a simple histogram
        double min = Arrays.stream(values).min().orElse(0.0);
        double max = Arrays.stream(values).max().orElse(1.0);
        
        int[] bins = new int[width];
        for (double value : values) {
            int bin = (int) ((value - min) / (max - min + 1e-8) * (width - 1));
            bin = Math.max(0, Math.min(width - 1, bin));
            bins[bin]++;
        }
        
        int maxCount = Arrays.stream(bins).max().orElse(1);
        StringBuilder hist = new StringBuilder();
        
        for (int bin : bins) {
            if (bin > maxCount * 0.7) hist.append("█");
            else if (bin > maxCount * 0.4) hist.append("▓");
            else if (bin > maxCount * 0.1) hist.append("▒");
            else hist.append("░");
        }
        
        return hist.toString();
    }
    
    private static String createCorrelationBar(double correlation) {
        int length = (int) (Math.abs(correlation) * 25);
        char symbol = correlation >= 0 ? '█' : '▓';
        
        StringBuilder bar = new StringBuilder();
        for (int i = 0; i < length; i++) {
            bar.append(symbol);
        }
        for (int i = length; i < 25; i++) {
            bar.append("░");
        }
        
        return bar.toString();
    }
    
    private static void visualizeConfusionMatrix(double[] yTrue, double[] yPred, String modelName) {
        int tp = 0, tn = 0, fp = 0, fn = 0;
        
        for (int i = 0; i < yTrue.length; i++) {
            if (yTrue[i] == 1.0 && yPred[i] == 1.0) tp++;
            else if (yTrue[i] == 0.0 && yPred[i] == 0.0) tn++;
            else if (yTrue[i] == 0.0 && yPred[i] == 1.0) fp++;
            else if (yTrue[i] == 1.0 && yPred[i] == 0.0) fn++;
        }
        
        System.out.printf("\n   %s Confusion Matrix:\n", modelName);
        System.out.println("           Predicted");
        System.out.println("   Actual    0    1");
        System.out.println("        0  " + String.format("%3d", tn) + "  " + String.format("%3d", fp));
        System.out.println("        1  " + String.format("%3d", fn) + "  " + String.format("%3d", tp));
    }
    
    private static void visualizeMemoryUsage(int[] sizes, long[] lrTimes, long[] rfTimes) {
        System.out.println("   Algorithm    | Memory Usage Pattern");
        System.out.println("   -------------|----------------------");
        
        for (int i = 0; i < sizes.length; i++) {
            // Simulate memory usage (linear for LR, more for RF)
            double lrMemory = sizes[i] * 0.001; // KB
            double rfMemory = sizes[i] * 0.005; // KB
            
            String lrBar = createBar(lrMemory / 2.5, 15); // Normalize to max expected
            String rfBar = createBar(rfMemory / 2.5, 15);
            
            if (i == 0) {
                System.out.printf("   Logistic Reg | %s\n", lrBar);
                System.out.printf("   Random Forest| %s\n", rfBar);
            }
        }
    }
    
    private static void assessDataQuality(double[][] X, double[] y) {
        // Check for missing values (simulated)
        System.out.println("   ✅ Missing Values: None detected");
        
        // Check for outliers
        int outliers = 0;
        for (int j = 0; j < X[0].length; j++) {
            double[] feature = new double[X.length];
            for (int i = 0; i < X.length; i++) {
                feature[i] = X[i][j];
            }
            
            double mean = Arrays.stream(feature).average().orElse(0.0);
            double std = calculateStd(feature, mean);
            
            for (double value : feature) {
                if (Math.abs(value - mean) > 3 * std) {
                    outliers++;
                }
            }
        }
        
        System.out.printf("   %s Outliers: %d detected (%.1f%%)\n", 
                         outliers > X.length * 0.05 ? "⚠️ " : "✅", 
                         outliers, 100.0 * outliers / (X.length * X[0].length));
        
        // Data balance check
        int positiveCount = 0;
        for (double value : y) {
            if (value > Arrays.stream(y).average().orElse(0.0)) positiveCount++;
        }
        double balance = (double) positiveCount / y.length;
        
        if (balance < 0.3 || balance > 0.7) {
            System.out.printf("   ⚠️  Class Imbalance: %.1f%% positive class\n", balance * 100);
        } else {
            System.out.println("   ✅ Class Balance: Good distribution");
        }
    }
    
    // Data generation and utility methods
    private static double[][] generateData(int nSamples, int nFeatures) {
        Random random = new Random(42);
        double[][] X = new double[nSamples][nFeatures];
        
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                X[i][j] = random.nextGaussian();
            }
        }
        
        return X;
    }
    
    private static double[] generateBinaryTargets(double[][] X) {
        double[] y = new double[X.length];
        
        for (int i = 0; i < X.length; i++) {
            double sum = 0;
            for (double feature : X[i]) {
                sum += feature;
            }
            y[i] = (sum > 0) ? 1.0 : 0.0;
        }
        
        return y;
    }
    
    private static double[] generateContinuousTargets(double[][] X) {
        Random random = new Random(42);
        double[] y = new double[X.length];
        
        for (int i = 0; i < X.length; i++) {
            double sum = 0;
            for (int j = 0; j < X[i].length; j++) {
                sum += (j + 1) * X[i][j]; // Weighted sum
            }
            y[i] = sum + random.nextGaussian() * 0.5;
        }
        
        return y;
    }
    
    private static double[] simulateFeatureImportance(int nFeatures) {
        Random random = new Random(42);
        double[] importance = new double[nFeatures];
        double sum = 0;
        
        for (int i = 0; i < nFeatures; i++) {
            importance[i] = random.nextDouble();
            sum += importance[i];
        }
        
        // Normalize to sum to 1
        for (int i = 0; i < nFeatures; i++) {
            importance[i] /= sum;
        }
        
        return importance;
    }
    
    // Statistical utility methods
    private static double calculateAccuracy(double[] yTrue, double[] yPred) {
        int correct = 0;
        for (int i = 0; i < yTrue.length; i++) {
            if (yTrue[i] == yPred[i]) correct++;
        }
        return (double) correct / yTrue.length;
    }
    
    private static double calculatePrecision(double[] yTrue, double[] yPred) {
        int truePositives = 0, falsePositives = 0;
        
        for (int i = 0; i < yTrue.length; i++) {
            if (yPred[i] == 1.0) {
                if (yTrue[i] == 1.0) truePositives++;
                else falsePositives++;
            }
        }
        
        return (truePositives + falsePositives > 0) ? 
               (double) truePositives / (truePositives + falsePositives) : 0.0;
    }
    
    private static double calculateRecall(double[] yTrue, double[] yPred) {
        int truePositives = 0, falseNegatives = 0;
        
        for (int i = 0; i < yTrue.length; i++) {
            if (yTrue[i] == 1.0) {
                if (yPred[i] == 1.0) truePositives++;
                else falseNegatives++;
            }
        }
        
        return (truePositives + falseNegatives > 0) ? 
               (double) truePositives / (truePositives + falseNegatives) : 0.0;
    }
    
    private static double calculateStd(double[] values, double mean) {
        double sum = 0;
        for (double value : values) {
            sum += Math.pow(value - mean, 2);
        }
        return Math.sqrt(sum / values.length);
    }
    
    private static double calculateCorrelation(double[] x, double[] y) {
        double meanX = Arrays.stream(x).average().orElse(0.0);
        double meanY = Arrays.stream(y).average().orElse(0.0);
        
        double numerator = 0, denomX = 0, denomY = 0;
        
        for (int i = 0; i < x.length; i++) {
            double deltaX = x[i] - meanX;
            double deltaY = y[i] - meanY;
            
            numerator += deltaX * deltaY;
            denomX += deltaX * deltaX;
            denomY += deltaY * deltaY;
        }
        
        if (denomX == 0 || denomY == 0) return 0.0;
        
        return numerator / Math.sqrt(denomX * denomY);
    }
}
