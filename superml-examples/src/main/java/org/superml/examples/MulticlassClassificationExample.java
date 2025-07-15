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
import java.util.Arrays;
import java.util.Random;

/**
 * Comprehensive example demonstrating multiclass classification with SuperML.
 * Shows different algorithms, evaluation techniques, and performance comparison.
 */
public class MulticlassClassificationExample {

    public static void main(String[] args) {
        System.out.println("=== SuperML 2.0.0 - Multiclass Classification Example ===\n");
        
        // Demo 1: Basic multiclass classification
        demonstrateBasicMulticlass();
        
        // Demo 2: Algorithm comparison
        demonstrateAlgorithmComparison();
        
        // Demo 3: Multiclass evaluation metrics
        demonstrateMulticlassMetrics();
        
        // Demo 4: Class imbalance handling
        demonstrateClassImbalance();
        
        System.out.println("\n=== Multiclass Classification Demo Complete! ===");
    }
    
    private static void demonstrateBasicMulticlass() {
        System.out.println("🎯 Basic Multiclass Classification");
        System.out.println("==================================");
        
        try {
            // Generate multiclass dataset (3 classes)
            double[][] X = generateMulticlassData(300, 4, 3);
            double[] y = generateMulticlassLabels(X, 3);
            
            // Split data
            int trainSize = (int)(X.length * 0.8);
            double[][] XTrain = Arrays.copyOfRange(X, 0, trainSize);
            double[] yTrain = Arrays.copyOfRange(y, 0, trainSize);
            double[][] XTest = Arrays.copyOfRange(X, trainSize, X.length);
            double[] yTest = Arrays.copyOfRange(y, trainSize, y.length);
            
            System.out.printf("📊 Dataset: %d samples, %d features, %d classes\n", 
                             X.length, X[0].length, getUniqueClasses(y));
            System.out.printf("   Train: %d samples, Test: %d samples\n\n", 
                             XTrain.length, XTest.length);
            
            // Train Logistic Regression with multinomial strategy
            LogisticRegression lr = new LogisticRegression()
                    .setMultiClass("multinomial")
                    .setMaxIter(1000)
                    .setLearningRate(0.01);
            
            lr.fit(XTrain, yTrain);
            
            // Make predictions
            double[] predictions = lr.predict(XTest);
            double[][] probabilities = lr.predictProba(XTest);
            
            // Calculate accuracy
            double accuracy = calculateAccuracy(yTest, predictions);
            
            System.out.printf("🤖 Logistic Regression (Multinomial):\n");
            System.out.printf("   Accuracy: %.3f\n", accuracy);
            System.out.printf("   Classes: %s\n", Arrays.toString(lr.getClasses()));
            
            // Show sample predictions with probabilities
            System.out.println("\n📈 Sample Predictions:");
            System.out.println("   Sample | True | Pred | Prob Class 0 | Prob Class 1 | Prob Class 2");
            System.out.println("   -------|------|------|-------------|-------------|-------------");
            
            for (int i = 0; i < Math.min(5, predictions.length); i++) {
                System.out.printf("   %6d |  %.0f   |  %.0f   |    %.3f     |    %.3f     |    %.3f\n",
                                 i + 1, yTest[i], predictions[i], 
                                 probabilities[i][0], probabilities[i][1], probabilities[i][2]);
            }
            
            System.out.println("   ✅ Basic multiclass classification completed!");
            
        } catch (Exception e) {
            System.err.println("   ❌ Error in basic multiclass: " + e.getMessage());
        }
    }
    
    private static void demonstrateAlgorithmComparison() {
        System.out.println("\n🔍 Algorithm Comparison");
        System.out.println("=======================");
        
        try {
            // Generate dataset
            double[][] X = generateMulticlassData(400, 5, 4);
            double[] y = generateMulticlassLabels(X, 4);
            
            // Split data
            int trainSize = (int)(X.length * 0.75);
            double[][] XTrain = Arrays.copyOfRange(X, 0, trainSize);
            double[] yTrain = Arrays.copyOfRange(y, 0, trainSize);
            double[][] XTest = Arrays.copyOfRange(X, trainSize, X.length);
            double[] yTest = Arrays.copyOfRange(y, trainSize, y.length);
            
            System.out.printf("📊 Dataset: %d classes, %d features\n", 
                             getUniqueClasses(y), X[0].length);
            System.out.printf("   Training: %d samples, Testing: %d samples\n\n", 
                             XTrain.length, XTest.length);
            
            // Algorithm 1: Logistic Regression (One-vs-Rest)
            LogisticRegression lrOvR = new LogisticRegression()
                    .setMultiClass("ovr")
                    .setMaxIter(1000);
            lrOvR.fit(XTrain, yTrain);
            double[] predOvR = lrOvR.predict(XTest);
            double accOvR = calculateAccuracy(yTest, predOvR);
            
            // Algorithm 2: Logistic Regression (Multinomial)
            LogisticRegression lrMulti = new LogisticRegression()
                    .setMultiClass("multinomial")
                    .setMaxIter(1000);
            lrMulti.fit(XTrain, yTrain);
            double[] predMulti = lrMulti.predict(XTest);
            double accMulti = calculateAccuracy(yTest, predMulti);
            
            // Algorithm 3: Decision Tree
            DecisionTree dt = new DecisionTree()
                    .setMaxDepth(10)
                    .setMinSamplesSplit(5);
            dt.fit(XTrain, yTrain);
            double[] predDT = dt.predict(XTest);
            double accDT = calculateAccuracy(yTest, predDT);
            
            // Algorithm 4: Random Forest
            RandomForest rf = new RandomForest()
                    .setNEstimators(50)
                    .setMaxDepth(8);
            rf.fit(XTrain, yTrain);
            double[] predRF = rf.predict(XTest);
            double accRF = calculateAccuracy(yTest, predRF);
            
            // Display results
            System.out.println("🏆 Algorithm Performance Comparison:");
            System.out.println("   Algorithm                    | Accuracy | Strategy");
            System.out.println("   -----------------------------|----------|----------");
            System.out.printf("   Logistic Regression (OvR)   |  %.3f   | One-vs-Rest\n", accOvR);
            System.out.printf("   Logistic Regression (Multi) |  %.3f   | Multinomial\n", accMulti);
            System.out.printf("   Decision Tree                |  %.3f   | Native\n", accDT);
            System.out.printf("   Random Forest                |  %.3f   | Native\n", accRF);
            
            // Find best algorithm
            String[] algorithms = {"LR (OvR)", "LR (Multi)", "Decision Tree", "Random Forest"};
            double[] accuracies = {accOvR, accMulti, accDT, accRF};
            int bestIdx = 0;
            for (int i = 1; i < accuracies.length; i++) {
                if (accuracies[i] > accuracies[bestIdx]) {
                    bestIdx = i;
                }
            }
            
            System.out.printf("\n🥇 Best Performance: %s (%.3f accuracy)\n", 
                             algorithms[bestIdx], accuracies[bestIdx]);
            System.out.println("   ✅ Algorithm comparison completed!");
            
        } catch (Exception e) {
            System.err.println("   ❌ Error in algorithm comparison: " + e.getMessage());
        }
    }
    
    private static void demonstrateMulticlassMetrics() {
        System.out.println("\n📊 Multiclass Evaluation Metrics");
        System.out.println("=================================");
        
        try {
            // Generate dataset
            double[][] X = generateMulticlassData(200, 3, 3);
            double[] y = generateMulticlassLabels(X, 3);
            
            // Train model
            RandomForest rf = new RandomForest()
                    .setNEstimators(30)
                    .setMaxDepth(6);
            rf.fit(X, y);
            
            // Predictions
            double[] predictions = rf.predict(X);
            
            // Calculate metrics
            double accuracy = calculateAccuracy(y, predictions);
            double[] precisionPerClass = calculatePrecisionPerClass(y, predictions, 3);
            double[] recallPerClass = calculateRecallPerClass(y, predictions, 3);
            double[] f1PerClass = calculateF1PerClass(precisionPerClass, recallPerClass);
            
            double macroPrecision = Arrays.stream(precisionPerClass).average().orElse(0.0);
            double macroRecall = Arrays.stream(recallPerClass).average().orElse(0.0);
            double macroF1 = Arrays.stream(f1PerClass).average().orElse(0.0);
            
            System.out.printf("🎯 Overall Metrics:\n");
            System.out.printf("   Accuracy: %.3f\n", accuracy);
            System.out.printf("   Macro Precision: %.3f\n", macroPrecision);
            System.out.printf("   Macro Recall: %.3f\n", macroRecall);
            System.out.printf("   Macro F1-Score: %.3f\n\n", macroF1);
            
            System.out.println("📈 Per-Class Metrics:");
            System.out.println("   Class | Precision | Recall | F1-Score");
            System.out.println("   ------|-----------|--------|----------");
            
            for (int i = 0; i < 3; i++) {
                System.out.printf("   %5d |   %.3f   | %.3f  |  %.3f\n", 
                                 i, precisionPerClass[i], recallPerClass[i], f1PerClass[i]);
            }
            
            // Confusion matrix-like summary
            int[][] confusionMatrix = calculateConfusionMatrix(y, predictions, 3);
            System.out.println("\n🔢 Confusion Matrix:");
            System.out.println("   Predicted:");
            System.out.print("   True\\    | ");
            for (int i = 0; i < 3; i++) {
                System.out.printf("  %d  ", i);
            }
            System.out.println();
            System.out.println("   ---------|-------------");
            
            for (int i = 0; i < 3; i++) {
                System.out.printf("        %d   | ", i);
                for (int j = 0; j < 3; j++) {
                    System.out.printf("%3d ", confusionMatrix[i][j]);
                }
                System.out.println();
            }
            
            System.out.println("\n   ✅ Multiclass metrics evaluation completed!");
            
        } catch (Exception e) {
            System.err.println("   ❌ Error in metrics evaluation: " + e.getMessage());
        }
    }
    
    private static void demonstrateClassImbalance() {
        System.out.println("\n⚖️  Class Imbalance Handling");
        System.out.println("=============================");
        
        try {
            // Generate imbalanced dataset
            double[][] X = generateImbalancedMulticlassData(300, 4);
            double[] y = generateImbalancedLabels(X);
            
            // Show class distribution
            int[] classCounts = new int[3];
            for (double label : y) {
                classCounts[(int)label]++;
            }
            
            System.out.println("📊 Class Distribution:");
            for (int i = 0; i < 3; i++) {
                double percentage = 100.0 * classCounts[i] / y.length;
                System.out.printf("   Class %d: %d samples (%.1f%%)\n", 
                                 i, classCounts[i], percentage);
            }
            
            // Split data
            int trainSize = (int)(X.length * 0.8);
            double[][] XTrain = Arrays.copyOfRange(X, 0, trainSize);
            double[] yTrain = Arrays.copyOfRange(y, 0, trainSize);
            double[][] XTest = Arrays.copyOfRange(X, trainSize, X.length);
            double[] yTest = Arrays.copyOfRange(y, trainSize, y.length);
            
            // Train model without handling imbalance
            RandomForest rfStandard = new RandomForest()
                    .setNEstimators(50);
            rfStandard.fit(XTrain, yTrain);
            double[] predStandard = rfStandard.predict(XTest);
            
            // Train model with balanced approach (simulated)
            RandomForest rfBalanced = new RandomForest()
                    .setNEstimators(100)
                    .setMaxDepth(8);
            rfBalanced.fit(XTrain, yTrain);
            double[] predBalanced = rfBalanced.predict(XTest);
            
            // Calculate metrics for both approaches
            double[] recallStandard = calculateRecallPerClass(yTest, predStandard, 3);
            double[] recallBalanced = calculateRecallPerClass(yTest, predBalanced, 3);
            
            System.out.println("\n🔍 Imbalance Impact Analysis:");
            System.out.println("   Approach     | Class 0 Recall | Class 1 Recall | Class 2 Recall");
            System.out.println("   -------------|----------------|----------------|----------------");
            System.out.printf("   Standard     |     %.3f      |     %.3f      |     %.3f\n",
                             recallStandard[0], recallStandard[1], recallStandard[2]);
            System.out.printf("   Enhanced     |     %.3f      |     %.3f      |     %.3f\n",
                             recallBalanced[0], recallBalanced[1], recallBalanced[2]);
            
            double avgRecallStandard = Arrays.stream(recallStandard).average().orElse(0.0);
            double avgRecallBalanced = Arrays.stream(recallBalanced).average().orElse(0.0);
            
            System.out.printf("\n📈 Average Recall Improvement: %.3f → %.3f\n", 
                             avgRecallStandard, avgRecallBalanced);
            
            System.out.println("💡 Recommendations for Class Imbalance:");
            System.out.println("   - Use stratified sampling for train/test splits");
            System.out.println("   - Consider class weights in algorithms that support them");
            System.out.println("   - Use ensemble methods like Random Forest");
            System.out.println("   - Monitor per-class recall, not just overall accuracy");
            
            System.out.println("\n   ✅ Class imbalance demonstration completed!");
            
        } catch (Exception e) {
            System.err.println("   ❌ Error in class imbalance demo: " + e.getMessage());
        }
    }
    
    // Utility methods
    private static double[][] generateMulticlassData(int nSamples, int nFeatures, int nClasses) {
        Random random = new Random(42);
        double[][] X = new double[nSamples][nFeatures];
        
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                X[i][j] = random.nextGaussian();
            }
        }
        
        return X;
    }
    
    private static double[] generateMulticlassLabels(double[][] X, int nClasses) {
        Random random = new Random(42);
        double[] y = new double[X.length];
        
        for (int i = 0; i < X.length; i++) {
            // Create separable classes based on features
            double sum = 0;
            for (double feature : X[i]) {
                sum += feature;
            }
            
            if (nClasses == 3) {
                if (sum < -0.5) y[i] = 0;
                else if (sum < 0.5) y[i] = 1;
                else y[i] = 2;
            } else if (nClasses == 4) {
                if (sum < -1.0) y[i] = 0;
                else if (sum < 0.0) y[i] = 1;
                else if (sum < 1.0) y[i] = 2;
                else y[i] = 3;
            } else {
                y[i] = random.nextInt(nClasses);
            }
        }
        
        return y;
    }
    
    private static double[][] generateImbalancedMulticlassData(int nSamples, int nFeatures) {
        return generateMulticlassData(nSamples, nFeatures, 3);
    }
    
    private static double[] generateImbalancedLabels(double[][] X) {
        Random random = new Random(42);
        double[] y = new double[X.length];
        
        for (int i = 0; i < X.length; i++) {
            double prob = random.nextDouble();
            // Create imbalanced distribution: 60% class 0, 30% class 1, 10% class 2
            if (prob < 0.60) y[i] = 0;
            else if (prob < 0.90) y[i] = 1;
            else y[i] = 2;
        }
        
        return y;
    }
    
    private static int getUniqueClasses(double[] y) {
        return (int) Arrays.stream(y).distinct().count();
    }
    
    private static double calculateAccuracy(double[] yTrue, double[] yPred) {
        int correct = 0;
        for (int i = 0; i < yTrue.length; i++) {
            if (yTrue[i] == yPred[i]) correct++;
        }
        return (double) correct / yTrue.length;
    }
    
    private static double[] calculatePrecisionPerClass(double[] yTrue, double[] yPred, int nClasses) {
        double[] precision = new double[nClasses];
        
        for (int cls = 0; cls < nClasses; cls++) {
            int truePositives = 0;
            int falsePositives = 0;
            
            for (int i = 0; i < yTrue.length; i++) {
                if (yPred[i] == cls) {
                    if (yTrue[i] == cls) truePositives++;
                    else falsePositives++;
                }
            }
            
            precision[cls] = (truePositives + falsePositives > 0) ? 
                            (double) truePositives / (truePositives + falsePositives) : 0.0;
        }
        
        return precision;
    }
    
    private static double[] calculateRecallPerClass(double[] yTrue, double[] yPred, int nClasses) {
        double[] recall = new double[nClasses];
        
        for (int cls = 0; cls < nClasses; cls++) {
            int truePositives = 0;
            int falseNegatives = 0;
            
            for (int i = 0; i < yTrue.length; i++) {
                if (yTrue[i] == cls) {
                    if (yPred[i] == cls) truePositives++;
                    else falseNegatives++;
                }
            }
            
            recall[cls] = (truePositives + falseNegatives > 0) ? 
                         (double) truePositives / (truePositives + falseNegatives) : 0.0;
        }
        
        return recall;
    }
    
    private static double[] calculateF1PerClass(double[] precision, double[] recall) {
        double[] f1 = new double[precision.length];
        
        for (int i = 0; i < precision.length; i++) {
            f1[i] = (precision[i] + recall[i] > 0) ? 
                   2 * precision[i] * recall[i] / (precision[i] + recall[i]) : 0.0;
        }
        
        return f1;
    }
    
    private static int[][] calculateConfusionMatrix(double[] yTrue, double[] yPred, int nClasses) {
        int[][] matrix = new int[nClasses][nClasses];
        
        for (int i = 0; i < yTrue.length; i++) {
            int trueClass = (int) yTrue[i];
            int predClass = (int) yPred[i];
            if (trueClass < nClasses && predClass < nClasses) {
                matrix[trueClass][predClass]++;
            }
        }
        
        return matrix;
    }
}
