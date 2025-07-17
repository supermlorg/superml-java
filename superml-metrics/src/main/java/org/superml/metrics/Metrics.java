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

package org.superml.metrics;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Classification and regression metrics.
 * Similar to sklearn.metrics
 */
public class Metrics {
    
    // Classification metrics
    
    /**
     * Calculate accuracy score.
     * @param yTrue true labels
     * @param yPred predicted labels
     * @return accuracy score
     */
    public static double accuracy(double[] yTrue, double[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        int correct = 0;
        for (int i = 0; i < yTrue.length; i++) {
            if (yTrue[i] == yPred[i]) {
                correct++;
            }
        }
        return (double) correct / yTrue.length;
    }
    
    /**
     * Calculate precision score for binary classification.
     * @param yTrue true labels
     * @param yPred predicted labels
     * @return precision score
     */
    public static double precision(double[] yTrue, double[] yPred) {
        int truePositives = 0;
        int falsePositives = 0;
        
        for (int i = 0; i < yTrue.length; i++) {
            if (yPred[i] == 1.0) {
                if (yTrue[i] == 1.0) {
                    truePositives++;
                } else {
                    falsePositives++;
                }
            }
        }
        
        return truePositives + falsePositives == 0 ? 0.0 : 
               (double) truePositives / (truePositives + falsePositives);
    }
    
    /**
     * Calculate recall score for binary classification.
     * @param yTrue true labels
     * @param yPred predicted labels
     * @return recall score
     */
    public static double recall(double[] yTrue, double[] yPred) {
        int truePositives = 0;
        int falseNegatives = 0;
        
        for (int i = 0; i < yTrue.length; i++) {
            if (yTrue[i] == 1.0) {
                if (yPred[i] == 1.0) {
                    truePositives++;
                } else {
                    falseNegatives++;
                }
            }
        }
        
        return truePositives + falseNegatives == 0 ? 0.0 : 
               (double) truePositives / (truePositives + falseNegatives);
    }
    
    /**
     * Calculate F1 score for binary classification.
     * @param yTrue true labels
     * @param yPred predicted labels
     * @return F1 score
     */
    public static double f1Score(double[] yTrue, double[] yPred) {
        double p = precision(yTrue, yPred);
        double r = recall(yTrue, yPred);
        return p + r == 0.0 ? 0.0 : 2 * (p * r) / (p + r);
    }
    
    /**
     * Generate a confusion matrix.
     * @param yTrue true labels
     * @param yPred predicted labels
     * @return confusion matrix as 2D array
     */
    public static int[][] confusionMatrix(double[] yTrue, double[] yPred) {
        // Find number of unique classes
        int maxClass = 0;
        for (int i = 0; i < yTrue.length; i++) {
            maxClass = Math.max(maxClass, Math.max((int) yTrue[i], (int) yPred[i]));
        }
        int numClasses = maxClass + 1;
        
        int[][] matrix = new int[numClasses][numClasses];
        
        for (int i = 0; i < yTrue.length; i++) {
            int trueLabel = (int) yTrue[i];
            int predLabel = (int) yPred[i];
            matrix[trueLabel][predLabel]++;
        }
        
        return matrix;
    }
    
    /**
     * Generate a classification report.
     * @param yTrue true labels
     * @param yPred predicted labels
     * @return map containing metrics
     */
    public static Map<String, Double> classificationReport(double[] yTrue, double[] yPred) {
        Map<String, Double> report = new HashMap<>();
        
        report.put("accuracy", accuracy(yTrue, yPred));
        report.put("precision", precision(yTrue, yPred));
        report.put("recall", recall(yTrue, yPred));
        report.put("f1_score", f1Score(yTrue, yPred));
        
        return report;
    }
    
    // Regression metrics
    
    /**
     * Calculate mean squared error.
     * @param yTrue true values
     * @param yPred predicted values
     * @return MSE
     */
    public static double meanSquaredError(double[] yTrue, double[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        double sum = 0.0;
        for (int i = 0; i < yTrue.length; i++) {
            sum += Math.pow(yTrue[i] - yPred[i], 2);
        }
        return sum / yTrue.length;
    }
    
    /**
     * Calculate root mean squared error.
     * @param yTrue true values
     * @param yPred predicted values
     * @return RMSE
     */
    public static double rootMeanSquaredError(double[] yTrue, double[] yPred) {
        return Math.sqrt(meanSquaredError(yTrue, yPred));
    }
    
    /**
     * Calculate mean absolute error.
     * @param yTrue true values
     * @param yPred predicted values
     * @return MAE
     */
    public static double meanAbsoluteError(double[] yTrue, double[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        double sum = 0.0;
        for (int i = 0; i < yTrue.length; i++) {
            sum += Math.abs(yTrue[i] - yPred[i]);
        }
        return sum / yTrue.length;
    }
    
    /**
     * Calculate R² (coefficient of determination) regression score.
     * @param yTrue true values
     * @param yPred predicted values
     * @return R² score
     */
    public static double r2Score(double[] yTrue, double[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        double mean = Arrays.stream(yTrue).average().orElse(0.0);
        double totalSumSquares = 0.0;
        double residualSumSquares = 0.0;
        
        for (int i = 0; i < yTrue.length; i++) {
            totalSumSquares += Math.pow(yTrue[i] - mean, 2);
            residualSumSquares += Math.pow(yTrue[i] - yPred[i], 2);
        }
        
        return 1.0 - (residualSumSquares / totalSumSquares);
    }
    
    /**
     * Calculate ROC AUC score for binary classification.
     * @param yTrue true binary labels (0 or 1)
     * @param yPred predicted probabilities
     * @return ROC AUC score
     */
    public static double rocAuc(double[] yTrue, double[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        // Create array of (prediction, label) pairs
        int n = yTrue.length;
        double[][] pairs = new double[n][2];
        for (int i = 0; i < n; i++) {
            pairs[i][0] = yPred[i];  // prediction
            pairs[i][1] = yTrue[i];  // true label
        }
        
        // Sort by prediction score (descending)
        Arrays.sort(pairs, (a, b) -> Double.compare(b[0], a[0]));
        
        // Count positive and negative samples
        int positives = 0;
        int negatives = 0;
        for (int i = 0; i < n; i++) {
            if (pairs[i][1] == 1.0) positives++;
            else negatives++;
        }
        
        if (positives == 0 || negatives == 0) {
            return 0.5; // No discrimination possible
        }
        
        // Calculate AUC using trapezoidal rule
        double auc = 0.0;
        int truePositives = 0;
        int falsePositives = 0;
        
        for (int i = 0; i < n; i++) {
            if (pairs[i][1] == 1.0) {
                truePositives++;
            } else {
                falsePositives++;
                // Add area under curve
                auc += (double) truePositives / positives;
            }
        }
        
        return auc / negatives;
    }
}
