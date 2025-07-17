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
import java.util.Map;
import java.util.HashMap;

/**
 * Neural network specific metrics for model evaluation.
 * Provides specialized metrics for deep learning models.
 */
public class NeuralNetworkMetrics {
    
    /**
     * Calculate cross-entropy loss for binary classification
     */
    public static double binaryCrossEntropy(double[] yTrue, double[] yPred) {
        double loss = 0.0;
        int n = yTrue.length;
        
        for (int i = 0; i < n; i++) {
            double y = yTrue[i];
            double p = Math.max(1e-15, Math.min(1 - 1e-15, yPred[i])); // Clip predictions
            loss += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
        }
        
        return loss / n;
    }
    
    /**
     * Calculate categorical cross-entropy loss for multi-class classification
     */
    public static double categoricalCrossEntropy(double[][] yTrue, double[][] yPred) {
        double loss = 0.0;
        int n = yTrue.length;
        int classes = yTrue[0].length;
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < classes; j++) {
                double y = yTrue[i][j];
                double p = Math.max(1e-15, Math.min(1 - 1e-15, yPred[i][j]));
                loss += -y * Math.log(p);
            }
        }
        
        return loss / n;
    }
    
    /**
     * Calculate sparse categorical cross-entropy loss
     */
    public static double sparseCategoricalCrossEntropy(int[] yTrue, double[][] yPred) {
        double loss = 0.0;
        int n = yTrue.length;
        
        for (int i = 0; i < n; i++) {
            int trueClass = yTrue[i];
            double p = Math.max(1e-15, Math.min(1 - 1e-15, yPred[i][trueClass]));
            loss += -Math.log(p);
        }
        
        return loss / n;
    }
    
    /**
     * Calculate mean squared error loss for regression
     */
    public static double meanSquaredError(double[] yTrue, double[] yPred) {
        double loss = 0.0;
        int n = yTrue.length;
        
        for (int i = 0; i < n; i++) {
            double diff = yTrue[i] - yPred[i];
            loss += diff * diff;
        }
        
        return loss / n;
    }
    
    /**
     * Calculate mean absolute error loss for regression
     */
    public static double meanAbsoluteError(double[] yTrue, double[] yPred) {
        double loss = 0.0;
        int n = yTrue.length;
        
        for (int i = 0; i < n; i++) {
            loss += Math.abs(yTrue[i] - yPred[i]);
        }
        
        return loss / n;
    }
    
    /**
     * Calculate top-k categorical accuracy
     */
    public static double topKAccuracy(int[] yTrue, double[][] yPred, int k) {
        int correct = 0;
        int n = yTrue.length;
        
        for (int i = 0; i < n; i++) {
            int trueClass = yTrue[i];
            
            // Get top-k predictions
            double[] predictions = yPred[i].clone();
            Integer[] indices = new Integer[predictions.length];
            for (int j = 0; j < indices.length; j++) {
                indices[j] = j;
            }
            
            Arrays.sort(indices, (a, b) -> Double.compare(predictions[b], predictions[a]));
            
            // Check if true class is in top-k
            for (int j = 0; j < Math.min(k, indices.length); j++) {
                if (indices[j] == trueClass) {
                    correct++;
                    break;
                }
            }
        }
        
        return (double) correct / n;
    }
    
    /**
     * Calculate perplexity for language models
     */
    public static double perplexity(double[] yTrue, double[] yPred) {
        double crossEntropy = binaryCrossEntropy(yTrue, yPred);
        return Math.exp(crossEntropy);
    }
    
    /**
     * Calculate comprehensive neural network metrics
     */
    public static Map<String, Double> comprehensiveMetrics(double[] yTrue, double[] yPred, 
                                                          String taskType) {
        Map<String, Double> metrics = new HashMap<>();
        
        switch (taskType.toLowerCase()) {
            case "binary_classification":
                metrics.put("binary_cross_entropy", binaryCrossEntropy(yTrue, yPred));
                metrics.put("accuracy", Metrics.accuracy(yTrue, yPred));
                metrics.put("precision", Metrics.precision(yTrue, yPred));
                metrics.put("recall", Metrics.recall(yTrue, yPred));
                metrics.put("f1_score", Metrics.f1Score(yTrue, yPred));
                metrics.put("auc_roc", Metrics.rocAuc(yTrue, yPred));
                break;
                
            case "regression":
                metrics.put("mse", meanSquaredError(yTrue, yPred));
                metrics.put("mae", meanAbsoluteError(yTrue, yPred));
                metrics.put("rmse", Math.sqrt(meanSquaredError(yTrue, yPred)));
                metrics.put("r2_score", Metrics.r2Score(yTrue, yPred));
                break;
                
            default:
                // Default to classification metrics
                metrics.put("accuracy", Metrics.accuracy(yTrue, yPred));
                metrics.put("binary_cross_entropy", binaryCrossEntropy(yTrue, yPred));
        }
        
        return metrics;
    }
    
    /**
     * Calculate gradient norm for monitoring training
     */
    public static double gradientNorm(double[][] gradients) {
        double norm = 0.0;
        
        for (double[] layer : gradients) {
            for (double grad : layer) {
                norm += grad * grad;
            }
        }
        
        return Math.sqrt(norm);
    }
    
    /**
     * Calculate layer-wise activation statistics
     */
    public static Map<String, Double> activationStats(double[][] activations) {
        Map<String, Double> stats = new HashMap<>();
        
        if (activations.length == 0) return stats;
        
        // Flatten all activations
        double[] allActivations = new double[activations.length * activations[0].length];
        int idx = 0;
        for (double[] layer : activations) {
            System.arraycopy(layer, 0, allActivations, idx, layer.length);
            idx += layer.length;
        }
        
        // Calculate statistics
        Arrays.sort(allActivations);
        double mean = Arrays.stream(allActivations).average().orElse(0.0);
        double std = Math.sqrt(Arrays.stream(allActivations)
            .map(x -> Math.pow(x - mean, 2))
            .average().orElse(0.0));
        
        stats.put("mean", mean);
        stats.put("std", std);
        stats.put("min", allActivations[0]);
        stats.put("max", allActivations[allActivations.length - 1]);
        stats.put("median", allActivations[allActivations.length / 2]);
        
        return stats;
    }
    
    /**
     * Calculate class-wise metrics for multi-class problems
     */
    public static Map<String, Map<String, Double>> classWiseMetrics(int[] yTrue, int[] yPred, 
                                                                   int numClasses) {
        Map<String, Map<String, Double>> classMetrics = new HashMap<>();
        
        for (int cls = 0; cls < numClasses; cls++) {
            Map<String, Double> metrics = new HashMap<>();
            
            // Convert to binary problem for this class
            double[] binaryTrue = new double[yTrue.length];
            double[] binaryPred = new double[yPred.length];
            
            for (int i = 0; i < yTrue.length; i++) {
                binaryTrue[i] = yTrue[i] == cls ? 1.0 : 0.0;
                binaryPred[i] = yPred[i] == cls ? 1.0 : 0.0;
            }
            
            metrics.put("precision", Metrics.precision(binaryTrue, binaryPred));
            metrics.put("recall", Metrics.recall(binaryTrue, binaryPred));
            metrics.put("f1_score", Metrics.f1Score(binaryTrue, binaryPred));
            
            classMetrics.put("class_" + cls, metrics);
        }
        
        return classMetrics;
    }
    
    /**
     * Calculate training convergence metrics
     */
    public static Map<String, Double> convergenceMetrics(double[] lossHistory) {
        Map<String, Double> metrics = new HashMap<>();
        
        if (lossHistory.length < 2) {
            return metrics;
        }
        
        // Loss improvement over last epoch
        double lastImprovement = lossHistory[lossHistory.length - 2] - lossHistory[lossHistory.length - 1];
        metrics.put("last_improvement", lastImprovement);
        
        // Average improvement over last 10 epochs
        int windowSize = Math.min(10, lossHistory.length - 1);
        double avgImprovement = 0.0;
        for (int i = lossHistory.length - windowSize - 1; i < lossHistory.length - 1; i++) {
            avgImprovement += lossHistory[i] - lossHistory[i + 1];
        }
        avgImprovement /= windowSize;
        metrics.put("avg_improvement", avgImprovement);
        
        // Loss volatility (standard deviation of recent losses)
        double[] recentLosses = Arrays.copyOfRange(lossHistory, 
            Math.max(0, lossHistory.length - windowSize), lossHistory.length);
        double mean = Arrays.stream(recentLosses).average().orElse(0.0);
        double volatility = Math.sqrt(Arrays.stream(recentLosses)
            .map(x -> Math.pow(x - mean, 2))
            .average().orElse(0.0));
        metrics.put("loss_volatility", volatility);
        
        return metrics;
    }
}
