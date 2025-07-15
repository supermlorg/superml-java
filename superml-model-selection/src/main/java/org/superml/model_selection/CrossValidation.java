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

package org.superml.model_selection;

import org.superml.core.Classifier;
import org.superml.core.Regressor;
import org.superml.metrics.Metrics;

import java.util.*;
import java.util.concurrent.*;

/**
 * Cross-validation utilities for model evaluation.
 * Supports k-fold cross-validation with various metrics.
 */
public class CrossValidation {
    
    /**
     * Results from cross-validation evaluation
     */
    public static class CrossValidationResults {
        private final Map<String, double[]> scores;
        private final Map<String, Double> meanScores;
        private final Map<String, Double> stdScores;
        private final int nFolds;
        
        public CrossValidationResults(Map<String, double[]> scores, int nFolds) {
            this.scores = new HashMap<>(scores);
            this.nFolds = nFolds;
            this.meanScores = new HashMap<>();
            this.stdScores = new HashMap<>();
            
            // Calculate mean and standard deviation for each metric
            for (Map.Entry<String, double[]> entry : scores.entrySet()) {
                String metric = entry.getKey();
                double[] values = entry.getValue();
                
                double mean = Arrays.stream(values).average().orElse(0.0);
                double variance = Arrays.stream(values)
                    .map(x -> Math.pow(x - mean, 2))
                    .average().orElse(0.0);
                double std = Math.sqrt(variance);
                
                meanScores.put(metric, mean);
                stdScores.put(metric, std);
            }
        }
        
        public Map<String, double[]> getScores() { return new HashMap<>(scores); }
        public Map<String, Double> getMeanScores() { return new HashMap<>(meanScores); }
        public Map<String, Double> getStdScores() { return new HashMap<>(stdScores); }
        public int getNumFolds() { return nFolds; }
        
        public double getMeanScore(String metric) {
            return meanScores.getOrDefault(metric, 0.0);
        }
        
        public double getStdScore(String metric) {
            return stdScores.getOrDefault(metric, 0.0);
        }
        
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("Cross-Validation Results (").append(nFolds).append(" folds):\n");
            for (String metric : meanScores.keySet()) {
                sb.append(String.format("%s: %.4f (+/- %.4f)\n", 
                    metric, getMeanScore(metric), getStdScore(metric) * 2));
            }
            return sb.toString();
        }
    }
    
    /**
     * Configuration for cross-validation
     */
    public static class CrossValidationConfig {
        private int nFolds = 5;
        private boolean shuffle = true;
        private Long randomSeed = null;
        private boolean parallel = false;
        private Set<String> metrics = new HashSet<>();
        
        public CrossValidationConfig() {
            // Default metrics for classification
            metrics.add("accuracy");
            metrics.add("precision");
            metrics.add("recall");
            metrics.add("f1");
        }
        
        public CrossValidationConfig setFolds(int nFolds) {
            if (nFolds < 2) throw new IllegalArgumentException("Number of folds must be >= 2");
            this.nFolds = nFolds;
            return this;
        }
        
        public CrossValidationConfig setShuffle(boolean shuffle) {
            this.shuffle = shuffle;
            return this;
        }
        
        public CrossValidationConfig setRandomSeed(long seed) {
            this.randomSeed = seed;
            return this;
        }
        
        public CrossValidationConfig setParallel(boolean parallel) {
            this.parallel = parallel;
            return this;
        }
        
        public CrossValidationConfig setMetrics(String... metrics) {
            this.metrics = new HashSet<>(Arrays.asList(metrics));
            return this;
        }
        
        public CrossValidationConfig addMetric(String metric) {
            this.metrics.add(metric);
            return this;
        }
        
        // Getters
        public int getFolds() { return nFolds; }
        public boolean isShuffle() { return shuffle; }
        public Long getRandomSeed() { return randomSeed; }
        public boolean isParallel() { return parallel; }
        public Set<String> getMetrics() { return new HashSet<>(metrics); }
    }
    
    /**
     * Perform k-fold cross-validation on a classifier
     */
    public static CrossValidationResults crossValidate(
            Classifier classifier, 
            double[][] X, 
            double[] y, 
            CrossValidationConfig config) {
        
        if (X.length != y.length) {
            throw new IllegalArgumentException("X and y must have the same number of samples");
        }
        
        // Create fold indices
        int[][] foldIndices = createFolds(X.length, config.getFolds(), 
                                        config.isShuffle(), config.getRandomSeed());
        
        Map<String, double[]> allScores = new HashMap<>();
        for (String metric : config.getMetrics()) {
            allScores.put(metric, new double[config.getFolds()]);
        }
        
        if (config.isParallel()) {
            evaluateParallel(classifier, X, y, foldIndices, allScores, config);
        } else {
            evaluateSequential(classifier, X, y, foldIndices, allScores, config);
        }
        
        return new CrossValidationResults(allScores, config.getFolds());
    }
    
    /**
     * Convenience method with default configuration
     */
    public static CrossValidationResults crossValidate(
            Classifier classifier, 
            double[][] X, 
            double[] y) {
        return crossValidate(classifier, X, y, new CrossValidationConfig());
    }
    
    /**
     * Convenience method with custom number of folds
     */
    public static CrossValidationResults crossValidate(
            Classifier classifier, 
            double[][] X, 
            double[] y, 
            int nFolds) {
        return crossValidate(classifier, X, y, 
                           new CrossValidationConfig().setFolds(nFolds));
    }
    
    /**
     * Create fold indices for cross-validation
     */
    private static int[][] createFolds(int nSamples, int nFolds, boolean shuffle, Long seed) {
        // Create array of indices
        Integer[] indices = new Integer[nSamples];
        for (int i = 0; i < nSamples; i++) {
            indices[i] = i;
        }
        
        // Shuffle if requested
        if (shuffle) {
            Random random = seed != null ? new Random(seed) : new Random();
            Collections.shuffle(Arrays.asList(indices), random);
        }
        
        // Create folds
        int[][] folds = new int[nFolds][];
        int foldSize = nSamples / nFolds;
        int remainder = nSamples % nFolds;
        
        int startIdx = 0;
        for (int fold = 0; fold < nFolds; fold++) {
            int currentFoldSize = foldSize + (fold < remainder ? 1 : 0);
            folds[fold] = new int[currentFoldSize];
            
            for (int i = 0; i < currentFoldSize; i++) {
                folds[fold][i] = indices[startIdx + i];
            }
            startIdx += currentFoldSize;
        }
        
        return folds;
    }
    
    /**
     * Sequential evaluation of folds
     */
    private static void evaluateSequential(
            Classifier classifier,
            double[][] X,
            double[] y,
            int[][] foldIndices,
            Map<String, double[]> allScores,
            CrossValidationConfig config) {
        
        for (int fold = 0; fold < foldIndices.length; fold++) {
            Map<String, Double> foldScores = evaluateFold(classifier, X, y, 
                                                        foldIndices, fold, config);
            
            for (Map.Entry<String, Double> entry : foldScores.entrySet()) {
                allScores.get(entry.getKey())[fold] = entry.getValue();
            }
        }
    }
    
    /**
     * Parallel evaluation of folds
     */
    private static void evaluateParallel(
            Classifier classifier,
            double[][] X,
            double[] y,
            int[][] foldIndices,
            Map<String, double[]> allScores,
            CrossValidationConfig config) {
        
        int nThreads = Math.min(foldIndices.length, 
                               Runtime.getRuntime().availableProcessors());
        ExecutorService executor = Executors.newFixedThreadPool(nThreads);
        
        List<Future<Map<String, Double>>> futures = new ArrayList<>();
        
        for (int fold = 0; fold < foldIndices.length; fold++) {
            final int currentFold = fold;
            Future<Map<String, Double>> future = executor.submit(() -> {
                return evaluateFold(classifier, X, y, foldIndices, currentFold, config);
            });
            futures.add(future);
        }
        
        try {
            for (int fold = 0; fold < futures.size(); fold++) {
                Map<String, Double> foldScores = futures.get(fold).get();
                for (Map.Entry<String, Double> entry : foldScores.entrySet()) {
                    allScores.get(entry.getKey())[fold] = entry.getValue();
                }
            }
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException("Error in parallel cross-validation", e);
        } finally {
            executor.shutdown();
        }
    }
    
    /**
     * Evaluate a single fold
     */
    private static Map<String, Double> evaluateFold(
            Classifier classifier,
            double[][] X,
            double[] y,
            int[][] foldIndices,
            int testFold,
            CrossValidationConfig config) {
        
        // Split data into train and test sets
        int[] testIndices = foldIndices[testFold];
        List<Integer> trainIndicesList = new ArrayList<>();
        
        for (int fold = 0; fold < foldIndices.length; fold++) {
            if (fold != testFold) {
                for (int idx : foldIndices[fold]) {
                    trainIndicesList.add(idx);
                }
            }
        }
        
        int[] trainIndices = trainIndicesList.stream().mapToInt(i -> i).toArray();
        
        // Create train and test datasets
        double[][] XTrain = new double[trainIndices.length][];
        double[] yTrain = new double[trainIndices.length];
        double[][] XTest = new double[testIndices.length][];
        double[] yTest = new double[testIndices.length];
        
        for (int i = 0; i < trainIndices.length; i++) {
            XTrain[i] = X[trainIndices[i]].clone();
            yTrain[i] = y[trainIndices[i]];
        }
        
        for (int i = 0; i < testIndices.length; i++) {
            XTest[i] = X[testIndices[i]].clone();
            yTest[i] = y[testIndices[i]];
        }
        
        // Create a copy of the classifier to avoid interference between folds
        Classifier foldClassifier = cloneClassifier(classifier);
        
        // Train and predict
        foldClassifier.fit(XTrain, yTrain);
        double[] yPred = foldClassifier.predict(XTest);
        
        // Calculate metrics
        Map<String, Double> scores = new HashMap<>();
        for (String metric : config.getMetrics()) {
            double score = calculateMetric(metric, yTest, yPred);
            scores.put(metric, score);
        }
        
        return scores;
    }
    
    /**
     * Clone a classifier for fold evaluation
     * Note: This is a simple approach. In practice, you might want a more sophisticated cloning mechanism.
     */
    private static Classifier cloneClassifier(Classifier classifier) {
        try {
            // For now, create a new instance of the same class
            return classifier.getClass().getDeclaredConstructor().newInstance();
        } catch (Exception e) {
            throw new RuntimeException("Unable to clone classifier: " + classifier.getClass(), e);
        }
    }
    
    /**
     * Calculate specific metric
     */
    private static double calculateMetric(String metric, double[] yTrue, double[] yPred) {
        switch (metric.toLowerCase()) {
            case "accuracy":
                return Metrics.accuracy(yTrue, yPred);
            case "precision":
                return Metrics.precision(yTrue, yPred);
            case "recall":
                return Metrics.recall(yTrue, yPred);
            case "f1":
                return Metrics.f1Score(yTrue, yPred);
            case "auc":
            case "roc_auc":
                // Note: This would require probability predictions
                // For now, return a placeholder
                return 0.5; // Placeholder - would need proper AUC implementation
            default:
                throw new IllegalArgumentException("Unknown metric: " + metric);
        }
    }
    
    /**
     * Cross-validation for regression models
     */
    public static CrossValidationResults crossValidateRegression(
            Regressor regressor,
            double[][] X,
            double[] y,
            CrossValidationConfig config) {
        
        // Set appropriate regression metrics
        config.setMetrics("mse", "mae", "r2");
        
        if (X.length != y.length) {
            throw new IllegalArgumentException("X and y must have the same number of samples");
        }
        
        int[][] foldIndices = createFolds(X.length, config.getFolds(), 
                                        config.isShuffle(), config.getRandomSeed());
        
        Map<String, double[]> allScores = new HashMap<>();
        for (String metric : config.getMetrics()) {
            allScores.put(metric, new double[config.getFolds()]);
        }
        
        for (int fold = 0; fold < foldIndices.length; fold++) {
            Map<String, Double> foldScores = evaluateRegressionFold(regressor, X, y, 
                                                                  foldIndices, fold, config);
            
            for (Map.Entry<String, Double> entry : foldScores.entrySet()) {
                allScores.get(entry.getKey())[fold] = entry.getValue();
            }
        }
        
        return new CrossValidationResults(allScores, config.getFolds());
    }
    
    /**
     * Evaluate a single regression fold
     */
    private static Map<String, Double> evaluateRegressionFold(
            Regressor regressor,
            double[][] X,
            double[] y,
            int[][] foldIndices,
            int testFold,
            CrossValidationConfig config) {
        
        // Similar to classification fold evaluation but for regression
        int[] testIndices = foldIndices[testFold];
        List<Integer> trainIndicesList = new ArrayList<>();
        
        for (int fold = 0; fold < foldIndices.length; fold++) {
            if (fold != testFold) {
                for (int idx : foldIndices[fold]) {
                    trainIndicesList.add(idx);
                }
            }
        }
        
        int[] trainIndices = trainIndicesList.stream().mapToInt(i -> i).toArray();
        
        // Create datasets
        double[][] XTrain = new double[trainIndices.length][];
        double[] yTrain = new double[trainIndices.length];
        double[][] XTest = new double[testIndices.length][];
        double[] yTest = new double[testIndices.length];
        
        for (int i = 0; i < trainIndices.length; i++) {
            XTrain[i] = X[trainIndices[i]].clone();
            yTrain[i] = y[trainIndices[i]];
        }
        
        for (int i = 0; i < testIndices.length; i++) {
            XTest[i] = X[testIndices[i]].clone();
            yTest[i] = y[testIndices[i]];
        }
        
        // Clone regressor
        Regressor foldRegressor = cloneRegressor(regressor);
        
        // Train and predict
        foldRegressor.fit(XTrain, yTrain);
        double[] yPred = foldRegressor.predict(XTest);
        
        // Calculate regression metrics
        Map<String, Double> scores = new HashMap<>();
        for (String metric : config.getMetrics()) {
            double score = calculateRegressionMetric(metric, yTest, yPred);
            scores.put(metric, score);
        }
        
        return scores;
    }
    
    /**
     * Clone a regressor
     */
    private static Regressor cloneRegressor(Regressor regressor) {
        try {
            return regressor.getClass().getDeclaredConstructor().newInstance();
        } catch (Exception e) {
            throw new RuntimeException("Unable to clone regressor: " + regressor.getClass(), e);
        }
    }
    
    /**
     * Calculate regression metrics
     */
    private static double calculateRegressionMetric(String metric, double[] yTrue, double[] yPred) {
        switch (metric.toLowerCase()) {
            case "mse":
                return meanSquaredError(yTrue, yPred);
            case "mae":
                return meanAbsoluteError(yTrue, yPred);
            case "r2":
                return r2Score(yTrue, yPred);
            default:
                throw new IllegalArgumentException("Unknown regression metric: " + metric);
        }
    }
    
    private static double meanSquaredError(double[] yTrue, double[] yPred) {
        double sum = 0.0;
        for (int i = 0; i < yTrue.length; i++) {
            double diff = yTrue[i] - yPred[i];
            sum += diff * diff;
        }
        return sum / yTrue.length;
    }
    
    private static double meanAbsoluteError(double[] yTrue, double[] yPred) {
        double sum = 0.0;
        for (int i = 0; i < yTrue.length; i++) {
            sum += Math.abs(yTrue[i] - yPred[i]);
        }
        return sum / yTrue.length;
    }
    
    private static double r2Score(double[] yTrue, double[] yPred) {
        double mean = Arrays.stream(yTrue).average().orElse(0.0);
        double totalSumSquares = 0.0;
        double residualSumSquares = 0.0;
        
        for (int i = 0; i < yTrue.length; i++) {
            totalSumSquares += Math.pow(yTrue[i] - mean, 2);
            residualSumSquares += Math.pow(yTrue[i] - yPred[i], 2);
        }
        
        return 1.0 - (residualSumSquares / totalSumSquares);
    }
}
