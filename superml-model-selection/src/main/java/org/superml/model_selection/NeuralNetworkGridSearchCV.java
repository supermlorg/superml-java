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

import org.superml.core.Estimator;
import org.superml.core.SupervisedLearner;
import org.superml.neural.MLPClassifier;
import org.superml.neural.CNNClassifier;
import org.superml.neural.RNNClassifier;
import org.superml.pipeline.Pipeline;

import java.util.*;

/**
 * Neural network specific model selection utilities.
 * Provides specialized hyperparameter tuning for deep learning models.
 */
public class NeuralNetworkGridSearchCV {
    
    private final Map<String, Object[]> paramGrid;
    private final String scoring;
    private final int cv;
    private final boolean verbose;
    
    /**
     * Results from grid search
     */
    public static class GridSearchResult {
        public final Map<String, Object> bestParams;
        public final double bestScore;
        public final Estimator bestEstimator;
        public final List<Map<String, Object>> cvResults;
        
        public GridSearchResult(Map<String, Object> bestParams, double bestScore,
                               Estimator bestEstimator, List<Map<String, Object>> cvResults) {
            this.bestParams = bestParams;
            this.bestScore = bestScore;
            this.bestEstimator = bestEstimator;
            this.cvResults = cvResults;
        }
    }
    
    public NeuralNetworkGridSearchCV(Map<String, Object[]> paramGrid, String scoring, int cv) {
        this(paramGrid, scoring, cv, false);
    }
    
    public NeuralNetworkGridSearchCV(Map<String, Object[]> paramGrid, String scoring, int cv, boolean verbose) {
        this.paramGrid = paramGrid;
        this.scoring = scoring;
        this.cv = cv;
        this.verbose = verbose;
    }
    
    /**
     * Perform grid search for neural network models
     */
    public GridSearchResult fit(Estimator estimator, double[][] X, double[] y) {
        List<Map<String, Object>> paramCombinations = generateParameterCombinations();
        
        double bestScore = Double.NEGATIVE_INFINITY;
        Map<String, Object> bestParams = null;
        Estimator bestEstimator = null;
        List<Map<String, Object>> cvResults = new ArrayList<>();
        
        for (int i = 0; i < paramCombinations.size(); i++) {
            Map<String, Object> params = paramCombinations.get(i);
            
            if (verbose) {
                System.out.printf("Fitting %d/%d parameter combinations...%n", i + 1, paramCombinations.size());
                System.out.println("Parameters: " + params);
            }
            
            // Create estimator copy with current parameters
            Estimator currentEstimator = createEstimatorCopy(estimator, params);
            
            // Perform cross-validation
            double[] scores = performCrossValidation(currentEstimator, X, y);
            double meanScore = Arrays.stream(scores).average().orElse(0.0);
            double stdScore = calculateStd(scores, meanScore);
            
            // Store results
            Map<String, Object> result = new HashMap<>();
            result.put("params", params);
            result.put("mean_test_score", meanScore);
            result.put("std_test_score", stdScore);
            result.put("scores", scores);
            cvResults.add(result);
            
            // Update best if this is better
            if (meanScore > bestScore) {
                bestScore = meanScore;
                bestParams = params;
                bestEstimator = createEstimatorCopy(estimator, params);
            }
            
            if (verbose) {
                System.out.printf("Score: %.4f (+/- %.4f)%n", meanScore, stdScore * 2);
                System.out.println();
            }
        }
        
        // Train best estimator on full dataset
        if (bestEstimator instanceof SupervisedLearner) {
            ((SupervisedLearner) bestEstimator).fit(X, y);
        }
        
        return new GridSearchResult(bestParams, bestScore, bestEstimator, cvResults);
    }
    
    /**
     * Generate all parameter combinations from grid
     */
    private List<Map<String, Object>> generateParameterCombinations() {
        List<Map<String, Object>> combinations = new ArrayList<>();
        List<String> paramNames = new ArrayList<>(paramGrid.keySet());
        
        generateCombinationsRecursive(combinations, new HashMap<>(), paramNames, 0);
        return combinations;
    }
    
    private void generateCombinationsRecursive(List<Map<String, Object>> combinations,
                                             Map<String, Object> current,
                                             List<String> paramNames, int index) {
        if (index == paramNames.size()) {
            combinations.add(new HashMap<>(current));
            return;
        }
        
        String paramName = paramNames.get(index);
        Object[] values = paramGrid.get(paramName);
        
        for (Object value : values) {
            current.put(paramName, value);
            generateCombinationsRecursive(combinations, current, paramNames, index + 1);
        }
        current.remove(paramName);
    }
    
    /**
     * Create copy of estimator with given parameters
     */
    private Estimator createEstimatorCopy(Estimator estimator, Map<String, Object> params) {
        try {
            Estimator copy;
            
            // Create new instance of the same type
            if (estimator instanceof MLPClassifier) {
                copy = new MLPClassifier();
            } else if (estimator instanceof CNNClassifier) {
                copy = new CNNClassifier();
            } else if (estimator instanceof RNNClassifier) {
                copy = new RNNClassifier();
            } else if (estimator instanceof Pipeline) {
                // For pipelines, would need more complex copying logic
                copy = estimator; // Placeholder
            } else {
                // Generic approach using reflection
                copy = estimator.getClass().getDeclaredConstructor().newInstance();
            }
            
            // Set parameters
            copy.setParams(params);
            return copy;
            
        } catch (Exception e) {
            throw new RuntimeException("Failed to create estimator copy", e);
        }
    }
    
    /**
     * Perform k-fold cross-validation
     */
    private double[] performCrossValidation(Estimator estimator, double[][] X, double[] y) {
        double[] scores = new double[cv];
        int foldSize = X.length / cv;
        
        for (int fold = 0; fold < cv; fold++) {
            int start = fold * foldSize;
            int end = (fold == cv - 1) ? X.length : start + foldSize;
            
            // Create train/val splits
            List<double[]> trainX = new ArrayList<>();
            List<Double> trainY = new ArrayList<>();
            List<double[]> valX = new ArrayList<>();
            List<Double> valY = new ArrayList<>();
            
            for (int i = 0; i < X.length; i++) {
                if (i >= start && i < end) {
                    valX.add(X[i]);
                    valY.add(y[i]);
                } else {
                    trainX.add(X[i]);
                    trainY.add(y[i]);
                }
            }
            
            // Convert to arrays
            double[][] XTrain = trainX.toArray(new double[0][]);
            double[] yTrain = trainY.stream().mapToDouble(Double::doubleValue).toArray();
            double[][] XVal = valX.toArray(new double[0][]);
            double[] yVal = valY.stream().mapToDouble(Double::doubleValue).toArray();
            
            // Train and evaluate
            if (estimator instanceof SupervisedLearner) {
                SupervisedLearner learner = (SupervisedLearner) estimator;
                learner.fit(XTrain, yTrain);
                scores[fold] = learner.score(XVal, yVal);
            } else {
                scores[fold] = 0.0; // Cannot evaluate unsupervised learners
            }
        }
        
        return scores;
    }
    
    /**
     * Calculate standard deviation
     */
    private double calculateStd(double[] values, double mean) {
        double variance = 0.0;
        for (double value : values) {
            variance += Math.pow(value - mean, 2);
        }
        return Math.sqrt(variance / values.length);
    }
    
    /**
     * Create standard parameter grids for neural networks
     */
    public static class StandardGrids {
        
        /**
         * Default MLP parameter grid
         */
        public static Map<String, Object[]> mlpGrid() {
            Map<String, Object[]> grid = new HashMap<>();
            grid.put("hidden_layer_sizes", new Object[]{
                new int[]{64}, new int[]{128}, new int[]{64, 32}, new int[]{128, 64}, new int[]{128, 64, 32}
            });
            grid.put("learning_rate", new Object[]{0.001, 0.01, 0.1});
            grid.put("activation", new Object[]{"relu", "tanh", "sigmoid"});
            grid.put("max_iter", new Object[]{100, 200, 300});
            return grid;
        }
        
        /**
         * Default CNN parameter grid
         */
        public static Map<String, Object[]> cnnGrid() {
            Map<String, Object[]> grid = new HashMap<>();
            grid.put("learning_rate", new Object[]{0.001, 0.01, 0.1});
            grid.put("max_epochs", new Object[]{30, 50, 100});
            grid.put("batch_size", new Object[]{16, 32, 64});
            return grid;
        }
        
        /**
         * Default RNN parameter grid
         */
        public static Map<String, Object[]> rnnGrid() {
            Map<String, Object[]> grid = new HashMap<>();
            grid.put("hidden_size", new Object[]{32, 64, 128});
            grid.put("num_layers", new Object[]{1, 2, 3});
            grid.put("learning_rate", new Object[]{0.001, 0.01, 0.1});
            grid.put("cell_type", new Object[]{"LSTM", "GRU"});
            grid.put("max_epochs", new Object[]{50, 75, 100});
            return grid;
        }
        
        /**
         * Quick parameter grid for fast experimentation
         */
        public static Map<String, Object[]> quickGrid() {
            Map<String, Object[]> grid = new HashMap<>();
            grid.put("learning_rate", new Object[]{0.01, 0.1});
            grid.put("max_iter", new Object[]{50, 100});
            return grid;
        }
    }
    
    /**
     * Random search variant for neural networks
     */
    public static class RandomizedSearchCV {
        private final Map<String, Object[]> paramDistributions;
        private final int nIter;
        private final String scoring;
        private final int cv;
        private final Random random;
        
        public RandomizedSearchCV(Map<String, Object[]> paramDistributions, int nIter,
                                 String scoring, int cv) {
            this.paramDistributions = paramDistributions;
            this.nIter = nIter;
            this.scoring = scoring;
            this.cv = cv;
            this.random = new Random(42);
        }
        
        /**
         * Perform randomized search
         */
        public GridSearchResult fit(Estimator estimator, double[][] X, double[] y) {
            NeuralNetworkGridSearchCV gridSearch = new NeuralNetworkGridSearchCV(
                generateRandomGrid(), scoring, cv, true);
            return gridSearch.fit(estimator, X, y);
        }
        
        /**
         * Generate random parameter combinations
         */
        private Map<String, Object[]> generateRandomGrid() {
            Map<String, Object[]> randomGrid = new HashMap<>();
            
            for (String paramName : paramDistributions.keySet()) {
                Object[] allValues = paramDistributions.get(paramName);
                Set<Object> selectedValues = new HashSet<>();
                
                // Randomly select values for this parameter
                int numValues = Math.min(nIter, allValues.length);
                while (selectedValues.size() < numValues) {
                    selectedValues.add(allValues[random.nextInt(allValues.length)]);
                }
                
                randomGrid.put(paramName, selectedValues.toArray());
            }
            
            return randomGrid;
        }
    }
}
