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

import java.util.*;
import java.util.concurrent.*;
import java.lang.reflect.Method;

/**
 * Hyperparameter tuning utilities supporting Grid Search, Random Search, and basic optimization.
 */
public class HyperparameterTuning {
    
    /**
     * Parameter specification for hyperparameter tuning
     */
    public static class ParameterSpec {
        private final String parameterName;
        private final Object[] values;
        private final ParameterType type;
        
        public enum ParameterType {
            DISCRETE, CONTINUOUS, INTEGER
        }
        
        private ParameterSpec(String parameterName, Object[] values, ParameterType type) {
            this.parameterName = parameterName;
            this.values = values;
            this.type = type;
        }
        
        public static ParameterSpec discrete(String name, Object... values) {
            return new ParameterSpec(name, values, ParameterType.DISCRETE);
        }
        
        public static ParameterSpec continuous(String name, double min, double max, int steps) {
            Double[] values = new Double[steps];
            double step = (max - min) / (steps - 1);
            for (int i = 0; i < steps; i++) {
                values[i] = min + i * step;
            }
            return new ParameterSpec(name, values, ParameterType.CONTINUOUS);
        }
        
        public static ParameterSpec integer(String name, int min, int max) {
            Integer[] values = new Integer[max - min + 1];
            for (int i = 0; i < values.length; i++) {
                values[i] = min + i;
            }
            return new ParameterSpec(name, values, ParameterType.INTEGER);
        }
        
        public String getName() { return parameterName; }
        public Object[] getValues() { return values.clone(); }
        public ParameterType getType() { return type; }
        
        public Object getRandomValue(Random random) {
            return values[random.nextInt(values.length)];
        }
    }
    
    /**
     * Parameter combination for a specific hyperparameter configuration
     */
    public static class ParameterCombination {
        private final Map<String, Object> parameters;
        
        public ParameterCombination(Map<String, Object> parameters) {
            this.parameters = new HashMap<>(parameters);
        }
        
        public Map<String, Object> getParameters() {
            return new HashMap<>(parameters);
        }
        
        public Object get(String parameterName) {
            return parameters.get(parameterName);
        }
        
        @Override
        public String toString() {
            return parameters.toString();
        }
        
        @Override
        public boolean equals(Object obj) {
            if (this == obj) return true;
            if (obj == null || getClass() != obj.getClass()) return false;
            ParameterCombination that = (ParameterCombination) obj;
            return Objects.equals(parameters, that.parameters);
        }
        
        @Override
        public int hashCode() {
            return Objects.hash(parameters);
        }
    }
    
    /**
     * Results from hyperparameter tuning
     */
    public static class TuningResults {
        private final List<ParameterCombination> allCombinations;
        private final List<Double> allScores;
        private final ParameterCombination bestParameters;
        private final double bestScore;
        private final String scoringMetric;
        private final long searchTime;
        
        public TuningResults(List<ParameterCombination> combinations, 
                           List<Double> scores, 
                           String scoringMetric,
                           long searchTime) {
            this.allCombinations = new ArrayList<>(combinations);
            this.allScores = new ArrayList<>(scores);
            this.scoringMetric = scoringMetric;
            this.searchTime = searchTime;
            
            // Find best parameters
            int bestIndex = 0;
            double bestScoreValue = scores.get(0);
            
            for (int i = 1; i < scores.size(); i++) {
                if (scores.get(i) > bestScoreValue) {
                    bestScoreValue = scores.get(i);
                    bestIndex = i;
                }
            }
            
            this.bestParameters = combinations.get(bestIndex);
            this.bestScore = bestScoreValue;
        }
        
        public ParameterCombination getBestParameters() { return bestParameters; }
        public double getBestScore() { return bestScore; }
        public String getScoringMetric() { return scoringMetric; }
        public long getSearchTimeMs() { return searchTime; }
        public List<ParameterCombination> getAllCombinations() { return new ArrayList<>(allCombinations); }
        public List<Double> getAllScores() { return new ArrayList<>(allScores); }
        
        @Override
        public String toString() {
            return String.format(
                "Hyperparameter Tuning Results:\n" +
                "Best Score (%s): %.4f\n" +
                "Best Parameters: %s\n" +
                "Total Combinations Evaluated: %d\n" +
                "Search Time: %d ms",
                scoringMetric, bestScore, bestParameters, 
                allCombinations.size(), searchTime
            );
        }
    }
    
    /**
     * Configuration for hyperparameter tuning
     */
    public static class TuningConfig {
        private String scoringMetric = "accuracy";
        private int cvFolds = 5;
        private boolean parallel = false;
        private Long randomSeed = null;
        private boolean verbose = false;
        private int maxIterations = -1; // -1 means no limit (for grid search)
        
        public TuningConfig setScoringMetric(String metric) {
            this.scoringMetric = metric;
            return this;
        }
        
        public TuningConfig setCvFolds(int folds) {
            this.cvFolds = folds;
            return this;
        }
        
        public TuningConfig setParallel(boolean parallel) {
            this.parallel = parallel;
            return this;
        }
        
        public TuningConfig setRandomSeed(long seed) {
            this.randomSeed = seed;
            return this;
        }
        
        public TuningConfig setVerbose(boolean verbose) {
            this.verbose = verbose;
            return this;
        }
        
        public TuningConfig setMaxIterations(int maxIterations) {
            this.maxIterations = maxIterations;
            return this;
        }
        
        // Getters
        public String getScoringMetric() { return scoringMetric; }
        public int getCvFolds() { return cvFolds; }
        public boolean isParallel() { return parallel; }
        public Long getRandomSeed() { return randomSeed; }
        public boolean isVerbose() { return verbose; }
        public int getMaxIterations() { return maxIterations; }
    }
    
    /**
     * Grid Search: Exhaustive search over parameter grid
     */
    public static class GridSearch {
        
        public static TuningResults search(Classifier classifier,
                                         double[][] X,
                                         double[] y,
                                         List<ParameterSpec> parameterSpecs,
                                         TuningConfig config) {
            
            long startTime = System.currentTimeMillis();
            
            // Generate all parameter combinations
            List<ParameterCombination> combinations = generateAllCombinations(parameterSpecs);
            
            if (config.isVerbose()) {
                System.out.println("Grid Search: Evaluating " + combinations.size() + " combinations");
            }
            
            // Evaluate all combinations
            List<Double> scores = evaluateCombinations(classifier, X, y, combinations, config);
            
            long endTime = System.currentTimeMillis();
            
            return new TuningResults(combinations, scores, config.getScoringMetric(), 
                                   endTime - startTime);
        }
        
        public static TuningResults searchRegressor(Regressor regressor,
                                                   double[][] X,
                                                   double[] y,
                                                   List<ParameterSpec> parameterSpecs,
                                                   TuningConfig config) {
            
            long startTime = System.currentTimeMillis();
            
            // Generate all parameter combinations
            List<ParameterCombination> combinations = generateAllCombinations(parameterSpecs);
            
            if (config.isVerbose()) {
                System.out.println("Grid Search (Regression): Evaluating " + combinations.size() + " combinations");
            }
            
            // Evaluate all combinations
            List<Double> scores = evaluateRegressorCombinations(regressor, X, y, combinations, config);
            
            long endTime = System.currentTimeMillis();
            
            return new TuningResults(combinations, scores, config.getScoringMetric(), 
                                   endTime - startTime);
        }
        
        private static List<ParameterCombination> generateAllCombinations(List<ParameterSpec> specs) {
            List<ParameterCombination> combinations = new ArrayList<>();
            generateCombinationsRecursive(specs, 0, new HashMap<>(), combinations);
            return combinations;
        }
        
        private static void generateCombinationsRecursive(List<ParameterSpec> specs,
                                                        int specIndex,
                                                        Map<String, Object> currentParams,
                                                        List<ParameterCombination> combinations) {
            if (specIndex == specs.size()) {
                combinations.add(new ParameterCombination(currentParams));
                return;
            }
            
            ParameterSpec spec = specs.get(specIndex);
            for (Object value : spec.getValues()) {
                Map<String, Object> newParams = new HashMap<>(currentParams);
                newParams.put(spec.getName(), value);
                generateCombinationsRecursive(specs, specIndex + 1, newParams, combinations);
            }
        }
    }
    
    /**
     * Random Search: Randomly sample parameter combinations
     */
    public static class RandomSearch {
        
        public static TuningResults search(Classifier classifier,
                                         double[][] X,
                                         double[] y,
                                         List<ParameterSpec> parameterSpecs,
                                         TuningConfig config) {
            
            long startTime = System.currentTimeMillis();
            
            Random random = config.getRandomSeed() != null ? 
                           new Random(config.getRandomSeed()) : new Random();
            
            int maxIterations = config.getMaxIterations() > 0 ? 
                              config.getMaxIterations() : 100; // Default 100 iterations
            
            if (config.isVerbose()) {
                System.out.println("Random Search: Evaluating " + maxIterations + " random combinations");
            }
            
            // Generate random combinations
            Set<ParameterCombination> uniqueCombinations = new HashSet<>();
            List<ParameterCombination> combinations = new ArrayList<>();
            
            int attempts = 0;
            while (combinations.size() < maxIterations && attempts < maxIterations * 10) {
                ParameterCombination combination = generateRandomCombination(parameterSpecs, random);
                if (uniqueCombinations.add(combination)) {
                    combinations.add(combination);
                }
                attempts++;
            }
            
            // Evaluate combinations
            List<Double> scores = evaluateCombinations(classifier, X, y, combinations, config);
            
            long endTime = System.currentTimeMillis();
            
            return new TuningResults(combinations, scores, config.getScoringMetric(), 
                                   endTime - startTime);
        }
        
        public static TuningResults searchRegressor(Regressor regressor,
                                                   double[][] X,
                                                   double[] y,
                                                   List<ParameterSpec> parameterSpecs,
                                                   TuningConfig config) {
            
            long startTime = System.currentTimeMillis();
            
            // Generate random combinations
            Random random = config.getRandomSeed() != null ? 
                           new Random(config.getRandomSeed()) : new Random();
            
            int maxIterations = config.getMaxIterations() > 0 ? 
                              config.getMaxIterations() : 100;
            
            if (config.isVerbose()) {
                System.out.println("Random Search (Regression): Evaluating " + maxIterations + " random combinations");
            }
            
            // Generate random combinations
            Set<ParameterCombination> uniqueCombinations = new HashSet<>();
            List<ParameterCombination> combinations = new ArrayList<>();
            
            int attempts = 0;
            while (combinations.size() < maxIterations && attempts < maxIterations * 10) {
                ParameterCombination combination = generateRandomCombination(parameterSpecs, random);
                if (uniqueCombinations.add(combination)) {
                    combinations.add(combination);
                }
                attempts++;
            }
            
            // Evaluate combinations
            List<Double> scores = evaluateRegressorCombinations(regressor, X, y, combinations, config);
            
            long endTime = System.currentTimeMillis();
            
            return new TuningResults(combinations, scores, config.getScoringMetric(), 
                                   endTime - startTime);
        }
        
        private static ParameterCombination generateRandomCombination(List<ParameterSpec> specs, Random random) {
            Map<String, Object> parameters = new HashMap<>();
            for (ParameterSpec spec : specs) {
                parameters.put(spec.getName(), spec.getRandomValue(random));
            }
            return new ParameterCombination(parameters);
        }
    }
    
    /**
     * Evaluate parameter combinations using cross-validation
     */
    private static List<Double> evaluateCombinations(Classifier classifier,
                                                   double[][] X,
                                                   double[] y,
                                                   List<ParameterCombination> combinations,
                                                   TuningConfig config) {
        
        List<Double> scores = new ArrayList<>();
        
        if (config.isParallel()) {
            scores = evaluateParallel(classifier, X, y, combinations, config);
        } else {
            for (int i = 0; i < combinations.size(); i++) {
                ParameterCombination combination = combinations.get(i);
                double score = evaluateSingleCombination(classifier, X, y, combination, config);
                scores.add(score);
                
                if (config.isVerbose()) {
                    System.out.printf("Combination %d/%d: %s -> %.4f%n", 
                                    i + 1, combinations.size(), combination, score);
                }
            }
        }
        
        return scores;
    }
    
    /**
     * Parallel evaluation of parameter combinations
     */
    private static List<Double> evaluateParallel(Classifier classifier,
                                                double[][] X,
                                                double[] y,
                                                List<ParameterCombination> combinations,
                                                TuningConfig config) {
        
        int nThreads = Math.min(combinations.size(), Runtime.getRuntime().availableProcessors());
        ExecutorService executor = Executors.newFixedThreadPool(nThreads);
        
        List<Future<Double>> futures = new ArrayList<>();
        
        for (ParameterCombination combination : combinations) {
            Future<Double> future = executor.submit(() -> {
                return evaluateSingleCombination(classifier, X, y, combination, config);
            });
            futures.add(future);
        }
        
        List<Double> scores = new ArrayList<>();
        try {
            for (int i = 0; i < futures.size(); i++) {
                double score = futures.get(i).get();
                scores.add(score);
                
                if (config.isVerbose()) {
                    System.out.printf("Combination %d/%d completed -> %.4f%n", 
                                    i + 1, combinations.size(), score);
                }
            }
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException("Error in parallel hyperparameter tuning", e);
        } finally {
            executor.shutdown();
        }
        
        return scores;
    }
    
    /**
     * Evaluate a single parameter combination
     */
    private static double evaluateSingleCombination(Classifier classifier,
                                                  double[][] X,
                                                  double[] y,
                                                  ParameterCombination combination,
                                                  TuningConfig config) {
        
        // Apply parameters to classifier
        Classifier configuredClassifier = applyParameters(classifier, combination);
        
        // Perform cross-validation
        CrossValidation.CrossValidationConfig cvConfig = 
            new CrossValidation.CrossValidationConfig()
                .setFolds(config.getCvFolds())
                .addMetric(config.getScoringMetric());
        
        if (config.getRandomSeed() != null) {
            cvConfig.setRandomSeed(config.getRandomSeed());
        }
        
        CrossValidation.CrossValidationResults results = 
            CrossValidation.crossValidate(configuredClassifier, X, y, cvConfig);
        
        return results.getMeanScore(config.getScoringMetric());
    }
    
    /**
     * Apply parameters to classifier using reflection
     */
    private static Classifier applyParameters(Classifier classifier, ParameterCombination combination) {
        try {
            // Create new instance
            Classifier newClassifier = classifier.getClass().getDeclaredConstructor().newInstance();
            
            // Apply parameters using setter methods
            for (Map.Entry<String, Object> entry : combination.getParameters().entrySet()) {
                String paramName = entry.getKey();
                Object value = entry.getValue();
                
                // Try to find setter method
                String setterName = "set" + paramName.substring(0, 1).toUpperCase() + 
                                  paramName.substring(1);
                
                Method setter = findSetter(newClassifier.getClass(), setterName, value.getClass());
                if (setter != null) {
                    setter.invoke(newClassifier, value);
                } else {
                    // Try direct field access or alternative setter patterns
                    applyParameterAlternative(newClassifier, paramName, value);
                }
            }
            
            return newClassifier;
        } catch (Exception e) {
            throw new RuntimeException("Failed to apply parameters to classifier", e);
        }
    }
    
    /**
     * Find appropriate setter method
     */
    private static Method findSetter(Class<?> clazz, String setterName, Class<?> paramType) {
        try {
            // Try exact match first
            return clazz.getMethod(setterName, paramType);
        } catch (NoSuchMethodException e) {
            // Try with primitive types
            try {
                if (paramType == Double.class) {
                    return clazz.getMethod(setterName, double.class);
                } else if (paramType == Integer.class) {
                    return clazz.getMethod(setterName, int.class);
                } else if (paramType == Boolean.class) {
                    return clazz.getMethod(setterName, boolean.class);
                }
            } catch (NoSuchMethodException ex) {
                // Ignore and return null
            }
        }
        return null;
    }
    
    /**
     * Alternative parameter application methods
     */
    private static void applyParameterAlternative(Classifier classifier, String paramName, Object value) {
        // For now, just log that parameter couldn't be applied
        // In a full implementation, you might use field access or other mechanisms
        System.err.println("Warning: Could not apply parameter " + paramName + " to " + 
                          classifier.getClass().getSimpleName());
    }
    
    /**
     * Convenience method for quick grid search
     */
    public static TuningResults gridSearch(Classifier classifier,
                                         double[][] X,
                                         double[] y,
                                         ParameterSpec... specs) {
        return GridSearch.search(classifier, X, y, Arrays.asList(specs), new TuningConfig());
    }
    
    /**
     * Convenience method for quick random search
     */
    public static TuningResults randomSearch(Classifier classifier,
                                           double[][] X,
                                           double[] y,
                                           int iterations,
                                           ParameterSpec... specs) {
        return RandomSearch.search(classifier, X, y, Arrays.asList(specs), 
                                 new TuningConfig().setMaxIterations(iterations));
    }
    
    /**
     * Convenience method for quick regressor grid search
     */
    public static TuningResults gridSearchRegressor(Regressor regressor,
                                                   double[][] X,
                                                   double[] y,
                                                   ParameterSpec... specs) {
        TuningConfig config = new TuningConfig().setScoringMetric("r2");
        return GridSearch.searchRegressor(regressor, X, y, Arrays.asList(specs), config);
    }
    
    /**
     * Convenience method for quick regressor random search
     */
    public static TuningResults randomSearchRegressor(Regressor regressor,
                                                     double[][] X,
                                                     double[] y,
                                                     int iterations,
                                                     ParameterSpec... specs) {
        TuningConfig config = new TuningConfig()
            .setMaxIterations(iterations)
            .setScoringMetric("r2");
        return RandomSearch.searchRegressor(regressor, X, y, Arrays.asList(specs), config);
    }
    
    /**
     * Evaluate regressor parameter combinations using cross-validation
     */
    private static List<Double> evaluateRegressorCombinations(Regressor regressor,
                                                             double[][] X,
                                                             double[] y,
                                                             List<ParameterCombination> combinations,
                                                             TuningConfig config) {
        
        List<Double> scores = new ArrayList<>();
        
        if (config.isParallel()) {
            scores = evaluateRegressorParallel(regressor, X, y, combinations, config);
        } else {
            for (int i = 0; i < combinations.size(); i++) {
                ParameterCombination combination = combinations.get(i);
                double score = evaluateSingleRegressorCombination(regressor, X, y, combination, config);
                scores.add(score);
                
                if (config.isVerbose()) {
                    System.out.printf("Combination %d/%d: %s -> %.4f%n", 
                                    i + 1, combinations.size(), combination, score);
                }
            }
        }
        
        return scores;
    }
    
    /**
     * Parallel evaluation of regressor parameter combinations
     */
    private static List<Double> evaluateRegressorParallel(Regressor regressor,
                                                         double[][] X,
                                                         double[] y,
                                                         List<ParameterCombination> combinations,
                                                         TuningConfig config) {
        
        int nThreads = Math.min(combinations.size(), Runtime.getRuntime().availableProcessors());
        ExecutorService executor = Executors.newFixedThreadPool(nThreads);
        
        List<Future<Double>> futures = new ArrayList<>();
        
        for (ParameterCombination combination : combinations) {
            Future<Double> future = executor.submit(() -> {
                return evaluateSingleRegressorCombination(regressor, X, y, combination, config);
            });
            futures.add(future);
        }
        
        List<Double> scores = new ArrayList<>();
        try {
            for (int i = 0; i < futures.size(); i++) {
                double score = futures.get(i).get();
                scores.add(score);
                
                if (config.isVerbose()) {
                    System.out.printf("Combination %d/%d completed -> %.4f%n", 
                                    i + 1, combinations.size(), score);
                }
            }
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException("Error in parallel hyperparameter tuning", e);
        } finally {
            executor.shutdown();
        }
        
        return scores;
    }
    
    /**
     * Evaluate a single regressor parameter combination
     */
    private static double evaluateSingleRegressorCombination(Regressor regressor,
                                                            double[][] X,
                                                            double[] y,
                                                            ParameterCombination combination,
                                                            TuningConfig config) {
        
        // Apply parameters to regressor
        Regressor configuredRegressor = applyRegressorParameters(regressor, combination);
        
        // Perform cross-validation
        CrossValidation.CrossValidationConfig cvConfig = 
            new CrossValidation.CrossValidationConfig()
                .setFolds(config.getCvFolds())
                .addMetric(config.getScoringMetric());
        
        if (config.getRandomSeed() != null) {
            cvConfig.setRandomSeed(config.getRandomSeed());
        }
        
        CrossValidation.CrossValidationResults results = 
            CrossValidation.crossValidateRegression(configuredRegressor, X, y, cvConfig);
        
        return results.getMeanScore(config.getScoringMetric());
    }
    
    /**
     * Apply parameters to regressor using reflection
     */
    private static Regressor applyRegressorParameters(Regressor regressor, ParameterCombination combination) {
        try {
            // Create new instance
            Regressor newRegressor = regressor.getClass().getDeclaredConstructor().newInstance();
            
            // Apply parameters using setter methods
            for (Map.Entry<String, Object> entry : combination.getParameters().entrySet()) {
                String paramName = entry.getKey();
                Object value = entry.getValue();
                
                // Try to find setter method
                String setterName = "set" + paramName.substring(0, 1).toUpperCase() + 
                                  paramName.substring(1);
                
                Method setter = findSetter(newRegressor.getClass(), setterName, value.getClass());
                if (setter != null) {
                    setter.invoke(newRegressor, value);
                } else {
                    // Try direct field access or alternative setter patterns
                    applyParameterAlternative(newRegressor, paramName, value);
                }
            }
            
            return newRegressor;
        } catch (Exception e) {
            throw new RuntimeException("Failed to apply parameters to regressor", e);
        }
    }
    
    /**
     * Alternative parameter application methods for regressors
     */
    private static void applyParameterAlternative(Regressor regressor, String paramName, Object value) {
        // For now, just log that parameter couldn't be applied
        // In a full implementation, you might use field access or other mechanisms
        System.err.println("Warning: Could not apply parameter " + paramName + " to " + 
                          regressor.getClass().getSimpleName());
    }
}
