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

package org.superml.autotrainer;

import org.superml.cluster.KMeans;
import org.superml.metrics.ClusteringMetrics;

import java.util.*;
import java.util.concurrent.*;
import java.util.stream.IntStream;

/**
 * AutoTrainer for Clustering Models
 * 
 * Provides automated hyperparameter optimization for unsupervised learning algorithms.
 * Supports various optimization strategies and evaluation metrics for clustering tasks.
 * 
 * Supported Algorithms:
 * - KMeans: Automated cluster number detection and parameter optimization
 * 
 * Features:
 * - Grid search and random search optimization
 * - Multiple internal validation metrics
 * - Optimal cluster number detection (elbow method, silhouette analysis)
 * - Cross-validation for parameter stability
 * - Parallel hyperparameter evaluation
 * - Early stopping based on convergence criteria
 * 
 * Example usage:
 * ```java
 * double[][] data = loadData();
 * 
 * ClusteringAutoTrainer trainer = new ClusteringAutoTrainer()
 *     .setOptimizationStrategy("grid_search")
 *     .setMetric("silhouette")
 *     .setCV(5)
 *     .setNJobs(4);
 * 
 * KMeans bestModel = trainer.fitKMeans(data);
 * AutoTrainerResult result = trainer.getLastResult();
 * ```
 * 
 * @author SuperML Team
 * @version 2.1.0
 */
public class ClusteringAutoTrainer {
    
    // Model types
    public enum ModelType {
        KMEANS
    }
    
    // Optimization settings
    private String optimizationStrategy = "grid_search"; // "grid_search", "random_search"
    private String metric = "silhouette"; // "silhouette", "inertia", "calinski_harabasz", "davies_bouldin"
    private int cv = 5; // Cross-validation folds for stability
    private int nJobs = 1; // Number of parallel jobs
    private int randomState = 42;
    private boolean verbose = true;
    
    // Search settings
    private int nIterRandom = 20; // For random search
    
    // Results tracking
    private AutoTrainerResult lastResult;
    
    public ClusteringAutoTrainer() {}
    
    // ========================================
    // MAIN TRAINING METHODS
    // ========================================
    
    /**
     * Automatically train and optimize KMeans clustering
     */
    public KMeans fitKMeans(double[][] X) {
        return fitKMeans(X, null);
    }
    
    /**
     * Train KMeans with custom parameter ranges
     */
    public KMeans fitKMeans(double[][] X, Map<String, Object> paramRanges) {
        logInfo("Starting KMeans AutoTrainer optimization...");
        
        // Generate parameter search space
        List<Map<String, Object>> searchSpace = generateKMeansSearchSpace(paramRanges);
        
        // Perform parameter optimization
        AutoTrainerResult result = optimizeParameters(ModelType.KMEANS, X, searchSpace);
        
        // Create and train best model
        KMeans bestModel = (KMeans) createModel(ModelType.KMEANS, result.bestParams);
        bestModel.fit(X);
        
        this.lastResult = result;
        logInfo(String.format("KMeans optimization completed. Best score: %.4f", result.bestScore));
        
        return bestModel;
    }
    
    // ========================================
    // PARAMETER OPTIMIZATION
    // ========================================
    
    private AutoTrainerResult optimizeParameters(ModelType modelType, double[][] X, 
                                                List<Map<String, Object>> searchSpace) {
        AutoTrainerResult result = new AutoTrainerResult();
        result.modelType = modelType.toString();
        result.searchSpaceSize = searchSpace.size();
        result.startTime = System.currentTimeMillis();
        
        double bestScore = Double.NEGATIVE_INFINITY;
        Map<String, Object> bestParams = null;
        List<Map<String, Object>> allResults = new ArrayList<>();
        
        logInfo(String.format("Evaluating %d parameter combinations...", searchSpace.size()));
        
        // Create thread pool for parallel evaluation
        ExecutorService executor = Executors.newFixedThreadPool(nJobs);
        List<Future<ParameterEvaluation>> futures = new ArrayList<>();
        
        // Submit evaluation tasks
        for (Map<String, Object> params : searchSpace) {
            Future<ParameterEvaluation> future = executor.submit(() -> 
                evaluateParameters(modelType, X, params));
            futures.add(future);
        }
        
        // Collect results
        try {
            for (int i = 0; i < futures.size(); i++) {
                ParameterEvaluation eval = futures.get(i).get();
                
                Map<String, Object> evalResult = new HashMap<>();
                evalResult.put("params", eval.params);
                evalResult.put("score", eval.score);
                evalResult.put("std", eval.std);
                allResults.add(evalResult);
                
                if (eval.score > bestScore) {
                    bestScore = eval.score;
                    bestParams = eval.params;
                }
                
                if (verbose && (i + 1) % Math.max(1, searchSpace.size() / 10) == 0) {
                    logInfo(String.format("Progress: %d/%d evaluations completed", 
                           i + 1, searchSpace.size()));
                }
            }
        } catch (Exception e) {
            throw new RuntimeException("Error during parameter optimization", e);
        } finally {
            executor.shutdown();
        }
        
        result.bestScore = bestScore;
        result.bestParams = bestParams;
        result.allResults = allResults;
        result.endTime = System.currentTimeMillis();
        result.executionTime = result.endTime - result.startTime;
        
        return result;
    }
    
    private ParameterEvaluation evaluateParameters(ModelType modelType, double[][] X, 
                                                  Map<String, Object> params) {
        ParameterEvaluation eval = new ParameterEvaluation();
        eval.params = params;
        
        try {
            // Perform cross-validation for stability
            double[] scores = new double[cv];
            
            for (int fold = 0; fold < cv; fold++) {
                // For clustering, we use different random states for stability assessment
                Map<String, Object> foldParams = new HashMap<>(params);
                foldParams.put("random_state", randomState + fold);
                
                // Create and train model
                Object model = createModel(modelType, foldParams);
                if (model instanceof KMeans) {
                    ((KMeans) model).fit(X);
                    
                    // Evaluate clustering quality
                    scores[fold] = evaluateClusteringModel(model, X);
                } else {
                    throw new IllegalArgumentException("Unsupported model type: " + modelType);
                }
            }
            
            // Calculate mean and standard deviation
            eval.score = Arrays.stream(scores).average().orElse(0.0);
            eval.std = Math.sqrt(Arrays.stream(scores)
                .map(score -> Math.pow(score - eval.score, 2))
                .average().orElse(0.0));
                
        } catch (Exception e) {
            eval.score = Double.NEGATIVE_INFINITY;
            eval.std = Double.POSITIVE_INFINITY;
        }
        
        return eval;
    }
    
    private double evaluateClusteringModel(Object model, double[][] X) {
        if (model instanceof KMeans) {
            KMeans kmeans = (KMeans) model;
            
            // Get cluster assignments
            int[] intPredictions = kmeans.predict(X);
            double[] predictions = Arrays.stream(intPredictions).asDoubleStream().toArray();
            
            // Calculate requested metric
            switch (metric.toLowerCase()) {
                case "silhouette":
                    return ClusteringMetrics.silhouetteScore(X, predictions);
                case "inertia":
                    // For inertia, return negative value (lower is better)
                    return -kmeans.getInertia();
                case "calinski_harabasz":
                    return ClusteringMetrics.calinskiHarabaszIndex(X, predictions);
                case "davies_bouldin":
                    // For Davies-Bouldin, return negative value (lower is better)
                    return -ClusteringMetrics.daviesBouldinIndex(X, predictions);
                default:
                    return ClusteringMetrics.silhouetteScore(X, predictions);
            }
        }
        
        return 0.0;
    }
    
    // ========================================
    // SEARCH SPACE GENERATION
    // ========================================
    
    private List<Map<String, Object>> generateKMeansSearchSpace(Map<String, Object> customRanges) {
        List<Map<String, Object>> searchSpace = new ArrayList<>();
        
        // Default parameter ranges
        int[] nClusters = customRanges != null && customRanges.containsKey("n_clusters") ?
            (int[]) customRanges.get("n_clusters") : new int[]{2, 3, 4, 5, 6, 7, 8, 10, 12, 15};
        
        int[] maxIter = customRanges != null && customRanges.containsKey("max_iter") ?
            (int[]) customRanges.get("max_iter") : new int[]{100, 300, 500};
            
        double[] tolerance = customRanges != null && customRanges.containsKey("tol") ?
            (double[]) customRanges.get("tol") : new double[]{1e-4, 1e-5, 1e-6};
            
        String[] initMethods = customRanges != null && customRanges.containsKey("init") ?
            (String[]) customRanges.get("init") : new String[]{"k-means++", "random"};
        
        // Generate all combinations
        for (int k : nClusters) {
            for (int iter : maxIter) {
                for (double tol : tolerance) {
                    for (String init : initMethods) {
                        Map<String, Object> params = new HashMap<>();
                        params.put("n_clusters", k);
                        params.put("max_iter", iter);
                        params.put("tol", tol);
                        params.put("init", init);
                        params.put("random_state", randomState);
                        searchSpace.add(params);
                    }
                }
            }
        }
        
        // Apply search strategy
        if ("random_search".equals(optimizationStrategy) && searchSpace.size() > nIterRandom) {
            Collections.shuffle(searchSpace, new Random(randomState));
            searchSpace = searchSpace.subList(0, nIterRandom);
        }
        
        return searchSpace;
    }
    
    // ========================================
    // MODEL CREATION
    // ========================================
    
    private Object createModel(ModelType modelType, Map<String, Object> params) {
        switch (modelType) {
            case KMEANS:
                KMeans kmeans = new KMeans();
                kmeans.setNClusters((Integer) params.get("n_clusters"))
                      .setMaxIter((Integer) params.get("max_iter"))
                      .setTolerance((Double) params.get("tol"))
                      .setInitMethod((String) params.get("init"))
                      .setRandomState((Integer) params.get("random_state"));
                return kmeans;
                
            default:
                throw new IllegalArgumentException("Unsupported model type: " + modelType);
        }
    }
    
    // ========================================
    // OPTIMAL CLUSTER NUMBER DETECTION
    // ========================================
    
    /**
     * Find optimal number of clusters using elbow method
     */
    public ElbowResult findOptimalClusters(double[][] X, int minClusters, int maxClusters) {
        logInfo("Finding optimal number of clusters using elbow method...");
        
        List<Integer> clusterRange = IntStream.rangeClosed(minClusters, maxClusters)
            .boxed().collect(ArrayList::new, ArrayList::add, ArrayList::addAll);
        
        double[] inertias = new double[clusterRange.size()];
        double[] silhouettes = new double[clusterRange.size()];
        
        for (int i = 0; i < clusterRange.size(); i++) {
            int k = clusterRange.get(i);
            
            KMeans kmeans = new KMeans(k, 300, randomState);
            kmeans.fit(X);
            
            inertias[i] = kmeans.getInertia();
            
            if (k > 1) { // Silhouette requires at least 2 clusters
                int[] predictions = kmeans.predict(X);
                double[] doublePredictions = Arrays.stream(predictions).asDoubleStream().toArray();
                silhouettes[i] = ClusteringMetrics.silhouetteScore(X, doublePredictions);
            } else {
                silhouettes[i] = 0.0;
            }
        }
        
        // Find elbow using rate of change
        int elbowIndex = findElbowPoint(inertias);
        
        // Find best silhouette
        int bestSilhouetteIndex = 0;
        for (int i = 1; i < silhouettes.length; i++) {
            if (silhouettes[i] > silhouettes[bestSilhouetteIndex]) {
                bestSilhouetteIndex = i;
            }
        }
        
        ElbowResult result = new ElbowResult();
        result.clusterRange = clusterRange.stream().mapToInt(Integer::intValue).toArray();
        result.inertias = inertias;
        result.silhouettes = silhouettes;
        result.elbowPoint = clusterRange.get(elbowIndex);
        result.bestSilhouette = clusterRange.get(bestSilhouetteIndex);
        result.recommendedClusters = result.bestSilhouette; // Prefer silhouette over elbow
        
        logInfo(String.format("Optimal clusters: %d (elbow: %d, silhouette: %d)", 
               result.recommendedClusters, result.elbowPoint, result.bestSilhouette));
        
        return result;
    }
    
    private int findElbowPoint(double[] inertias) {
        // Simple elbow detection using second derivative
        if (inertias.length < 3) return 0;
        
        double[] firstDiff = new double[inertias.length - 1];
        for (int i = 0; i < firstDiff.length; i++) {
            firstDiff[i] = inertias[i + 1] - inertias[i];
        }
        
        double[] secondDiff = new double[firstDiff.length - 1];
        for (int i = 0; i < secondDiff.length; i++) {
            secondDiff[i] = firstDiff[i + 1] - firstDiff[i];
        }
        
        // Find point with maximum second derivative (most curvature)
        int maxIndex = 0;
        for (int i = 1; i < secondDiff.length; i++) {
            if (secondDiff[i] > secondDiff[maxIndex]) {
                maxIndex = i;
            }
        }
        
        return maxIndex + 1; // Adjust for offset
    }
    
    // ========================================
    // GETTERS AND SETTERS
    // ========================================
    
    public ClusteringAutoTrainer setOptimizationStrategy(String strategy) {
        this.optimizationStrategy = strategy;
        return this;
    }
    
    public ClusteringAutoTrainer setMetric(String metric) {
        this.metric = metric;
        return this;
    }
    
    public ClusteringAutoTrainer setCV(int cv) {
        this.cv = cv;
        return this;
    }
    
    public ClusteringAutoTrainer setNJobs(int nJobs) {
        this.nJobs = nJobs;
        return this;
    }
    
    public ClusteringAutoTrainer setRandomState(int randomState) {
        this.randomState = randomState;
        return this;
    }
    
    public ClusteringAutoTrainer setVerbose(boolean verbose) {
        this.verbose = verbose;
        return this;
    }
    
    public ClusteringAutoTrainer setNIterRandom(int nIterRandom) {
        this.nIterRandom = nIterRandom;
        return this;
    }
    
    public AutoTrainerResult getLastResult() {
        return lastResult;
    }
    
    // ========================================
    // UTILITY METHODS
    // ========================================
    
    private void logInfo(String message) {
        if (verbose) {
            System.out.println("[ClusteringAutoTrainer] " + message);
        }
    }
    
    // ========================================
    // RESULT CLASSES
    // ========================================
    
    public static class AutoTrainerResult {
        public String modelType;
        public double bestScore;
        public Map<String, Object> bestParams;
        public int searchSpaceSize;
        public List<Map<String, Object>> allResults;
        public long startTime;
        public long endTime;
        public long executionTime;
    }
    
    private static class ParameterEvaluation {
        public Map<String, Object> params;
        public double score;
        public double std;
    }
    
    public static class ElbowResult {
        public int[] clusterRange;
        public double[] inertias;
        public double[] silhouettes;
        public int elbowPoint;
        public int bestSilhouette;
        public int recommendedClusters;
    }
}
