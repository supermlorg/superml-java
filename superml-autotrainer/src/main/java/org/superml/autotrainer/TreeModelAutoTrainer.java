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

import org.superml.core.BaseEstimator;
import org.superml.core.Classifier;
import org.superml.core.Regressor;
import org.superml.tree.*;
import java.util.*;
import java.util.concurrent.*;

/**
 * Specialized AutoTrainer for Tree-based models with domain expertise.
 * Provides optimized hyperparameter tuning for decision trees, random forests,
 * gradient boosting, extra trees, and AdaBoost algorithms.
 */
public class TreeModelAutoTrainer {
    
    public enum TreeModelType {
        DECISION_TREE,
        RANDOM_FOREST, 
        GRADIENT_BOOSTING,
        AUTO_SELECT
    }
    
    public enum ProblemType {
        CLASSIFICATION,
        REGRESSION,
        AUTO_DETECT
    }
    
    public enum SearchStrategy {
        GRID_SEARCH,
        RANDOM_SEARCH,
        BAYESIAN_OPTIMIZATION
    }
    
    private final ExecutorService executorService;
    private final Random random;
    private int cvFolds = 5;
    private double testSize = 0.2;
    private String primaryMetric = "accuracy"; // or "r2" for regression
    
    public TreeModelAutoTrainer() {
        this.executorService = Executors.newFixedThreadPool(
            Runtime.getRuntime().availableProcessors());
        this.random = new Random(42);
    }
    
    public TreeModelAutoTrainer(int parallelJobs, int randomSeed) {
        this.executorService = Executors.newFixedThreadPool(parallelJobs);
        this.random = new Random(randomSeed);
    }
    
    /**
     * Main auto-training method for tree models
     */
    public TreeAutoTrainingResult autoTrain(double[][] X, double[] y, TreeModelType modelType) {
        System.out.println("🌳 Starting Tree Model Auto-Training...");
        TreeAutoTrainingResult result = new TreeAutoTrainingResult();
        result.startTime = System.currentTimeMillis();
        
        // Detect problem type
        ProblemType problemType = detectProblemType(y);
        result.problemType = problemType;
        
        // Set appropriate metric
        if (problemType == ProblemType.CLASSIFICATION) {
            primaryMetric = "accuracy";
        } else {
            primaryMetric = "r2";
        }
        
        // Auto-select model if needed
        if (modelType == TreeModelType.AUTO_SELECT) {
            modelType = autoSelectTreeModel(X, y, problemType);
            System.out.println("Auto-selected model: " + modelType);
        }
        
        // Generate search space
        List<HyperparameterSet> searchSpace = generateTreeSearchSpace(X, y, modelType, problemType);
        System.out.println("Generated " + searchSpace.size() + " hyperparameter combinations");
        
        // Optimize hyperparameters
        HyperparameterOptimizationResult optResult = optimizeHyperparameters(
            X, y, modelType, searchSpace, problemType);
        
        result.bestModel = optResult.bestModel;
        result.bestParameters = optResult.bestParameters;
        result.bestScore = optResult.bestScore;
        result.bestModelType = modelType.toString();
        result.optimizationHistory = optResult.history;
        
        result.endTime = System.currentTimeMillis();
        result.optimizationTime = (result.endTime - result.startTime) / 1000.0;
        
        System.out.printf("🏆 Best %s Score: %.4f%n", primaryMetric, result.bestScore);
        System.out.printf("⏱️ Total optimization time: %.2f seconds%n", result.optimizationTime);
        
        return result;
    }
    
    /**
     * Create ensemble of diverse tree models
     */
    public TreeEnsembleResult createTreeEnsemble(double[][] X, double[] y) {
        System.out.println("🌲 Creating Tree Model Ensemble...");
        TreeEnsembleResult result = new TreeEnsembleResult();
        result.startTime = System.currentTimeMillis();
        
        ProblemType problemType = detectProblemType(y);
        
        // Train different tree models
        List<BaseEstimator> models = new ArrayList<>();
        List<Double> scores = new ArrayList<>();
        
        // Decision Tree
        TreeAutoTrainingResult dtResult = autoTrain(X, y, TreeModelType.DECISION_TREE);
        models.add(dtResult.bestModel);
        scores.add(dtResult.bestScore);
        
        // Random Forest
        TreeAutoTrainingResult rfResult = autoTrain(X, y, TreeModelType.RANDOM_FOREST);
        models.add(rfResult.bestModel);
        scores.add(rfResult.bestScore);
        
        // Gradient Boosting
        TreeAutoTrainingResult gbResult = autoTrain(X, y, TreeModelType.GRADIENT_BOOSTING);
        models.add(gbResult.bestModel);
        scores.add(gbResult.bestScore);
        
        result.models = models;
        result.individualScores = scores;
        
        // Create ensemble predictor
        result.ensemblePredictor = new TreeEnsemblePredictor(models, problemType);
        
        // Evaluate ensemble
        if (problemType == ProblemType.CLASSIFICATION) {
            result.ensembleScore = evaluateClassificationEnsemble(result.ensemblePredictor, X, y);
        } else {
            result.ensembleScore = evaluateRegressionEnsemble(result.ensemblePredictor, X, y);
        }
        
        result.endTime = System.currentTimeMillis();
        result.ensembleTime = (result.endTime - result.startTime) / 1000.0;
        
        double bestIndividual = scores.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
        System.out.printf("🎯 Ensemble Score: %.4f vs Best Individual: %.4f%n", 
                         result.ensembleScore, bestIndividual);
        
        return result;
    }
    
    /**
     * Feature importance analysis across different tree models
     */
    public TreeFeatureImportanceResult analyzeFeatureImportance(double[][] X, double[] y, 
                                                               String[] featureNames) {
        System.out.println("🔍 Analyzing Feature Importance with Tree Models...");
        TreeFeatureImportanceResult result = new TreeFeatureImportanceResult();
        result.startTime = System.currentTimeMillis();
        
        ProblemType problemType = detectProblemType(y);
        
        // Random Forest Feature Importance
        RandomForest rf = new RandomForest()
            .setNEstimators(100)
            .setMaxDepth(10)
            .setRandomState(42);
        rf.fit(X, y);
        result.rfImportance = rf.getFeatureImportances();
        
        // Gradient Boosting Feature Importance
        GradientBoosting gb = new GradientBoosting()
            .setNEstimators(100)
            .setLearningRate(0.1)
            .setMaxDepth(6)
            .setRandomState(42);
        gb.fit(X, y);
        result.gbImportance = gb.getFeatureImportances();
        
        // Single Decision Tree Feature Importance (simplified calculation)
        DecisionTree dt = new DecisionTree()
            .setMaxDepth(10)
            .setRandomState(42);
        dt.fit(X, y);
        // For now, use uniform importance since DecisionTree doesn't have getFeatureImportances
        result.dtImportance = new double[X[0].length];
        java.util.Arrays.fill(result.dtImportance, 1.0 / X[0].length);
        
        // Calculate consensus importance
        result.consensusImportance = calculateConsensusImportance(
            result.rfImportance, result.gbImportance, result.dtImportance);
        
        // Feature ranking
        result.featureRanking = rankFeatures(result.consensusImportance, featureNames);
        
        result.endTime = System.currentTimeMillis();
        result.analysisTime = (result.endTime - result.startTime) / 1000.0;
        
        System.out.println("📊 Top 5 Important Features:");
        for (int i = 0; i < Math.min(5, result.featureRanking.size()); i++) {
            FeatureImportance fi = result.featureRanking.get(i);
            System.out.printf("  %d. %s: %.4f%n", i+1, fi.featureName, fi.importance);
        }
        
        return result;
    }
    
    // ================== Private Helper Methods ==================
    
    private ProblemType detectProblemType(double[] y) {
        Set<Double> uniqueValues = new HashSet<>();
        for (double value : y) {
            uniqueValues.add(value);
        }
        
        // If less than 20 unique values and all are integers, likely classification
        if (uniqueValues.size() < 20) {
            boolean allIntegers = uniqueValues.stream()
                .allMatch(v -> v == Math.floor(v));
            if (allIntegers) {
                return ProblemType.CLASSIFICATION;
            }
        }
        
        return ProblemType.REGRESSION;
    }
    
    private TreeModelType autoSelectTreeModel(double[][] X, double[] y, ProblemType problemType) {
        int nSamples = X.length;
        int nFeatures = X[0].length;
        
        // Quick heuristics for model selection
        if (nSamples < 1000) {
            return TreeModelType.DECISION_TREE; // Simple datasets
        } else if (nFeatures > nSamples / 10) {
            return TreeModelType.RANDOM_FOREST; // High-dimensional data
        } else {
            return TreeModelType.GRADIENT_BOOSTING; // General purpose
        }
    }
    
    private List<HyperparameterSet> generateTreeSearchSpace(double[][] X, double[] y, 
                                                           TreeModelType modelType, 
                                                           ProblemType problemType) {
        List<HyperparameterSet> searchSpace = new ArrayList<>();
        int nSamples = X.length;
        int nFeatures = X[0].length;
        
        switch (modelType) {
            case DECISION_TREE:
                searchSpace = generateDecisionTreeSearchSpace(nSamples, nFeatures, problemType);
                break;
            case RANDOM_FOREST:
                searchSpace = generateRandomForestSearchSpace(nSamples, nFeatures, problemType);
                break;
            case GRADIENT_BOOSTING:
                searchSpace = generateGradientBoostingSearchSpace(nSamples, nFeatures, problemType);
                break;
            case AUTO_SELECT:
                // This should not happen as auto-selection is done earlier
                throw new IllegalStateException("AUTO_SELECT should have been resolved earlier");
        }
        
        return searchSpace;
    }
    
    private List<HyperparameterSet> generateDecisionTreeSearchSpace(int nSamples, int nFeatures, 
                                                                   ProblemType problemType) {
        List<HyperparameterSet> searchSpace = new ArrayList<>();
        
        // Decision Tree parameters based on data size
        int[] maxDepths = {3, 5, 7, 10, 15, 20};
        int[] minSamplesSplits = {2, 5, 10, 20};
        int[] minSamplesLeafs = {1, 2, 5, 10};
        String[] criteria = problemType == ProblemType.CLASSIFICATION ? 
            new String[]{"gini", "entropy"} : new String[]{"mse"};
        
        for (int maxDepth : maxDepths) {
            for (int minSamplesSplit : minSamplesSplits) {
                for (int minSamplesLeaf : minSamplesLeafs) {
                    for (String criterion : criteria) {
                        HyperparameterSet params = new HyperparameterSet();
                        params.put("maxDepth", maxDepth);
                        params.put("minSamplesSplit", minSamplesSplit);
                        params.put("minSamplesLeaf", minSamplesLeaf);
                        params.put("criterion", criterion);
                        params.put("randomState", 42);
                        
                        searchSpace.add(params);
                    }
                }
            }
        }
        
        return searchSpace;
    }
    
    private List<HyperparameterSet> generateRandomForestSearchSpace(int nSamples, int nFeatures, 
                                                                   ProblemType problemType) {
        List<HyperparameterSet> searchSpace = new ArrayList<>();
        
        // Random Forest parameters
        int[] nEstimators = {50, 100, 200, 300};
        int[] maxDepths = {5, 10, 15, 20, 25};
        int[] minSamplesSplits = {2, 5, 10};
        int[] minSamplesLeafs = {1, 2, 5};
        double[] maxSamplesFractions = {0.8, 0.9, 1.0};
        
        for (int nEst : nEstimators) {
            for (int maxDepth : maxDepths) {
                for (int minSamplesSplit : minSamplesSplits) {
                    for (int minSamplesLeaf : minSamplesLeafs) {
                        for (double maxSamples : maxSamplesFractions) {
                            HyperparameterSet params = new HyperparameterSet();
                            params.put("nEstimators", nEst);
                            params.put("maxDepth", maxDepth);
                            params.put("minSamplesSplit", minSamplesSplit);
                            params.put("minSamplesLeaf", minSamplesLeaf);
                            params.put("maxSamples", maxSamples);
                            params.put("bootstrap", true);
                            params.put("randomState", 42);
                            
                            if (problemType == ProblemType.CLASSIFICATION) {
                                params.put("criterion", "gini");
                            } else {
                                params.put("criterion", "mse");
                            }
                            
                            searchSpace.add(params);
                        }
                    }
                }
            }
        }
        
        return searchSpace;
    }
    
    private List<HyperparameterSet> generateGradientBoostingSearchSpace(int nSamples, int nFeatures, 
                                                                       ProblemType problemType) {
        List<HyperparameterSet> searchSpace = new ArrayList<>();
        
        // Gradient Boosting parameters
        int[] nEstimators = {50, 100, 200, 300};
        double[] learningRates = {0.01, 0.05, 0.1, 0.2};
        int[] maxDepths = {3, 4, 5, 6, 7};
        double[] subsampleRates = {0.8, 0.9, 1.0};
        
        for (int nEst : nEstimators) {
            for (double lr : learningRates) {
                for (int maxDepth : maxDepths) {
                    for (double subsample : subsampleRates) {
                        HyperparameterSet params = new HyperparameterSet();
                        params.put("nEstimators", nEst);
                        params.put("learningRate", lr);
                        params.put("maxDepth", maxDepth);
                        params.put("subsample", subsample);
                        params.put("randomState", 42);
                        
                        searchSpace.add(params);
                    }
                }
            }
        }
        
        return searchSpace;
    }
    
    private HyperparameterOptimizationResult optimizeHyperparameters(double[][] X, double[] y, 
                                                                    TreeModelType modelType,
                                                                    List<HyperparameterSet> searchSpace,
                                                                    ProblemType problemType) {
        HyperparameterOptimizationResult result = new HyperparameterOptimizationResult();
        result.history = new ArrayList<>();
        
        double bestScore = Double.NEGATIVE_INFINITY;
        BaseEstimator bestModel = null;
        HyperparameterSet bestParams = null;
        
        int completed = 0;
        for (HyperparameterSet params : searchSpace) {
            try {
                // Create model instance
                BaseEstimator model = createModelInstance(modelType, params);
                
                // Cross-validation evaluation
                double score = crossValidateModel(model, X, y, problemType);
                
                // Track optimization history
                OptimizationStep step = new OptimizationStep();
                step.parameters = new HyperparameterSet(params);
                step.score = score;
                step.model = model;
                result.history.add(step);
                
                // Update best result
                if (score > bestScore) {
                    bestScore = score;
                    bestModel = model;
                    bestParams = new HyperparameterSet(params);
                }
                
                completed++;
                if (completed % 20 == 0) {
                    System.out.printf("Progress: %d/%d evaluations (Best: %.4f)%n", 
                                    completed, searchSpace.size(), bestScore);
                }
                
            } catch (Exception e) {
                System.err.println("Evaluation failed: " + e.getMessage());
            }
        }
        
        result.bestModel = bestModel;
        result.bestParameters = bestParams;
        result.bestScore = bestScore;
        
        return result;
    }
    
    private BaseEstimator createModelInstance(TreeModelType modelType, HyperparameterSet params) {
        switch (modelType) {
            case DECISION_TREE:
                DecisionTree dt = new DecisionTree();
                dt.setMaxDepth(params.getInt("maxDepth", 10))
                  .setMinSamplesSplit(params.getInt("minSamplesSplit", 2))
                  .setMinSamplesLeaf(params.getInt("minSamplesLeaf", 1))
                  .setCriterion(params.getString("criterion", "gini"))
                  .setRandomState(params.getInt("randomState", 42));
                return dt;
                
            case RANDOM_FOREST:
                RandomForest rf = new RandomForest();
                rf.setNEstimators(params.getInt("nEstimators", 100))
                  .setMaxDepth(params.getInt("maxDepth", 10))
                  .setMinSamplesSplit(params.getInt("minSamplesSplit", 2))
                  .setMinSamplesLeaf(params.getInt("minSamplesLeaf", 1))
                  .setMaxSamples(params.getDouble("maxSamples", 1.0))
                  .setBootstrap(params.getBoolean("bootstrap", true))
                  .setCriterion(params.getString("criterion", "gini"))
                  .setRandomState(params.getInt("randomState", 42));
                return rf;
                
            case GRADIENT_BOOSTING:
                GradientBoosting gb = new GradientBoosting();
                gb.setNEstimators(params.getInt("nEstimators", 100))
                  .setLearningRate(params.getDouble("learningRate", 0.1))
                  .setMaxDepth(params.getInt("maxDepth", 3))
                  .setSubsample(params.getDouble("subsample", 1.0))
                  .setRandomState(params.getInt("randomState", 42));
                return gb;
                
            default:
                throw new IllegalArgumentException("Unknown model type: " + modelType);
        }
    }
    
    private double crossValidateModel(BaseEstimator model, double[][] X, double[] y, 
                                     ProblemType problemType) {
        int nSamples = X.length;
        int foldSize = nSamples / cvFolds;
        double totalScore = 0.0;
        
        for (int fold = 0; fold < cvFolds; fold++) {
            // Split data
            int testStart = fold * foldSize;
            int testEnd = (fold == cvFolds - 1) ? nSamples : (fold + 1) * foldSize;
            
            DataSplit split = splitData(X, y, testStart, testEnd);
            
            // Train and evaluate
            if (problemType == ProblemType.CLASSIFICATION) {
                Classifier classifier = (Classifier) model;
                classifier.fit(split.XTrain, split.yTrain);
                double[] predictions = classifier.predict(split.XTest);
                totalScore += calculateAccuracy(split.yTest, predictions);
            } else {
                Regressor regressor = (Regressor) model;
                regressor.fit(split.XTrain, split.yTrain);
                double[] predictions = regressor.predict(split.XTest);
                totalScore += calculateR2Score(split.yTest, predictions);
            }
        }
        
        return totalScore / cvFolds;
    }
    
    private DataSplit splitData(double[][] X, double[] y, int testStart, int testEnd) {
        int nSamples = X.length;
        int nFeatures = X[0].length;
        int testSize = testEnd - testStart;
        int trainSize = nSamples - testSize;
        
        DataSplit split = new DataSplit();
        split.XTrain = new double[trainSize][nFeatures];
        split.yTrain = new double[trainSize];
        split.XTest = new double[testSize][nFeatures];
        split.yTest = new double[testSize];
        
        int trainIdx = 0, testIdx = 0;
        for (int i = 0; i < nSamples; i++) {
            if (i >= testStart && i < testEnd) {
                split.XTest[testIdx] = X[i].clone();
                split.yTest[testIdx] = y[i];
                testIdx++;
            } else {
                split.XTrain[trainIdx] = X[i].clone();
                split.yTrain[trainIdx] = y[i];
                trainIdx++;
            }
        }
        
        return split;
    }
    
    private double calculateAccuracy(double[] actual, double[] predicted) {
        int correct = 0;
        for (int i = 0; i < actual.length; i++) {
            if (Math.abs(actual[i] - predicted[i]) < 1e-6) {
                correct++;
            }
        }
        return (double) correct / actual.length;
    }
    
    private double calculateR2Score(double[] actual, double[] predicted) {
        double meanActual = java.util.Arrays.stream(actual).average().orElse(0.0);
        
        double totalSumSquares = 0.0;
        double residualSumSquares = 0.0;
        
        for (int i = 0; i < actual.length; i++) {
            totalSumSquares += Math.pow(actual[i] - meanActual, 2);
            residualSumSquares += Math.pow(actual[i] - predicted[i], 2);
        }
        
        return 1.0 - (residualSumSquares / totalSumSquares);
    }
    
    private double evaluateClassificationEnsemble(TreeEnsemblePredictor predictor, 
                                                 double[][] X, double[] y) {
        double[] predictions = predictor.predict(X);
        return calculateAccuracy(y, predictions);
    }
    
    private double evaluateRegressionEnsemble(TreeEnsemblePredictor predictor, 
                                            double[][] X, double[] y) {
        double[] predictions = predictor.predict(X);
        return calculateR2Score(y, predictions);
    }
    
    private double[] calculateConsensusImportance(double[] rf, double[] gb, double[] dt) {
        int nFeatures = rf.length;
        double[] consensus = new double[nFeatures];
        
        for (int i = 0; i < nFeatures; i++) {
            consensus[i] = (rf[i] + gb[i] + dt[i]) / 3.0;
        }
        
        return consensus;
    }
    
    private List<FeatureImportance> rankFeatures(double[] importance, String[] featureNames) {
        List<FeatureImportance> ranking = new ArrayList<>();
        
        for (int i = 0; i < importance.length; i++) {
            FeatureImportance fi = new FeatureImportance();
            fi.featureIndex = i;
            fi.featureName = featureNames != null && i < featureNames.length ? 
                           featureNames[i] : "feature_" + i;
            fi.importance = importance[i];
            ranking.add(fi);
        }
        
        ranking.sort((a, b) -> Double.compare(b.importance, a.importance));
        
        return ranking;
    }
    
    // ================== Result Classes ==================
    
    public static class TreeAutoTrainingResult {
        public BaseEstimator bestModel;
        public HyperparameterSet bestParameters;
        public double bestScore;
        public String bestModelType;
        public List<OptimizationStep> optimizationHistory;
        public ProblemType problemType;
        public long startTime;
        public long endTime;
        public double optimizationTime;
    }
    
    public static class TreeEnsembleResult {
        public List<BaseEstimator> models;
        public List<Double> individualScores;
        public TreeEnsemblePredictor ensemblePredictor;
        public double ensembleScore;
        public long startTime;
        public long endTime;
        public double ensembleTime;
    }
    
    public static class TreeFeatureImportanceResult {
        public double[] rfImportance;
        public double[] gbImportance;
        public double[] dtImportance;
        public double[] consensusImportance;
        public List<FeatureImportance> featureRanking;
        public long startTime;
        public long endTime;
        public double analysisTime;
    }
    
    public static class HyperparameterOptimizationResult {
        public BaseEstimator bestModel;
        public HyperparameterSet bestParameters;
        public double bestScore;
        public List<OptimizationStep> history;
    }
    
    public static class OptimizationStep {
        public HyperparameterSet parameters;
        public double score;
        public BaseEstimator model;
    }
    
    public static class FeatureImportance {
        public int featureIndex;
        public String featureName;
        public double importance;
    }
    
    public static class DataSplit {
        public double[][] XTrain;
        public double[] yTrain;
        public double[][] XTest;
        public double[] yTest;
    }
    
    public static class HyperparameterSet extends HashMap<String, Object> {
        public HyperparameterSet() {
            super();
        }
        
        public HyperparameterSet(HyperparameterSet other) {
            super(other);
        }
        
        public int getInt(String key, int defaultValue) {
            Object value = get(key);
            return value instanceof Integer ? (Integer) value : defaultValue;
        }
        
        public double getDouble(String key, double defaultValue) {
            Object value = get(key);
            return value instanceof Double ? (Double) value : defaultValue;
        }
        
        public String getString(String key, String defaultValue) {
            Object value = get(key);
            return value instanceof String ? (String) value : defaultValue;
        }
        
        public boolean getBoolean(String key, boolean defaultValue) {
            Object value = get(key);
            return value instanceof Boolean ? (Boolean) value : defaultValue;
        }
    }
    
    public static class TreeEnsemblePredictor {
        private final List<BaseEstimator> models;
        private final ProblemType problemType;
        
        public TreeEnsemblePredictor(List<BaseEstimator> models, ProblemType problemType) {
            this.models = models;
            this.problemType = problemType;
        }
        
        public double[] predict(double[][] X) {
            int nSamples = X.length;
            double[] finalPredictions = new double[nSamples];
            
            if (problemType == ProblemType.CLASSIFICATION) {
                // Majority voting for classification
                for (int i = 0; i < nSamples; i++) {
                    Map<Double, Integer> votes = new HashMap<>();
                    
                    for (BaseEstimator model : models) {
                        Classifier classifier = (Classifier) model;
                        double[] predictions = classifier.predict(new double[][]{X[i]});
                        double prediction = predictions[0];
                        
                        votes.put(prediction, votes.getOrDefault(prediction, 0) + 1);
                    }
                    
                    // Find majority vote
                    finalPredictions[i] = votes.entrySet().stream()
                        .max(Map.Entry.comparingByValue())
                        .map(Map.Entry::getKey)
                        .orElse(0.0);
                }
            } else {
                // Averaging for regression
                for (int i = 0; i < nSamples; i++) {
                    double sum = 0.0;
                    
                    for (BaseEstimator model : models) {
                        Regressor regressor = (Regressor) model;
                        double[] predictions = regressor.predict(new double[][]{X[i]});
                        sum += predictions[0];
                    }
                    
                    finalPredictions[i] = sum / models.size();
                }
            }
            
            return finalPredictions;
        }
    }
    
    public void shutdown() {
        executorService.shutdown();
        try {
            if (!executorService.awaitTermination(60, TimeUnit.SECONDS)) {
                executorService.shutdownNow();
            }
        } catch (InterruptedException e) {
            executorService.shutdownNow();
        }
    }
}
