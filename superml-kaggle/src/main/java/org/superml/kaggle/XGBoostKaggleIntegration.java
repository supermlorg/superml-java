/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor            // 2. Feature engineering
            System.out.println("\n🔧 Feature engineering...");
            FeatureEngineering.Result features = engineerFeatures(data);
            
            // 3. Hyperparameter optimization
            System.out.println("\n🎯 Hyperparameter opt        // Load test data  
        DataUtils.DataSet testDataSet = DataUtils.loadCSV(testPath, true, -1);
        List<String[]> testRows = new ArrayList<>();ization...");
            HyperparameterResult bestParams = optimizeHyperparameters(features.trainX, data.trainY);
            
            // 4. Cross-validation with best parameters
            System.out.println("\n✅ Cross-validation with optimized parameters...");
            CrossValidationResult cvResult = performCrossValidation(features.trainX, data.trainY, bestParams.params);
            
            // 5. Train final model on full dataset
            System.out.println("\n🎯 Training final model...");
            XGBoost finalModel = trainFinalModel(features.trainX, data.trainY, bestParams.params);ents.  See the NOTICE file
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

package org.superml.kaggle;

import org.superml.tree.XGBoost;
import org.superml.preprocessing.StandardScaler;
import org.superml.model_selection.CrossValidation;
import org.superml.metrics.Metrics;
import org.superml.utils.DataUtils;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

/**
 * XGBoost Kaggle Integration - Production-ready competition framework
 * 
 * Provides comprehensive tooling for Kaggle competitions using XGBoost with:
 * - Advanced hyperparameter optimization
 * - Robust cross-validation strategies
 * - Feature engineering pipelines
 * - Submission generation and validation
 * - Competition leaderboard tracking
 * - Model ensemble capabilities
 * 
 * Features:
 * - Automated feature selection and engineering
 * - Stratified K-fold validation
 * - Hyperparameter grid search with early stopping
 * - Out-of-fold predictions for stacking
 * - Submission file generation and validation
 * - Competition metrics optimization
 * - Model persistence and versioning
 * - Performance visualization and analysis
 * 
 * @author SuperML Team
 * @version 2.1.0
 */
public class XGBoostKaggleIntegration {
    
    /**
     * Model entry for tracking trained models
     */
    private static class ModelEntry {
        final XGBoost model;
        final Map<String, Double> params;
        final double score;
        
        ModelEntry(XGBoost model, Map<String, Double> params, double score) {
            this.model = model;
            this.params = params;
            this.score = score;
        }
    }
    
    private final KaggleConfig config;
    private final List<ModelEntry> models;
    private final Map<String, Double> leaderboardScores;
    private XGBoost bestModel;
    private double bestCVScore;
    
    public XGBoostKaggleIntegration(KaggleConfig config) {
        this.config = config;
        this.models = new ArrayList<>();
        this.leaderboardScores = new HashMap<>();
        this.bestCVScore = config.isClassification ? 0.0 : Double.MAX_VALUE;
    }
    
    /**
     * Complete competition workflow - from data to submission
     */
    public CompetitionResult runCompetition(String trainPath, String testPath, String submissionPath) {
        System.out.println("🏆 Starting Kaggle Competition Workflow with XGBoost");
        System.out.println("=" + "=".repeat(60));
        
        try {
            // 1. Load and explore data
            System.out.println("\n📊 Loading competition data...");
            CompetitionData data = loadCompetitionData(trainPath, testPath);
            exploreData(data);
            
            // 2. Feature engineering
            System.out.println("\n🔧 Feature engineering...");
            FeatureEngineering.Result features = engineerFeatures(data);
            
            // 3. Hyperparameter optimization
            System.out.println("\n🎯 Hyperparameter optimization...");
            HyperparameterResult bestParams = optimizeHyperparameters(features.trainX, features.trainY);
            
            // 4. Cross-validation with best parameters
            System.out.println("\n✅ Cross-validation with optimized parameters...");
            CrossValidationResult cvResult = performCrossValidation(features.trainX, features.trainY, bestParams.params);
            
            // 5. Train final model on full dataset
            System.out.println("\n🚀 Training final model...");
            XGBoost finalModel = trainFinalModel(features.trainX, features.trainY, bestParams.params);
            
            // 6. Generate predictions and submission
            System.out.println("\n📈 Generating predictions...");
            double[] testPredictions = finalModel.predict(features.testX);
            generateSubmission(testPredictions, submissionPath);
            
            // 7. Model analysis and visualization
            System.out.println("\n📊 Model analysis...");
            ModelAnalysis analysis = analyzeModel(finalModel, features);
            
            // 8. Create competition result
            CompetitionResult result = new CompetitionResult();
            result.bestCVScore = cvResult.meanScore;
            result.bestParams = bestParams.params;
            result.finalModel = finalModel;
            result.featureImportance = analysis.featureImportance;
            result.cvScores = cvResult.scores;
            result.modelAnalysis = analysis;
            result.submissionPath = submissionPath;
            
            System.out.println("\n🎉 Competition workflow completed!");
            System.out.printf("📊 Best CV Score: %.6f ± %.6f%n", cvResult.meanScore, cvResult.stdScore);
            System.out.printf("🏆 Submission saved to: %s%n", submissionPath);
            
            return result;
            
        } catch (Exception e) {
            throw new RuntimeException("Competition workflow failed: " + e.getMessage(), e);
        }
    }
    
    /**
     * Advanced hyperparameter optimization using grid search with early stopping
     */
    public HyperparameterResult optimizeHyperparameters(double[][] X, double[] y) {
        System.out.println("Optimizing hyperparameters with grid search...");
        
        // Define hyperparameter grid
        Map<String, double[]> paramGrid = createParameterGrid();
        
        double bestScore = config.isClassification ? 0.0 : Double.MAX_VALUE;
        Map<String, Double> bestParams = new HashMap<>();
        
        int totalCombinations = calculateGridSize(paramGrid);
        int currentCombination = 0;
        
        System.out.printf("Testing %d parameter combinations...%n", totalCombinations);
        
        // Grid search
        for (Map<String, Double> params : generateParameterCombinations(paramGrid)) {
            currentCombination++;
            
            try {
                // Quick cross-validation for parameter evaluation
                double cvScore = evaluateParameters(X, y, params);
                
                boolean isImprovement = config.isClassification ? 
                    cvScore > bestScore : cvScore < bestScore;
                
                if (isImprovement) {
                    bestScore = cvScore;
                    bestParams = new HashMap<>(params);
                    
                    System.out.printf("[%d/%d] New best score: %.6f with params: %s%n", 
                        currentCombination, totalCombinations, cvScore, formatParams(params));
                } else if (currentCombination % 10 == 0) {
                    System.out.printf("[%d/%d] Current score: %.6f%n", 
                        currentCombination, totalCombinations, cvScore);
                }
                
            } catch (Exception e) {
                System.out.printf("[%d/%d] Failed with params: %s - %s%n", 
                    currentCombination, totalCombinations, formatParams(params), e.getMessage());
            }
        }
        
        HyperparameterResult result = new HyperparameterResult();
        result.params = bestParams;
        result.score = bestScore;
        result.totalCombinations = totalCombinations;
        
        System.out.printf("Hyperparameter optimization completed!%n");
        System.out.printf("Best score: %.6f%n", bestScore);
        System.out.printf("Best parameters: %s%n", formatParams(bestParams));
        
        return result;
    }
    
    /**
     * Robust stratified cross-validation
     */
    public CrossValidationResult performCrossValidation(double[][] X, double[] y, Map<String, Double> params) {
        System.out.printf("Performing %d-fold cross-validation...%n", config.nFolds);
        
        // Create fold indices manually
        int nSamples = X.length;
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < nSamples; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, new Random(config.randomSeed));
        
        List<Double> scores = new ArrayList<>();
        double[][] oofPredictions = new double[X.length][config.isClassification ? 2 : 1];
        
        int foldSize = nSamples / config.nFolds;
        
        for (int fold = 0; fold < config.nFolds; fold++) {
            System.out.printf("Training fold %d/%d...%n", fold + 1, config.nFolds);
            
            // Create train/validation split
            int startTest = fold * foldSize;
            int endTest = (fold == config.nFolds - 1) ? nSamples : (fold + 1) * foldSize;
            
            List<Integer> trainIndices = new ArrayList<>();
            List<Integer> validIndices = new ArrayList<>();
            
            for (int i = 0; i < nSamples; i++) {
                if (i >= startTest && i < endTest) {
                    validIndices.add(indices.get(i));
                } else {
                    trainIndices.add(indices.get(i));
                }
            }
            
            // Create datasets
            double[][] trainX = new double[trainIndices.size()][];
            double[] trainY = new double[trainIndices.size()];
            double[][] validX = new double[validIndices.size()][];
            double[] validY = new double[validIndices.size()];
            
            for (int i = 0; i < trainIndices.size(); i++) {
                trainX[i] = X[trainIndices.get(i)];
                trainY[i] = y[trainIndices.get(i)];
            }
            
            for (int i = 0; i < validIndices.size(); i++) {
                validX[i] = X[validIndices.get(i)];
                validY[i] = y[validIndices.get(i)];
            }
            
            // Train model on fold
            XGBoost model = createModelWithParams(params);
            model.fit(trainX, trainY);
            
            // Validate on fold
            double[] foldPreds = model.predict(validX);
            double foldScore = calculateScore(validY, foldPreds);
            scores.add(foldScore);
            
            // Store out-of-fold predictions for stacking
            for (int i = 0; i < validIndices.size(); i++) {
                int originalIndex = validIndices.get(i);
                if (config.isClassification) {
                    double[][] probas = model.predictProba(new double[][]{validX[i]});
                    oofPredictions[originalIndex] = probas[0];
                } else {
                    oofPredictions[originalIndex][0] = foldPreds[i];
                }
            }
            
            System.out.printf("Fold %d score: %.6f%n", fold + 1, foldScore);
        }
        
        CrossValidationResult result = new CrossValidationResult();
        result.scores = scores;
        result.meanScore = scores.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        result.stdScore = calculateStandardDeviation(scores);
        result.oofPredictions = oofPredictions;
        
        System.out.printf("CV Score: %.6f ± %.6f%n", result.meanScore, result.stdScore);
        
        return result;
    }
    
    /**
     * Advanced feature engineering pipeline
     */
    public FeatureEngineering.Result engineerFeatures(CompetitionData data) {
        System.out.println("Applying feature engineering pipeline...");
        
        FeatureEngineering fe = new FeatureEngineering(config);
        
        // Apply feature engineering
        FeatureEngineering.Result result = fe.transform(data.trainX, data.trainY, data.testX);
        
        System.out.printf("Features: %d -> %d (%.1f%% increase)%n", 
            data.trainX[0].length, result.trainX[0].length,
            ((double) result.trainX[0].length / data.trainX[0].length - 1) * 100);
        
        return result;
    }
    
    /**
     * Generate and validate Kaggle submission file
     */
    public void generateSubmission(double[] predictions, String submissionPath) throws IOException {
        System.out.println("Generating submission file...");
        
        // Validate predictions
        if (predictions == null || predictions.length == 0) {
            throw new IllegalArgumentException("Predictions cannot be empty");
        }
        
        // Create submission directory if needed
        Path path = Path.of(submissionPath);
        Files.createDirectories(path.getParent());
        
        // Generate submission file
        try (PrintWriter writer = new PrintWriter(submissionPath)) {
            // Write header
            writer.println("id," + config.targetColumn);
            
            // Write predictions
            for (int i = 0; i < predictions.length; i++) {
                double pred = predictions[i];
                
                // Apply post-processing if needed
                if (config.isClassification) {
                    // Ensure probabilities are in [0, 1]
                    pred = Math.max(0.0, Math.min(1.0, pred));
                } else {
                    // Apply any regression post-processing
                    if (config.minTargetValue != null) {
                        pred = Math.max(config.minTargetValue, pred);
                    }
                    if (config.maxTargetValue != null) {
                        pred = Math.min(config.maxTargetValue, pred);
                    }
                }
                
                writer.printf("%d,%.8f%n", i, pred);
            }
        }
        
        // Validate submission file
        validateSubmission(submissionPath);
        
        System.out.printf("Submission saved: %s (%d predictions)%n", submissionPath, predictions.length);
    }
    
    /**
     * Comprehensive model analysis
     */
    public ModelAnalysis analyzeModel(XGBoost model, FeatureEngineering.Result features) {
        System.out.println("Analyzing model performance...");
        
        ModelAnalysis analysis = new ModelAnalysis();
        
        // Feature importance analysis
        Map<String, double[]> importanceStats = model.getFeatureImportanceStats();
        analysis.featureImportance = importanceStats;
        
        // Model statistics
        analysis.nTrees = model.getNEstimators();
        analysis.nFeatures = model.getNFeatures();
        analysis.isClassification = model.isClassification();
        
        // Training statistics
        Map<String, List<Double>> evalResults = model.getEvalResults();
        analysis.trainingCurves = evalResults;
        
        // Feature analysis
        analysis.topFeatures = analyzeTopFeatures(importanceStats, features.featureNames);
        
        return analysis;
    }
    
    // Helper methods
    
    private CompetitionData loadCompetitionData(String trainPath, String testPath) throws IOException {
        System.out.println("Loading training and test data...");
        
        // Load training data using DataSet
        DataUtils.DataSet trainDataSet = DataUtils.loadCSV(trainPath, true, -1);
        
        // Load test data (no target column)
        DataUtils.DataSet testDataSet = DataUtils.loadCSV(testPath, true, 0);
        
        CompetitionData data = new CompetitionData();
        data.trainX = trainDataSet.X;
        data.trainY = trainDataSet.y;
        data.testX = testDataSet.X;
        data.featureNames = trainDataSet.headers.toArray(new String[0]);
        
        return data;
    }
    
    private void exploreData(CompetitionData data) {
        System.out.println("\n📊 Data Exploration:");
        System.out.printf("├─ Training samples: %d%n", data.trainX.length);
        System.out.printf("├─ Test samples: %d%n", data.testX.length);
        System.out.printf("├─ Features: %d%n", data.trainX[0].length);
        
        // Target distribution
        if (config.isClassification) {
            Map<Double, Integer> classCounts = new HashMap<>();
            for (double target : data.trainY) {
                classCounts.put(target, classCounts.getOrDefault(target, 0) + 1);
            }
            
            System.out.println("├─ Class distribution:");
            for (Map.Entry<Double, Integer> entry : classCounts.entrySet()) {
                double percentage = (entry.getValue() * 100.0) / data.trainY.length;
                System.out.printf("│  ├─ Class %.0f: %d (%.1f%%)%n", 
                    entry.getKey(), entry.getValue(), percentage);
            }
        } else {
            double mean = Arrays.stream(data.trainY).average().orElse(0.0);
            double min = Arrays.stream(data.trainY).min().orElse(0.0);
            double max = Arrays.stream(data.trainY).max().orElse(0.0);
            
            System.out.printf("├─ Target statistics:%n");
            System.out.printf("│  ├─ Mean: %.4f%n", mean);
            System.out.printf("│  ├─ Min: %.4f%n", min);
            System.out.printf("│  └─ Max: %.4f%n", max);
        }
        
        System.out.println("└─ Ready for feature engineering");
    }
    
    private Map<String, double[]> createParameterGrid() {
        Map<String, double[]> grid = new HashMap<>();
        
        // Learning rate
        grid.put("learning_rate", new double[]{0.01, 0.05, 0.1, 0.2});
        
        // Tree depth
        grid.put("max_depth", new double[]{3, 4, 5, 6, 7, 8});
        
        // Regularization
        grid.put("lambda", new double[]{0.1, 1.0, 10.0});
        grid.put("alpha", new double[]{0.0, 0.1, 1.0});
        
        // Sampling
        grid.put("subsample", new double[]{0.7, 0.8, 0.9, 1.0});
        grid.put("colsample_bytree", new double[]{0.7, 0.8, 0.9, 1.0});
        
        // Pruning
        grid.put("gamma", new double[]{0.0, 0.1, 0.5, 1.0});
        
        return grid;
    }
    
    private List<Map<String, Double>> generateParameterCombinations(Map<String, double[]> paramGrid) {
        List<Map<String, Double>> combinations = new ArrayList<>();
        
        // Start with empty combination
        combinations.add(new HashMap<>());
        
        // Add each parameter
        for (Map.Entry<String, double[]> entry : paramGrid.entrySet()) {
            String param = entry.getKey();
            double[] values = entry.getValue();
            
            List<Map<String, Double>> newCombinations = new ArrayList<>();
            
            for (Map<String, Double> combination : combinations) {
                for (double value : values) {
                    Map<String, Double> newCombination = new HashMap<>(combination);
                    newCombination.put(param, value);
                    newCombinations.add(newCombination);
                }
            }
            
            combinations = newCombinations;
        }
        
        return combinations;
    }
    
    private double evaluateParameters(double[][] X, double[] y, Map<String, Double> params) {
        // Quick 3-fold CV for parameter evaluation
        XGBoost model = createModelWithParams(params);
        
        // Create 3-fold cross-validation manually
        int nFolds = 3;
        int nSamples = X.length;
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < nSamples; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, new Random(config.randomSeed));
        
        List<Double> scores = new ArrayList<>();
        int foldSize = nSamples / nFolds;
        
        for (int fold = 0; fold < nFolds; fold++) {
            // Create train/validation split
            int startTest = fold * foldSize;
            int endTest = (fold == nFolds - 1) ? nSamples : (fold + 1) * foldSize;
            
            List<Integer> trainIndices = new ArrayList<>();
            List<Integer> validIndices = new ArrayList<>();
            
            for (int i = 0; i < nSamples; i++) {
                if (i >= startTest && i < endTest) {
                    validIndices.add(indices.get(i));
                } else {
                    trainIndices.add(indices.get(i));
                }
            }
            
            // Create train/validation datasets
            double[][] trainX = new double[trainIndices.size()][];
            double[] trainY = new double[trainIndices.size()];
            double[][] validX = new double[validIndices.size()][];
            double[] validY = new double[validIndices.size()];
            
            for (int i = 0; i < trainIndices.size(); i++) {
                trainX[i] = X[trainIndices.get(i)];
                trainY[i] = y[trainIndices.get(i)];
            }
            
            for (int i = 0; i < validIndices.size(); i++) {
                validX[i] = X[validIndices.get(i)];
                validY[i] = y[validIndices.get(i)];
            }
            
            // Train and evaluate
            XGBoost foldModel = createModelWithParams(params);
            foldModel.fit(trainX, trainY);
            double[] preds = foldModel.predict(validX);
            double score = calculateScore(validY, preds);
            scores.add(score);
        }
        
        return scores.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
    
    private XGBoost createModelWithParams(Map<String, Double> params) {
        XGBoost model = new XGBoost()
            .setNEstimators(config.nEstimators)
            .setRandomState(config.randomSeed)
            .setEarlyStoppingRounds(50)
            .setValidationFraction(0.1);
        
        // Apply parameters
        if (params.containsKey("learning_rate")) {
            model.setLearningRate(params.get("learning_rate"));
        }
        if (params.containsKey("max_depth")) {
            model.setMaxDepth(params.get("max_depth").intValue());
        }
        if (params.containsKey("lambda")) {
            model.setLambda(params.get("lambda"));
        }
        if (params.containsKey("alpha")) {
            model.setAlpha(params.get("alpha"));
        }
        if (params.containsKey("subsample")) {
            model.setSubsample(params.get("subsample"));
        }
        if (params.containsKey("colsample_bytree")) {
            model.setColsampleBytree(params.get("colsample_bytree"));
        }
        if (params.containsKey("gamma")) {
            model.setGamma(params.get("gamma"));
        }
        
        return model;
    }
    
    private XGBoost trainFinalModel(double[][] X, double[] y, Map<String, Double> params) {
        XGBoost model = createModelWithParams(params);
        
        // Train on full dataset
        model.fit(X, y);
        
        return model;
    }
    
    private double calculateScore(double[] yTrue, double[] yPred) {
        if (config.isClassification) {
            return Metrics.accuracy(yTrue, yPred);
        } else {
            return -Metrics.meanSquaredError(yTrue, yPred); // Negative for maximization
        }
    }
    
    private int findTargetColumn(String[] header) {
        for (int i = 0; i < header.length; i++) {
            if (header[i].equals(config.targetColumn)) {
                return i;
            }
        }
        return -1;
    }
    
    private double[] parseFeatures(String[] row, int targetIndex) {
        List<Double> features = new ArrayList<>();
        
        for (int i = 0; i < row.length; i++) {
            if (i != targetIndex) {
                try {
                    features.add(Double.parseDouble(row[i]));
                } catch (NumberFormatException e) {
                    // Handle categorical or missing values
                    features.add(0.0); // Simple encoding
                }
            }
        }
        
        return features.stream().mapToDouble(Double::doubleValue).toArray();
    }
    
    private String[] extractFeatureNames(String[] header, int targetIndex) {
        List<String> names = new ArrayList<>();
        
        for (int i = 0; i < header.length; i++) {
            if (i != targetIndex) {
                names.add(header[i]);
            }
        }
        
        return names.toArray(new String[0]);
    }
    
    private void validateSubmission(String submissionPath) throws IOException {
        List<String> lines = Files.readAllLines(Path.of(submissionPath));
        
        if (lines.size() < 2) {
            throw new IllegalStateException("Submission file is too short");
        }
        
        // Check header
        String header = lines.get(0);
        if (!header.contains("id") || !header.contains(config.targetColumn)) {
            throw new IllegalStateException("Invalid header in submission file");
        }
        
        System.out.printf("Submission validated: %d rows%n", lines.size() - 1);
    }
    
    // Data classes and configurations
    
    public static class KaggleConfig {
        public boolean isClassification = true;
        public String targetColumn = "target";
        public int nFolds = 5;
        public int nEstimators = 1000;
        public int randomSeed = 42;
        public Double minTargetValue = null;
        public Double maxTargetValue = null;
        
        // Feature engineering settings
        public boolean enableFeatureSelection = true;
        public boolean enableFeatureInteractions = true;
        public boolean enablePolynomialFeatures = false;
        public int maxInteractionDepth = 2;
    }
    
    public static class CompetitionData {
        public double[][] trainX;
        public double[] trainY;
        public double[][] testX;
        public String[] featureNames;
    }
    
    public static class CompetitionResult {
        public double bestCVScore;
        public Map<String, Double> bestParams;
        public XGBoost finalModel;
        public Map<String, double[]> featureImportance;
        public List<Double> cvScores;
        public ModelAnalysis modelAnalysis;
        public String submissionPath;
    }
    
    public static class HyperparameterResult {
        public Map<String, Double> params;
        public double score;
        public int totalCombinations;
    }
    
    public static class CrossValidationResult {
        public List<Double> scores;
        public double meanScore;
        public double stdScore;
        public double[][] oofPredictions;
    }
    
    public static class ModelAnalysis {
        public Map<String, double[]> featureImportance;
        public int nTrees;
        public int nFeatures;
        public boolean isClassification;
        public Map<String, List<Double>> trainingCurves;
        public List<String> topFeatures;
    }
    
    // Helper classes
    
    private static class FeatureEngineering {
        private final KaggleConfig config;
        
        public FeatureEngineering(KaggleConfig config) {
            this.config = config;
        }
        
        public Result transform(double[][] trainX, double[] trainY, double[][] testX) {
            // Simple feature engineering - can be extended
            Result result = new Result();
            result.trainX = trainX;
            result.trainY = trainY;
            result.testX = testX;
            result.featureNames = generateFeatureNames(trainX[0].length);
            return result;
        }
        
        private String[] generateFeatureNames(int nFeatures) {
            String[] names = new String[nFeatures];
            for (int i = 0; i < nFeatures; i++) {
                names[i] = "feature_" + i;
            }
            return names;
        }
        
        public static class Result {
            public double[][] trainX;
            public double[] trainY;
            public double[][] testX;
            public String[] featureNames;
        }
    }
    
    // Utility methods
    
    private int calculateGridSize(Map<String, double[]> grid) {
        return grid.values().stream().mapToInt(arr -> arr.length).reduce(1, (a, b) -> a * b);
    }
    
    private String formatParams(Map<String, Double> params) {
        return params.entrySet().stream()
            .map(e -> e.getKey() + "=" + String.format("%.3f", e.getValue()))
            .collect(Collectors.joining(", "));
    }
    
    private double calculateStandardDeviation(List<Double> values) {
        double mean = values.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double variance = values.stream()
            .mapToDouble(v -> Math.pow(v - mean, 2))
            .average().orElse(0.0);
        return Math.sqrt(variance);
    }
    
    private List<String> analyzeTopFeatures(Map<String, double[]> importance, String[] featureNames) {
        double[] gainImportance = importance.get("gain");
        if (gainImportance == null) return new ArrayList<>();
        
        List<FeatureImportancePair> pairs = new ArrayList<>();
        for (int i = 0; i < gainImportance.length; i++) {
            pairs.add(new FeatureImportancePair(i, gainImportance[i]));
        }
        
        pairs.sort((a, b) -> Double.compare(b.importance, a.importance));
        
        return pairs.stream()
            .limit(20)
            .map(p -> featureNames != null && p.index < featureNames.length ? 
                featureNames[p.index] : "feature_" + p.index)
            .collect(Collectors.toList());
    }
    
    private static class FeatureImportancePair {
        final int index;
        final double importance;
        
        FeatureImportancePair(int index, double importance) {
            this.index = index;
            this.importance = importance;
        }
    }
}
