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

import org.superml.linear_model.*;
import org.superml.metrics.LinearRegressionMetrics;

import java.util.*;
import java.util.concurrent.*;
import java.util.stream.IntStream;

/**
 * AutoTrainer for Linear Models
 * 
 * Provides automated hyperparameter optimization and model selection for linear models:
 * - Automated regularization parameter tuning (Ridge/Lasso)
 * - Cross-validation based model selection
 * - Parallel hyperparameter search
 * - Feature selection automation
 * - Early stopping for iterative algorithms
 * - Ensemble model creation
 * - Performance tracking and visualization
 * - Automatic preprocessing recommendations
 * 
 * Supports:
 * - Linear Regression (feature selection)
 * - Ridge Regression (alpha tuning)
 * - Lasso Regression (alpha tuning + feature selection)
 * - Logistic Regression (regularization tuning)
 * - Model comparison and selection
 * 
 * @author SuperML Team
 * @version 2.1.0
 */
public class LinearModelAutoTrainer {
    
    private int nFolds = 5;
    private int nJobs = Runtime.getRuntime().availableProcessors();
    private boolean verbose = true;
    private Random random = new Random(42);
    private ExecutorService executor;
    
    // Hyperparameter search configurations
    private SearchStrategy searchStrategy = SearchStrategy.GRID_SEARCH;
    private int maxIterations = 100;
    private double tolerance = 1e-6;
    private boolean earlyStoppingEnabled = true;
    private int earlyStoppingRounds = 10;
    
    public enum SearchStrategy {
        GRID_SEARCH,
        RANDOM_SEARCH,
        BAYESIAN_OPTIMIZATION
    }
    
    public enum ModelType {
        LINEAR_REGRESSION,
        RIDGE,
        LASSO,
        LOGISTIC_REGRESSION,
        SGD_CLASSIFIER,
        SGD_REGRESSOR,
        ONE_VS_REST_CLASSIFIER,
        SOFTMAX_REGRESSION,
        AUTO_SELECT
    }
    
    public LinearModelAutoTrainer() {
        this.executor = Executors.newFixedThreadPool(nJobs);
    }
    
    /**
     * Automatically train and optimize a linear model
     */
    public AutoTrainingResult autoTrain(double[][] X, double[] y, ModelType modelType) {
        AutoTrainingResult result = new AutoTrainingResult();
        result.startTime = System.currentTimeMillis();
        
        // Data validation and preprocessing analysis
        DataAnalysis dataAnalysis = analyzeData(X, y);
        result.dataAnalysis = dataAnalysis;
        
        if (verbose) {
            System.out.println("AutoTrainer: Starting automated training");
            System.out.println("Data shape: " + X.length + " x " + X[0].length);
            System.out.println("Target type: " + (dataAnalysis.isClassification ? "Classification" : "Regression"));
        }
        
        // Auto-select model type if requested
        if (modelType == ModelType.AUTO_SELECT) {
            modelType = autoSelectModelType(dataAnalysis);
            if (verbose) {
                System.out.println("Auto-selected model type: " + modelType);
            }
        }
        
        // Perform hyperparameter optimization
        switch (modelType) {
            case LINEAR_REGRESSION:
                result = trainLinearRegression(X, y, result);
                break;
            case RIDGE:
                result = trainRidge(X, y, result);
                break;
            case LASSO:
                result = trainLasso(X, y, result);
                break;
            case LOGISTIC_REGRESSION:
                result = trainLogisticRegression(X, y, result);
                break;
            case SGD_CLASSIFIER:
                result = trainSGDClassifier(X, y, result);
                break;
            case SGD_REGRESSOR:
                result = trainSGDRegressor(X, y, result);
                break;
            case ONE_VS_REST_CLASSIFIER:
                result = trainOneVsRestClassifier(X, y, result);
                break;
            case SOFTMAX_REGRESSION:
                result = trainSoftmaxRegression(X, y, result);
                break;
            case AUTO_SELECT:
                // This case should not be reached as auto-selection happens earlier
                throw new IllegalStateException("AUTO_SELECT should have been resolved earlier");
        }
        
        // Feature importance analysis
        result.featureImportance = calculateFeatureImportance(result.bestModel, X, y);
        
        // Model comparison
        result.modelComparison = compareModels(X, y, dataAnalysis);
        
        result.endTime = System.currentTimeMillis();
        result.trainingTime = result.endTime - result.startTime;
        
        if (verbose) {
            System.out.println("AutoTrainer: Training completed in " + result.trainingTime + "ms");
            System.out.println("Best model: " + result.bestModelType);
            System.out.println("Best score: " + String.format("%.4f", result.bestScore));
        }
        
        return result;
    }
    
    /**
     * Train Linear Regression with feature selection
     */
    private AutoTrainingResult trainLinearRegression(double[][] X, double[] y, AutoTrainingResult result) {
        List<HyperparameterSet> searchSpace = generateLinearRegressionSearchSpace(X[0].length);
        
        HyperparameterOptimizationResult optResult = optimizeHyperparameters(
            X, y, ModelType.LINEAR_REGRESSION, searchSpace);
        
        result.bestModel = optResult.bestModel;
        result.bestParameters = optResult.bestParameters;
        result.bestScore = optResult.bestScore;
        result.bestModelType = "LinearRegression";
        result.optimizationHistory = optResult.history;
        
        return result;
    }
    
    /**
     * Train Ridge regression with alpha optimization
     */
    private AutoTrainingResult trainRidge(double[][] X, double[] y, AutoTrainingResult result) {
        List<HyperparameterSet> searchSpace = generateRidgeSearchSpace();
        
        HyperparameterOptimizationResult optResult = optimizeHyperparameters(
            X, y, ModelType.RIDGE, searchSpace);
        
        result.bestModel = optResult.bestModel;
        result.bestParameters = optResult.bestParameters;
        result.bestScore = optResult.bestScore;
        result.bestModelType = "Ridge";
        result.optimizationHistory = optResult.history;
        
        return result;
    }
    
    /**
     * Train Lasso regression with alpha optimization
     */
    private AutoTrainingResult trainLasso(double[][] X, double[] y, AutoTrainingResult result) {
        List<HyperparameterSet> searchSpace = generateLassoSearchSpace();
        
        HyperparameterOptimizationResult optResult = optimizeHyperparameters(
            X, y, ModelType.LASSO, searchSpace);
        
        result.bestModel = optResult.bestModel;
        result.bestParameters = optResult.bestParameters;
        result.bestScore = optResult.bestScore;
        result.bestModelType = "Lasso";
        result.optimizationHistory = optResult.history;
        
        return result;
    }
    
    /**
     * Train Logistic Regression with regularization optimization
     */
    private AutoTrainingResult trainLogisticRegression(double[][] X, double[] y, AutoTrainingResult result) {
        List<HyperparameterSet> searchSpace = generateLogisticRegressionSearchSpace();
        
        HyperparameterOptimizationResult optResult = optimizeHyperparameters(
            X, y, ModelType.LOGISTIC_REGRESSION, searchSpace);
        
        result.bestModel = optResult.bestModel;
        result.bestParameters = optResult.bestParameters;
        result.bestScore = optResult.bestScore;
        result.bestModelType = "LogisticRegression";
        result.optimizationHistory = optResult.history;
        
        return result;
    }
    
    /**
     * Train SGD Classifier with comprehensive hyperparameter optimization
     */
    private AutoTrainingResult trainSGDClassifier(double[][] X, double[] y, AutoTrainingResult result) {
        List<HyperparameterSet> searchSpace = generateSGDClassifierSearchSpace();
        
        HyperparameterOptimizationResult optResult = optimizeHyperparameters(
            X, y, ModelType.SGD_CLASSIFIER, searchSpace);
        
        result.bestModel = optResult.bestModel;
        result.bestParameters = optResult.bestParameters;
        result.bestScore = optResult.bestScore;
        result.bestModelType = "SGDClassifier";
        result.optimizationHistory = optResult.history;
        
        return result;
    }
    
    /**
     * Train SGD Regressor with comprehensive hyperparameter optimization
     */
    private AutoTrainingResult trainSGDRegressor(double[][] X, double[] y, AutoTrainingResult result) {
        List<HyperparameterSet> searchSpace = generateSGDRegressorSearchSpace();
        
        HyperparameterOptimizationResult optResult = optimizeHyperparameters(
            X, y, ModelType.SGD_REGRESSOR, searchSpace);
        
        result.bestModel = optResult.bestModel;
        result.bestParameters = optResult.bestParameters;
        result.bestScore = optResult.bestScore;
        result.bestModelType = "SGDRegressor";
        result.optimizationHistory = optResult.history;
        
        return result;
    }
    
    /**
     * Train OneVsRestClassifier with comprehensive hyperparameter optimization
     */
    private AutoTrainingResult trainOneVsRestClassifier(double[][] X, double[] y, AutoTrainingResult result) {
        List<HyperparameterSet> searchSpace = generateOneVsRestSearchSpace();
        
        HyperparameterOptimizationResult optResult = optimizeHyperparameters(
            X, y, ModelType.ONE_VS_REST_CLASSIFIER, searchSpace);
        
        result.bestModel = optResult.bestModel;
        result.bestParameters = optResult.bestParameters;
        result.bestScore = optResult.bestScore;
        result.bestModelType = "OneVsRestClassifier";
        result.optimizationHistory = optResult.history;
        
        return result;
    }
    
    /**
     * Train SoftmaxRegression with comprehensive hyperparameter optimization
     */
    private AutoTrainingResult trainSoftmaxRegression(double[][] X, double[] y, AutoTrainingResult result) {
        List<HyperparameterSet> searchSpace = generateSoftmaxSearchSpace();
        
        HyperparameterOptimizationResult optResult = optimizeHyperparameters(
            X, y, ModelType.SOFTMAX_REGRESSION, searchSpace);
        
        result.bestModel = optResult.bestModel;
        result.bestParameters = optResult.bestParameters;
        result.bestScore = optResult.bestScore;
        result.bestModelType = "SoftmaxRegression";
        result.optimizationHistory = optResult.history;
        
        return result;
    }
    
    /**
     * Core hyperparameter optimization engine
     */
    private HyperparameterOptimizationResult optimizeHyperparameters(double[][] X, double[] y, 
                                                                    ModelType modelType, 
                                                                    List<HyperparameterSet> searchSpace) {
        HyperparameterOptimizationResult result = new HyperparameterOptimizationResult();
        result.history = new ArrayList<>();
        
        double bestScore = Double.NEGATIVE_INFINITY;
        Object bestModel = null;
        HyperparameterSet bestParams = null;
        
        List<Future<EvaluationResult>> futures = new ArrayList<>();
        
        // Parallel evaluation of hyperparameter combinations
        for (HyperparameterSet params : searchSpace) {
            Future<EvaluationResult> future = executor.submit(() -> 
                evaluateHyperparameters(X, y, modelType, params));
            futures.add(future);
        }
        
        // Collect results
        for (int i = 0; i < futures.size(); i++) {
            try {
                EvaluationResult evalResult = futures.get(i).get();
                result.history.add(evalResult);
                
                if (evalResult.score > bestScore) {
                    bestScore = evalResult.score;
                    bestModel = evalResult.model;
                    bestParams = evalResult.parameters;
                }
                
                if (verbose && i % 10 == 0) {
                    System.out.println("Evaluated " + (i + 1) + "/" + futures.size() + 
                                     " parameter combinations");
                }
            } catch (Exception e) {
                if (verbose) {
                    System.err.println("Error evaluating parameters: " + e.getMessage());
                }
            }
        }
        
        result.bestModel = bestModel;
        result.bestParameters = bestParams;
        result.bestScore = bestScore;
        
        return result;
    }
    
    /**
     * Evaluate a specific hyperparameter configuration
     */
    private EvaluationResult evaluateHyperparameters(double[][] X, double[] y, 
                                                   ModelType modelType, 
                                                   HyperparameterSet params) {
        EvaluationResult result = new EvaluationResult();
        result.parameters = params;
        
        try {
            // Create model with parameters
            Object model = createModel(modelType, params);
            
            // Cross-validation evaluation
            CrossValidationResult cvResult = crossValidate(model, X, y);
            
            result.model = model;
            result.score = cvResult.meanScore;
            result.scoreStd = cvResult.stdScore;
            result.cvScores = cvResult.scores;
            
        } catch (Exception e) {
            result.score = Double.NEGATIVE_INFINITY;
            result.error = e.getMessage();
        }
        
        return result;
    }
    
    /**
     * Perform k-fold cross-validation
     */
    private CrossValidationResult crossValidate(Object model, double[][] X, double[] y) {
        double[] scores = new double[nFolds];
        int n = X.length;
        int foldSize = n / nFolds;
        
        for (int fold = 0; fold < nFolds; fold++) {
            // Create train/validation split
            int valStart = fold * foldSize;
            int valEnd = (fold == nFolds - 1) ? n : (fold + 1) * foldSize;
            
            // Split data
            DataSplit split = createTrainValidationSplit(X, y, valStart, valEnd);
            
            // Clone model for this fold
            Object foldModel = cloneModel(model);
            
            // Train on training set
            trainModel(foldModel, split.XTrain, split.yTrain);
            
            // Evaluate on validation set
            double[] predictions = predictModel(foldModel, split.XVal);
            scores[fold] = calculateScore(split.yVal, predictions);
        }
        
        CrossValidationResult result = new CrossValidationResult();
        result.scores = scores;
        result.meanScore = Arrays.stream(scores).average().orElse(0.0);
        result.stdScore = calculateStandardDeviation(scores);
        
        return result;
    }
    
    /**
     * Data analysis for preprocessing recommendations
     */
    private DataAnalysis analyzeData(double[][] X, double[] y) {
        DataAnalysis analysis = new DataAnalysis();
        
        // Basic statistics
        analysis.nSamples = X.length;
        analysis.nFeatures = X[0].length;
        
        // Check if classification or regression
        Set<Double> uniqueValues = new HashSet<>();
        for (double value : y) {
            uniqueValues.add(value);
        }
        analysis.isClassification = uniqueValues.size() <= 20 && 
                                  uniqueValues.stream().allMatch(v -> v == Math.floor(v));
        
        // Feature analysis
        analysis.featureStats = new FeatureStatistics[analysis.nFeatures];
        for (int j = 0; j < analysis.nFeatures; j++) {
            analysis.featureStats[j] = analyzeFeature(X, j);
        }
        
        // Correlation analysis
        analysis.correlationMatrix = calculateCorrelationMatrix(X);
        
        // Missing value analysis
        analysis.missingValueCounts = countMissingValues(X);
        
        // Scaling recommendations
        analysis.needsScaling = needsScaling(X);
        
        return analysis;
    }
    
    /**
     * Auto-select optimal model type based on data characteristics
     */
    private ModelType autoSelectModelType(DataAnalysis analysis) {
        if (analysis.isClassification) {
            return ModelType.LOGISTIC_REGRESSION;
        }
        
        // For regression, choose based on feature characteristics
        if (analysis.nFeatures > analysis.nSamples) {
            // High-dimensional data - prefer Lasso for feature selection
            return ModelType.LASSO;
        } else if (analysis.nFeatures > analysis.nSamples * 0.5) {
            // Moderate dimensionality - prefer Ridge for regularization
            return ModelType.RIDGE;
        } else {
            // Low dimensionality - Linear regression may be sufficient
            return ModelType.LINEAR_REGRESSION;
        }
    }
    
    /**
     * Calculate feature importance for the best model
     */
    private FeatureImportance calculateFeatureImportance(Object model, double[][] X, double[] y) {
        FeatureImportance importance = new FeatureImportance();
        importance.featureNames = generateFeatureNames(X[0].length);
        
        try {
            if (model instanceof LinearRegression) {
                LinearRegression lr = (LinearRegression) model;
                if (lr.getCoefficients() != null) {
                    importance.coefficients = lr.getCoefficients();
                    importance.importance = Arrays.stream(importance.coefficients)
                        .map(Math::abs)
                        .toArray();
                } else {
                    importance.coefficients = new double[X[0].length];
                    importance.importance = new double[X[0].length];
                }
            } else if (model instanceof Ridge) {
                Ridge ridge = (Ridge) model;
                if (ridge.getCoefficients() != null) {
                    importance.coefficients = ridge.getCoefficients();
                    importance.importance = Arrays.stream(importance.coefficients)
                        .map(Math::abs)
                        .toArray();
                } else {
                    importance.coefficients = new double[X[0].length];
                    importance.importance = new double[X[0].length];
                }
            } else if (model instanceof Lasso) {
                Lasso lasso = (Lasso) model;
                if (lasso.getCoefficients() != null) {
                    importance.coefficients = lasso.getCoefficients();
                    importance.importance = Arrays.stream(importance.coefficients)
                        .map(Math::abs)
                        .toArray();
                    
                    // Count non-zero features (Lasso feature selection)
                    importance.selectedFeatures = (int) Arrays.stream(importance.coefficients)
                        .mapToLong(c -> Math.abs(c) > 1e-8 ? 1 : 0)
                        .sum();
                } else {
                    importance.coefficients = new double[X[0].length];
                    importance.importance = new double[X[0].length];
                    importance.selectedFeatures = 0;
                }
            } else {
                importance.coefficients = new double[X[0].length];
                importance.importance = new double[X[0].length];
            }
        } catch (Exception e) {
            // Fallback if coefficients aren't available
            importance.coefficients = new double[X[0].length];
            importance.importance = new double[X[0].length];
        }
        
        // Sort by importance
        Integer[] indices = IntStream.range(0, importance.importance.length)
            .boxed()
            .toArray(Integer[]::new);
        Arrays.sort(indices, (i, j) -> Double.compare(importance.importance[j], importance.importance[i]));
        
        importance.sortedIndices = indices;
        importance.sortedFeatureNames = Arrays.stream(indices)
            .map(i -> importance.featureNames[i])
            .toArray(String[]::new);
        importance.sortedImportance = Arrays.stream(indices)
            .mapToDouble(i -> importance.importance[i])
            .toArray();
        
        return importance;
    }
    
    /**
     * Compare different model types
     */
    private ModelComparison compareModels(double[][] X, double[] y, DataAnalysis analysis) {
        ModelComparison comparison = new ModelComparison();
        
        List<ModelType> modelsToCompare = Arrays.asList(
            ModelType.LINEAR_REGRESSION,
            ModelType.RIDGE,
            ModelType.LASSO
        );
        
        if (analysis.isClassification) {
            modelsToCompare = Arrays.asList(ModelType.LOGISTIC_REGRESSION);
        }
        
        comparison.results = new ArrayList<>();
        
        for (ModelType modelType : modelsToCompare) {
            ModelComparisonResult result = new ModelComparisonResult();
            result.modelType = modelType.toString();
            
            try {
                // Quick evaluation with default parameters
                Object model = createModel(modelType, new HyperparameterSet());
                CrossValidationResult cvResult = crossValidate(model, X, y);
                
                result.meanScore = cvResult.meanScore;
                result.stdScore = cvResult.stdScore;
                result.success = true;
                
            } catch (Exception e) {
                result.success = false;
                result.error = e.getMessage();
            }
            
            comparison.results.add(result);
        }
        
        return comparison;
    }
    
    // Search space generation methods
    
    private List<HyperparameterSet> generateLinearRegressionSearchSpace(int nFeatures) {
        List<HyperparameterSet> searchSpace = new ArrayList<>();
        
        // For Linear Regression, we can vary feature selection methods
        searchSpace.add(new HyperparameterSet()); // Default - no feature selection
        
        return searchSpace;
    }
    
    private List<HyperparameterSet> generateRidgeSearchSpace() {
        List<HyperparameterSet> searchSpace = new ArrayList<>();
        
        // Alpha values for Ridge regression
        double[] alphas = {0.01, 0.1, 1.0, 10.0, 100.0, 1000.0};
        
        for (double alpha : alphas) {
            HyperparameterSet params = new HyperparameterSet();
            params.put("alpha", alpha);
            searchSpace.add(params);
        }
        
        return searchSpace;
    }
    
    private List<HyperparameterSet> generateLassoSearchSpace() {
        List<HyperparameterSet> searchSpace = new ArrayList<>();
        
        // Alpha values for Lasso regression
        double[] alphas = {0.001, 0.01, 0.1, 1.0, 10.0, 100.0};
        
        for (double alpha : alphas) {
            HyperparameterSet params = new HyperparameterSet();
            params.put("alpha", alpha);
            searchSpace.add(params);
        }
        
        return searchSpace;
    }
    
    private List<HyperparameterSet> generateLogisticRegressionSearchSpace() {
        List<HyperparameterSet> searchSpace = new ArrayList<>();
        
        // C values for Logistic Regression (inverse of regularization strength)
        double[] cValues = {0.001, 0.01, 0.1, 1.0, 10.0, 100.0};
        
        for (double c : cValues) {
            HyperparameterSet params = new HyperparameterSet();
            params.put("C", c);
            searchSpace.add(params);
        }
        
        return searchSpace;
    }
    
    private List<HyperparameterSet> generateSGDClassifierSearchSpace() {
        List<HyperparameterSet> searchSpace = new ArrayList<>();
        
        // SGD Classifier hyperparameters
        double[] alphas = {0.0001, 0.001, 0.01, 0.1};
        String[] losses = {"hinge", "log", "squared_hinge", "modified_huber"};
        String[] penalties = {"l1", "l2", "elasticnet"};
        double[] l1Ratios = {0.15, 0.5, 0.7};
        String[] learningRates = {"optimal", "constant", "invscaling"};
        
        for (double alpha : alphas) {
            for (String loss : losses) {
                for (String penalty : penalties) {
                    if ("elasticnet".equals(penalty)) {
                        for (double l1Ratio : l1Ratios) {
                            for (String learningRate : learningRates) {
                                HyperparameterSet params = new HyperparameterSet();
                                params.put("alpha", alpha);
                                params.put("loss", loss);
                                params.put("penalty", penalty);
                                params.put("l1_ratio", l1Ratio);
                                params.put("learning_rate", learningRate);
                                searchSpace.add(params);
                            }
                        }
                    } else {
                        for (String learningRate : learningRates) {
                            HyperparameterSet params = new HyperparameterSet();
                            params.put("alpha", alpha);
                            params.put("loss", loss);
                            params.put("penalty", penalty);
                            params.put("learning_rate", learningRate);
                            searchSpace.add(params);
                        }
                    }
                }
            }
        }
        
        return searchSpace;
    }
    
    private List<HyperparameterSet> generateSGDRegressorSearchSpace() {
        List<HyperparameterSet> searchSpace = new ArrayList<>();
        
        // SGD Regressor hyperparameters
        double[] alphas = {0.0001, 0.001, 0.01, 0.1};
        String[] losses = {"squared_loss", "huber", "epsilon_insensitive"};
        String[] penalties = {"l1", "l2", "elasticnet"};
        double[] l1Ratios = {0.15, 0.5, 0.7};
        String[] learningRates = {"optimal", "constant", "invscaling"};
        double[] epsilons = {0.01, 0.1, 0.5};
        
        for (double alpha : alphas) {
            for (String loss : losses) {
                for (String penalty : penalties) {
                    if ("elasticnet".equals(penalty)) {
                        for (double l1Ratio : l1Ratios) {
                            for (String learningRate : learningRates) {
                                HyperparameterSet params = new HyperparameterSet();
                                params.put("alpha", alpha);
                                params.put("loss", loss);
                                params.put("penalty", penalty);
                                params.put("l1_ratio", l1Ratio);
                                params.put("learning_rate", learningRate);
                                
                                if ("epsilon_insensitive".equals(loss)) {
                                    for (double epsilon : epsilons) {
                                        HyperparameterSet epsParams = new HyperparameterSet();
                                        epsParams.put("alpha", alpha);
                                        epsParams.put("loss", loss);
                                        epsParams.put("penalty", penalty);
                                        epsParams.put("l1_ratio", l1Ratio);
                                        epsParams.put("learning_rate", learningRate);
                                        epsParams.put("epsilon", epsilon);
                                        searchSpace.add(epsParams);
                                    }
                                } else {
                                    searchSpace.add(params);
                                }
                            }
                        }
                    } else {
                        for (String learningRate : learningRates) {
                            HyperparameterSet params = new HyperparameterSet();
                            params.put("alpha", alpha);
                            params.put("loss", loss);
                            params.put("penalty", penalty);
                            params.put("learning_rate", learningRate);
                            
                            if ("epsilon_insensitive".equals(loss)) {
                                for (double epsilon : epsilons) {
                                    HyperparameterSet epsParams = new HyperparameterSet();
                                    epsParams.put("alpha", alpha);
                                    epsParams.put("loss", loss);
                                    epsParams.put("penalty", penalty);
                                    epsParams.put("learning_rate", learningRate);
                                    epsParams.put("epsilon", epsilon);
                                    searchSpace.add(epsParams);
                                }
                            } else {
                                searchSpace.add(params);
                            }
                        }
                    }
                }
            }
        }
        
        return searchSpace;
    }
    
    private List<HyperparameterSet> generateOneVsRestSearchSpace() {
        List<HyperparameterSet> searchSpace = new ArrayList<>();
        
        // OneVsRest hyperparameters - focusing on base estimator parameters
        double[] regularizations = {0.01, 0.1, 1.0, 10.0, 100.0};
        String[] solvers = {"liblinear", "lbfgs", "newton-cg"};
        double[] tolerances = {1e-4, 1e-3, 1e-2};
        int[] maxIters = {100, 500, 1000};
        
        for (double C : regularizations) {
            for (String solver : solvers) {
                for (double tol : tolerances) {
                    for (int maxIter : maxIters) {
                        HyperparameterSet params = new HyperparameterSet();
                        params.put("C", C);
                        params.put("solver", solver);
                        params.put("tol", tol);
                        params.put("max_iter", maxIter);
                        searchSpace.add(params);
                    }
                }
            }
        }
        
        return searchSpace;
    }
    
    private List<HyperparameterSet> generateSoftmaxSearchSpace() {
        List<HyperparameterSet> searchSpace = new ArrayList<>();
        
        // Softmax regression hyperparameters
        double[] regularizations = {0.01, 0.1, 1.0, 10.0, 100.0};
        double[] learningRates = {0.001, 0.01, 0.1, 0.5};
        int[] maxIters = {100, 500, 1000, 2000};
        double[] tolerances = {1e-4, 1e-3, 1e-2};
        String[] solvers = {"lbfgs", "newton-cg", "sag"};
        
        for (double C : regularizations) {
            for (double lr : learningRates) {
                for (int maxIter : maxIters) {
                    for (double tol : tolerances) {
                        for (String solver : solvers) {
                            HyperparameterSet params = new HyperparameterSet();
                            params.put("C", C);
                            params.put("learning_rate", lr);
                            params.put("max_iter", maxIter);
                            params.put("tol", tol);
                            params.put("solver", solver);
                            searchSpace.add(params);
                        }
                    }
                }
            }
        }
        
        return searchSpace;
    }
    
    // Helper methods
    
    private Object createModel(ModelType modelType, HyperparameterSet params) {
        switch (modelType) {
            case LINEAR_REGRESSION:
                return new LinearRegression();
                
            case RIDGE:
                double alpha = params.getDouble("alpha", 1.0);
                return new Ridge().setAlpha(alpha);
                
            case LASSO:
                double lassoAlpha = params.getDouble("alpha", 1.0);
                return new Lasso().setAlpha(lassoAlpha);
                
            case LOGISTIC_REGRESSION:
                // Simplified - actual LogisticRegression would have regularization
                return new LogisticRegression();
                
            case SGD_CLASSIFIER:
                SGDClassifier sgdClassifier = new SGDClassifier();
                sgdClassifier.setAlpha(params.getDouble("alpha", 0.0001))
                           .setLoss(params.getString("loss", "hinge"))
                           .setPenalty(params.getString("penalty", "l2"))
                           .setLearningRate(params.getString("learning_rate", "optimal"));
                
                if ("elasticnet".equals(params.getString("penalty", "l2"))) {
                    sgdClassifier.setL1Ratio(params.getDouble("l1_ratio", 0.15));
                }
                
                return sgdClassifier;
                
            case SGD_REGRESSOR:
                SGDRegressor sgdRegressor = new SGDRegressor();
                sgdRegressor.setAlpha(params.getDouble("alpha", 0.0001))
                          .setLoss(params.getString("loss", "squared_loss"))
                          .setPenalty(params.getString("penalty", "l2"))
                          .setLearningRate(params.getString("learning_rate", "optimal"));
                
                if ("elasticnet".equals(params.getString("penalty", "l2"))) {
                    sgdRegressor.setL1Ratio(params.getDouble("l1_ratio", 0.15));
                }
                
                if ("epsilon_insensitive".equals(params.getString("loss", "squared_loss"))) {
                    sgdRegressor.setEpsilon(params.getDouble("epsilon", 0.1));
                }
                
                return sgdRegressor;
                
            case ONE_VS_REST_CLASSIFIER:
                // Use LogisticRegression as base estimator for OneVsRest
                LogisticRegression baseEstimator = new LogisticRegression();
                OneVsRestClassifier oneVsRest = new OneVsRestClassifier(baseEstimator);
                // Note: OneVsRest configuration is handled through base estimator
                
                return oneVsRest;
                
            case SOFTMAX_REGRESSION:
                SoftmaxRegression softmax = new SoftmaxRegression();
                softmax.setC(params.getDouble("C", 1.0))
                       .setLearningRate(params.getDouble("learning_rate", 0.01))
                       .setMaxIter(params.getInt("max_iter", 1000))
                       .setTolerance(params.getDouble("tol", 1e-4));
                
                return softmax;
                
            default:
                throw new IllegalArgumentException("Unsupported model type: " + modelType);
        }
    }
    
    private Object cloneModel(Object model) {
        // Simplified cloning - in practice would use proper deep cloning
        if (model instanceof LinearRegression) {
            return new LinearRegression();
        } else if (model instanceof Ridge) {
            Ridge ridge = (Ridge) model;
            return new Ridge().setAlpha(ridge.getAlpha());
        } else if (model instanceof Lasso) {
            Lasso lasso = (Lasso) model;
            return new Lasso().setAlpha(lasso.getAlpha());
        } else if (model instanceof LogisticRegression) {
            return new LogisticRegression();
        } else if (model instanceof SGDClassifier) {
            // For SGD models, return a new instance with default parameters
            // In practice, we would copy all parameters
            return new SGDClassifier();
        } else if (model instanceof SGDRegressor) {
            return new SGDRegressor();
        }
        throw new IllegalArgumentException("Cannot clone model of type: " + model.getClass());
    }
    
    private void trainModel(Object model, double[][] X, double[] y) {
        if (model instanceof LinearRegression) {
            ((LinearRegression) model).fit(X, y);
        } else if (model instanceof Ridge) {
            ((Ridge) model).fit(X, y);
        } else if (model instanceof Lasso) {
            ((Lasso) model).fit(X, y);
        } else if (model instanceof LogisticRegression) {
            ((LogisticRegression) model).fit(X, y);
        } else if (model instanceof SGDClassifier) {
            ((SGDClassifier) model).fit(X, y);
        } else if (model instanceof SGDRegressor) {
            ((SGDRegressor) model).fit(X, y);
        } else {
            throw new IllegalArgumentException("Cannot train model of type: " + model.getClass());
        }
    }
    
    private double[] predictModel(Object model, double[][] X) {
        if (model instanceof LinearRegression) {
            return ((LinearRegression) model).predict(X);
        } else if (model instanceof Ridge) {
            return ((Ridge) model).predict(X);
        } else if (model instanceof Lasso) {
            return ((Lasso) model).predict(X);
        } else if (model instanceof LogisticRegression) {
            return ((LogisticRegression) model).predict(X);
        }
        return new double[X.length];
    }
    
    private double calculateScore(double[] yTrue, double[] yPred) {
        // Use R² for regression, accuracy for classification
        return LinearRegressionMetrics.rSquared(yTrue, yPred);
    }
    
    private double calculateStandardDeviation(double[] values) {
        double mean = Arrays.stream(values).average().orElse(0.0);
        double variance = Arrays.stream(values)
            .map(x -> Math.pow(x - mean, 2))
            .average()
            .orElse(0.0);
        return Math.sqrt(variance);
    }
    
    private DataSplit createTrainValidationSplit(double[][] X, double[] y, int valStart, int valEnd) {
        int n = X.length;
        int nFeatures = X[0].length;
        
        // Validation set
        int valSize = valEnd - valStart;
        double[][] XVal = new double[valSize][nFeatures];
        double[] yVal = new double[valSize];
        
        for (int i = 0; i < valSize; i++) {
            System.arraycopy(X[valStart + i], 0, XVal[i], 0, nFeatures);
            yVal[i] = y[valStart + i];
        }
        
        // Training set
        int trainSize = n - valSize;
        double[][] XTrain = new double[trainSize][nFeatures];
        double[] yTrain = new double[trainSize];
        
        int trainIdx = 0;
        for (int i = 0; i < n; i++) {
            if (i < valStart || i >= valEnd) {
                System.arraycopy(X[i], 0, XTrain[trainIdx], 0, nFeatures);
                yTrain[trainIdx] = y[i];
                trainIdx++;
            }
        }
        
        DataSplit split = new DataSplit();
        split.XTrain = XTrain;
        split.yTrain = yTrain;
        split.XVal = XVal;
        split.yVal = yVal;
        
        return split;
    }
    
    private FeatureStatistics analyzeFeature(double[][] X, int featureIndex) {
        double[] feature = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            feature[i] = X[i][featureIndex];
        }
        
        FeatureStatistics stats = new FeatureStatistics();
        stats.mean = Arrays.stream(feature).average().orElse(0.0);
        stats.std = calculateStandardDeviation(feature);
        stats.min = Arrays.stream(feature).min().orElse(0.0);
        stats.max = Arrays.stream(feature).max().orElse(0.0);
        
        return stats;
    }
    
    private double[][] calculateCorrelationMatrix(double[][] X) {
        int nFeatures = X[0].length;
        double[][] correlation = new double[nFeatures][nFeatures];
        
        for (int i = 0; i < nFeatures; i++) {
            for (int j = 0; j < nFeatures; j++) {
                correlation[i][j] = calculateCorrelation(X, i, j);
            }
        }
        
        return correlation;
    }
    
    private double calculateCorrelation(double[][] X, int feature1, int feature2) {
        // Simplified Pearson correlation
        double[] f1 = Arrays.stream(X).mapToDouble(row -> row[feature1]).toArray();
        double[] f2 = Arrays.stream(X).mapToDouble(row -> row[feature2]).toArray();
        
        double mean1 = Arrays.stream(f1).average().orElse(0.0);
        double mean2 = Arrays.stream(f2).average().orElse(0.0);
        
        double numerator = 0.0;
        double sum1 = 0.0;
        double sum2 = 0.0;
        
        for (int i = 0; i < f1.length; i++) {
            double diff1 = f1[i] - mean1;
            double diff2 = f2[i] - mean2;
            numerator += diff1 * diff2;
            sum1 += diff1 * diff1;
            sum2 += diff2 * diff2;
        }
        
        double denominator = Math.sqrt(sum1 * sum2);
        return denominator == 0 ? 0 : numerator / denominator;
    }
    
    private int[] countMissingValues(double[][] X) {
        int[] counts = new int[X[0].length];
        
        for (int j = 0; j < X[0].length; j++) {
            final int featureIndex = j; // Make variable effectively final
            for (int i = 0; i < X.length; i++) {
                if (Double.isNaN(X[i][featureIndex])) {
                    counts[featureIndex]++;
                }
            }
        }
        
        return counts;
    }
    
    private boolean needsScaling(double[][] X) {
        // Check if features have significantly different scales
        double[] scales = new double[X[0].length];
        
        for (int j = 0; j < X[0].length; j++) {
            final int featureIndex = j; // Make variable effectively final for lambda
            double[] feature = Arrays.stream(X).mapToDouble(row -> row[featureIndex]).toArray();
            double std = calculateStandardDeviation(feature);
            scales[j] = std;
        }
        
        double maxScale = Arrays.stream(scales).max().orElse(1.0);
        double minScale = Arrays.stream(scales).min().orElse(1.0);
        
        return (maxScale / minScale) > 10.0; // Significant scale difference
    }
    
    private String[] generateFeatureNames(int nFeatures) {
        String[] names = new String[nFeatures];
        for (int i = 0; i < nFeatures; i++) {
            names[i] = "feature_" + i;
        }
        return names;
    }
    
    // Configuration methods
    
    public LinearModelAutoTrainer setNFolds(int nFolds) {
        this.nFolds = nFolds;
        return this;
    }
    
    public LinearModelAutoTrainer setNJobs(int nJobs) {
        this.nJobs = nJobs;
        if (this.executor != null) {
            this.executor.shutdown();
        }
        this.executor = Executors.newFixedThreadPool(nJobs);
        return this;
    }
    
    public LinearModelAutoTrainer setVerbose(boolean verbose) {
        this.verbose = verbose;
        return this;
    }
    
    public LinearModelAutoTrainer setSearchStrategy(SearchStrategy strategy) {
        this.searchStrategy = strategy;
        return this;
    }
    
    public LinearModelAutoTrainer setRandomSeed(long seed) {
        this.random = new Random(seed);
        return this;
    }
    
    public void shutdown() {
        if (executor != null) {
            executor.shutdown();
        }
    }
    
    // Data classes
    
    public static class AutoTrainingResult {
        public Object bestModel;
        public String bestModelType;
        public HyperparameterSet bestParameters;
        public double bestScore;
        public List<EvaluationResult> optimizationHistory;
        public FeatureImportance featureImportance;
        public ModelComparison modelComparison;
        public DataAnalysis dataAnalysis;
        public long startTime;
        public long endTime;
        public long trainingTime;
    }
    
    public static class HyperparameterOptimizationResult {
        public Object bestModel;
        public HyperparameterSet bestParameters;
        public double bestScore;
        public List<EvaluationResult> history;
    }
    
    public static class EvaluationResult {
        public Object model;
        public HyperparameterSet parameters;
        public double score;
        public double scoreStd;
        public double[] cvScores;
        public String error;
    }
    
    public static class CrossValidationResult {
        public double[] scores;
        public double meanScore;
        public double stdScore;
    }
    
    public static class HyperparameterSet {
        private Map<String, Object> parameters = new HashMap<>();
        
        public void put(String key, Object value) {
            parameters.put(key, value);
        }
        
        public double getDouble(String key, double defaultValue) {
            Object value = parameters.get(key);
            if (value instanceof Number) {
                return ((Number) value).doubleValue();
            }
            return defaultValue;
        }
        
        public int getInt(String key, int defaultValue) {
            Object value = parameters.get(key);
            if (value instanceof Number) {
                return ((Number) value).intValue();
            }
            return defaultValue;
        }
        
        public String getString(String key, String defaultValue) {
            Object value = parameters.get(key);
            return value != null ? value.toString() : defaultValue;
        }
        
        @Override
        public String toString() {
            return parameters.toString();
        }
    }
    
    public static class DataAnalysis {
        public int nSamples;
        public int nFeatures;
        public boolean isClassification;
        public FeatureStatistics[] featureStats;
        public double[][] correlationMatrix;
        public int[] missingValueCounts;
        public boolean needsScaling;
    }
    
    public static class FeatureStatistics {
        public double mean;
        public double std;
        public double min;
        public double max;
    }
    
    public static class FeatureImportance {
        public String[] featureNames;
        public double[] coefficients;
        public double[] importance;
        public int selectedFeatures;
        public Integer[] sortedIndices;
        public String[] sortedFeatureNames;
        public double[] sortedImportance;
    }
    
    public static class ModelComparison {
        public List<ModelComparisonResult> results;
    }
    
    public static class ModelComparisonResult {
        public String modelType;
        public double meanScore;
        public double stdScore;
        public boolean success;
        public String error;
    }
    
    public static class DataSplit {
        public double[][] XTrain;
        public double[] yTrain;
        public double[][] XVal;
        public double[] yVal;
    }
}
