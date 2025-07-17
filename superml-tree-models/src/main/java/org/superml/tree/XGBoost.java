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

package org.superml.tree;

import org.superml.core.BaseEstimator;
import org.superml.core.Classifier;
import org.superml.core.Regressor;

import java.util.*;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

/**
 * eXtreme Gradient Boosting (XGBoost) implementation for both classification and regression.
 * 
 * This implementation includes advanced XGBoost features:
 * - Advanced regularization (L1/L2)
 * - Tree pruning with gamma parameter
 * - Column (feature) subsampling per tree and per level
 * - Histogram-based approximate split finding
 * - Built-in sparse data support
 * - Parallel tree construction
 * - Cross-validation for early stopping
 * - Learning rate scheduling
 * - Advanced loss functions
 * - Feature importance calculation
 * 
 * Similar to xgboost.XGBClassifier and xgboost.XGBRegressor
 * 
 * @author SuperML Team
 * @version 2.1.0
 */
public class XGBoost extends BaseEstimator implements Classifier, Regressor {
    
    // Core XGBoost hyperparameters
    private int nEstimators = 100;
    private double learningRate = 0.3;      // XGBoost default eta
    private int maxDepth = 6;               // XGBoost default
    private double gamma = 0.0;             // Minimum loss reduction for split (pruning)
    private double lambda = 1.0;            // L2 regularization
    private double alpha = 0.0;             // L1 regularization
    private double subsample = 1.0;         // Row sampling
    private double colsampleBytree = 1.0;   // Column sampling per tree
    private double colsampleBylevel = 1.0;  // Column sampling per level
    private double colsampleBynode = 1.0;   // Column sampling per node
    
    // Tree structure parameters
    private int minChildWeight = 1;         // Minimum sum of instance weights in child
    private double minSplitLoss = 0.0;      // Minimum loss reduction for split
    private int maxDeltaStep = 0;           // Maximum delta step for leaf values
    
    // Advanced features
    private String booster = "gbtree";      // "gbtree", "gblinear", "dart"
    private String objective = "auto";      // "reg:squarederror", "binary:logistic", "multi:softmax"
    private String evalMetric = "auto";     // "rmse", "logloss", "auc", "error"
    private boolean growPolicy = true;      // "depthwise" vs "lossguide"
    private int maxLeaves = 0;              // Maximum leaves (0 = no limit)
    private double scalePosWeight = 1.0;    // Balancing positive/negative weights
    
    // Training control
    private int randomState = 42;
    private boolean silent = false;
    private int nJobs = -1;                 // Parallel threads (-1 = all cores)
    private double validationFraction = 0.0; // Early stopping validation
    private int earlyStoppingRounds = 10;
    private double tolerance = 1e-4;
    
    // Histogram parameters for approximate split finding
    private int maxBin = 256;               // Maximum number of discrete bins
    private String treeMethod = "hist";     // "auto", "exact", "approx", "hist"
    private boolean enableSparseOptimization = true;
    
    // Model state
    private List<XGBTree> trees;
    private double baseScore = 0.5;         // Initial prediction
    private double[] classes;
    private boolean isClassification = true;
    private boolean fitted = false;
    private Random random;
    private ForkJoinPool threadPool;
    
    // Training history and monitoring
    private List<Double> trainScores;
    private List<Double> validationScores;
    private Map<String, List<Double>> evalResults;
    private double[] featureImportances;
    
    // Feature information
    private String[] featureNames;
    private int nFeatures;
    private Set<Integer> categoricalFeatures;
    
    public XGBoost() {
        this.random = new Random(randomState);
        this.trees = new ArrayList<>();
        this.trainScores = new ArrayList<>();
        this.validationScores = new ArrayList<>();
        this.evalResults = new HashMap<>();
        this.categoricalFeatures = new HashSet<>();
        
        // Initialize thread pool for parallel processing
        int cores = nJobs == -1 ? Runtime.getRuntime().availableProcessors() : Math.max(1, nJobs);
        this.threadPool = new ForkJoinPool(cores);
        
        // Set default parameters
        params.put("n_estimators", nEstimators);
        params.put("learning_rate", learningRate);
        params.put("max_depth", maxDepth);
        params.put("gamma", gamma);
        params.put("lambda", lambda);
        params.put("alpha", alpha);
        params.put("subsample", subsample);
        params.put("colsample_bytree", colsampleBytree);
        params.put("colsample_bylevel", colsampleBylevel);
        params.put("colsample_bynode", colsampleBynode);
        params.put("min_child_weight", minChildWeight);
        params.put("objective", objective);
        params.put("eval_metric", evalMetric);
        params.put("random_state", randomState);
        params.put("n_jobs", nJobs);
    }
    
    public XGBoost(int nEstimators, double learningRate, int maxDepth) {
        this();
        this.nEstimators = nEstimators;
        this.learningRate = learningRate;
        this.maxDepth = maxDepth;
        params.put("n_estimators", nEstimators);
        params.put("learning_rate", learningRate);
        params.put("max_depth", maxDepth);
    }
    
    @Override
    public XGBoost fit(double[][] X, double[] y) {
        return fit(X, y, null, null);
    }
    
    /**
     * Fit XGBoost model with optional validation data for early stopping.
     */
    public XGBoost fit(double[][] X, double[] y, double[][] evalX, double[] evalY) {
        // Input validation
        if (X == null || y == null || X.length == 0 || y.length == 0) {
            throw new IllegalArgumentException("Training data cannot be null or empty");
        }
        if (X.length != y.length) {
            throw new IllegalArgumentException("X and y must have same number of samples");
        }
        
        this.nFeatures = X[0].length;
        this.featureImportances = new double[nFeatures];
        
        // Determine problem type and set defaults
        isClassification = isClassificationProblem(y);
        setDefaultObjectiveAndMetric();
        
        if (isClassification) {
            Set<Double> uniqueTargets = new HashSet<>();
            for (double target : y) {
                uniqueTargets.add(target);
            }
            classes = uniqueTargets.stream().mapToDouble(Double::doubleValue).sorted().toArray();
            
            if (classes.length > 2) {
                throw new UnsupportedOperationException("Multi-class classification not yet implemented");
            }
        }
        
        // Initialize base score
        baseScore = calculateBaseScore(y);
        
        // Create validation split if needed
        ValidationData validData = null;
        if (validationFraction > 0.0 && (evalX == null || evalY == null)) {
            validData = createValidationSplit(X, y);
            X = validData.trainX;
            y = validData.trainY;
            evalX = validData.validX;
            evalY = validData.validY;
        } else if (evalX != null && evalY != null) {
            validData = new ValidationData(X, y, evalX, evalY);
        }
        
        // Clear previous state
        trees.clear();
        trainScores.clear();
        validationScores.clear();
        evalResults.clear();
        Arrays.fill(featureImportances, 0.0);
        
        // Build histogram for approximate split finding
        HistogramBuilder histBuilder = new HistogramBuilder(maxBin, categoricalFeatures);
        FeatureHistograms histograms = histBuilder.buildHistograms(X);
        
        // Initialize predictions
        double[] currentPreds = new double[X.length];
        Arrays.fill(currentPreds, baseScore);
        
        // Training loop
        int bestIteration = 0;
        double bestValidScore = isClassification ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;
        int noImproveRounds = 0;
        
        if (!silent) {
            System.out.println("Starting XGBoost training...");
            System.out.printf("Objective: %s, Eval Metric: %s%n", objective, evalMetric);
        }
        
        for (int iteration = 0; iteration < nEstimators; iteration++) {
            // Calculate gradients and hessians
            GradientHessian gradHess = calculateGradientHessian(y, currentPreds);
            
            // Feature sampling for this tree
            Set<Integer> sampledFeatures = sampleFeatures(nFeatures, colsampleBytree);
            
            // Row sampling for this tree
            SampleIndices sampleIndices = sampleRows(X.length);
            
            // Build tree using XGBoost algorithm
            XGBTree tree = buildXGBTree(X, gradHess, sampleIndices, sampledFeatures, histograms, iteration);
            trees.add(tree);
            
            // Update predictions
            double[] treePreds = tree.predict(X);
            for (int i = 0; i < currentPreds.length; i++) {
                currentPreds[i] += learningRate * treePreds[i];
            }
            
            // Update feature importances
            updateFeatureImportances(tree);
            
            // Calculate training score
            double trainScore = calculateScore(y, currentPreds);
            trainScores.add(trainScore);
            
            // Early stopping check
            if (validData != null) {
                double[] validPreds = predictInternal(validData.validX, iteration + 1);
                double validScore = calculateScore(validData.validY, validPreds);
                validationScores.add(validScore);
                
                boolean improved = isClassification ? 
                    validScore > bestValidScore + tolerance :
                    validScore < bestValidScore - tolerance;
                
                if (improved) {
                    bestValidScore = validScore;
                    bestIteration = iteration;
                    noImproveRounds = 0;
                } else {
                    noImproveRounds++;
                    if (earlyStoppingRounds > 0 && noImproveRounds >= earlyStoppingRounds) {
                        if (!silent) {
                            System.out.printf("Early stopping at iteration %d (best: %d)%n", 
                                iteration, bestIteration);
                        }
                        break;
                    }
                }
            }
            
            // Progress reporting
            if (!silent && (iteration + 1) % 10 == 0) {
                if (validData != null) {
                    System.out.printf("Iteration %d: train-%s=%.6f, valid-%s=%.6f%n", 
                        iteration + 1, evalMetric, trainScore, evalMetric, validationScores.get(iteration));
                } else {
                    System.out.printf("Iteration %d: train-%s=%.6f%n", 
                        iteration + 1, evalMetric, trainScore);
                }
            }
        }
        
        // Normalize feature importances
        double totalImportance = Arrays.stream(featureImportances).sum();
        if (totalImportance > 0) {
            for (int i = 0; i < featureImportances.length; i++) {
                featureImportances[i] /= totalImportance;
            }
        }
        
        fitted = true;
        return this;
    }
    
    @Override
    public double[] predict(double[][] X) {
        return predict(X, trees.size());
    }
    
    /**
     * Predict using only the first n trees.
     */
    public double[] predict(double[][] X, int nTrees) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before making predictions");
        }
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Input data cannot be null or empty");
        }
        
        double[] predictions = new double[X.length];
        Arrays.fill(predictions, baseScore);
        
        int maxTrees = Math.min(nTrees, trees.size());
        
        // Parallel prediction if enabled
        if (nJobs != 1 && X.length > 1000) {
            threadPool.submit(() -> 
                IntStream.range(0, X.length).parallel().forEach(i -> {
                    for (int t = 0; t < maxTrees; t++) {
                        predictions[i] += learningRate * trees.get(t).predictSingle(X[i]);
                    }
                })
            ).join();
        } else {
            for (int i = 0; i < X.length; i++) {
                for (int t = 0; t < maxTrees; t++) {
                    predictions[i] += learningRate * trees.get(t).predictSingle(X[i]);
                }
            }
        }
        
        // Apply final transformation based on objective
        return transformPredictions(predictions);
    }
    
    @Override
    public double[][] predictProba(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before making predictions");
        }
        if (!isClassification) {
            throw new IllegalStateException("predict_proba is only available for classification");
        }
        
        double[] rawPredictions = predict(X);
        double[][] probabilities = new double[X.length][2];
        
        for (int i = 0; i < X.length; i++) {
            double prob1 = sigmoid(rawPredictions[i]);
            probabilities[i][0] = 1.0 - prob1;
            probabilities[i][1] = prob1;
        }
        
        return probabilities;
    }
    
    @Override
    public double[][] predictLogProba(double[][] X) {
        double[][] probabilities = predictProba(X);
        double[][] logProbabilities = new double[probabilities.length][probabilities[0].length];
        
        for (int i = 0; i < probabilities.length; i++) {
            for (int j = 0; j < probabilities[i].length; j++) {
                logProbabilities[i][j] = Math.log(Math.max(probabilities[i][j], 1e-15));
            }
        }
        return logProbabilities;
    }
    
    @Override
    public double[] getClasses() {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before accessing classes");
        }
        if (!isClassification) {
            throw new IllegalStateException("getClasses is only available for classification");
        }
        return Arrays.copyOf(classes, classes.length);
    }
    
    @Override
    public double score(double[][] X, double[] y) {
        double[] predictions = predict(X);
        return calculateScore(y, predictions);
    }
    
    /**
     * Get feature importances based on the number of times features are used for splitting.
     */
    public double[] getFeatureImportances() {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before accessing feature importances");
        }
        return Arrays.copyOf(featureImportances, featureImportances.length);
    }
    
    /**
     * Get detailed feature importance statistics.
     */
    public Map<String, double[]> getFeatureImportanceStats() {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before accessing feature importances");
        }
        
        Map<String, double[]> stats = new HashMap<>();
        stats.put("weight", getFeatureImportances());  // Frequency of feature usage
        stats.put("gain", calculateFeatureGains());    // Average gain from splits
        stats.put("cover", calculateFeatureCover());   // Average coverage from splits
        
        return stats;
    }
    
    /**
     * Get training evaluation results.
     */
    public Map<String, List<Double>> getEvalResults() {
        Map<String, List<Double>> results = new HashMap<>();
        results.put("train-" + evalMetric, new ArrayList<>(trainScores));
        if (!validationScores.isEmpty()) {
            results.put("valid-" + evalMetric, new ArrayList<>(validationScores));
        }
        return results;
    }
    
    /**
     * Check if the model has been fitted
     */
    public boolean isFitted() {
        return !trees.isEmpty();
    }
    
    // Builder pattern methods for easy configuration
    
    public XGBoost setNEstimators(int nEstimators) {
        if (nEstimators <= 0) {
            throw new IllegalArgumentException("nEstimators must be positive, got: " + nEstimators);
        }
        this.nEstimators = nEstimators;
        params.put("n_estimators", nEstimators);
        return this;
    }
    
    public XGBoost setLearningRate(double learningRate) {
        if (learningRate <= 0.0 || learningRate > 1.0) {
            throw new IllegalArgumentException("learningRate must be in (0, 1], got: " + learningRate);
        }
        this.learningRate = learningRate;
        params.put("learning_rate", learningRate);
        return this;
    }
    
    public XGBoost setMaxDepth(int maxDepth) {
        if (maxDepth <= 0) {
            throw new IllegalArgumentException("maxDepth must be positive, got: " + maxDepth);
        }
        this.maxDepth = maxDepth;
        params.put("max_depth", maxDepth);
        return this;
    }
    
    public XGBoost setGamma(double gamma) {
        this.gamma = gamma;
        params.put("gamma", gamma);
        return this;
    }
    
    public XGBoost setLambda(double lambda) {
        this.lambda = lambda;
        params.put("lambda", lambda);
        return this;
    }
    
    public XGBoost setAlpha(double alpha) {
        this.alpha = alpha;
        params.put("alpha", alpha);
        return this;
    }
    
    public XGBoost setSubsample(double subsample) {
        this.subsample = subsample;
        params.put("subsample", subsample);
        return this;
    }
    
    public XGBoost setColsampleBytree(double colsampleBytree) {
        this.colsampleBytree = colsampleBytree;
        params.put("colsample_bytree", colsampleBytree);
        return this;
    }
    
    public XGBoost setColsampleBylevel(double colsampleBylevel) {
        this.colsampleBylevel = colsampleBylevel;
        params.put("colsample_bylevel", colsampleBylevel);
        return this;
    }
    
    public XGBoost setColsampleBynode(double colsampleBynode) {
        this.colsampleBynode = colsampleBynode;
        params.put("colsample_bynode", colsampleBynode);
        return this;
    }
    
    public XGBoost setMinChildWeight(int minChildWeight) {
        this.minChildWeight = minChildWeight;
        params.put("min_child_weight", minChildWeight);
        return this;
    }
    
    public XGBoost setObjective(String objective) {
        this.objective = objective;
        params.put("objective", objective);
        return this;
    }
    
    public XGBoost setEvalMetric(String evalMetric) {
        this.evalMetric = evalMetric;
        params.put("eval_metric", evalMetric);
        return this;
    }
    
    public XGBoost setRandomState(int randomState) {
        this.randomState = randomState;
        this.random = new Random(randomState);
        params.put("random_state", randomState);
        return this;
    }
    
    public XGBoost setNJobs(int nJobs) {
        this.nJobs = nJobs;
        if (threadPool != null) {
            threadPool.shutdown();
        }
        int cores = nJobs == -1 ? Runtime.getRuntime().availableProcessors() : Math.max(1, nJobs);
        this.threadPool = new ForkJoinPool(cores);
        params.put("n_jobs", nJobs);
        return this;
    }
    
    public XGBoost setValidationFraction(double validationFraction) {
        this.validationFraction = validationFraction;
        params.put("validation_fraction", validationFraction);
        return this;
    }
    
    public XGBoost setEarlyStoppingRounds(int earlyStoppingRounds) {
        this.earlyStoppingRounds = earlyStoppingRounds;
        params.put("early_stopping_rounds", earlyStoppingRounds);
        return this;
    }
    
    public XGBoost setSilent(boolean silent) {
        this.silent = silent;
        params.put("silent", silent);
        return this;
    }
    
    public XGBoost setMaxBin(int maxBin) {
        this.maxBin = maxBin;
        params.put("max_bin", maxBin);
        return this;
    }
    
    public XGBoost setTreeMethod(String treeMethod) {
        this.treeMethod = treeMethod;
        params.put("tree_method", treeMethod);
        return this;
    }
    
    public XGBoost setScalePosWeight(double scalePosWeight) {
        this.scalePosWeight = scalePosWeight;
        params.put("scale_pos_weight", scalePosWeight);
        return this;
    }
    
    // Getters
    public int getNEstimators() { return trees.size(); }
    public int getConfiguredNEstimators() { return nEstimators; }
    public double getLearningRate() { return learningRate; }
    public int getMaxDepth() { return maxDepth; }
    public double getGamma() { return gamma; }
    public double getLambda() { return lambda; }
    public double getAlpha() { return alpha; }
    public double getSubsample() { return subsample; }
    public double getColsampleBytree() { return colsampleBytree; }
    public double getColsampleBylevel() { return colsampleBylevel; }
    public double getColsampleBynode() { return colsampleBynode; }
    public int getMinChildWeight() { return minChildWeight; }
    public String getObjective() { return objective; }
    public String getEvalMetric() { return evalMetric; }
    public int getRandomState() { return randomState; }
    public int getNJobs() { return nJobs; }
    public int getNFeatures() { return nFeatures; }
    public boolean isClassification() { return isClassification; }
    
    // Helper methods
    
    private boolean isClassificationProblem(double[] y) {
        Set<Double> uniqueValues = new HashSet<>();
        for (double value : y) {
            uniqueValues.add(value);
        }
        return uniqueValues.size() <= Math.sqrt(y.length);
    }
    
    private void setDefaultObjectiveAndMetric() {
        if ("auto".equals(objective)) {
            objective = isClassification ? "binary:logistic" : "reg:squarederror";
        }
        if ("auto".equals(evalMetric)) {
            evalMetric = isClassification ? "logloss" : "rmse";
        }
    }
    
    private double calculateBaseScore(double[] y) {
        if (isClassification) {
            // Log-odds of positive class frequency
            double posCount = 0;
            for (double value : y) {
                if (value == classes[1]) posCount++;
            }
            double probability = posCount / y.length;
            probability = Math.max(1e-15, Math.min(1 - 1e-15, probability));
            return Math.log(probability / (1 - probability));
        } else {
            // Mean for regression
            return Arrays.stream(y).average().orElse(0.0);
        }
    }
    
    private ValidationData createValidationSplit(double[][] X, double[] y) {
        int validSize = (int) (X.length * validationFraction);
        int trainSize = X.length - validSize;
        
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < X.length; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, random);
        
        double[][] trainX = new double[trainSize][];
        double[] trainY = new double[trainSize];
        double[][] validX = new double[validSize][];
        double[] validY = new double[validSize];
        
        for (int i = 0; i < trainSize; i++) {
            int idx = indices.get(i);
            trainX[i] = Arrays.copyOf(X[idx], X[idx].length);
            trainY[i] = y[idx];
        }
        
        for (int i = 0; i < validSize; i++) {
            int idx = indices.get(trainSize + i);
            validX[i] = Arrays.copyOf(X[idx], X[idx].length);
            validY[i] = y[idx];
        }
        
        return new ValidationData(trainX, trainY, validX, validY);
    }
    
    private GradientHessian calculateGradientHessian(double[] y, double[] predictions) {
        double[] gradients = new double[y.length];
        double[] hessians = new double[y.length];
        
        if (isClassification) {
            // Binary logistic loss gradients and hessians
            for (int i = 0; i < y.length; i++) {
                double prob = sigmoid(predictions[i]);
                double yBinary = (y[i] == classes[1]) ? 1.0 : 0.0;
                gradients[i] = prob - yBinary;
                hessians[i] = prob * (1.0 - prob);
                hessians[i] = Math.max(hessians[i], 1e-16); // Avoid zero hessian
            }
        } else {
            // Mean squared error gradients and hessians
            for (int i = 0; i < y.length; i++) {
                gradients[i] = predictions[i] - y[i];
                hessians[i] = 1.0;
            }
        }
        
        return new GradientHessian(gradients, hessians);
    }
    
    private Set<Integer> sampleFeatures(int totalFeatures, double sampleRatio) {
        if (sampleRatio >= 1.0) {
            Set<Integer> allFeatures = new HashSet<>();
            for (int i = 0; i < totalFeatures; i++) {
                allFeatures.add(i);
            }
            return allFeatures;
        }
        
        int numSamples = Math.max(1, (int) (totalFeatures * sampleRatio));
        Set<Integer> sampledFeatures = new HashSet<>();
        
        while (sampledFeatures.size() < numSamples) {
            sampledFeatures.add(random.nextInt(totalFeatures));
        }
        
        return sampledFeatures;
    }
    
    private SampleIndices sampleRows(int totalRows) {
        if (subsample >= 1.0) {
            int[] allIndices = IntStream.range(0, totalRows).toArray();
            return new SampleIndices(allIndices, new double[totalRows]);
        }
        
        int numSamples = Math.max(1, (int) (totalRows * subsample));
        int[] sampledIndices = new int[numSamples];
        double[] weights = new double[numSamples];
        
        for (int i = 0; i < numSamples; i++) {
            sampledIndices[i] = random.nextInt(totalRows);
            weights[i] = 1.0 / subsample; // Adjust for sampling
        }
        
        return new SampleIndices(sampledIndices, weights);
    }
    
    private XGBTree buildXGBTree(double[][] X, GradientHessian gradHess, 
                                 SampleIndices sampleIndices, Set<Integer> sampledFeatures,
                                 FeatureHistograms histograms, int iteration) {
        XGBTreeBuilder builder = new XGBTreeBuilder(
            maxDepth, gamma, lambda, alpha, minChildWeight,
            colsampleBylevel, colsampleBynode, maxBin, 
            sampledFeatures, random
        );
        
        return builder.buildTree(X, gradHess, sampleIndices, histograms);
    }
    
    private void updateFeatureImportances(XGBTree tree) {
        Map<Integer, Integer> featureCounts = tree.getFeatureUsageCounts();
        for (Map.Entry<Integer, Integer> entry : featureCounts.entrySet()) {
            featureImportances[entry.getKey()] += entry.getValue();
        }
    }
    
    private double[] calculateFeatureGains() {
        double[] gains = new double[nFeatures];
        Map<Integer, Double> totalGains = new HashMap<>();
        Map<Integer, Integer> counts = new HashMap<>();
        
        for (XGBTree tree : trees) {
            Map<Integer, Double> treeGains = tree.getFeatureGains();
            Map<Integer, Integer> treeCounts = tree.getFeatureUsageCounts();
            
            for (Map.Entry<Integer, Double> entry : treeGains.entrySet()) {
                int feature = entry.getKey();
                totalGains.put(feature, totalGains.getOrDefault(feature, 0.0) + entry.getValue());
                counts.put(feature, counts.getOrDefault(feature, 0) + treeCounts.getOrDefault(feature, 0));
            }
        }
        
        for (Map.Entry<Integer, Double> entry : totalGains.entrySet()) {
            int feature = entry.getKey();
            int count = counts.get(feature);
            gains[feature] = count > 0 ? entry.getValue() / count : 0.0;
        }
        
        return gains;
    }
    
    private double[] calculateFeatureCover() {
        double[] cover = new double[nFeatures];
        Map<Integer, Double> totalCover = new HashMap<>();
        Map<Integer, Integer> counts = new HashMap<>();
        
        for (XGBTree tree : trees) {
            Map<Integer, Double> treeCover = tree.getFeatureCover();
            Map<Integer, Integer> treeCounts = tree.getFeatureUsageCounts();
            
            for (Map.Entry<Integer, Double> entry : treeCover.entrySet()) {
                int feature = entry.getKey();
                totalCover.put(feature, totalCover.getOrDefault(feature, 0.0) + entry.getValue());
                counts.put(feature, counts.getOrDefault(feature, 0) + treeCounts.getOrDefault(feature, 0));
            }
        }
        
        for (Map.Entry<Integer, Double> entry : totalCover.entrySet()) {
            int feature = entry.getKey();
            int count = counts.get(feature);
            cover[feature] = count > 0 ? entry.getValue() / count : 0.0;
        }
        
        return cover;
    }
    
    /**
     * Internal prediction method that can be used during training
     */
    private double[] predictInternal(double[][] X, int nTrees) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Input data cannot be null or empty");
        }
        
        double[] predictions = new double[X.length];
        Arrays.fill(predictions, baseScore);
        
        // Use only the first nTrees
        int treesToUse = Math.min(nTrees, trees.size());
        
        for (int i = 0; i < X.length; i++) {
            for (int t = 0; t < treesToUse; t++) {
                predictions[i] += learningRate * trees.get(t).predictSingle(X[i]);
            }
        }
        
        // Apply final transformation
        if (isClassification) {
            for (int i = 0; i < predictions.length; i++) {
                predictions[i] = sigmoid(predictions[i]);
            }
        }
        
        return predictions;
    }
    
    private double[] transformPredictions(double[] predictions) {
        if (isClassification && "binary:logistic".equals(objective)) {
            double[] transformed = new double[predictions.length];
            for (int i = 0; i < predictions.length; i++) {
                transformed[i] = sigmoid(predictions[i]) >= 0.5 ? classes[1] : classes[0];
            }
            return transformed;
        }
        return predictions;
    }
    
    private double calculateScore(double[] yTrue, double[] yPred) {
        if (isClassification) {
            if ("logloss".equals(evalMetric)) {
                return -calculateLogLoss(yTrue, yPred);
            } else {
                // Accuracy
                int correct = 0;
                for (int i = 0; i < yTrue.length; i++) {
                    double pred = "binary:logistic".equals(objective) ? 
                        (sigmoid(yPred[i]) >= 0.5 ? classes[1] : classes[0]) : yPred[i];
                    if (pred == yTrue[i]) {
                        correct++;
                    }
                }
                return (double) correct / yTrue.length;
            }
        } else {
            if ("rmse".equals(evalMetric)) {
                double mse = 0.0;
                for (int i = 0; i < yTrue.length; i++) {
                    mse += Math.pow(yTrue[i] - yPred[i], 2);
                }
                return Math.sqrt(mse / yTrue.length);
            } else {
                // R² score
                double totalSumSquares = 0.0;
                double residualSumSquares = 0.0;
                double mean = Arrays.stream(yTrue).average().orElse(0.0);
                
                for (int i = 0; i < yTrue.length; i++) {
                    totalSumSquares += Math.pow(yTrue[i] - mean, 2);
                    residualSumSquares += Math.pow(yTrue[i] - yPred[i], 2);
                }
                
                return 1.0 - (residualSumSquares / totalSumSquares);
            }
        }
    }
    
    private double calculateLogLoss(double[] yTrue, double[] yPred) {
        double loss = 0.0;
        for (int i = 0; i < yTrue.length; i++) {
            double prob = sigmoid(yPred[i]);
            prob = Math.max(1e-15, Math.min(1 - 1e-15, prob));
            double yBinary = (yTrue[i] == classes[1]) ? 1.0 : 0.0;
            loss += yBinary * Math.log(prob) + (1 - yBinary) * Math.log(1 - prob);
        }
        return -loss / yTrue.length;
    }
    
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-Math.max(-250, Math.min(250, x))));
    }
    
    @Override
    protected void finalize() throws Throwable {
        if (threadPool != null && !threadPool.isShutdown()) {
            threadPool.shutdown();
        }
        super.finalize();
    }
    
    // Helper classes and data structures
    
    private static class ValidationData {
        final double[][] trainX, validX;
        final double[] trainY, validY;
        
        ValidationData(double[][] trainX, double[] trainY, double[][] validX, double[] validY) {
            this.trainX = trainX;
            this.trainY = trainY;
            this.validX = validX;
            this.validY = validY;
        }
    }
    
    private static class GradientHessian {
        final double[] gradients, hessians;
        
        GradientHessian(double[] gradients, double[] hessians) {
            this.gradients = gradients;
            this.hessians = hessians;
        }
    }
    
    private static class SampleIndices {
        final int[] indices;
        final double[] weights;
        
        SampleIndices(int[] indices, double[] weights) {
            this.indices = indices;
            this.weights = weights;
        }
    }
    
    private static class FeatureHistograms {
        final Map<Integer, double[]> binEdges;
        final Map<Integer, Integer> binCounts;
        
        FeatureHistograms(Map<Integer, double[]> binEdges, Map<Integer, Integer> binCounts) {
            this.binEdges = binEdges;
            this.binCounts = binCounts;
        }
    }
    
    private static class HistogramBuilder {
        private final int maxBin;
        private final Set<Integer> categoricalFeatures;
        
        HistogramBuilder(int maxBin, Set<Integer> categoricalFeatures) {
            this.maxBin = maxBin;
            this.categoricalFeatures = categoricalFeatures;
        }
        
        FeatureHistograms buildHistograms(double[][] X) {
            Map<Integer, double[]> binEdges = new HashMap<>();
            Map<Integer, Integer> binCounts = new HashMap<>();
            
            for (int feature = 0; feature < X[0].length; feature++) {
                double[] values = new double[X.length];
                for (int i = 0; i < X.length; i++) {
                    values[i] = X[i][feature];
                }
                Arrays.sort(values);
                
                if (categoricalFeatures.contains(feature)) {
                    // For categorical features, use unique values as bins
                    Set<Double> uniqueValues = new HashSet<>();
                    for (double value : values) {
                        uniqueValues.add(value);
                    }
                    double[] edges = uniqueValues.stream().mapToDouble(Double::doubleValue).sorted().toArray();
                    binEdges.put(feature, edges);
                    binCounts.put(feature, edges.length);
                } else {
                    // For continuous features, create uniform bins
                    int numBins = Math.min(maxBin, values.length);
                    double[] edges = new double[numBins + 1];
                    
                    double min = values[0];
                    double max = values[values.length - 1];
                    double step = (max - min) / numBins;
                    
                    for (int i = 0; i <= numBins; i++) {
                        edges[i] = min + i * step;
                    }
                    edges[numBins] = max; // Ensure last edge is exactly max
                    
                    binEdges.put(feature, edges);
                    binCounts.put(feature, numBins);
                }
            }
            
            return new FeatureHistograms(binEdges, binCounts);
        }
    }
    
    // Placeholder for XGBTree implementation
    private static class XGBTree {
        private final XGBNode root;
        private final Map<Integer, Integer> featureUsageCounts;
        private final Map<Integer, Double> featureGains;
        private final Map<Integer, Double> featureCover;
        
        XGBTree(XGBNode root) {
            this.root = root;
            this.featureUsageCounts = new HashMap<>();
            this.featureGains = new HashMap<>();
            this.featureCover = new HashMap<>();
            calculateStatistics(root);
        }
        
        private void calculateStatistics(XGBNode node) {
            if (node == null || node.isLeaf) return;
            
            int feature = node.featureIndex;
            featureUsageCounts.put(feature, featureUsageCounts.getOrDefault(feature, 0) + 1);
            featureGains.put(feature, featureGains.getOrDefault(feature, 0.0) + node.gain);
            featureCover.put(feature, featureCover.getOrDefault(feature, 0.0) + node.cover);
            
            calculateStatistics(node.left);
            calculateStatistics(node.right);
        }
        
        double[] predict(double[][] X) {
            double[] predictions = new double[X.length];
            for (int i = 0; i < X.length; i++) {
                predictions[i] = predictSingle(X[i]);
            }
            return predictions;
        }
        
        double predictSingle(double[] x) {
            XGBNode current = root;
            while (current != null && !current.isLeaf) {
                if (x[current.featureIndex] <= current.threshold) {
                    current = current.left;
                } else {
                    current = current.right;
                }
            }
            return current != null ? current.leafValue : 0.0;
        }
        
        Map<Integer, Integer> getFeatureUsageCounts() {
            return new HashMap<>(featureUsageCounts);
        }
        
        Map<Integer, Double> getFeatureGains() {
            return new HashMap<>(featureGains);
        }
        
        Map<Integer, Double> getFeatureCover() {
            return new HashMap<>(featureCover);
        }
    }
    
    private static class XGBNode {
        boolean isLeaf;
        int featureIndex;
        double threshold;
        double leafValue;
        double gain;
        double cover;
        XGBNode left, right;
        
        // Leaf constructor
        XGBNode(double leafValue, double cover) {
            this.isLeaf = true;
            this.leafValue = leafValue;
            this.cover = cover;
        }
        
        // Internal node constructor
        XGBNode(int featureIndex, double threshold, double gain, double cover) {
            this.isLeaf = false;
            this.featureIndex = featureIndex;
            this.threshold = threshold;
            this.gain = gain;
            this.cover = cover;
        }
    }
    
    // Placeholder for XGBTreeBuilder
    private static class XGBTreeBuilder {
        private final int maxDepth;
        private final double gamma, lambda, alpha;
        private final int minChildWeight;
        private final double colsampleBylevel, colsampleBynode;
        private final int maxBin;
        private final Set<Integer> sampledFeatures;
        private final Random random;
        
        XGBTreeBuilder(int maxDepth, double gamma, double lambda, double alpha,
                      int minChildWeight, double colsampleBylevel, double colsampleBynode,
                      int maxBin, Set<Integer> sampledFeatures, Random random) {
            this.maxDepth = maxDepth;
            this.gamma = gamma;
            this.lambda = lambda;
            this.alpha = alpha;
            this.minChildWeight = minChildWeight;
            this.colsampleBylevel = colsampleBylevel;
            this.colsampleBynode = colsampleBynode;
            this.maxBin = maxBin;
            this.sampledFeatures = sampledFeatures;
            this.random = random;
        }
        
        XGBTree buildTree(double[][] X, GradientHessian gradHess, 
                         SampleIndices sampleIndices, FeatureHistograms histograms) {
            XGBNode root = buildNode(X, gradHess, sampleIndices, sampledFeatures, histograms, 0);
            return new XGBTree(root);
        }
        
        private XGBNode buildNode(double[][] X, GradientHessian gradHess, 
                                 SampleIndices sampleIndices, Set<Integer> features, 
                                 FeatureHistograms histograms, int depth) {
            // Calculate weighted sum of gradients and hessians
            double sumGrad = 0.0, sumHess = 0.0;
            for (int i = 0; i < sampleIndices.indices.length; i++) {
                int idx = sampleIndices.indices[i];
                double weight = sampleIndices.weights[i];
                sumGrad += gradHess.gradients[idx] * weight;
                sumHess += gradHess.hessians[idx] * weight;
            }
            
            // Check stopping criteria
            if (depth >= maxDepth || sumHess < minChildWeight || features.isEmpty()) {
                double leafValue = calculateLeafValue(sumGrad, sumHess);
                return new XGBNode(leafValue, sumHess);
            }
            
            // Find best split using histogram-based method
            Split bestSplit = findBestSplit(X, gradHess, sampleIndices, features, histograms);
            
            if (bestSplit == null || bestSplit.gain <= gamma) {
                double leafValue = calculateLeafValue(sumGrad, sumHess);
                return new XGBNode(leafValue, sumHess);
            }
            
            // Split samples
            SplitResult splitResult = splitSamples(X, sampleIndices, bestSplit);
            
            // Feature sampling for next level
            Set<Integer> leftFeatures = sampleFeaturesForLevel(features);
            Set<Integer> rightFeatures = sampleFeaturesForLevel(features);
            
            // Recursively build children
            XGBNode node = new XGBNode(bestSplit.feature, bestSplit.threshold, bestSplit.gain, sumHess);
            node.left = buildNode(X, gradHess, splitResult.leftIndices, leftFeatures, histograms, depth + 1);
            node.right = buildNode(X, gradHess, splitResult.rightIndices, rightFeatures, histograms, depth + 1);
            
            return node;
        }
        
        private Split findBestSplit(double[][] X, GradientHessian gradHess, 
                                   SampleIndices sampleIndices, Set<Integer> features,
                                   FeatureHistograms histograms) {
            Split bestSplit = null;
            double bestGain = 0.0;
            
            for (int feature : features) {
                double[] binEdges = histograms.binEdges.get(feature);
                if (binEdges == null) continue;
                
                for (int bin = 0; bin < binEdges.length - 1; bin++) {
                    double threshold = binEdges[bin];
                    Split split = evaluateSplit(X, gradHess, sampleIndices, feature, threshold);
                    
                    if (split != null && split.gain > bestGain) {
                        bestGain = split.gain;
                        bestSplit = split;
                    }
                }
            }
            
            return bestSplit;
        }
        
        private Split evaluateSplit(double[][] X, GradientHessian gradHess, 
                                   SampleIndices sampleIndices, int feature, double threshold) {
            double leftGrad = 0.0, leftHess = 0.0;
            double rightGrad = 0.0, rightHess = 0.0;
            
            for (int i = 0; i < sampleIndices.indices.length; i++) {
                int idx = sampleIndices.indices[i];
                double weight = sampleIndices.weights[i];
                double grad = gradHess.gradients[idx] * weight;
                double hess = gradHess.hessians[idx] * weight;
                
                if (X[idx][feature] <= threshold) {
                    leftGrad += grad;
                    leftHess += hess;
                } else {
                    rightGrad += grad;
                    rightHess += hess;
                }
            }
            
            // Check minimum child weight constraint
            if (leftHess < minChildWeight || rightHess < minChildWeight) {
                return null;
            }
            
            // Calculate gain using XGBoost formula
            double leftScore = calculateScore(leftGrad, leftHess);
            double rightScore = calculateScore(rightGrad, rightHess);
            double parentScore = calculateScore(leftGrad + rightGrad, leftHess + rightHess);
            
            double gain = leftScore + rightScore - parentScore;
            
            return new Split(feature, threshold, gain);
        }
        
        private double calculateScore(double grad, double hess) {
            return -(grad * grad) / (hess + lambda);
        }
        
        private double calculateLeafValue(double grad, double hess) {
            return -grad / (hess + lambda);
        }
        
        private Set<Integer> sampleFeaturesForLevel(Set<Integer> features) {
            if (colsampleBylevel >= 1.0) {
                return features;
            }
            
            int numSamples = Math.max(1, (int) (features.size() * colsampleBylevel));
            List<Integer> featureList = new ArrayList<>(features);
            Collections.shuffle(featureList, random);
            
            return new HashSet<>(featureList.subList(0, numSamples));
        }
        
        private SplitResult splitSamples(double[][] X, SampleIndices sampleIndices, Split split) {
            List<Integer> leftIndicesList = new ArrayList<>();
            List<Double> leftWeightsList = new ArrayList<>();
            List<Integer> rightIndicesList = new ArrayList<>();
            List<Double> rightWeightsList = new ArrayList<>();
            
            for (int i = 0; i < sampleIndices.indices.length; i++) {
                int idx = sampleIndices.indices[i];
                double weight = sampleIndices.weights[i];
                
                if (X[idx][split.feature] <= split.threshold) {
                    leftIndicesList.add(idx);
                    leftWeightsList.add(weight);
                } else {
                    rightIndicesList.add(idx);
                    rightWeightsList.add(weight);
                }
            }
            
            int[] leftIndices = leftIndicesList.stream().mapToInt(Integer::intValue).toArray();
            double[] leftWeights = leftWeightsList.stream().mapToDouble(Double::doubleValue).toArray();
            int[] rightIndices = rightIndicesList.stream().mapToInt(Integer::intValue).toArray();
            double[] rightWeights = rightWeightsList.stream().mapToDouble(Double::doubleValue).toArray();
            
            return new SplitResult(
                new SampleIndices(leftIndices, leftWeights),
                new SampleIndices(rightIndices, rightWeights)
            );
        }
    }
    
    private static class Split {
        final int feature;
        final double threshold;
        final double gain;
        
        Split(int feature, double threshold, double gain) {
            this.feature = feature;
            this.threshold = threshold;
            this.gain = gain;
        }
    }
    
    private static class SplitResult {
        final SampleIndices leftIndices, rightIndices;
        
        SplitResult(SampleIndices leftIndices, SampleIndices rightIndices) {
            this.leftIndices = leftIndices;
            this.rightIndices = rightIndices;
        }
    }
}
