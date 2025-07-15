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

/**
 * Gradient Boosting implementation for both classification and regression.
 * Similar to sklearn.ensemble.GradientBoostingClassifier and GradientBoostingRegressor
 */
public class GradientBoosting extends BaseEstimator implements Classifier, Regressor {
    
    // Hyperparameters
    private int nEstimators = 100;
    private double learningRate = 0.1;
    private int maxDepth = 3;
    private String criterion = "mse"; // "mse" for regression, "friedman_mse" for classification
    private int minSamplesSplit = 2;
    private int minSamplesLeaf = 1;
    private double minImpurityDecrease = 0.0;
    private int maxFeatures = -1; // -1 means use all features
    private double subsample = 1.0; // Fraction of samples for fitting individual base learners
    private int randomState = 42;
    private double validationFraction = 0.1; // For early stopping
    private int nIterNoChange = -1; // Early stopping rounds (-1 to disable)
    private double tol = 1e-4; // Tolerance for early stopping
    
    // Model components
    private List<DecisionTree> trees;
    private double initialPrediction;
    private double[] classes;
    private boolean isClassification = true;
    private boolean fitted = false;
    private Random random;
    
    // Training history
    private List<Double> trainScores;
    private List<Double> validationScores;
    
    public GradientBoosting() {
        this.random = new Random(randomState);
        this.trees = new ArrayList<>();
        this.trainScores = new ArrayList<>();
        this.validationScores = new ArrayList<>();
        
        params.put("n_estimators", nEstimators);
        params.put("learning_rate", learningRate);
        params.put("max_depth", maxDepth);
        params.put("criterion", criterion);
        params.put("min_samples_split", minSamplesSplit);
        params.put("min_samples_leaf", minSamplesLeaf);
        params.put("min_impurity_decrease", minImpurityDecrease);
        params.put("max_features", maxFeatures);
        params.put("subsample", subsample);
        params.put("random_state", randomState);
        params.put("validation_fraction", validationFraction);
        params.put("n_iter_no_change", nIterNoChange);
        params.put("tol", tol);
    }
    
    public GradientBoosting(int nEstimators, double learningRate, int maxDepth) {
        this();
        this.nEstimators = nEstimators;
        this.learningRate = learningRate;
        this.maxDepth = maxDepth;
        params.put("n_estimators", nEstimators);
        params.put("learning_rate", learningRate);
        params.put("max_depth", maxDepth);
    }
    
    @Override
    public GradientBoosting fit(double[][] X, double[] y) {
        // Determine if this is classification or regression
        isClassification = isClassificationProblem(y);
        
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
        
        // Set max_features if not specified
        if (maxFeatures == -1) {
            maxFeatures = X[0].length;
        }
        
        // Clear previous state
        trees.clear();
        trainScores.clear();
        validationScores.clear();
        
        // Prepare validation split if early stopping is enabled
        ValidationSplit split = null;
        if (nIterNoChange > 0) {
            split = createValidationSplit(X, y);
        }
        
        // Initialize predictions
        initializePredictions(y);
        
        // Current predictions for training set
        double[] currentPreds = new double[X.length];
        Arrays.fill(currentPreds, initialPrediction);
        
        // Boosting iterations
        int noImproveCount = 0;
        double bestValidationScore = Double.NEGATIVE_INFINITY;
        
        for (int iteration = 0; iteration < nEstimators; iteration++) {
            // Calculate residuals/gradients
            double[] residuals = calculateResiduals(y, currentPreds);
            
            // Create subsample if needed
            SubsampleData subsampleData = createSubsample(X, residuals, iteration);
            
            // Fit tree to residuals
            DecisionTree tree = new DecisionTree(criterion, maxDepth)
                    .setMinSamplesSplit(minSamplesSplit)
                    .setMinSamplesLeaf(minSamplesLeaf)
                    .setMinImpurityDecrease(minImpurityDecrease)
                    .setMaxFeatures(maxFeatures)
                    .setRandomState(randomState + iteration);
            
            tree.fit(subsampleData.X, subsampleData.y);
            trees.add(tree);
            
            // Update predictions
            double[] treePreds = tree.predict(X);
            for (int i = 0; i < currentPreds.length; i++) {
                currentPreds[i] += learningRate * treePreds[i];
            }
            
            // Calculate training score
            double trainScore = calculateScore(y, currentPreds);
            trainScores.add(trainScore);
            
            // Early stopping check
            if (split != null) {
                double[] validPreds = predictAtIteration(split.X, iteration + 1);
                double validScore = calculateScore(split.y, validPreds);
                validationScores.add(validScore);
                
                if (validScore > bestValidationScore + tol) {
                    bestValidationScore = validScore;
                    noImproveCount = 0;
                } else {
                    noImproveCount++;
                    if (noImproveCount >= nIterNoChange) {
                        // Early stopping
                        break;
                    }
                }
            }
        }
        
        fitted = true;
        return this;
    }
    
    @Override
    public double[] predict(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before making predictions");
        }
        
        return predictAtIteration(X, trees.size());
    }
    
    /**
     * Predict using only the first n estimators.
     */
    public double[] predictAtIteration(double[][] X, int nEstimators) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before making predictions");
        }
        
        double[] predictions = new double[X.length];
        Arrays.fill(predictions, initialPrediction);
        
        int maxIterations = Math.min(nEstimators, trees.size());
        
        for (int i = 0; i < maxIterations; i++) {
            double[] treePreds = trees.get(i).predict(X);
            for (int j = 0; j < predictions.length; j++) {
                predictions[j] += learningRate * treePreds[j];
            }
        }
        
        if (isClassification) {
            // Convert to probabilities and then to class predictions
            for (int i = 0; i < predictions.length; i++) {
                double prob = sigmoid(predictions[i]);
                predictions[i] = prob >= 0.5 ? classes[1] : classes[0];
            }
        }
        
        return predictions;
    }
    
    @Override
    public double[][] predictProba(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before making predictions");
        }
        
        if (!isClassification) {
            throw new IllegalStateException("predict_proba is only available for classification");
        }
        
        double[] rawPredictions = predictRaw(X);
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
    
    /**
     * Get raw predictions (before applying sigmoid for classification).
     */
    public double[] predictRaw(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before making predictions");
        }
        
        double[] predictions = new double[X.length];
        Arrays.fill(predictions, initialPrediction);
        
        for (DecisionTree tree : trees) {
            double[] treePreds = tree.predict(X);
            for (int i = 0; i < predictions.length; i++) {
                predictions[i] += learningRate * treePreds[i];
            }
        }
        
        return predictions;
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
        
        if (isClassification) {
            // Accuracy for classification
            int correct = 0;
            for (int i = 0; i < y.length; i++) {
                if (predictions[i] == y[i]) {
                    correct++;
                }
            }
            return (double) correct / y.length;
        } else {
            // R² score for regression
            return calculateR2Score(y, predictions);
        }
    }
    
    /**
     * Get feature importances based on total impurity decrease.
     */
    public double[] getFeatureImportances() {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before accessing feature importances");
        }
        
        // This is a simplified implementation
        // In a full implementation, you'd sum impurity decreases across all trees
        int nFeatures = maxFeatures;
        double[] importances = new double[nFeatures];
        
        // For now, return uniform importances
        // TODO: Implement proper feature importance calculation
        Arrays.fill(importances, 1.0 / nFeatures);
        
        return importances;
    }
    
    /**
     * Get training scores for each iteration.
     */
    public List<Double> getTrainScores() {
        return new ArrayList<>(trainScores);
    }
    
    /**
     * Get validation scores for each iteration (if early stopping was used).
     */
    public List<Double> getValidationScores() {
        return new ArrayList<>(validationScores);
    }
    
    // Helper methods
    
    private boolean isClassificationProblem(double[] y) {
        Set<Double> uniqueValues = new HashSet<>();
        for (double value : y) {
            uniqueValues.add(value);
        }
        return uniqueValues.size() <= Math.sqrt(y.length);
    }
    
    private void initializePredictions(double[] y) {
        if (isClassification) {
            // Initialize to log-odds
            double posCount = 0;
            for (double value : y) {
                if (value == classes[1]) posCount++;
            }
            double probability = posCount / y.length;
            probability = Math.max(1e-15, Math.min(1 - 1e-15, probability));
            initialPrediction = Math.log(probability / (1 - probability));
        } else {
            // Initialize to mean for regression
            initialPrediction = Arrays.stream(y).average().orElse(0.0);
        }
    }
    
    private double[] calculateResiduals(double[] y, double[] predictions) {
        double[] residuals = new double[y.length];
        
        if (isClassification) {
            // Binary classification: gradient of log-loss
            for (int i = 0; i < y.length; i++) {
                double prob = sigmoid(predictions[i]);
                double yBinary = (y[i] == classes[1]) ? 1.0 : 0.0;
                residuals[i] = yBinary - prob;
            }
        } else {
            // Regression: negative gradient of MSE
            for (int i = 0; i < y.length; i++) {
                residuals[i] = y[i] - predictions[i];
            }
        }
        
        return residuals;
    }
    
    private SubsampleData createSubsample(double[][] X, double[] residuals, int iteration) {
        if (subsample >= 1.0) {
            return new SubsampleData(X, residuals);
        }
        
        Random subRandom = new Random(randomState + iteration);
        int sampleSize = (int) (X.length * subsample);
        
        double[][] sampleX = new double[sampleSize][];
        double[] sampleY = new double[sampleSize];
        
        Set<Integer> selectedIndices = new HashSet<>();
        while (selectedIndices.size() < sampleSize) {
            selectedIndices.add(subRandom.nextInt(X.length));
        }
        
        int idx = 0;
        for (int index : selectedIndices) {
            sampleX[idx] = Arrays.copyOf(X[index], X[index].length);
            sampleY[idx] = residuals[index];
            idx++;
        }
        
        return new SubsampleData(sampleX, sampleY);
    }
    
    private ValidationSplit createValidationSplit(double[][] X, double[] y) {
        int validationSize = (int) (X.length * validationFraction);
        int trainSize = X.length - validationSize;
        
        // Shuffle indices
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < X.length; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, random);
        
        // Split data
        double[][] trainX = new double[trainSize][];
        double[] trainY = new double[trainSize];
        double[][] validX = new double[validationSize][];
        double[] validY = new double[validationSize];
        
        for (int i = 0; i < trainSize; i++) {
            int idx = indices.get(i);
            trainX[i] = Arrays.copyOf(X[idx], X[idx].length);
            trainY[i] = y[idx];
        }
        
        for (int i = 0; i < validationSize; i++) {
            int idx = indices.get(trainSize + i);
            validX[i] = Arrays.copyOf(X[idx], X[idx].length);
            validY[i] = y[idx];
        }
        
        return new ValidationSplit(trainX, trainY, validX, validY);
    }
    
    private double calculateScore(double[] yTrue, double[] yPred) {
        if (isClassification) {
            // Convert raw predictions to class predictions
            double[] classPreds = new double[yPred.length];
            for (int i = 0; i < yPred.length; i++) {
                double prob = sigmoid(yPred[i]);
                classPreds[i] = prob >= 0.5 ? classes[1] : classes[0];
            }
            
            // Calculate accuracy
            int correct = 0;
            for (int i = 0; i < yTrue.length; i++) {
                if (classPreds[i] == yTrue[i]) {
                    correct++;
                }
            }
            return (double) correct / yTrue.length;
        } else {
            // R² score for regression
            return calculateR2Score(yTrue, yPred);
        }
    }
    
    private double calculateR2Score(double[] yTrue, double[] yPred) {
        double totalSumSquares = 0.0;
        double residualSumSquares = 0.0;
        double mean = Arrays.stream(yTrue).average().orElse(0.0);
        
        for (int i = 0; i < yTrue.length; i++) {
            totalSumSquares += Math.pow(yTrue[i] - mean, 2);
            residualSumSquares += Math.pow(yTrue[i] - yPred[i], 2);
        }
        
        return 1.0 - (residualSumSquares / totalSumSquares);
    }
    
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
    
    // Getters and setters
    
    public int getNEstimators() { return nEstimators; }
    public GradientBoosting setNEstimators(int nEstimators) {
        this.nEstimators = nEstimators;
        params.put("n_estimators", nEstimators);
        return this;
    }
    
    public double getLearningRate() { return learningRate; }
    public GradientBoosting setLearningRate(double learningRate) {
        this.learningRate = learningRate;
        params.put("learning_rate", learningRate);
        return this;
    }
    
    public int getMaxDepth() { return maxDepth; }
    public GradientBoosting setMaxDepth(int maxDepth) {
        this.maxDepth = maxDepth;
        params.put("max_depth", maxDepth);
        return this;
    }
    
    public String getCriterion() { return criterion; }
    public GradientBoosting setCriterion(String criterion) {
        this.criterion = criterion;
        params.put("criterion", criterion);
        return this;
    }
    
    public int getMinSamplesSplit() { return minSamplesSplit; }
    public GradientBoosting setMinSamplesSplit(int minSamplesSplit) {
        this.minSamplesSplit = minSamplesSplit;
        params.put("min_samples_split", minSamplesSplit);
        return this;
    }
    
    public int getMinSamplesLeaf() { return minSamplesLeaf; }
    public GradientBoosting setMinSamplesLeaf(int minSamplesLeaf) {
        this.minSamplesLeaf = minSamplesLeaf;
        params.put("min_samples_leaf", minSamplesLeaf);
        return this;
    }
    
    public double getMinImpurityDecrease() { return minImpurityDecrease; }
    public GradientBoosting setMinImpurityDecrease(double minImpurityDecrease) {
        this.minImpurityDecrease = minImpurityDecrease;
        params.put("min_impurity_decrease", minImpurityDecrease);
        return this;
    }
    
    public int getMaxFeatures() { return maxFeatures; }
    public GradientBoosting setMaxFeatures(int maxFeatures) {
        this.maxFeatures = maxFeatures;
        params.put("max_features", maxFeatures);
        return this;
    }
    
    public double getSubsample() { return subsample; }
    public GradientBoosting setSubsample(double subsample) {
        this.subsample = subsample;
        params.put("subsample", subsample);
        return this;
    }
    
    public int getRandomState() { return randomState; }
    public GradientBoosting setRandomState(int randomState) {
        this.randomState = randomState;
        this.random = new Random(randomState);
        params.put("random_state", randomState);
        return this;
    }
    
    public double getValidationFraction() { return validationFraction; }
    public GradientBoosting setValidationFraction(double validationFraction) {
        this.validationFraction = validationFraction;
        params.put("validation_fraction", validationFraction);
        return this;
    }
    
    public int getNIterNoChange() { return nIterNoChange; }
    public GradientBoosting setNIterNoChange(int nIterNoChange) {
        this.nIterNoChange = nIterNoChange;
        params.put("n_iter_no_change", nIterNoChange);
        return this;
    }
    
    public double getTol() { return tol; }
    public GradientBoosting setTol(double tol) {
        this.tol = tol;
        params.put("tol", tol);
        return this;
    }
    
    /**
     * Get the individual trees in the ensemble.
     */
    public List<DecisionTree> getTrees() {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before accessing trees");
        }
        return new ArrayList<>(trees);
    }
    
    // Helper classes
    
    private static class SubsampleData {
        final double[][] X;
        final double[] y;
        
        SubsampleData(double[][] X, double[] y) {
            this.X = X;
            this.y = y;
        }
    }
    
    private static class ValidationSplit {
        final double[][] X;
        final double[] y;
        
        ValidationSplit(double[][] trainX, double[] trainY, double[][] validX, double[] validY) {
            this.X = validX;
            this.y = validY;
        }
    }
}
