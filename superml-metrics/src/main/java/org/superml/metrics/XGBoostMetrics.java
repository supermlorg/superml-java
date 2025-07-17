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

import org.superml.tree.XGBoost;

import java.util.*;
import java.util.stream.IntStream;

/**
 * XGBoost-specific metrics and evaluation utilities
 * 
 * Provides specialized metrics for XGBoost models including:
 * - Training curve analysis
 * - Feature importance metrics
 * - Model complexity analysis
 * - Overfitting detection
 * - Competition-specific metrics
 * - Learning curve generation
 * 
 * @author SuperML Team
 * @version 2.1.0
 */
public class XGBoostMetrics {
    
    /**
     * Calculate log loss (logistic loss) for XGBoost classification
     */
    public static double logLoss(double[] yTrue, double[] yProbabilities) {
        if (yTrue.length != yProbabilities.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        double loss = 0.0;
        for (int i = 0; i < yTrue.length; i++) {
            double p = Math.max(1e-15, Math.min(1 - 1e-15, yProbabilities[i]));
            if (yTrue[i] == 1.0) {
                loss -= Math.log(p);
            } else {
                loss -= Math.log(1 - p);
            }
        }
        
        return loss / yTrue.length;
    }
    
    /**
     * Calculate AUC-ROC score for binary classification
     */
    public static double aucRoc(double[] yTrue, double[] yScores) {
        if (yTrue.length != yScores.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        // Create pairs and sort by score descending
        List<ScorePair> pairs = new ArrayList<>();
        for (int i = 0; i < yTrue.length; i++) {
            pairs.add(new ScorePair(yTrue[i], yScores[i]));
        }
        pairs.sort((a, b) -> Double.compare(b.score, a.score));
        
        int positives = (int) Arrays.stream(yTrue).sum();
        int negatives = yTrue.length - positives;
        
        if (positives == 0 || negatives == 0) {
            return 0.5; // No discrimination possible
        }
        
        double auc = 0.0;
        int truePositives = 0;
        
        for (ScorePair pair : pairs) {
            if (pair.label == 1.0) {
                truePositives++;
            } else {
                auc += truePositives; // Area under curve contribution
            }
        }
        
        return auc / (positives * negatives);
    }
    
    /**
     * Calculate RMSE (Root Mean Square Error) for regression
     */
    public static double rmse(double[] yTrue, double[] yPred) {
        return Math.sqrt(Metrics.meanSquaredError(yTrue, yPred));
    }
    
    /**
     * Calculate Mean Absolute Percentage Error (MAPE)
     */
    public static double mape(double[] yTrue, double[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        double sum = 0.0;
        int validCount = 0;
        
        for (int i = 0; i < yTrue.length; i++) {
            if (Math.abs(yTrue[i]) > 1e-10) { // Avoid division by zero
                sum += Math.abs((yTrue[i] - yPred[i]) / yTrue[i]);
                validCount++;
            }
        }
        
        return validCount > 0 ? (sum / validCount) * 100.0 : Double.NaN;
    }
    
    /**
     * Comprehensive XGBoost model evaluation
     */
    public static XGBoostEvaluation evaluateModel(XGBoost model, double[][] XTest, double[] yTest) {
        if (!model.isFitted()) {
            throw new IllegalStateException("Model must be fitted before evaluation");
        }
        
        XGBoostEvaluation eval = new XGBoostEvaluation();
        
        // Generate predictions
        double[] predictions = model.predict(XTest);
        eval.predictions = predictions;
        
        // Calculate metrics based on problem type
        if (model.isClassification()) {
            eval.accuracy = Metrics.accuracy(yTest, predictions);
            eval.precision = Metrics.precision(yTest, predictions);
            eval.recall = Metrics.recall(yTest, predictions);
            eval.f1Score = Metrics.f1Score(yTest, predictions);
            
            // Probabilistic metrics
            double[][] probabilities = model.predictProba(XTest);
            double[] positiveProbabilities = new double[probabilities.length];
            for (int i = 0; i < probabilities.length; i++) {
                positiveProbabilities[i] = probabilities[i][1]; // Positive class probability
            }
            
            eval.logLoss = logLoss(yTest, positiveProbabilities);
            eval.aucRoc = aucRoc(yTest, positiveProbabilities);
            
        } else {
            eval.mse = Metrics.meanSquaredError(yTest, predictions);
            eval.rmse = rmse(yTest, predictions);
            eval.mae = Metrics.meanAbsoluteError(yTest, predictions);
            eval.mape = mape(yTest, predictions);
            eval.r2Score = Metrics.r2Score(yTest, predictions);
        }
        
        // Model complexity metrics
        eval.nTrees = model.getNEstimators();
        eval.nFeatures = model.getNFeatures();
        eval.isClassification = model.isClassification();
        
        // Feature importance analysis
        eval.featureImportance = analyzeFeatureImportance(model);
        
        // Training metrics
        eval.trainingMetrics = analyzeTrainingMetrics(model);
        
        return eval;
    }
    
    /**
     * Analyze feature importance across multiple metrics
     */
    public static FeatureImportanceAnalysis analyzeFeatureImportance(XGBoost model) {
        Map<String, double[]> importanceStats = model.getFeatureImportanceStats();
        
        FeatureImportanceAnalysis analysis = new FeatureImportanceAnalysis();
        analysis.weightImportance = importanceStats.get("weight");
        analysis.gainImportance = importanceStats.get("gain");
        analysis.coverImportance = importanceStats.get("cover");
        
        // Calculate summary statistics
        if (analysis.gainImportance != null) {
            analysis.topFeatures = getTopFeatures(analysis.gainImportance, 10);
            analysis.importanceRatio = calculateImportanceRatio(analysis.gainImportance);
            analysis.effectiveFeatures = countEffectiveFeatures(analysis.gainImportance, 0.01);
        }
        
        return analysis;
    }
    
    /**
     * Analyze training metrics and detect overfitting
     */
    public static TrainingAnalysis analyzeTrainingMetrics(XGBoost model) {
        Map<String, List<Double>> evalResults = model.getEvalResults();
        
        TrainingAnalysis analysis = new TrainingAnalysis();
        analysis.evalResults = evalResults;
        
        // Analyze training curve
        List<Double> trainScores = evalResults.get("train-logloss");
        List<Double> validScores = evalResults.get("valid-logloss");
        
        if (trainScores != null && !trainScores.isEmpty()) {
            analysis.finalTrainScore = trainScores.get(trainScores.size() - 1);
            analysis.initialTrainScore = trainScores.get(0);
            analysis.trainingImprovement = analysis.initialTrainScore - analysis.finalTrainScore;
        }
        
        if (validScores != null && !validScores.isEmpty()) {
            analysis.finalValidScore = validScores.get(validScores.size() - 1);
            analysis.bestValidScore = validScores.stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
            analysis.overfittingDetected = detectOverfitting(trainScores, validScores);
        }
        
        analysis.converged = checkConvergence(trainScores);
        
        return analysis;
    }
    
    /**
     * Generate learning curves for XGBoost model
     */
    public static LearningCurveResult generateLearningCurves(XGBoost model, double[][] X, double[] y, 
                                                           int[] trainSizes, int cvFolds) {
        LearningCurveResult result = new LearningCurveResult();
        result.trainSizes = trainSizes;
        result.trainScores = new double[trainSizes.length][];
        result.validScores = new double[trainSizes.length][];
        
        for (int i = 0; i < trainSizes.length; i++) {
            int trainSize = trainSizes[i];
            
            // Generate scores for this training size
            double[] foldTrainScores = new double[cvFolds];
            double[] foldValidScores = new double[cvFolds];
            
            for (int fold = 0; fold < cvFolds; fold++) {
                // Create train/valid split for this fold
                LearningCurveSplit split = createLearningCurveSplit(X, y, trainSize, fold, cvFolds);
                
                // Train model on subset
                XGBoost foldModel = createModelCopy(model);
                foldModel.fit(split.trainX, split.trainY);
                
                // Evaluate on train and validation sets
                double[] trainPreds = foldModel.predict(split.trainX);
                double[] validPreds = foldModel.predict(split.validX);
                
                foldTrainScores[fold] = calculateScore(split.trainY, trainPreds, model.isClassification());
                foldValidScores[fold] = calculateScore(split.validY, validPreds, model.isClassification());
            }
            
            result.trainScores[i] = foldTrainScores;
            result.validScores[i] = foldValidScores;
        }
        
        return result;
    }
    
    /**
     * Competition-specific metrics for Kaggle
     */
    public static CompetitionMetrics calculateCompetitionMetrics(double[] yTrue, double[] yPred, 
                                                               String competitionType) {
        CompetitionMetrics metrics = new CompetitionMetrics();
        metrics.competitionType = competitionType;
        
        switch (competitionType.toLowerCase()) {
            case "binary_classification":
                metrics.primaryMetric = aucRoc(yTrue, yPred);
                metrics.primaryMetricName = "AUC-ROC";
                metrics.secondaryMetric = logLoss(yTrue, yPred);
                metrics.secondaryMetricName = "Log Loss";
                break;
                
            case "regression":
                metrics.primaryMetric = rmse(yTrue, yPred);
                metrics.primaryMetricName = "RMSE";
                metrics.secondaryMetric = Metrics.meanAbsoluteError(yTrue, yPred);
                metrics.secondaryMetricName = "MAE";
                break;
                
            case "multiclass":
                metrics.primaryMetric = Metrics.accuracy(yTrue, yPred);
                metrics.primaryMetricName = "Accuracy";
                metrics.secondaryMetric = Metrics.f1Score(yTrue, yPred);
                metrics.secondaryMetricName = "F1-Score";
                break;
                
            default:
                throw new IllegalArgumentException("Unknown competition type: " + competitionType);
        }
        
        return metrics;
    }
    
    // Helper methods
    
    private static List<Integer> getTopFeatures(double[] importance, int topK) {
        return IntStream.range(0, importance.length)
            .boxed()
            .sorted((i, j) -> Double.compare(importance[j], importance[i]))
            .limit(topK)
            .collect(ArrayList::new, ArrayList::add, ArrayList::addAll);
    }
    
    private static double calculateImportanceRatio(double[] importance) {
        double sum = Arrays.stream(importance).sum();
        if (sum == 0) return 0.0;
        
        double max = Arrays.stream(importance).max().orElse(0.0);
        return max / sum;
    }
    
    private static int countEffectiveFeatures(double[] importance, double threshold) {
        double totalImportance = Arrays.stream(importance).sum();
        double cutoff = totalImportance * threshold;
        
        return (int) Arrays.stream(importance).filter(imp -> imp >= cutoff).count();
    }
    
    private static boolean detectOverfitting(List<Double> trainScores, List<Double> validScores) {
        if (trainScores == null || validScores == null || trainScores.size() < 10) {
            return false;
        }
        
        // Check if validation score stopped improving while training score continues to improve
        int size = Math.min(trainScores.size(), validScores.size());
        int windowSize = Math.min(5, size / 4);
        
        double recentValidImprovement = validScores.get(size - 1) - validScores.get(size - windowSize);
        double recentTrainImprovement = trainScores.get(size - 1) - trainScores.get(size - windowSize);
        
        // Overfitting if training improves but validation doesn't (for loss metrics, lower is better)
        return recentTrainImprovement < -0.001 && recentValidImprovement > -0.001;
    }
    
    private static boolean checkConvergence(List<Double> scores) {
        if (scores == null || scores.size() < 10) return false;
        
        int windowSize = Math.min(5, scores.size() / 4);
        double recentChange = Math.abs(scores.get(scores.size() - 1) - scores.get(scores.size() - windowSize));
        
        return recentChange < 0.001; // Converged if change is very small
    }
    
    private static double calculateScore(double[] yTrue, double[] yPred, boolean isClassification) {
        return isClassification ? Metrics.accuracy(yTrue, yPred) : -rmse(yTrue, yPred);
    }
    
    private static XGBoost createModelCopy(XGBoost original) {
        return new XGBoost()
            .setNEstimators(original.getConfiguredNEstimators())
            .setLearningRate(original.getLearningRate())
            .setMaxDepth(original.getMaxDepth())
            .setGamma(original.getGamma())
            .setLambda(original.getLambda())
            .setAlpha(original.getAlpha())
            .setSubsample(original.getSubsample())
            .setColsampleBytree(original.getColsampleBytree())
            .setMinChildWeight(original.getMinChildWeight())
            .setRandomState(original.getRandomState());
    }
    
    private static LearningCurveSplit createLearningCurveSplit(double[][] X, double[] y, 
                                                             int trainSize, int fold, int nFolds) {
        // Simple implementation - can be improved with proper CV splitting
        Random random = new Random(42 + fold);
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < X.length; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, random);
        
        int validSize = X.length / nFolds;
        int validStart = fold * validSize;
        int validEnd = Math.min(validStart + validSize, X.length);
        
        List<Integer> trainIndices = new ArrayList<>();
        List<Integer> validIndices = new ArrayList<>();
        
        for (int i = 0; i < X.length; i++) {
            if (i >= validStart && i < validEnd) {
                validIndices.add(indices.get(i));
            } else if (trainIndices.size() < trainSize) {
                trainIndices.add(indices.get(i));
            }
        }
        
        LearningCurveSplit split = new LearningCurveSplit();
        split.trainX = trainIndices.stream().map(i -> X[i]).toArray(double[][]::new);
        split.trainY = trainIndices.stream().mapToDouble(i -> y[i]).toArray();
        split.validX = validIndices.stream().map(i -> X[i]).toArray(double[][]::new);
        split.validY = validIndices.stream().mapToDouble(i -> y[i]).toArray();
        
        return split;
    }
    
    // Data classes
    
    public static class XGBoostEvaluation {
        // Classification metrics
        public double accuracy;
        public double precision;
        public double recall;
        public double f1Score;
        public double logLoss;
        public double aucRoc;
        
        // Regression metrics
        public double mse;
        public double rmse;
        public double mae;
        public double mape;
        public double r2Score;
        
        // Model info
        public int nTrees;
        public int nFeatures;
        public boolean isClassification;
        public double[] predictions;
        
        // Analysis results
        public FeatureImportanceAnalysis featureImportance;
        public TrainingAnalysis trainingMetrics;
        
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("XGBoost Model Evaluation\n");
            sb.append("=" + "=".repeat(30) + "\n");
            sb.append(String.format("Model Type: %s\n", isClassification ? "Classification" : "Regression"));
            sb.append(String.format("Trees: %d, Features: %d\n", nTrees, nFeatures));
            
            if (isClassification) {
                sb.append(String.format("Accuracy: %.4f\n", accuracy));
                sb.append(String.format("Precision: %.4f\n", precision));
                sb.append(String.format("Recall: %.4f\n", recall));
                sb.append(String.format("F1-Score: %.4f\n", f1Score));
                sb.append(String.format("Log Loss: %.4f\n", logLoss));
                sb.append(String.format("AUC-ROC: %.4f\n", aucRoc));
            } else {
                sb.append(String.format("RMSE: %.4f\n", rmse));
                sb.append(String.format("MAE: %.4f\n", mae));
                sb.append(String.format("MAPE: %.2f%%\n", mape));
                sb.append(String.format("R²: %.4f\n", r2Score));
            }
            
            return sb.toString();
        }
    }
    
    public static class FeatureImportanceAnalysis {
        public double[] weightImportance;
        public double[] gainImportance;
        public double[] coverImportance;
        public List<Integer> topFeatures;
        public double importanceRatio;
        public int effectiveFeatures;
    }
    
    public static class TrainingAnalysis {
        public Map<String, List<Double>> evalResults;
        public double finalTrainScore;
        public double finalValidScore;
        public double initialTrainScore;
        public double bestValidScore;
        public double trainingImprovement;
        public boolean overfittingDetected;
        public boolean converged;
    }
    
    public static class LearningCurveResult {
        public int[] trainSizes;
        public double[][] trainScores;
        public double[][] validScores;
    }
    
    public static class CompetitionMetrics {
        public String competitionType;
        public double primaryMetric;
        public String primaryMetricName;
        public double secondaryMetric;
        public String secondaryMetricName;
    }
    
    private static class ScorePair {
        final double label;
        final double score;
        
        ScorePair(double label, double score) {
            this.label = label;
            this.score = score;
        }
    }
    
    private static class LearningCurveSplit {
        double[][] trainX;
        double[] trainY;
        double[][] validX;
        double[] validY;
    }
}
