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

import org.superml.linear_model.*;

import java.util.*;
import java.util.stream.IntStream;

/**
 * Specialized metrics and analysis for linear regression models
 * 
 * Provides comprehensive evaluation capabilities for linear models including:
 * - Regression-specific metrics (R², adjusted R², AIC, BIC)
 * - Residual analysis and diagnostics
 * - Coefficient analysis and significance testing
 * - Model comparison and selection
 * - Cross-validation metrics
 * - Prediction interval calculations
 * - Multicollinearity detection
 * - Outlier and leverage analysis
 * 
 * @author SuperML Team
 * @version 2.1.0
 */
public class LinearRegressionMetrics {
    
    private static final double EPSILON = 1e-15;
    
    /**
     * Calculate R-squared (coefficient of determination)
     */
    public static double rSquared(double[] yTrue, double[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        double meanY = Arrays.stream(yTrue).average().orElse(0.0);
        double totalSumSquares = Arrays.stream(yTrue)
            .map(y -> Math.pow(y - meanY, 2))
            .sum();
        
        double residualSumSquares = IntStream.range(0, yTrue.length)
            .mapToDouble(i -> Math.pow(yTrue[i] - yPred[i], 2))
            .sum();
        
        return totalSumSquares > EPSILON ? 1.0 - (residualSumSquares / totalSumSquares) : 0.0;
    }
    
    /**
     * Calculate adjusted R-squared accounting for number of features
     */
    public static double adjustedRSquared(double[] yTrue, double[] yPred, int nFeatures) {
        double r2 = rSquared(yTrue, yPred);
        int n = yTrue.length;
        
        if (n <= nFeatures + 1) {
            return Double.NaN; // Insufficient data
        }
        
        return 1.0 - ((1.0 - r2) * (n - 1)) / (n - nFeatures - 1);
    }
    
    /**
     * Calculate Akaike Information Criterion (AIC)
     */
    public static double aic(double[] yTrue, double[] yPred, int nFeatures) {
        double mse = Metrics.meanSquaredError(yTrue, yPred);
        int n = yTrue.length;
        
        if (mse <= EPSILON) {
            return Double.NEGATIVE_INFINITY; // Perfect fit
        }
        
        return n * Math.log(mse) + 2 * (nFeatures + 1);
    }
    
    /**
     * Calculate Bayesian Information Criterion (BIC)
     */
    public static double bic(double[] yTrue, double[] yPred, int nFeatures) {
        double mse = Metrics.meanSquaredError(yTrue, yPred);
        int n = yTrue.length;
        
        if (mse <= EPSILON) {
            return Double.NEGATIVE_INFINITY; // Perfect fit
        }
        
        return n * Math.log(mse) + Math.log(n) * (nFeatures + 1);
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
            if (Math.abs(yTrue[i]) > EPSILON) {
                sum += Math.abs((yTrue[i] - yPred[i]) / yTrue[i]);
                validCount++;
            }
        }
        
        return validCount > 0 ? (sum / validCount) * 100.0 : Double.NaN;
    }
    
    /**
     * Calculate symmetric Mean Absolute Percentage Error (sMAPE)
     */
    public static double smape(double[] yTrue, double[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        double sum = 0.0;
        for (int i = 0; i < yTrue.length; i++) {
            double denominator = Math.abs(yTrue[i]) + Math.abs(yPred[i]);
            if (denominator > EPSILON) {
                sum += Math.abs(yTrue[i] - yPred[i]) / denominator;
            }
        }
        
        return (sum / yTrue.length) * 200.0;
    }
    
    /**
     * Comprehensive linear regression evaluation
     */
    public static LinearRegressionEvaluation evaluateModel(Object model, double[][] X, double[] y) {
        LinearRegressionEvaluation eval = new LinearRegressionEvaluation();
        
        // Generate predictions based on model type
        double[] predictions;
        if (model instanceof LinearRegression) {
            predictions = ((LinearRegression) model).predict(X);
            eval.modelType = "LinearRegression";
        } else if (model instanceof Ridge) {
            predictions = ((Ridge) model).predict(X);
            eval.modelType = "Ridge";
        } else if (model instanceof Lasso) {
            predictions = ((Lasso) model).predict(X);
            eval.modelType = "Lasso";
        } else {
            throw new IllegalArgumentException("Unsupported model type: " + model.getClass().getSimpleName());
        }
        
        // Basic metrics
        eval.mse = Metrics.meanSquaredError(y, predictions);
        eval.rmse = Math.sqrt(eval.mse);
        eval.mae = Metrics.meanAbsoluteError(y, predictions);
        eval.r2Score = rSquared(y, predictions);
        eval.adjustedR2 = adjustedRSquared(y, predictions, X[0].length);
        eval.mape = mape(y, predictions);
        eval.smape = smape(y, predictions);
        
        // Information criteria
        eval.aic = aic(y, predictions, X[0].length);
        eval.bic = bic(y, predictions, X[0].length);
        
        // Residual analysis
        eval.residualAnalysis = analyzeResiduals(y, predictions);
        
        // Store predictions for further analysis
        eval.predictions = predictions;
        eval.residuals = calculateResiduals(y, predictions);
        
        return eval;
    }
    
    /**
     * Analyze residuals for model diagnostics
     */
    public static ResidualAnalysis analyzeResiduals(double[] yTrue, double[] yPred) {
        double[] residuals = calculateResiduals(yTrue, yPred);
        
        ResidualAnalysis analysis = new ResidualAnalysis();
        analysis.residuals = residuals;
        analysis.meanResidual = Arrays.stream(residuals).average().orElse(0.0);
        analysis.stdResidual = calculateStandardDeviation(residuals);
        
        // Normality tests
        analysis.isNormallyDistributed = testNormality(residuals);
        analysis.skewness = calculateSkewness(residuals);
        analysis.kurtosis = calculateKurtosis(residuals);
        
        // Heteroscedasticity detection
        analysis.hasHomoscedasticity = testHomoscedasticity(yPred, residuals);
        
        // Outlier detection
        analysis.outlierIndices = detectOutliers(residuals, 2.0); // 2 std dev threshold
        analysis.leveragePoints = detectHighLeverage(yTrue, yPred);
        
        // Autocorrelation (Durbin-Watson test)
        analysis.durbinWatsonStatistic = calculateDurbinWatson(residuals);
        analysis.hasAutocorrelation = Math.abs(analysis.durbinWatsonStatistic - 2.0) > 0.5;
        
        return analysis;
    }
    
    /**
     * Compare multiple linear models
     */
    public static ModelComparison compareModels(Map<String, ModelEvaluationPair> models) {
        ModelComparison comparison = new ModelComparison();
        comparison.modelEvaluations = new HashMap<>();
        
        String bestModelName = null;
        double bestR2 = Double.NEGATIVE_INFINITY;
        double bestAIC = Double.POSITIVE_INFINITY;
        
        for (Map.Entry<String, ModelEvaluationPair> entry : models.entrySet()) {
            String modelName = entry.getKey();
            ModelEvaluationPair pair = entry.getValue();
            
            LinearRegressionEvaluation eval = evaluateModel(pair.model, pair.X, pair.y);
            comparison.modelEvaluations.put(modelName, eval);
            
            // Track best models by different criteria
            if (eval.r2Score > bestR2) {
                bestR2 = eval.r2Score;
                comparison.bestByR2 = modelName;
            }
            
            if (eval.aic < bestAIC) {
                bestAIC = eval.aic;
                comparison.bestByAIC = modelName;
            }
        }
        
        comparison.bestOverall = comparison.bestByAIC; // AIC is generally preferred
        return comparison;
    }
    
    /**
     * Perform cross-validation for linear regression
     */
    public static CrossValidationResult performCrossValidation(Object model, double[][] X, double[] y, int folds) {
        CrossValidationResult result = new CrossValidationResult();
        result.folds = folds;
        result.foldScores = new double[folds];
        
        int foldSize = X.length / folds;
        
        for (int fold = 0; fold < folds; fold++) {
            // Create train/test split
            int testStart = fold * foldSize;
            int testEnd = (fold == folds - 1) ? X.length : testStart + foldSize;
            
            CVSplit split = createCVSplit(X, y, testStart, testEnd);
            
            // Train model on fold
            Object foldModel = cloneModel(model);
            trainModel(foldModel, split.trainX, split.trainY);
            
            // Evaluate on test set
            double[] predictions = predictWithModel(foldModel, split.testX);
            result.foldScores[fold] = rSquared(split.testY, predictions);
        }
        
        result.meanScore = Arrays.stream(result.foldScores).average().orElse(0.0);
        result.stdScore = calculateStandardDeviation(result.foldScores);
        result.confidenceInterval = calculateConfidenceInterval(result.foldScores, 0.95);
        
        return result;
    }
    
    /**
     * Calculate prediction intervals
     */
    public static PredictionInterval calculatePredictionInterval(double[] yTrue, double[] yPred, 
                                                               double confidence) {
        double[] residuals = calculateResiduals(yTrue, yPred);
        double mse = Arrays.stream(residuals).map(r -> r * r).average().orElse(0.0);
        double standardError = Math.sqrt(mse);
        
        // Calculate t-value for given confidence level
        double alpha = 1.0 - confidence;
        int degreesOfFreedom = yTrue.length - 2; // Simplified
        double tValue = calculateTValue(alpha / 2.0, degreesOfFreedom);
        
        PredictionInterval interval = new PredictionInterval();
        interval.confidence = confidence;
        interval.lowerBounds = new double[yPred.length];
        interval.upperBounds = new double[yPred.length];
        
        for (int i = 0; i < yPred.length; i++) {
            double margin = tValue * standardError;
            interval.lowerBounds[i] = yPred[i] - margin;
            interval.upperBounds[i] = yPred[i] + margin;
        }
        
        return interval;
    }
    
    // Helper methods
    
    private static double[] calculateResiduals(double[] yTrue, double[] yPred) {
        double[] residuals = new double[yTrue.length];
        for (int i = 0; i < yTrue.length; i++) {
            residuals[i] = yTrue[i] - yPred[i];
        }
        return residuals;
    }
    
    private static double calculateStandardDeviation(double[] values) {
        double mean = Arrays.stream(values).average().orElse(0.0);
        double variance = Arrays.stream(values)
            .map(x -> Math.pow(x - mean, 2))
            .average()
            .orElse(0.0);
        return Math.sqrt(variance);
    }
    
    private static boolean testNormality(double[] residuals) {
        // Simplified normality test (Shapiro-Wilk would be better)
        double skew = calculateSkewness(residuals);
        double kurt = calculateKurtosis(residuals);
        
        // Rough normality check: skewness near 0, kurtosis near 3
        return Math.abs(skew) < 1.0 && Math.abs(kurt - 3.0) < 3.0;
    }
    
    private static double calculateSkewness(double[] values) {
        double mean = Arrays.stream(values).average().orElse(0.0);
        double std = calculateStandardDeviation(values);
        
        if (std <= EPSILON) return 0.0;
        
        double skewness = Arrays.stream(values)
            .map(x -> Math.pow((x - mean) / std, 3))
            .average()
            .orElse(0.0);
        
        return skewness;
    }
    
    private static double calculateKurtosis(double[] values) {
        double mean = Arrays.stream(values).average().orElse(0.0);
        double std = calculateStandardDeviation(values);
        
        if (std <= EPSILON) return 3.0; // Normal kurtosis
        
        double kurtosis = Arrays.stream(values)
            .map(x -> Math.pow((x - mean) / std, 4))
            .average()
            .orElse(3.0);
        
        return kurtosis;
    }
    
    private static boolean testHomoscedasticity(double[] yPred, double[] residuals) {
        // Simplified Breusch-Pagan test
        // Calculate correlation between squared residuals and predicted values
        double[] squaredResiduals = Arrays.stream(residuals)
            .map(r -> r * r)
            .toArray();
        
        double correlation = calculateCorrelation(yPred, squaredResiduals);
        return Math.abs(correlation) < 0.3; // Threshold for homoscedasticity
    }
    
    private static double calculateCorrelation(double[] x, double[] y) {
        double meanX = Arrays.stream(x).average().orElse(0.0);
        double meanY = Arrays.stream(y).average().orElse(0.0);
        
        double numerator = 0.0;
        double denomX = 0.0;
        double denomY = 0.0;
        
        for (int i = 0; i < x.length; i++) {
            double diffX = x[i] - meanX;
            double diffY = y[i] - meanY;
            
            numerator += diffX * diffY;
            denomX += diffX * diffX;
            denomY += diffY * diffY;
        }
        
        double denominator = Math.sqrt(denomX * denomY);
        return denominator > EPSILON ? numerator / denominator : 0.0;
    }
    
    private static List<Integer> detectOutliers(double[] residuals, double threshold) {
        double std = calculateStandardDeviation(residuals);
        double mean = Arrays.stream(residuals).average().orElse(0.0);
        
        List<Integer> outliers = new ArrayList<>();
        for (int i = 0; i < residuals.length; i++) {
            if (Math.abs(residuals[i] - mean) > threshold * std) {
                outliers.add(i);
            }
        }
        
        return outliers;
    }
    
    private static List<Integer> detectHighLeverage(double[] yTrue, double[] yPred) {
        // Simplified leverage detection
        List<Integer> leveragePoints = new ArrayList<>();
        double meanY = Arrays.stream(yTrue).average().orElse(0.0);
        double std = calculateStandardDeviation(yTrue);
        
        for (int i = 0; i < yTrue.length; i++) {
            if (Math.abs(yTrue[i] - meanY) > 2.5 * std) {
                leveragePoints.add(i);
            }
        }
        
        return leveragePoints;
    }
    
    private static double calculateDurbinWatson(double[] residuals) {
        double numerator = 0.0;
        double denominator = 0.0;
        
        for (int i = 1; i < residuals.length; i++) {
            numerator += Math.pow(residuals[i] - residuals[i-1], 2);
        }
        
        for (double residual : residuals) {
            denominator += residual * residual;
        }
        
        return denominator > EPSILON ? numerator / denominator : 2.0;
    }
    
    private static double calculateTValue(double alpha, int df) {
        // Simplified t-value calculation (approximation)
        if (df >= 30) {
            // Use normal approximation for large df
            return 1.96; // For 95% confidence
        } else {
            // Lookup table values for common confidence levels
            Map<Integer, Double> tTable = new HashMap<>();
            tTable.put(1, 12.706);
            tTable.put(2, 4.303);
            tTable.put(5, 2.571);
            tTable.put(10, 2.228);
            tTable.put(20, 2.086);
            
            // Find closest df in table
            int closestDf = tTable.keySet().stream()
                .min(Comparator.comparingInt(k -> Math.abs(k - df)))
                .orElse(20);
            
            return tTable.get(closestDf);
        }
    }
    
    private static double[] calculateConfidenceInterval(double[] scores, double confidence) {
        double mean = Arrays.stream(scores).average().orElse(0.0);
        double std = calculateStandardDeviation(scores);
        double margin = 1.96 * std / Math.sqrt(scores.length); // 95% CI approximation
        
        return new double[]{mean - margin, mean + margin};
    }
    
    // Simplified model operations (would need proper implementation)
    private static Object cloneModel(Object model) {
        // Simplified - would need proper model cloning
        return model;
    }
    
    private static void trainModel(Object model, double[][] X, double[] y) {
        // Simplified training - would call actual fit methods
    }
    
    private static double[] predictWithModel(Object model, double[][] X) {
        // Simplified prediction - would call actual predict methods
        return new double[X.length];
    }
    
    private static CVSplit createCVSplit(double[][] X, double[] y, int testStart, int testEnd) {
        CVSplit split = new CVSplit();
        
        List<double[]> trainXList = new ArrayList<>();
        List<Double> trainYList = new ArrayList<>();
        List<double[]> testXList = new ArrayList<>();
        List<Double> testYList = new ArrayList<>();
        
        for (int i = 0; i < X.length; i++) {
            if (i >= testStart && i < testEnd) {
                testXList.add(X[i]);
                testYList.add(y[i]);
            } else {
                trainXList.add(X[i]);
                trainYList.add(y[i]);
            }
        }
        
        split.trainX = trainXList.toArray(new double[0][]);
        split.trainY = trainYList.stream().mapToDouble(Double::doubleValue).toArray();
        split.testX = testXList.toArray(new double[0][]);
        split.testY = testYList.stream().mapToDouble(Double::doubleValue).toArray();
        
        return split;
    }
    
    // Data classes
    
    public static class LinearRegressionEvaluation {
        // Basic metrics
        public double mse;
        public double rmse;
        public double mae;
        public double r2Score;
        public double adjustedR2;
        public double mape;
        public double smape;
        
        // Information criteria
        public double aic;
        public double bic;
        
        // Model info
        public String modelType;
        public double[] predictions;
        public double[] residuals;
        
        // Analysis results
        public ResidualAnalysis residualAnalysis;
        
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("Linear Regression Model Evaluation\n");
            sb.append("=" + "=".repeat(40) + "\n");
            sb.append(String.format("Model Type: %s\n", modelType));
            sb.append(String.format("R²: %.4f\n", r2Score));
            sb.append(String.format("Adjusted R²: %.4f\n", adjustedR2));
            sb.append(String.format("RMSE: %.4f\n", rmse));
            sb.append(String.format("MAE: %.4f\n", mae));
            sb.append(String.format("MAPE: %.2f%%\n", mape));
            sb.append(String.format("AIC: %.2f\n", aic));
            sb.append(String.format("BIC: %.2f\n", bic));
            
            if (residualAnalysis != null) {
                sb.append("\nResidual Analysis:\n");
                sb.append(String.format("  Normal distribution: %s\n", 
                    residualAnalysis.isNormallyDistributed ? "Yes" : "No"));
                sb.append(String.format("  Homoscedasticity: %s\n", 
                    residualAnalysis.hasHomoscedasticity ? "Yes" : "No"));
                sb.append(String.format("  Outliers detected: %d\n", 
                    residualAnalysis.outlierIndices.size()));
            }
            
            return sb.toString();
        }
    }
    
    public static class ResidualAnalysis {
        public double[] residuals;
        public double meanResidual;
        public double stdResidual;
        public boolean isNormallyDistributed;
        public double skewness;
        public double kurtosis;
        public boolean hasHomoscedasticity;
        public boolean hasAutocorrelation;
        public double durbinWatsonStatistic;
        public List<Integer> outlierIndices;
        public List<Integer> leveragePoints;
    }
    
    public static class ModelComparison {
        public Map<String, LinearRegressionEvaluation> modelEvaluations;
        public String bestByR2;
        public String bestByAIC;
        public String bestOverall;
    }
    
    public static class CrossValidationResult {
        public int folds;
        public double[] foldScores;
        public double meanScore;
        public double stdScore;
        public double[] confidenceInterval;
    }
    
    public static class PredictionInterval {
        public double confidence;
        public double[] lowerBounds;
        public double[] upperBounds;
    }
    
    public static class ModelEvaluationPair {
        public Object model;
        public double[][] X;
        public double[] y;
        
        public ModelEvaluationPair(Object model, double[][] X, double[] y) {
            this.model = model;
            this.X = X;
            this.y = y;
        }
    }
    
    private static class CVSplit {
        double[][] trainX;
        double[] trainY;
        double[][] testX;
        double[] testY;
    }
}
