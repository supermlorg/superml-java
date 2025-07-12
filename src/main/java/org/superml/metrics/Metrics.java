package org.superml.metrics;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Classification and regression metrics.
 * Similar to sklearn.metrics
 */
public class Metrics {
    
    // Classification metrics
    
    /**
     * Calculate accuracy score.
     * @param yTrue true labels
     * @param yPred predicted labels
     * @return accuracy score
     */
    public static double accuracy(double[] yTrue, double[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        int correct = 0;
        for (int i = 0; i < yTrue.length; i++) {
            if (yTrue[i] == yPred[i]) {
                correct++;
            }
        }
        return (double) correct / yTrue.length;
    }
    
    /**
     * Calculate precision score for binary classification.
     * @param yTrue true labels
     * @param yPred predicted labels
     * @return precision score
     */
    public static double precision(double[] yTrue, double[] yPred) {
        int truePositives = 0;
        int falsePositives = 0;
        
        for (int i = 0; i < yTrue.length; i++) {
            if (yPred[i] == 1.0) {
                if (yTrue[i] == 1.0) {
                    truePositives++;
                } else {
                    falsePositives++;
                }
            }
        }
        
        return truePositives + falsePositives == 0 ? 0.0 : 
               (double) truePositives / (truePositives + falsePositives);
    }
    
    /**
     * Calculate recall score for binary classification.
     * @param yTrue true labels
     * @param yPred predicted labels
     * @return recall score
     */
    public static double recall(double[] yTrue, double[] yPred) {
        int truePositives = 0;
        int falseNegatives = 0;
        
        for (int i = 0; i < yTrue.length; i++) {
            if (yTrue[i] == 1.0) {
                if (yPred[i] == 1.0) {
                    truePositives++;
                } else {
                    falseNegatives++;
                }
            }
        }
        
        return truePositives + falseNegatives == 0 ? 0.0 : 
               (double) truePositives / (truePositives + falseNegatives);
    }
    
    /**
     * Calculate F1 score for binary classification.
     * @param yTrue true labels
     * @param yPred predicted labels
     * @return F1 score
     */
    public static double f1Score(double[] yTrue, double[] yPred) {
        double p = precision(yTrue, yPred);
        double r = recall(yTrue, yPred);
        return p + r == 0.0 ? 0.0 : 2 * (p * r) / (p + r);
    }
    
    /**
     * Generate a confusion matrix.
     * @param yTrue true labels
     * @param yPred predicted labels
     * @return confusion matrix as 2D array
     */
    public static int[][] confusionMatrix(double[] yTrue, double[] yPred) {
        // For binary classification
        int[][] matrix = new int[2][2];
        
        for (int i = 0; i < yTrue.length; i++) {
            int trueLabel = (int) yTrue[i];
            int predLabel = (int) yPred[i];
            matrix[trueLabel][predLabel]++;
        }
        
        return matrix;
    }
    
    /**
     * Generate a classification report.
     * @param yTrue true labels
     * @param yPred predicted labels
     * @return map containing metrics
     */
    public static Map<String, Double> classificationReport(double[] yTrue, double[] yPred) {
        Map<String, Double> report = new HashMap<>();
        
        report.put("accuracy", accuracy(yTrue, yPred));
        report.put("precision", precision(yTrue, yPred));
        report.put("recall", recall(yTrue, yPred));
        report.put("f1_score", f1Score(yTrue, yPred));
        
        return report;
    }
    
    // Regression metrics
    
    /**
     * Calculate mean squared error.
     * @param yTrue true values
     * @param yPred predicted values
     * @return MSE
     */
    public static double meanSquaredError(double[] yTrue, double[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        double sum = 0.0;
        for (int i = 0; i < yTrue.length; i++) {
            sum += Math.pow(yTrue[i] - yPred[i], 2);
        }
        return sum / yTrue.length;
    }
    
    /**
     * Calculate root mean squared error.
     * @param yTrue true values
     * @param yPred predicted values
     * @return RMSE
     */
    public static double rootMeanSquaredError(double[] yTrue, double[] yPred) {
        return Math.sqrt(meanSquaredError(yTrue, yPred));
    }
    
    /**
     * Calculate mean absolute error.
     * @param yTrue true values
     * @param yPred predicted values
     * @return MAE
     */
    public static double meanAbsoluteError(double[] yTrue, double[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        double sum = 0.0;
        for (int i = 0; i < yTrue.length; i++) {
            sum += Math.abs(yTrue[i] - yPred[i]);
        }
        return sum / yTrue.length;
    }
    
    /**
     * Calculate R² (coefficient of determination) regression score.
     * @param yTrue true values
     * @param yPred predicted values
     * @return R² score
     */
    public static double r2Score(double[] yTrue, double[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        double mean = Arrays.stream(yTrue).average().orElse(0.0);
        double totalSumSquares = 0.0;
        double residualSumSquares = 0.0;
        
        for (int i = 0; i < yTrue.length; i++) {
            totalSumSquares += Math.pow(yTrue[i] - mean, 2);
            residualSumSquares += Math.pow(yTrue[i] - yPred[i], 2);
        }
        
        return 1.0 - (residualSumSquares / totalSumSquares);
    }
}
