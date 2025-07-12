package org.superml.linear_model;

import org.superml.core.BaseEstimator;
import org.superml.core.Regressor;
import java.util.Arrays;

/**
 * Lasso Regression with L1 regularization.
 * Similar to sklearn.linear_model.Lasso
 */
public class Lasso extends BaseEstimator implements Regressor {
    
    private double[] coefficients;
    private double intercept;
    private boolean fitted = false;
    private boolean fitIntercept = true;
    private double alpha = 1.0; // Regularization strength
    private int maxIter = 1000;
    private double tolerance = 1e-4;
    private boolean warm_start = false;
    
    public Lasso() {
        params.put("alpha", alpha);
        params.put("fit_intercept", fitIntercept);
        params.put("max_iter", maxIter);
        params.put("tol", tolerance);
        params.put("warm_start", warm_start);
    }
    
    public Lasso(double alpha) {
        this();
        this.alpha = alpha;
        params.put("alpha", alpha);
    }
    
    public Lasso(double alpha, int maxIter) {
        this();
        this.alpha = alpha;
        this.maxIter = maxIter;
        params.put("alpha", alpha);
        params.put("max_iter", maxIter);
    }
    
    @Override
    public Lasso fit(double[][] X, double[] y) {
        int nFeatures = X[0].length;
        
        // Standardize data for coordinate descent
        double[][] XStd = standardizeFeatures(X);
        double[] yStd = standardizeTarget(y);
        
        // Initialize coefficients
        if (!warm_start || coefficients == null) {
            coefficients = new double[nFeatures];
        }
        
        // Coordinate descent algorithm
        coordinateDescent(XStd, yStd);
        
        // Calculate intercept
        if (fitIntercept) {
            intercept = calculateIntercept(X, y);
        } else {
            intercept = 0.0;
        }
        
        fitted = true;
        return this;
    }
    
    @Override
    public double[] predict(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before making predictions");
        }
        
        double[] predictions = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = intercept;
            for (int j = 0; j < coefficients.length; j++) {
                predictions[i] += coefficients[j] * X[i][j];
            }
        }
        return predictions;
    }
    
    @Override
    public double score(double[][] X, double[] y) {
        double[] predictions = predict(X);
        return r2Score(y, predictions);
    }
    
    private void coordinateDescent(double[][] X, double[] y) {
        int nSamples = X.length;
        int nFeatures = X[0].length;
        
        // Pre-compute X^T * X diagonal (for efficiency)
        double[] XTX_diag = new double[nFeatures];
        for (int j = 0; j < nFeatures; j++) {
            for (int i = 0; i < nSamples; i++) {
                XTX_diag[j] += X[i][j] * X[i][j];
            }
        }
        
        for (int iter = 0; iter < maxIter; iter++) {
            double maxChange = 0.0;
            
            for (int j = 0; j < nFeatures; j++) {
                double oldCoeff = coefficients[j];
                
                // Calculate partial residual
                double partialResidual = 0.0;
                for (int i = 0; i < nSamples; i++) {
                    double prediction = 0.0;
                    for (int k = 0; k < nFeatures; k++) {
                        if (k != j) {
                            prediction += coefficients[k] * X[i][k];
                        }
                    }
                    partialResidual += X[i][j] * (y[i] - prediction);
                }
                
                // Soft thresholding (coordinate descent update for Lasso)
                coefficients[j] = softThreshold(partialResidual, alpha) / XTX_diag[j];
                
                double change = Math.abs(coefficients[j] - oldCoeff);
                maxChange = Math.max(maxChange, change);
            }
            
            // Check convergence
            if (maxChange < tolerance) {
                break;
            }
        }
    }
    
    private double softThreshold(double value, double threshold) {
        if (value > threshold) {
            return value - threshold;
        } else if (value < -threshold) {
            return value + threshold;
        } else {
            return 0.0;
        }
    }
    
    private double[][] standardizeFeatures(double[][] X) {
        int nSamples = X.length;
        int nFeatures = X[0].length;
        double[][] XStd = new double[nSamples][nFeatures];
        
        // Calculate means and standard deviations
        double[] means = new double[nFeatures];
        double[] stds = new double[nFeatures];
        
        for (int j = 0; j < nFeatures; j++) {
            double sum = 0.0;
            for (int i = 0; i < nSamples; i++) {
                sum += X[i][j];
            }
            means[j] = sum / nSamples;
            
            double sumSquares = 0.0;
            for (int i = 0; i < nSamples; i++) {
                sumSquares += Math.pow(X[i][j] - means[j], 2);
            }
            stds[j] = Math.sqrt(sumSquares / nSamples);
            if (stds[j] == 0.0) stds[j] = 1.0; // Avoid division by zero
        }
        
        // Standardize
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                XStd[i][j] = (X[i][j] - means[j]) / stds[j];
            }
        }
        
        return XStd;
    }
    
    private double[] standardizeTarget(double[] y) {
        int nSamples = y.length;
        double[] yStd = new double[nSamples];
        
        double mean = Arrays.stream(y).average().orElse(0.0);
        
        for (int i = 0; i < nSamples; i++) {
            yStd[i] = y[i] - mean;
        }
        
        return yStd;
    }
    
    private double calculateIntercept(double[][] X, double[] y) {
        // Calculate means
        double[] xMeans = new double[X[0].length];
        for (int j = 0; j < X[0].length; j++) {
            for (int i = 0; i < X.length; i++) {
                xMeans[j] += X[i][j];
            }
            xMeans[j] /= X.length;
        }
        
        double yMean = Arrays.stream(y).average().orElse(0.0);
        
        // intercept = y_mean - sum(coef * x_mean)
        double interceptValue = yMean;
        for (int j = 0; j < coefficients.length; j++) {
            interceptValue -= coefficients[j] * xMeans[j];
        }
        
        return interceptValue;
    }
    
    private double r2Score(double[] yTrue, double[] yPred) {
        double totalSumSquares = 0.0;
        double residualSumSquares = 0.0;
        double mean = Arrays.stream(yTrue).average().orElse(0.0);
        
        for (int i = 0; i < yTrue.length; i++) {
            totalSumSquares += Math.pow(yTrue[i] - mean, 2);
            residualSumSquares += Math.pow(yTrue[i] - yPred[i], 2);
        }
        
        return 1.0 - (residualSumSquares / totalSumSquares);
    }
    
    /**
     * Get the number of non-zero coefficients (sparsity measure).
     * @return number of non-zero coefficients
     */
    public int getNumNonZeroCoefficients() {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before accessing coefficients");
        }
        
        int count = 0;
        for (double coeff : coefficients) {
            if (Math.abs(coeff) > 1e-10) {
                count++;
            }
        }
        return count;
    }
    
    /**
     * Get indices of non-zero coefficients.
     * @return array of indices with non-zero coefficients
     */
    public int[] getNonZeroIndices() {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before accessing coefficients");
        }
        
        return java.util.stream.IntStream.range(0, coefficients.length)
                .filter(i -> Math.abs(coefficients[i]) > 1e-10)
                .toArray();
    }
    
    // Getters
    public double[] getCoefficients() {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before accessing coefficients");
        }
        return Arrays.copyOf(coefficients, coefficients.length);
    }
    
    public double getIntercept() {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before accessing intercept");
        }
        return intercept;
    }
    
    // Setters
    public Lasso setAlpha(double alpha) {
        this.alpha = alpha;
        params.put("alpha", alpha);
        return this;
    }
    
    public Lasso setFitIntercept(boolean fitIntercept) {
        this.fitIntercept = fitIntercept;
        params.put("fit_intercept", fitIntercept);
        return this;
    }
    
    public Lasso setMaxIter(int maxIter) {
        this.maxIter = maxIter;
        params.put("max_iter", maxIter);
        return this;
    }
    
    public Lasso setTolerance(double tolerance) {
        this.tolerance = tolerance;
        params.put("tol", tolerance);
        return this;
    }
    
    public Lasso setWarmStart(boolean warmStart) {
        this.warm_start = warmStart;
        params.put("warm_start", warmStart);
        return this;
    }
    
    public double getAlpha() { return alpha; }
    public boolean isFitIntercept() { return fitIntercept; }
    public int getMaxIter() { return maxIter; }
    public double getTolerance() { return tolerance; }
    public boolean isWarmStart() { return warm_start; }
}
