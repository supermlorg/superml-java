package org.superml.preprocessing;

import org.superml.core.BaseEstimator;
import org.superml.core.UnsupervisedLearner;
import java.util.Arrays;

/**
 * Standardize features by removing the mean and scaling to unit variance.
 * Similar to sklearn.preprocessing.StandardScaler
 */
public class StandardScaler extends BaseEstimator implements UnsupervisedLearner {
    
    private double[] mean;
    private double[] scale;
    private boolean fitted = false;
    private boolean withMean = true;
    private boolean withStd = true;
    
    public StandardScaler() {
        params.put("with_mean", withMean);
        params.put("with_std", withStd);
    }
    
    public StandardScaler(boolean withMean, boolean withStd) {
        this.withMean = withMean;
        this.withStd = withStd;
        params.put("with_mean", withMean);
        params.put("with_std", withStd);
    }
    
    @Override
    public StandardScaler fit(double[][] X) {
        int nSamples = X.length;
        int nFeatures = X[0].length;
        
        mean = new double[nFeatures];
        scale = new double[nFeatures];
        
        // Calculate means
        if (withMean) {
            for (int j = 0; j < nFeatures; j++) {
                double sum = 0.0;
                for (int i = 0; i < nSamples; i++) {
                    sum += X[i][j];
                }
                mean[j] = sum / nSamples;
            }
        } else {
            Arrays.fill(mean, 0.0);
        }
        
        // Calculate standard deviations
        if (withStd) {
            for (int j = 0; j < nFeatures; j++) {
                double sumSquares = 0.0;
                for (int i = 0; i < nSamples; i++) {
                    sumSquares += Math.pow(X[i][j] - mean[j], 2);
                }
                scale[j] = Math.sqrt(sumSquares / nSamples);
                // Avoid division by zero
                if (scale[j] == 0.0) {
                    scale[j] = 1.0;
                }
            }
        } else {
            Arrays.fill(scale, 1.0);
        }
        
        fitted = true;
        return this;
    }
    
    @Override
    public double[][] transform(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Scaler must be fitted before transforming data");
        }
        
        int nSamples = X.length;
        int nFeatures = X[0].length;
        double[][] transformed = new double[nSamples][nFeatures];
        
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                transformed[i][j] = (X[i][j] - mean[j]) / scale[j];
            }
        }
        
        return transformed;
    }
    
    /**
     * Scale back the data to the original representation.
     * @param X scaled data
     * @return original scale data
     */
    public double[][] inverseTransform(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Scaler must be fitted before inverse transforming data");
        }
        
        int nSamples = X.length;
        int nFeatures = X[0].length;
        double[][] original = new double[nSamples][nFeatures];
        
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                original[i][j] = X[i][j] * scale[j] + mean[j];
            }
        }
        
        return original;
    }
    
    // Getters
    public double[] getMean() {
        if (!fitted) {
            throw new IllegalStateException("Scaler must be fitted before accessing mean");
        }
        return Arrays.copyOf(mean, mean.length);
    }
    
    public double[] getScale() {
        if (!fitted) {
            throw new IllegalStateException("Scaler must be fitted before accessing scale");
        }
        return Arrays.copyOf(scale, scale.length);
    }
    
    public boolean isWithMean() {
        return withMean;
    }
    
    public StandardScaler setWithMean(boolean withMean) {
        this.withMean = withMean;
        params.put("with_mean", withMean);
        return this;
    }
    
    public boolean isWithStd() {
        return withStd;
    }
    
    public StandardScaler setWithStd(boolean withStd) {
        this.withStd = withStd;
        params.put("with_std", withStd);
        return this;
    }
}
