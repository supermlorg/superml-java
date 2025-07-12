package com.superml.core;

/**
 * Interface for regression algorithms.
 * Similar to sklearn.base.RegressorMixin
 */
public interface Regressor extends SupervisedLearner {
    
    /**
     * Predict continuous values for samples.
     * @param X features to predict on
     * @return continuous predictions
     */
    double[] predict(double[][] X);
}
