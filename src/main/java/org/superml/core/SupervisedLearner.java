package org.superml.core;

/**
 * Interface for supervised learning algorithms.
 * Similar to sklearn.base.ClassifierMixin and RegressorMixin
 */
public interface SupervisedLearner extends Estimator {
    
    /**
     * Fit the model to training data.
     * @param X training features
     * @param y training targets
     * @return this estimator instance
     */
    SupervisedLearner fit(double[][] X, double[] y);
    
    /**
     * Make predictions on new data.
     * @param X features to predict on
     * @return predictions
     */
    double[] predict(double[][] X);
    
    /**
     * Return the coefficient of determination R^2 of the prediction.
     * @param X test features
     * @param y true targets
     * @return R^2 score
     */
    double score(double[][] X, double[] y);
}
