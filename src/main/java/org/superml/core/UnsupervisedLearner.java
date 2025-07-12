package org.superml.core;

/**
 * Interface for unsupervised learning algorithms.
 * Similar to sklearn.base.ClusterMixin and TransformerMixin
 */
public interface UnsupervisedLearner extends Estimator {
    
    /**
     * Fit the model to data.
     * @param X training features
     * @return this estimator instance
     */
    UnsupervisedLearner fit(double[][] X);
    
    /**
     * Transform the data.
     * @param X data to transform
     * @return transformed data
     */
    double[][] transform(double[][] X);
    
    /**
     * Fit to data, then transform it.
     * @param X data to fit and transform
     * @return transformed data
     */
    default double[][] fitTransform(double[][] X) {
        return fit(X).transform(X);
    }
}
