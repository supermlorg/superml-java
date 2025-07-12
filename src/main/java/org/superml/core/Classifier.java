package org.superml.core;

/**
 * Interface for classification algorithms.
 * Similar to sklearn.base.ClassifierMixin
 */
public interface Classifier extends SupervisedLearner {
    
    /**
     * Predict class probabilities for samples.
     * @param X features to predict on
     * @return class probabilities
     */
    double[][] predictProba(double[][] X);
    
    /**
     * Predict log-probabilities for samples.
     * @param X features to predict on
     * @return log probabilities
     */
    double[][] predictLogProba(double[][] X);
    
    /**
     * Get the unique class labels.
     * @return array of class labels
     */
    double[] getClasses();
}
