package org.superml.core;

/**
 * Base interface for all estimators in SuperML.
 * Similar to sklearn.base.BaseEstimator
 */
public interface Estimator {
    
    /**
     * Get parameters for this estimator.
     * @return parameter map
     */
    java.util.Map<String, Object> getParams();
    
    /**
     * Set the parameters of this estimator.
     * @param params parameter map
     * @return this estimator instance
     */
    Estimator setParams(java.util.Map<String, Object> params);
}
