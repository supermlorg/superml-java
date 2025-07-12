package com.superml.pipeline;

import com.superml.core.BaseEstimator;
import com.superml.core.Estimator;
import com.superml.core.SupervisedLearner;
import com.superml.core.UnsupervisedLearner;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Pipeline for chaining preprocessing steps and estimators.
 * Similar to sklearn.pipeline.Pipeline
 */
public class Pipeline extends BaseEstimator implements SupervisedLearner {
    
    private static final long serialVersionUID = 1L;
    
    private List<PipelineStep> steps;
    private Map<String, Estimator> namedSteps;
    private boolean fitted = false;
    
    /**
     * Container for pipeline steps.
     */
    public static class PipelineStep implements Serializable {
        private static final long serialVersionUID = 1L;
        
        public final String name;
        public final Estimator estimator;
        
        public PipelineStep(String name, Estimator estimator) {
            this.name = name;
            this.estimator = estimator;
        }
    }
    
    public Pipeline() {
        this.steps = new ArrayList<>();
        this.namedSteps = new HashMap<>();
    }
    
    /**
     * Create pipeline from list of steps.
     * @param steps list of (name, estimator) pairs
     */
    public Pipeline(List<PipelineStep> steps) {
        this();
        for (PipelineStep step : steps) {
            addStep(step.name, step.estimator);
        }
    }
    
    /**
     * Add a step to the pipeline.
     * @param name step name
     * @param estimator estimator instance
     * @return this pipeline
     */
    public Pipeline addStep(String name, Estimator estimator) {
        steps.add(new PipelineStep(name, estimator));
        namedSteps.put(name, estimator);
        return this;
    }
    
    /**
     * Get a step by name.
     * @param name step name
     * @return estimator
     */
    public Estimator getStep(String name) {
        return namedSteps.get(name);
    }
    
    /**
     * Get all step names.
     * @return list of step names
     */
    public List<String> getStepNames() {
        return new ArrayList<>(namedSteps.keySet());
    }
    
    /**
     * Get the final estimator (last step).
     * @return final estimator
     */
    public Estimator getFinalEstimator() {
        if (steps.isEmpty()) {
            throw new IllegalStateException("Pipeline is empty");
        }
        return steps.get(steps.size() - 1).estimator;
    }
    
    @Override
    public Pipeline fit(double[][] X, double[] y) {
        double[][] currentX = X;
        
        // Fit all steps except the last one
        for (int i = 0; i < steps.size() - 1; i++) {
            PipelineStep step = steps.get(i);
            Estimator estimator = step.estimator;
            
            if (estimator instanceof UnsupervisedLearner) {
                UnsupervisedLearner transformer = (UnsupervisedLearner) estimator;
                transformer.fit(currentX);
                currentX = transformer.transform(currentX);
            } else {
                throw new IllegalArgumentException("All steps except the last must be transformers");
            }
        }
        
        // Fit the final estimator
        Estimator finalEstimator = getFinalEstimator();
        if (finalEstimator instanceof SupervisedLearner) {
            ((SupervisedLearner) finalEstimator).fit(currentX, y);
        } else if (finalEstimator instanceof UnsupervisedLearner) {
            ((UnsupervisedLearner) finalEstimator).fit(currentX);
        } else {
            throw new IllegalArgumentException("Final estimator must be a learner");
        }
        
        fitted = true;
        return this;
    }
    
    @Override
    public double[] predict(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Pipeline must be fitted before making predictions");
        }
        
        double[][] currentX = transform(X);
        
        // Make predictions with final estimator
        Estimator finalEstimator = getFinalEstimator();
        if (finalEstimator instanceof SupervisedLearner) {
            return ((SupervisedLearner) finalEstimator).predict(currentX);
        } else {
            throw new IllegalArgumentException("Final estimator must support prediction");
        }
    }
    
    @Override
    public double score(double[][] X, double[] y) {
        if (!fitted) {
            throw new IllegalStateException("Pipeline must be fitted before scoring");
        }
        
        double[][] currentX = transform(X);
        
        // Score with final estimator
        Estimator finalEstimator = getFinalEstimator();
        if (finalEstimator instanceof SupervisedLearner) {
            return ((SupervisedLearner) finalEstimator).score(currentX, y);
        } else {
            throw new IllegalArgumentException("Final estimator must support scoring");
        }
    }
    
    /**
     * Transform data through all transformer steps.
     * @param X input data
     * @return transformed data
     */
    public double[][] transform(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Pipeline must be fitted before transforming");
        }
        
        double[][] currentX = X;
        
        // Apply all transformations except the final estimator
        for (int i = 0; i < steps.size() - 1; i++) {
            PipelineStep step = steps.get(i);
            Estimator estimator = step.estimator;
            
            if (estimator instanceof UnsupervisedLearner) {
                UnsupervisedLearner transformer = (UnsupervisedLearner) estimator;
                currentX = transformer.transform(currentX);
            } else {
                throw new IllegalArgumentException("All steps except the last must be transformers");
            }
        }
        
        return currentX;
    }
    
    /**
     * Fit and transform data in one step.
     * @param X input data
     * @param y target values
     * @return transformed data
     */
    public double[][] fitTransform(double[][] X, double[] y) {
        fit(X, y);
        return transform(X);
    }
    
    /**
     * Get parameters for the entire pipeline.
     * @return parameter map with step names as prefixes
     */
    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> allParams = new HashMap<>(params);
        
        for (PipelineStep step : steps) {
            Map<String, Object> stepParams = step.estimator.getParams();
            for (Map.Entry<String, Object> entry : stepParams.entrySet()) {
                String key = step.name + "__" + entry.getKey();
                allParams.put(key, entry.getValue());
            }
        }
        
        return allParams;
    }
    
    /**
     * Set parameters for pipeline steps.
     * @param params parameter map with step names as prefixes
     * @return this pipeline
     */
    @Override
    public Pipeline setParams(Map<String, Object> params) {
        Map<String, Map<String, Object>> stepParams = new HashMap<>();
        
        for (Map.Entry<String, Object> entry : params.entrySet()) {
            String key = entry.getKey();
            Object value = entry.getValue();
            
            if (key.contains("__")) {
                // Parameter for a specific step
                String[] parts = key.split("__", 2);
                String stepName = parts[0];
                String paramName = parts[1];
                
                stepParams.computeIfAbsent(stepName, k -> new HashMap<>())
                          .put(paramName, value);
            } else {
                // Parameter for the pipeline itself
                this.params.put(key, value);
            }
        }
        
        // Set parameters for each step
        for (Map.Entry<String, Map<String, Object>> entry : stepParams.entrySet()) {
            String stepName = entry.getKey();
            Map<String, Object> stepParamMap = entry.getValue();
            
            Estimator estimator = namedSteps.get(stepName);
            if (estimator != null) {
                estimator.setParams(stepParamMap);
            }
        }
        
        return this;
    }
    
    /**
     * Set a parameter for a specific step.
     * @param stepName name of the step
     * @param paramName parameter name
     * @param value parameter value
     * @return this pipeline
     */
    public Pipeline setStepParam(String stepName, String paramName, Object value) {
        Estimator estimator = namedSteps.get(stepName);
        if (estimator == null) {
            throw new IllegalArgumentException("Step '" + stepName + "' not found");
        }
        
        Map<String, Object> stepParams = new HashMap<>();
        stepParams.put(paramName, value);
        estimator.setParams(stepParams);
        
        return this;
    }
    
    /**
     * Get a parameter for a specific step.
     * @param stepName name of the step
     * @param paramName parameter name
     * @return parameter value
     */
    public Object getStepParam(String stepName, String paramName) {
        Estimator estimator = namedSteps.get(stepName);
        if (estimator == null) {
            throw new IllegalArgumentException("Step '" + stepName + "' not found");
        }
        
        return estimator.getParams().get(paramName);
    }
    
    /**
     * Check if the pipeline is empty.
     * @return true if empty
     */
    public boolean isEmpty() {
        return steps.isEmpty();
    }
    
    /**
     * Get the number of steps.
     * @return number of steps
     */
    public int size() {
        return steps.size();
    }
    
    /**
     * Get a copy of all steps.
     * @return list of pipeline steps
     */
    public List<PipelineStep> getSteps() {
        return new ArrayList<>(steps);
    }
    
    /**
     * Create a string representation of the pipeline.
     * @return pipeline description
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("Pipeline(");
        
        for (int i = 0; i < steps.size(); i++) {
            PipelineStep step = steps.get(i);
            sb.append(step.name).append("=").append(step.estimator.getClass().getSimpleName());
            
            if (i < steps.size() - 1) {
                sb.append(", ");
            }
        }
        
        sb.append(")");
        return sb.toString();
    }
}
