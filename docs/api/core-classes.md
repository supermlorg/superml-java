---
title: "Core Classes API Reference"
description: "Comprehensive API documentation for SuperML Java 2.0.0 - 21 modules and 12+ algorithms"
layout: default
toc: true
search: true
---

# Core Classes API Reference

This document provides comprehensive API documentation for the core classes and interfaces in SuperML Java 2.0.0, covering all 21 modules and 12+ algorithms.

## 🏗️ Foundation Interfaces (superml-core)

### Estimator

The foundational interface that all ML components implement - provides consistent parameter management across the framework.

```java
package org.superml.core;

public interface Estimator {
    /**
     * Get all parameters of this estimator.
     * @return Map of parameter names to values
     */
    Map<String, Object> getParams();
    
    /**
     * Set parameters of this estimator.
     * @param params Map of parameter names to values
     * @return This estimator for method chaining
     */
    Estimator setParams(Map<String, Object> params);
    
    /**
     * Check if this estimator has been fitted.
     * @return true if fitted, false otherwise
     */
    boolean isFitted();
    
    /**
     * Validate parameters before training.
     * @throws IllegalArgumentException if parameters are invalid
     */
    void validateParameters();
}
```

**Usage Example:**
```java
var model = new LogisticRegression();
Map<String, Object> params = model.getParams();
model.setParams(Map.of("learningRate", 0.01, "maxIterations", 1000));
```

### SupervisedLearner

Interface for algorithms that learn from labeled data - the foundation for all classification and regression algorithms.

```java
package org.superml.core;

public interface SupervisedLearner extends Estimator {
    /**
     * Fit the model to training data.
     * @param X Training features matrix (n_samples x n_features)
     * @param y Training target values (n_samples)
     * @return This estimator for method chaining
     */
    SupervisedLearner fit(double[][] X, double[] y);
    
    /**
     * Make predictions on new data.
     * @param X Features matrix (n_samples x n_features) 
     * @return Predictions array (n_samples)
     */
    double[] predict(double[][] X);
    
    /**
     * Predict single sample.
     * @param sample Single feature vector (n_features)
     * @return Single prediction
     */
    double predict(double[] sample);
    
    /**
     * Calculate prediction score on test data.
     * @param X Test features matrix
     * @param y True target values
     * @return Model score (higher is better)
     */
    double score(double[][] X, double[] y);
}
```

### Classifier

Specialized interface for classification algorithms with probability prediction capabilities.

```java
package org.superml.core;

public interface Classifier extends SupervisedLearner {
    /**
     * Predict class probabilities.
     * @param X Features matrix (n_samples x n_features)
     * @return Probability matrix (n_samples x n_classes)
     */
    double[][] predictProba(double[][] X);
    
    /**
     * Predict class probabilities for single sample.
     * @param sample Single feature vector (n_features) 
     * @return Probability vector (n_classes)
     */
    double[] predictProba(double[] sample);
    
    /**
     * Get decision function values.
     * @param X Features matrix
     * @return Decision function values
     */
    double[][] decisionFunction(double[][] X);
    
    /**
     * Get unique class labels.
     * @return Array of class labels
     */
    double[] getClasses();
    
    /**
     * Get number of classes.
     * @return Number of classes
     */
    int getNumClasses();
}
```

### Regressor  

Specialized interface for regression algorithms.

```java
package org.superml.core;

public interface Regressor extends SupervisedLearner {
    /**
     * Calculate R² coefficient of determination.
     * @param X Test features matrix
     * @param y True target values
     * @return R² score
     */
    double r2Score(double[][] X, double[] y);
    
    /**
     * Calculate mean squared error.
     * @param X Test features matrix
     * @param y True target values
     * @return MSE value
     */
    double meanSquaredError(double[][] X, double[] y);
    
    /**
     * Calculate mean absolute error.
     * @param X Test features matrix  
     * @param y True target values
     * @return MAE value
     */
    double meanAbsoluteError(double[][] X, double[] y);
}
```

### UnsupervisedLearner

Interface for unsupervised learning algorithms like clustering.

```java
package org.superml.core;

public interface UnsupervisedLearner extends Estimator {
    /**
     * Fit the model to data.
     * @param X Training data matrix (n_samples x n_features)
     * @return This estimator for method chaining
     */
    UnsupervisedLearner fit(double[][] X);
    
    /**
     * Transform data using the fitted model.
     * @param X Data matrix to transform
     * @return Transformed data
     */
    double[][] transform(double[][] X);
    
    /**
     * Fit model and transform data in one step.
     * @param X Data matrix
     * @return Transformed data
     */
    double[][] fitTransform(double[][] X);
}
```

### BaseEstimator

Abstract base class providing common functionality for all estimators.

```java
package org.superml.core;

public abstract class BaseEstimator implements Estimator {
    protected Map<String, Object> parameters = new HashMap<>();
    protected boolean fitted = false;
    protected long trainingTime = 0;
    
    @Override
    public Map<String, Object> getParams() {
        return new HashMap<>(parameters);
    }
    
    @Override  
    public Estimator setParams(Map<String, Object> params) {
        this.parameters.putAll(params);
        validateParameters();
        return this;
    }
    
    @Override
    public boolean isFitted() {
        return fitted;
    }
    
    /**
     * Get training time in milliseconds.
     * @return Training time
     */
    public long getTrainingTime() {
        return trainingTime;
    }
    
    /**
     * Set parameter with type checking.
     * @param name Parameter name
     * @param value Parameter value
     * @param type Expected type
     * @return This estimator for chaining
     */
    protected <T> BaseEstimator setParameter(String name, T value, Class<T> type) {
        if (value != null && !type.isInstance(value)) {
            throw new IllegalArgumentException("Parameter " + name + " must be of type " + type.getSimpleName());
        }
        parameters.put(name, value);
        return this;
    }
    
    /**
     * Get parameter with type safety.
     * @param name Parameter name
     * @param type Expected type
     * @param defaultValue Default value if not set
     * @return Parameter value
     */
    protected <T> T getParameter(String name, Class<T> type, T defaultValue) {
        Object value = parameters.get(name);
        if (value == null) return defaultValue;
        if (!type.isInstance(value)) {
            throw new ClassCastException("Parameter " + name + " is not of type " + type.getSimpleName());
        }
        return type.cast(value);
    }
    
    /**
     * Validate input data dimensions.
     * @param X Feature matrix
     * @param y Target vector (can be null for unsupervised)
     */
    protected void validateInput(double[][] X, double[] y) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Input data X cannot be null or empty");
        }
        if (y != null && X.length != y.length) {
            throw new IllegalArgumentException("X and y must have same number of samples");
        }
        if (X[0].length == 0) {
            throw new IllegalArgumentException("Input data must have at least one feature");
        }
    }
}
```

## 🧮 Linear Models API (superml-linear-models)

### LogisticRegression

Advanced logistic regression with automatic multiclass support and regularization.

```java
package org.superml.linear_model;

public class LogisticRegression extends BaseEstimator implements Classifier {
    
    /**
     * Constructor with default parameters.
     */
    public LogisticRegression() {
        setDefaults();
    }
    
    /**
     * Set learning rate for gradient descent.
     * @param learningRate Learning rate (default: 0.01)
     * @return This instance for chaining
     */
    public LogisticRegression setLearningRate(double learningRate) {
        return (LogisticRegression) setParameter("learningRate", learningRate, Double.class);
    }
    
    /**
     * Set maximum number of iterations.
     * @param maxIter Maximum iterations (default: 1000)
     * @return This instance for chaining
     */
    public LogisticRegression setMaxIter(int maxIter) {
        return (LogisticRegression) setParameter("maxIter", maxIter, Integer.class);
    }
    
    /**
     * Set regularization parameter.
     * @param C Inverse regularization strength (default: 1.0)
     * @return This instance for chaining
     */
    public LogisticRegression setC(double C) {
        return (LogisticRegression) setParameter("C", C, Double.class);
    }
    
    /**
     * Set penalty type.
     * @param penalty "l1", "l2", or "elasticnet" (default: "l2")
     * @return This instance for chaining
     */
    public LogisticRegression setPenalty(String penalty) {
        return (LogisticRegression) setParameter("penalty", penalty, String.class);
    }
    
    /**
     * Set solver algorithm.
     * @param solver "newton-cg", "lbfgs", "liblinear", "sag", "saga" (default: "lbfgs")
     * @return This instance for chaining
     */
    public LogisticRegression setSolver(String solver) {
        return (LogisticRegression) setParameter("solver", solver, String.class);
    }
    
    /**
     * Set multiclass strategy.
     * @param multiClass "auto", "ovr", "multinomial" (default: "auto")
     * @return This instance for chaining
     */
    public LogisticRegression setMultiClass(String multiClass) {
        return (LogisticRegression) setParameter("multiClass", multiClass, String.class);
    }
    
    /**
     * Get model coefficients.
     * @return Coefficient matrix (n_classes x n_features) for multiclass, (n_features,) for binary
     */
    public double[][] getCoef() {
        if (!fitted) throw new IllegalStateException("Model must be fitted first");
        return coef.clone();
    }
    
    /**
     * Get intercept terms.
     * @return Intercept array (n_classes,) for multiclass, single value for binary
     */
    public double[] getIntercept() {
        if (!fitted) throw new IllegalStateException("Model must be fitted first");
        return intercept.clone();
    }
    
    // Implementation methods
    @Override
    public LogisticRegression fit(double[][] X, double[] y) { /* ... */ }
    
    @Override
    public double[] predict(double[][] X) { /* ... */ }
    
    @Override
    public double[][] predictProba(double[][] X) { /* ... */ }
    
    @Override
    public double[][] decisionFunction(double[][] X) { /* ... */ }
}
```

**Usage Example:**
```java
// Binary classification
var lr = new LogisticRegression()
    .setLearningRate(0.01)
    .setMaxIter(1000)
    .setC(1.0)
    .setPenalty("l2");

lr.fit(X_train, y_train);
double[] predictions = lr.predict(X_test);
double[][] probabilities = lr.predictProba(X_test);

// Multiclass automatically detected
var dataset = Datasets.loadIris();
lr.fit(dataset.X, dataset.y);  // Automatically uses multinomial
```

### LinearRegression

Ordinary least squares linear regression with multiple solver options.

```java
package org.superml.linear_model;

public class LinearRegression extends BaseEstimator implements Regressor {
    
    /**
     * Constructor with default parameters.
     */
    public LinearRegression() {
        setDefaults();
    }
    
    /**
     * Set whether to fit intercept.
     * @param fitIntercept Whether to fit intercept (default: true)
     * @return This instance for chaining
     */
    public LinearRegression setFitIntercept(boolean fitIntercept) {
        return (LinearRegression) setParameter("fitIntercept", fitIntercept, Boolean.class);
    }
    
    /**
     * Set solver method.
     * @param solver "normal" for normal equation, "svd" for SVD (default: "normal")
     * @return This instance for chaining
     */
    public LinearRegression setSolver(String solver) {
        return (LinearRegression) setParameter("solver", solver, String.class);
    }
    
    /**
     * Get model coefficients.
     * @return Coefficient array (n_features,)
     */
    public double[] getCoef() {
        if (!fitted) throw new IllegalStateException("Model must be fitted first");
        return coef.clone();
    }
    
    /**
     * Get intercept term.
     * @return Intercept value
     */
    public double getIntercept() {
        if (!fitted) throw new IllegalStateException("Model must be fitted first");
        return intercept;
    }
    
    @Override
    public LinearRegression fit(double[][] X, double[] y) { /* ... */ }
    
    @Override
    public double[] predict(double[][] X) { /* ... */ }
}
```

### Ridge

L2 regularized linear regression with closed-form solution.

```java
package org.superml.linear_model;

public class Ridge extends BaseEstimator implements Regressor {
    
    /**
     * Set regularization strength.
     * @param alpha Regularization strength (default: 1.0)
     * @return This instance for chaining
     */
    public Ridge setAlpha(double alpha) {
        return (Ridge) setParameter("alpha", alpha, Double.class);
    }
    
    /**
     * Set solver method.
     * @param solver "auto", "svd", "cholesky", "saga" (default: "auto")
     * @return This instance for chaining  
     */
    public Ridge setSolver(String solver) {
        return (Ridge) setParameter("solver", solver, String.class);
    }
    
    @Override
    public Ridge fit(double[][] X, double[] y) { /* ... */ }
}
```

### Lasso

L1 regularized linear regression with coordinate descent optimization.

```java
package org.superml.linear_model;

public class Lasso extends BaseEstimator implements Regressor {
    
    /**
     * Set regularization strength.
     * @param alpha Regularization strength (default: 1.0)
     * @return This instance for chaining
     */
    public Lasso setAlpha(double alpha) {
        return (Lasso) setParameter("alpha", alpha, Double.class);
    }
    
    /**
     * Set maximum iterations for coordinate descent.
     * @param maxIter Maximum iterations (default: 1000)
     * @return This instance for chaining
     */
    public Lasso setMaxIter(int maxIter) {
        return (Lasso) setParameter("maxIter", maxIter, Integer.class);
    }
    
    /**
     * Set convergence tolerance.
     * @param tol Tolerance for convergence (default: 1e-4)
     * @return This instance for chaining
     */
    public Lasso setTol(double tol) {
        return (Lasso) setParameter("tol", tol, Double.class);
    }
    
    /**
     * Get number of non-zero coefficients (sparsity).
     * @return Number of non-zero coefficients
     */
    public int getNumNonZeroCoef() {
        if (!fitted) throw new IllegalStateException("Model must be fitted first");
        return (int) Arrays.stream(coef).filter(c -> Math.abs(c) > 1e-10).count();
    }
    
    @Override
    public Lasso fit(double[][] X, double[] y) { /* ... */ }
}
```

## 🌳 Tree Models API (superml-tree-models)

### DecisionTreeClassifier

CART implementation for classification with comprehensive configuration options.

```java
package org.superml.tree;

public class DecisionTreeClassifier extends BaseEstimator implements Classifier {
    
    /**
     * Set splitting criterion.
     * @param criterion "gini" or "entropy" (default: "gini")
     * @return This instance for chaining
     */
    public DecisionTreeClassifier setCriterion(String criterion) {
        return (DecisionTreeClassifier) setParameter("criterion", criterion, String.class);
    }
    
    /**
     * Set maximum tree depth.
     * @param maxDepth Maximum depth (default: null for unlimited)
     * @return This instance for chaining
     */
    public DecisionTreeClassifier setMaxDepth(Integer maxDepth) {
        return (DecisionTreeClassifier) setParameter("maxDepth", maxDepth, Integer.class);
    }
    
    /**
     * Set minimum samples required to split a node.
     * @param minSamplesSplit Minimum samples to split (default: 2)
     * @return This instance for chaining
     */
    public DecisionTreeClassifier setMinSamplesSplit(int minSamplesSplit) {
        return (DecisionTreeClassifier) setParameter("minSamplesSplit", minSamplesSplit, Integer.class);
    }
    
    /**
     * Set minimum samples required in a leaf node.
     * @param minSamplesLeaf Minimum samples in leaf (default: 1)
     * @return This instance for chaining
     */
    public DecisionTreeClassifier setMinSamplesLeaf(int minSamplesLeaf) {
        return (DecisionTreeClassifier) setParameter("minSamplesLeaf", minSamplesLeaf, Integer.class);
    }
    
    /**
     * Get feature importances.
     * @return Feature importance array (n_features,)
     */
    public double[] getFeatureImportances() {
        if (!fitted) throw new IllegalStateException("Model must be fitted first");
        return featureImportances.clone();
    }
    
    /**
     * Get tree depth.
     * @return Actual tree depth
     */
    public int getDepth() {
        if (!fitted) throw new IllegalStateException("Model must be fitted first");
        return calculateDepth(root);
    }
    
    /**
     * Get number of leaves.
     * @return Number of leaf nodes
     */
    public int getNumLeaves() {
        if (!fitted) throw new IllegalStateException("Model must be fitted first");
        return countLeaves(root);
    }
    
    @Override
    public DecisionTreeClassifier fit(double[][] X, double[] y) { /* ... */ }
}
```

### RandomForestClassifier

Bootstrap aggregating ensemble with parallel training capabilities.

```java
package org.superml.tree;

public class RandomForestClassifier extends BaseEstimator implements Classifier {
    
    /**
     * Set number of trees in the forest.
     * @param nEstimators Number of trees (default: 100)
     * @return This instance for chaining
     */
    public RandomForestClassifier setNEstimators(int nEstimators) {
        return (RandomForestClassifier) setParameter("nEstimators", nEstimators, Integer.class);
    }
    
    /**
     * Set number of features to consider for best split.
     * @param maxFeatures "auto", "sqrt", "log2", or integer (default: "auto")
     * @return This instance for chaining
     */
    public RandomForestClassifier setMaxFeatures(Object maxFeatures) {
        return (RandomForestClassifier) setParameter("maxFeatures", maxFeatures, Object.class);
    }
    
    /**
     * Set bootstrap sampling.
     * @param bootstrap Whether to use bootstrap sampling (default: true)
     * @return This instance for chaining
     */
    public RandomForestClassifier setBootstrap(boolean bootstrap) {
        return (RandomForestClassifier) setParameter("bootstrap", bootstrap, Boolean.class);
    }
    
    /**
     * Set out-of-bag score calculation.
     * @param oobScore Whether to calculate OOB score (default: false)
     * @return This instance for chaining
     */
    public RandomForestClassifier setOobScore(boolean oobScore) {
        return (RandomForestClassifier) setParameter("oobScore", oobScore, Boolean.class);
    }
    
    /**
     * Set number of parallel jobs.
     * @param nJobs Number of parallel jobs (default: 1, -1 for all cores)
     * @return This instance for chaining
     */
    public RandomForestClassifier setNJobs(int nJobs) {
        return (RandomForestClassifier) setParameter("nJobs", nJobs, Integer.class);
    }
    
    /**
     * Get feature importances aggregated across all trees.
     * @return Feature importance array (n_features,)
     */
    public double[] getFeatureImportances() {
        if (!fitted) throw new IllegalStateException("Model must be fitted first");
        return featureImportances.clone();
    }
    
    /**
     * Get out-of-bag score if calculated.
     * @return OOB score
     */
    public double getOobScore() {
        if (!fitted) throw new IllegalStateException("Model must be fitted first");
        if (!getParameter("oobScore", Boolean.class, false)) {
            throw new IllegalStateException("OOB score not calculated. Set oobScore=true");
        }
        return oobScore;
    }
    
    /**
     * Get individual tree estimators.
     * @return List of fitted decision trees
     */
    public List<DecisionTreeClassifier> getEstimators() {
        if (!fitted) throw new IllegalStateException("Model must be fitted first");
        return new ArrayList<>(trees);
    }
    
    @Override
    public RandomForestClassifier fit(double[][] X, double[] y) { /* ... */ }
}
```

## 🔗 Clustering API (superml-clustering)

### KMeans

K-means clustering with k-means++ initialization and advanced convergence monitoring.

```java
package org.superml.cluster;

public class KMeans extends BaseEstimator implements UnsupervisedLearner {
    
    /**
     * Constructor with number of clusters.
     * @param k Number of clusters
     */
    public KMeans(int k) {
        setParameter("k", k, Integer.class);
        setDefaults();
    }
    
    /**
     * Set initialization method.
     * @param init "k-means++", "random" (default: "k-means++")
     * @return This instance for chaining
     */
    public KMeans setInit(String init) {
        return (KMeans) setParameter("init", init, String.class);
    }
    
    /**
     * Set number of random restarts.
     * @param nInit Number of initializations (default: 10)
     * @return This instance for chaining
     */
    public KMeans setNInit(int nInit) {
        return (KMeans) setParameter("nInit", nInit, Integer.class);
    }
    
    /**
     * Set maximum iterations.
     * @param maxIter Maximum iterations (default: 300)
     * @return This instance for chaining
     */
    public KMeans setMaxIter(int maxIter) {
        return (KMeans) setParameter("maxIter", maxIter, Integer.class);
    }
    
    /**
     * Set convergence tolerance.
     * @param tol Tolerance for convergence (default: 1e-4)
     * @return This instance for chaining
     */
    public KMeans setTol(double tol) {
        return (KMeans) setParameter("tol", tol, Double.class);
    }
    
    /**
     * Get cluster centers.
     * @return Cluster centers matrix (k x n_features)
     */
    public double[][] getClusterCenters() {
        if (!fitted) throw new IllegalStateException("Model must be fitted first");
        return clusterCenters.clone();
    }
    
    /**
     * Get final inertia (within-cluster sum of squared distances).
     * @return Inertia value
     */
    public double getInertia() {
        if (!fitted) throw new IllegalStateException("Model must be fitted first");
        return inertia;
    }
    
    /**
     * Get number of iterations until convergence.
     * @return Number of iterations
     */
    public int getNumIter() {
        if (!fitted) throw new IllegalStateException("Model must be fitted first");
        return numIter;
    }
    
    /**
     * Predict cluster labels for new data.
     * @param X Data matrix (n_samples x n_features)
     * @return Cluster labels (n_samples,)
     */
    public int[] predict(double[][] X) {
        if (!fitted) throw new IllegalStateException("Model must be fitted first");
        return assignToClusters(X);
    }
    
    @Override
    public KMeans fit(double[][] X) { /* ... */ }
    
    @Override
    public double[][] transform(double[][] X) { /* ... */ }
}
```

## 🎯 AutoML API (superml-autotrainer)

### AutoTrainer

Automated machine learning with intelligent algorithm selection and hyperparameter optimization.

```java
package org.superml.autotrainer;

public class AutoTrainer {
    
    /**
     * Automated ML for classification or regression with default configuration.
     * @param X Feature matrix
     * @param y Target values
     * @param taskType "classification" or "regression"
     * @return AutoML result with best model and performance
     */
    public static AutoMLResult autoML(double[][] X, double[] y, String taskType) {
        return autoMLWithConfig(X, y, new Config().setTaskType(taskType));
    }
    
    /**
     * Automated ML with custom configuration.
     * @param X Feature matrix
     * @param y Target values  
     * @param config AutoML configuration
     * @return AutoML result
     */
    public static AutoMLResult autoMLWithConfig(double[][] X, double[] y, Config config) {
        /* Implementation */
    }
    
    /**
     * Configuration class for AutoML.
     */
    public static class Config {
        private String taskType = "classification";
        private List<String> algorithms = Arrays.asList("logistic", "randomforest", "gradientboosting");
        private String searchStrategy = "grid";
        private int crossValidationFolds = 5;
        private int maxEvaluationTime = 300; // seconds
        private boolean ensembleMethods = false;
        private double testSize = 0.2;
        private int randomState = 42;
        
        public Config setTaskType(String taskType) {
            this.taskType = taskType;
            return this;
        }
        
        public Config setAlgorithms(String... algorithms) {
            this.algorithms = Arrays.asList(algorithms);
            return this;
        }
        
        public Config setSearchStrategy(String strategy) {
            this.searchStrategy = strategy;
            return this;
        }
        
        public Config setCrossValidationFolds(int folds) {
            this.crossValidationFolds = folds;
            return this;
        }
        
        public Config setMaxEvaluationTime(int seconds) {
            this.maxEvaluationTime = seconds;
            return this;
        }
        
        public Config setEnsembleMethods(boolean ensemble) {
            this.ensembleMethods = ensemble;
            return this;
        }
        
        // ... getters
    }
    
    /**
     * AutoML result containing best model and performance metrics.
     */
    public static class AutoMLResult {
        private final String bestAlgorithm;
        private final SupervisedLearner bestModel;
        private final double bestScore;
        private final Map<String, Object> bestParams;
        private final Map<String, Double> allScores;
        private final SupervisedLearner ensembleModel;
        private final double ensembleScore;
        
        public String getBestAlgorithm() { return bestAlgorithm; }
        public SupervisedLearner getBestModel() { return bestModel; }
        public double getBestScore() { return bestScore; }
        public Map<String, Object> getBestParams() { return bestParams; }
        public Map<String, Double> getAllScores() { return allScores; }
        public boolean hasEnsemble() { return ensembleModel != null; }
        public SupervisedLearner getEnsembleModel() { return ensembleModel; }
        public double getEnsembleScore() { return ensembleScore; }
    }
}
```

## 📊 Visualization API (superml-visualization)

### VisualizationFactory

Factory for creating dual-mode visualizations (XChart GUI + ASCII fallback).

```java
package org.superml.visualization;

public class VisualizationFactory {
    
    /**
     * Create dual-mode confusion matrix visualization.
     * @param yTrue True labels
     * @param yPred Predicted labels
     * @param classNames Class names for display
     * @return Dual-mode visualization
     */
    public static DualModeVisualization createDualModeConfusionMatrix(
            double[] yTrue, double[] yPred, String[] classNames) {
        return new DualModeConfusionMatrix(yTrue, yPred, classNames);
    }
    
    /**
     * Create professional XChart confusion matrix.
     * @param yTrue True labels
     * @param yPred Predicted labels
     * @param classNames Class names
     * @return XChart visualization
     */
    public static XChartVisualization createXChartConfusionMatrix(
            double[] yTrue, double[] yPred, String[] classNames) {
        return new XChartConfusionMatrix(yTrue, yPred, classNames);
    }
    
    /**
     * Create scatter plot with cluster highlighting.
     * @param X Data matrix
     * @param labels Cluster or class labels
     * @param title Chart title
     * @param xLabel X-axis label
     * @param yLabel Y-axis label
     * @return XChart scatter plot
     */
    public static XChartVisualization createXChartScatterPlot(
            double[][] X, double[] labels, String title, String xLabel, String yLabel) {
        return new XChartScatterPlot(X, labels, title, xLabel, yLabel);
    }
    
    /**
     * Create regression plot with predictions vs actual.
     * @param yTrue True values
     * @param yPred Predicted values
     * @param title Chart title
     * @return Regression plot visualization
     */
    public static DualModeVisualization createRegressionPlot(
            double[] yTrue, double[] yPred, String title) {
        return new RegressionPlot(yTrue, yPred, title);
    }
    
    /**
     * Create model performance comparison chart.
     * @param modelNames List of model names
     * @param scores List of model scores
     * @param title Chart title
     * @return Model comparison chart
     */
    public static XChartVisualization createModelComparisonChart(
            List<String> modelNames, List<Double> scores, String title) {
        return new ModelComparisonChart(modelNames, scores, title);
    }
}
```

## ⚡ Inference API (superml-inference)

### InferenceEngine

High-performance production inference engine with caching and monitoring.

```java
package org.superml.inference;

public class InferenceEngine {
    
    /**
     * Constructor with default configuration.
     */
    public InferenceEngine() {
        this.modelCache = new ModelCache();
        this.metricsCollector = new InferenceMetrics();
    }
    
    /**
     * Enable model caching.
     * @param enabled Whether to enable caching
     * @return This instance for chaining
     */
    public InferenceEngine setModelCache(boolean enabled) {
        this.cacheEnabled = enabled;
        return this;
    }
    
    /**
     * Set batch size for batch processing.
     * @param batchSize Batch size (default: 100)
     * @return This instance for chaining
     */
    public InferenceEngine setBatchSize(int batchSize) {
        this.batchSize = batchSize;
        return this;
    }
    
    /**
     * Enable performance monitoring.
     * @param enabled Whether to enable monitoring
     * @return This instance for chaining
     */
    public InferenceEngine setPerformanceMonitoring(boolean enabled) {
        this.monitoringEnabled = enabled;
        return this;
    }
    
    /**
     * Register a model for inference.
     * @param modelId Unique model identifier
     * @param model Trained model
     */
    public void registerModel(String modelId, SupervisedLearner model) {
        if (cacheEnabled) {
            modelCache.put(modelId, new LoadedModel(model, System.currentTimeMillis()));
        }
        metricsCollector.registerModel(modelId);
    }
    
    /**
     * Make predictions on batch data.
     * @param modelId Model identifier
     * @param X Feature matrix
     * @return Predictions array
     */
    public double[] predict(String modelId, double[][] X) {
        long startTime = System.nanoTime();
        
        try {
            LoadedModel loadedModel = getLoadedModel(modelId);
            double[] predictions = loadedModel.model.predict(X);
            
            if (monitoringEnabled) {
                long inferenceTime = System.nanoTime() - startTime;
                metricsCollector.recordInference(modelId, X.length, inferenceTime);
            }
            
            return predictions;
        } catch (Exception e) {
            if (monitoringEnabled) {
                metricsCollector.recordError(modelId);
            }
            throw new InferenceException("Prediction failed for model " + modelId, e);
        }
    }
    
    /**
     * Make asynchronous prediction for single sample.
     * @param modelId Model identifier
     * @param sample Feature vector
     * @return Future containing prediction
     */
    public CompletableFuture<Double> predictAsync(String modelId, double[] sample) {
        return CompletableFuture.supplyAsync(() -> {
            double[][] batchX = {sample};
            double[] predictions = predict(modelId, batchX);
            return predictions[0];
        });
    }
    
    /**
     * Get inference metrics for a model.
     * @param modelId Model identifier
     * @return Inference metrics
     */
    public InferenceMetrics.ModelMetrics getMetrics(String modelId) {
        return metricsCollector.getModelMetrics(modelId);
    }
    
    /**
     * Get last inference time in microseconds.
     * @return Last inference time
     */
    public long getLastInferenceTime() {
        return metricsCollector.getLastInferenceTime();
    }
}
```

## 💾 Persistence API (superml-persistence)

### ModelPersistence

Advanced model serialization with automatic statistics capture.

```java
package org.superml.persistence;

public class ModelPersistence {
    
    /**
     * Save model with automatic performance statistics.
     * @param model Trained model to save
     * @param modelName Model name/identifier
     * @param description Model description
     * @param X_test Test features for evaluation
     * @param y_test Test targets for evaluation
     * @return Path to saved model file
     */
    public static String saveWithStats(SupervisedLearner model, String modelName, 
                                     String description, double[][] X_test, double[] y_test) {
        ModelMetadata metadata = new ModelMetadata(modelName, description, model.getClass().getSimpleName());
        
        // Calculate performance statistics
        if (model instanceof Classifier) {
            Classifier classifier = (Classifier) model;
            double[] predictions = classifier.predict(X_test);
            var metrics = Metrics.classificationReport(y_test, predictions);
            metadata.addPerformanceMetric("accuracy", metrics.accuracy);
            metadata.addPerformanceMetric("f1_score", metrics.f1Score);
        }
        
        return save(model, modelName, metadata);
    }
    
    /**
     * Save model with metadata.
     * @param model Model to save
     * @param modelName Model name
     * @param metadata Model metadata
     * @return Path to saved model file
     */
    public static String save(SupervisedLearner model, String modelName, ModelMetadata metadata) {
        /* Implementation */
    }
    
    /**
     * Load model from file.
     * @param modelPath Path to model file
     * @return Loaded model
     */
    public static SupervisedLearner load(String modelPath) {
        /* Implementation */
    }
    
    /**
     * Load model with type safety.
     * @param modelPath Path to model file
     * @param modelClass Expected model class
     * @return Loaded model of specified type
     */
    public static <T extends SupervisedLearner> T load(String modelPath, Class<T> modelClass) {
        SupervisedLearner model = load(modelPath);
        if (!modelClass.isInstance(model)) {
            throw new ClassCastException("Loaded model is not of type " + modelClass.getSimpleName());
        }
        return modelClass.cast(model);
    }
}
```

## 📈 Metrics API (superml-metrics)

### Metrics

Comprehensive evaluation metrics for all ML tasks.

```java
package org.superml.metrics;

public class Metrics {
    
    /**
     * Calculate comprehensive classification metrics.
     * @param yTrue True labels
     * @param yPred Predicted labels
     * @return Classification metrics report
     */
    public static ClassificationReport classificationReport(double[] yTrue, double[] yPred) {
        return new ClassificationReport(yTrue, yPred);
    }
    
    /**
     * Calculate regression metrics.
     * @param yTrue True values
     * @param yPred Predicted values
     * @return Regression metrics report
     */
    public static RegressionReport regressionReport(double[] yTrue, double[] yPred) {
        return new RegressionReport(yTrue, yPred);
    }
    
    /**
     * Calculate confusion matrix.
     * @param yTrue True labels
     * @param yPred Predicted labels
     * @return Confusion matrix
     */
    public static int[][] confusionMatrix(double[] yTrue, double[] yPred) {
        /* Implementation */
    }
    
    /**
     * Classification metrics container.
     */
    public static class ClassificationReport {
        public final double accuracy;
        public final double precision;
        public final double recall;
        public final double f1Score;
        public final double[] precisionPerClass;
        public final double[] recallPerClass;
        public final double[] f1ScorePerClass;
        
        public ClassificationReport(double[] yTrue, double[] yPred) {
            /* Calculate metrics */
        }
    }
    
    /**
     * Regression metrics container.
     */
    public static class RegressionReport {
        public final double mse;
        public final double rmse;
        public final double mae;
        public final double r2Score;
        public final double adjustedR2;
        public final double mape;
        
        public RegressionReport(double[] yTrue, double[] yPred) {
            /* Calculate metrics */
        }
    }
}
```

---

This comprehensive API reference covers all major components of SuperML Java 2.0.0's 21-module architecture. Each class provides consistent, type-safe APIs with extensive configuration options and comprehensive functionality for building production-ready machine learning applications.
