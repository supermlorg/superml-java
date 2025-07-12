# Core Classes API Reference

This document provides comprehensive API documentation for the core classes and interfaces in SuperML Java.

## 🏗️ Base Interfaces

### Estimator

The foundational interface that all ML components implement.

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
}
```

**Usage Example:**
```java
var model = new LogisticRegression();
Map<String, Object> params = model.getParams();
model.setParams(Map.of("learningRate", 0.01, "maxIterations", 1000));
```

### SupervisedLearner

Interface for algorithms that learn from labeled data.

```java
package org.superml.core;

public interface SupervisedLearner extends Estimator {
    /**
     * Train the model on training data.
     * @param X Feature matrix (samples x features)
     * @param y Target values
     * @return This estimator for method chaining
     */
    SupervisedLearner fit(double[][] X, double[] y);
    
    /**
     * Make predictions on new data.
     * @param X Feature matrix (samples x features)
     * @return Predicted values
     */
    double[] predict(double[][] X);
    
    /**
     * Check if the model has been fitted.
     * @return true if fit() has been called
     */
    default boolean isFitted() {
        return true; // Implementations should override
    }
}
```

**Usage Example:**
```java
SupervisedLearner learner = new LogisticRegression();
learner.fit(trainingX, trainingY);
double[] predictions = learner.predict(testX);
```

### Classifier

Specialized interface for classification algorithms.

```java
package org.superml.core;

public interface Classifier extends SupervisedLearner {
    /**
     * Predict class probabilities.
     * @param X Feature matrix (samples x features)
     * @return Probability matrix (samples x classes)
     */
    double[][] predictProba(double[][] X);
    
    /**
     * Get the classes this classifier can predict.
     * @return Array of class labels
     */
    double[] getClasses();
}
```

**Usage Example:**
```java
Classifier classifier = new LogisticRegression();
classifier.fit(X, y);
double[][] probabilities = classifier.predictProba(X_test);
double[] classes = classifier.getClasses();
```

### Regressor

Interface for regression algorithms.

```java
package org.superml.core;

public interface Regressor extends SupervisedLearner {
    // Inherits fit() and predict() from SupervisedLearner
    // No additional methods required for basic regression
}
```

### UnsupervisedLearner

Interface for algorithms that learn from unlabeled data.

```java
package org.superml.core;

public interface UnsupervisedLearner extends Estimator {
    /**
     * Learn patterns from unlabeled data.
     * @param X Feature matrix (samples x features)
     * @return This estimator for method chaining
     */
    UnsupervisedLearner fit(double[][] X);
    
    /**
     * Transform or predict on new data.
     * @param X Feature matrix (samples x features)
     * @return Transformed data or cluster assignments
     */
    double[] transform(double[][] X);
    
    /**
     * Fit and transform in one step.
     * @param X Feature matrix (samples x features)
     * @return Transformed data or cluster assignments
     */
    default double[] fitTransform(double[][] X) {
        fit(X);
        return transform(X);
    }
}
```

## 🎯 Abstract Base Classes

### BaseEstimator

Abstract base class providing common parameter management functionality.

```java
package org.superml.core;

public abstract class BaseEstimator implements Estimator {
    protected Map<String, Object> parameters = new HashMap<>();
    protected boolean fitted = false;
    
    @Override
    public Map<String, Object> getParams() {
        return new HashMap<>(parameters);
    }
    
    @Override
    public Estimator setParams(Map<String, Object> params) {
        this.parameters.putAll(params);
        updateInternalParameters();
        return this;
    }
    
    /**
     * Update internal state when parameters change.
     * Subclasses should override to sync parameters.
     */
    protected void updateInternalParameters() {
        // Default implementation does nothing
    }
    
    /**
     * Validate that the model has been fitted.
     * @throws ModelNotFittedException if not fitted
     */
    protected void checkFitted() {
        if (!fitted) {
            throw new ModelNotFittedException(
                "Model must be fitted before making predictions");
        }
    }
    
    /**
     * Validate input data dimensions and content.
     * @param X Feature matrix
     * @throws IllegalArgumentException if invalid
     */
    protected void validateInput(double[][] X) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Input data cannot be null or empty");
        }
        
        int features = X[0].length;
        for (int i = 1; i < X.length; i++) {
            if (X[i].length != features) {
                throw new IllegalArgumentException(
                    "All samples must have the same number of features");
            }
        }
    }
    
    /**
     * Validate training data.
     * @param X Feature matrix
     * @param y Target values
     * @throws IllegalArgumentException if invalid
     */
    protected void validateInput(double[][] X, double[] y) {
        validateInput(X);
        if (y == null || y.length != X.length) {
            throw new IllegalArgumentException(
                "Target values must have same length as number of samples");
        }
    }
}
```

**Usage Example:**
```java
public class MyAlgorithm extends BaseEstimator implements Regressor {
    private double learningRate = 0.01;
    
    @Override
    protected void updateInternalParameters() {
        Object lr = parameters.get("learningRate");
        if (lr != null) {
            this.learningRate = ((Number) lr).doubleValue();
        }
    }
    
    @Override
    public MyAlgorithm fit(double[][] X, double[] y) {
        validateInput(X, y);
        // Training logic here
        this.fitted = true;
        return this;
    }
    
    @Override
    public double[] predict(double[][] X) {
        checkFitted();
        validateInput(X);
        // Prediction logic here
        return predictions;
    }
}
```

## 🔧 Parameter Management Patterns

### Fluent Interface Pattern

All estimators support method chaining for easy configuration:

```java
var model = new LogisticRegression()
    .setMaxIterations(1000)
    .setLearningRate(0.01)
    .setTolerance(1e-6)
    .setRandomState(42);
```

### Parameter Dictionary Pattern

Parameters can be set from maps for programmatic configuration:

```java
Map<String, Object> config = Map.of(
    "maxIterations", 1500,
    "learningRate", 0.001,
    "tolerance", 1e-8
);

model.setParams(config);
```

### Parameter Validation

Implementing custom parameter validation:

```java
public class ValidatedEstimator extends BaseEstimator {
    @Override
    protected void updateInternalParameters() {
        super.updateInternalParameters();
        validateParameters();
    }
    
    private void validateParameters() {
        Object lr = parameters.get("learningRate");
        if (lr != null) {
            double learningRate = ((Number) lr).doubleValue();
            if (learningRate <= 0 || learningRate > 1) {
                throw new IllegalArgumentException(
                    "Learning rate must be between 0 and 1");
            }
        }
    }
}
```

## 🛡️ Error Handling

### Custom Exceptions

```java
package org.superml.core;

/**
 * Base exception for all SuperML errors.
 */
public class SuperMLException extends RuntimeException {
    public SuperMLException(String message) {
        super(message);
    }
    
    public SuperMLException(String message, Throwable cause) {
        super(message, cause);
    }
}

/**
 * Thrown when operations are attempted on unfitted models.
 */
public class ModelNotFittedException extends SuperMLException {
    public ModelNotFittedException() {
        super("Model must be fitted before making predictions");
    }
    
    public ModelNotFittedException(String message) {
        super(message);
    }
}

/**
 * Thrown when model convergence fails.
 */
public class ConvergenceException extends SuperMLException {
    public ConvergenceException(String message) {
        super(message);
    }
}
```

### Validation Utilities

```java
package org.superml.core;

public class ValidationUtils {
    /**
     * Validate feature matrix.
     */
    public static void validateFeatureMatrix(double[][] X) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Feature matrix cannot be null or empty");
        }
        
        int features = X[0].length;
        if (features == 0) {
            throw new IllegalArgumentException("Feature matrix must have at least one feature");
        }
        
        for (int i = 0; i < X.length; i++) {
            if (X[i].length != features) {
                throw new IllegalArgumentException(
                    String.format("Sample %d has %d features, expected %d", 
                        i, X[i].length, features));
            }
            
            for (int j = 0; j < features; j++) {
                if (Double.isNaN(X[i][j]) || Double.isInfinite(X[i][j])) {
                    throw new IllegalArgumentException(
                        String.format("Invalid value at position [%d,%d]: %f", 
                            i, j, X[i][j]));
                }
            }
        }
    }
    
    /**
     * Validate target values.
     */
    public static void validateTargetValues(double[] y) {
        if (y == null || y.length == 0) {
            throw new IllegalArgumentException("Target values cannot be null or empty");
        }
        
        for (int i = 0; i < y.length; i++) {
            if (Double.isNaN(y[i]) || Double.isInfinite(y[i])) {
                throw new IllegalArgumentException(
                    String.format("Invalid target value at position %d: %f", i, y[i]));
            }
        }
    }
    
    /**
     * Validate that X and y have compatible dimensions.
     */
    public static void validateXy(double[][] X, double[] y) {
        validateFeatureMatrix(X);
        validateTargetValues(y);
        
        if (X.length != y.length) {
            throw new IllegalArgumentException(
                String.format("Number of samples in X (%d) must match length of y (%d)", 
                    X.length, y.length));
        }
    }
}
```

## 🔄 Lifecycle Management

### Model State

Models follow a clear lifecycle:

1. **Created**: Model is instantiated with default parameters
2. **Configured**: Parameters are set via fluent interface or parameter maps
3. **Fitted**: Model is trained on data via `fit()`
4. **Ready**: Model can make predictions via `predict()`

```java
// 1. Created
var model = new LogisticRegression();

// 2. Configured  
model.setMaxIterations(1000).setLearningRate(0.01);

// 3. Fitted
model.fit(X_train, y_train);

// 4. Ready
double[] predictions = model.predict(X_test);
```

### Thread Safety

- **Parameter Management**: Not thread-safe during configuration
- **Training**: Not thread-safe during `fit()`
- **Prediction**: Thread-safe for `predict()` after fitting
- **Recommendation**: Create separate instances for concurrent use

```java
// Thread-safe prediction usage
var model = new LogisticRegression();
model.fit(X_train, y_train);  // Single-threaded training

// Now safe for concurrent prediction
CompletableFuture<double[]> future1 = CompletableFuture.supplyAsync(
    () -> model.predict(X_test1));
CompletableFuture<double[]> future2 = CompletableFuture.supplyAsync(
    () -> model.predict(X_test2));
```

## 🎯 Best Practices

### 1. Always Validate Input

```java
@Override
public MyEstimator fit(double[][] X, double[] y) {
    ValidationUtils.validateXy(X, y);
    // Training logic
    return this;
}
```

### 2. Use Fluent Interfaces

```java
// Good: Method chaining
var model = new Algorithm()
    .setParam1(value1)
    .setParam2(value2)
    .fit(X, y);

// Less preferred: Separate calls
var model = new Algorithm();
model.setParam1(value1);
model.setParam2(value2);
model.fit(X, y);
```

### 3. Implement Proper toString()

```java
@Override
public String toString() {
    return String.format("%s(learningRate=%.3f, maxIterations=%d)", 
        getClass().getSimpleName(), learningRate, maxIterations);
}
```

### 4. Handle Edge Cases

```java
@Override
public double[] predict(double[][] X) {
    checkFitted();
    validateInput(X);
    
    if (X.length == 0) {
        return new double[0];  // Empty input, empty output
    }
    
    // Normal prediction logic
    return predictions;
}
```

### 5. Use Defensive Copying

```java
@Override
public double[] getCoefficients() {
    checkFitted();
    return Arrays.copyOf(coefficients, coefficients.length);
}
```

This API design ensures consistency, safety, and ease of use across all SuperML Java components while following established patterns from the broader Java and ML communities.
