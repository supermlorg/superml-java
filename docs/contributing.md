# Contributing Guide

Thank you for your interest in contributing to SuperML Java! This guide will help you get started with contributing to the project.

## 🎯 Ways to Contribute

### 1. Code Contributions
- **New Algorithms**: Implement additional ML algorithms
- **Performance Improvements**: Optimize existing algorithms
- **Bug Fixes**: Fix issues and improve stability
- **Features**: Add new functionality and capabilities

### 2. Documentation
- **API Documentation**: Improve method and class documentation
- **Tutorials**: Create guides and examples
- **Wiki Pages**: Add comprehensive documentation
- **Code Comments**: Improve code readability

### 3. Testing
- **Unit Tests**: Add tests for new features
- **Integration Tests**: Test component interactions
- **Performance Tests**: Benchmark and profiling
- **Edge Case Testing**: Test boundary conditions

### 4. Quality Assurance
- **Code Review**: Review pull requests
- **Issue Reporting**: Report bugs and improvements
- **Feature Requests**: Suggest new functionality
- **Performance Analysis**: Identify bottlenecks

## 🚀 Getting Started

### 1. Development Environment Setup

```bash
# Clone the repository
git clone https://github.com/superml/superml-java.git
cd superml-java

# Build the project
mvn clean compile

# Run tests
mvn test

# Generate documentation
mvn javadoc:javadoc
```

### 2. Project Structure

```
superml-java/
├── src/main/java/com/superml/
│   ├── core/                    # Base interfaces and classes
│   ├── linear_model/           # Linear algorithms
│   ├── cluster/                # Clustering algorithms
│   ├── preprocessing/          # Data preprocessing
│   ├── metrics/               # Evaluation metrics
│   ├── model_selection/       # Cross-validation and tuning
│   ├── pipeline/              # ML workflows
│   └── datasets/              # Data loading and Kaggle integration
├── src/test/java/              # Test files
├── docs/                      # Documentation
├── examples/                  # Usage examples
└── pom.xml                   # Maven configuration
```

### 3. Code Style Guidelines

#### Java Conventions
- Use **camelCase** for variables and methods
- Use **PascalCase** for classes and interfaces
- Use **ALL_CAPS** for constants
- Maximum line length: **120 characters**
- Indentation: **4 spaces** (no tabs)

#### Method Naming
```java
// Good: Descriptive and follows conventions
public double calculateMeanSquaredError(double[] actual, double[] predicted)

// Bad: Unclear abbreviations
public double calcMSE(double[] a, double[] p)
```

#### Documentation
```java
/**
 * Trains a logistic regression model using gradient descent.
 * 
 * @param X Feature matrix with shape (n_samples, n_features)
 * @param y Target values with shape (n_samples,)
 * @return This estimator for method chaining
 * @throws IllegalArgumentException if X and y have incompatible shapes
 */
public LogisticRegression fit(double[][] X, double[] y) {
    // Implementation
}
```

## 🧪 Testing Guidelines

### 1. Test Structure

Create tests in the corresponding test package:

```
src/test/java/com/superml/
├── core/                    # Core interface tests
├── linear_model/           # Algorithm tests
│   ├── LogisticRegressionTest.java
│   └── LinearRegressionTest.java
└── utils/                  # Test utilities
    └── TestDatasets.java
```

### 2. Test Categories

#### Unit Tests
Test individual methods and components:

```java
@Test
void testFitWithValidData() {
    // Arrange
    double[][] X = {{1, 2}, {3, 4}, {5, 6}};
    double[] y = {0, 1, 0};
    var model = new LogisticRegression();
    
    // Act
    model.fit(X, y);
    
    // Assert
    assertTrue(model.isFitted());
    assertNotNull(model.getCoefficients());
}

@Test
void testPredictThrowsWhenNotFitted() {
    // Arrange
    double[][] X = {{1, 2}, {3, 4}};
    var model = new LogisticRegression();
    
    // Act & Assert
    assertThrows(ModelNotFittedException.class, () -> model.predict(X));
}
```

#### Integration Tests
Test component interactions:

```java
@Test
void testPipelineWithScalerAndClassifier() {
    // Arrange
    var dataset = TestDatasets.makeClassification(100, 5, 2);
    var pipeline = new Pipeline()
        .addStep("scaler", new StandardScaler())
        .addStep("classifier", new LogisticRegression());
    
    // Act
    pipeline.fit(dataset.X, dataset.y);
    double[] predictions = pipeline.predict(dataset.X);
    
    // Assert
    assertEquals(dataset.X.length, predictions.length);
    double accuracy = Metrics.accuracy(dataset.y, predictions);
    assertTrue(accuracy > 0.8, "Pipeline should achieve reasonable accuracy");
}
```

#### Performance Tests
Test algorithm efficiency:

```java
@Test
void testTrainingPerformance() {
    // Large dataset
    var dataset = TestDatasets.makeClassification(10000, 20, 2);
    var model = new LogisticRegression();
    
    long startTime = System.currentTimeMillis();
    model.fit(dataset.X, dataset.y);
    long trainingTime = System.currentTimeMillis() - startTime;
    
    // Should complete within reasonable time
    assertTrue(trainingTime < 5000, "Training should complete within 5 seconds");
}
```

### 3. Test Utilities

Create reusable test utilities:

```java
public class TestDatasets {
    public static Dataset makeClassification(int samples, int features, int classes) {
        return makeClassification(samples, features, classes, 42);
    }
    
    public static Dataset makeClassification(int samples, int features, int classes, int seed) {
        Random random = new Random(seed);
        double[][] X = new double[samples][features];
        double[] y = new double[samples];
        
        // Generate synthetic data
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < features; j++) {
                X[i][j] = random.nextGaussian();
            }
            y[i] = random.nextInt(classes);
        }
        
        return new Dataset(X, y);
    }
}
```

## 🔧 Adding New Algorithms

### 1. Algorithm Implementation Template

```java
package com.superml.linear_model;

import com.superml.core.BaseEstimator;
import com.superml.core.Classifier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * My New Algorithm implementation.
 * 
 * This algorithm does X by using technique Y.
 * 
 * Parameters:
 * - parameter1: Description of parameter1 (default: defaultValue)
 * - parameter2: Description of parameter2 (default: defaultValue)
 * 
 * Example:
 * <pre>
 * var model = new MyNewAlgorithm()
 *     .setParameter1(value1)
 *     .setParameter2(value2);
 * model.fit(X, y);
 * double[] predictions = model.predict(X_test);
 * </pre>
 */
public class MyNewAlgorithm extends BaseEstimator implements Classifier {
    
    private static final Logger logger = LoggerFactory.getLogger(MyNewAlgorithm.class);
    
    // Model parameters (learned during training)
    private double[] weights;
    private double bias;
    private double[] classes;
    
    // Hyperparameters
    private double parameter1 = 1.0;
    private int parameter2 = 100;
    private double tolerance = 1e-6;
    private int maxIterations = 1000;
    private int randomState = -1;
    
    // Construction and configuration
    public MyNewAlgorithm() {
        // Initialize parameters map for base class
        parameters.put("parameter1", parameter1);
        parameters.put("parameter2", parameter2);
        parameters.put("tolerance", tolerance);
        parameters.put("maxIterations", maxIterations);
        parameters.put("randomState", randomState);
    }
    
    // Fluent interface methods
    public MyNewAlgorithm setParameter1(double parameter1) {
        this.parameter1 = parameter1;
        parameters.put("parameter1", parameter1);
        return this;
    }
    
    public MyNewAlgorithm setParameter2(int parameter2) {
        this.parameter2 = parameter2;
        parameters.put("parameter2", parameter2);
        return this;
    }
    
    // Additional fluent methods...
    
    @Override
    protected void updateInternalParameters() {
        Object p1 = parameters.get("parameter1");
        if (p1 != null) this.parameter1 = ((Number) p1).doubleValue();
        
        Object p2 = parameters.get("parameter2");
        if (p2 != null) this.parameter2 = ((Number) p2).intValue();
        
        // Update other parameters...
        
        validateParameters();
    }
    
    private void validateParameters() {
        if (parameter1 <= 0) {
            throw new IllegalArgumentException("parameter1 must be positive");
        }
        if (parameter2 <= 0) {
            throw new IllegalArgumentException("parameter2 must be positive");
        }
    }
    
    @Override
    public MyNewAlgorithm fit(double[][] X, double[] y) {
        validateInput(X, y);
        validateParameters();
        
        logger.info("Training {} with {} samples and {} features", 
            getClass().getSimpleName(), X.length, X[0].length);
        
        // Initialize model state
        initializeModel(X, y);
        
        // Main training algorithm
        trainModel(X, y);
        
        this.fitted = true;
        logger.info("Training completed in {} iterations", /* actual iterations */);
        
        return this;
    }
    
    @Override
    public double[] predict(double[][] X) {
        checkFitted();
        validateInput(X);
        
        double[] predictions = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = predictSample(X[i]);
        }
        
        return predictions;
    }
    
    @Override
    public double[][] predictProba(double[][] X) {
        checkFitted();
        validateInput(X);
        
        double[][] probabilities = new double[X.length][classes.length];
        for (int i = 0; i < X.length; i++) {
            probabilities[i] = predictProbaSample(X[i]);
        }
        
        return probabilities;
    }
    
    @Override
    public double[] getClasses() {
        checkFitted();
        return Arrays.copyOf(classes, classes.length);
    }
    
    // Algorithm-specific methods
    private void initializeModel(double[][] X, double[] y) {
        int features = X[0].length;
        this.weights = new double[features];
        this.bias = 0.0;
        this.classes = findUniqueClasses(y);
        
        // Initialize weights (e.g., random or zeros)
        if (randomState != -1) {
            Random random = new Random(randomState);
            for (int i = 0; i < weights.length; i++) {
                weights[i] = random.nextGaussian() * 0.01;
            }
        }
    }
    
    private void trainModel(double[][] X, double[] y) {
        // Main training loop
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            double previousLoss = computeLoss(X, y);
            
            // Update parameters (gradient descent, etc.)
            updateParameters(X, y);
            
            // Check convergence
            double currentLoss = computeLoss(X, y);
            if (Math.abs(previousLoss - currentLoss) < tolerance) {
                logger.debug("Converged after {} iterations", iteration + 1);
                break;
            }
        }
    }
    
    private double predictSample(double[] sample) {
        double logit = bias;
        for (int i = 0; i < weights.length; i++) {
            logit += weights[i] * sample[i];
        }
        return logit > 0 ? 1.0 : 0.0;  // Simple threshold
    }
    
    private double[] predictProbaSample(double[] sample) {
        double logit = bias;
        for (int i = 0; i < weights.length; i++) {
            logit += weights[i] * sample[i];
        }
        
        double prob1 = 1.0 / (1.0 + Math.exp(-logit));
        return new double[]{1.0 - prob1, prob1};
    }
    
    // Utility methods
    private double[] findUniqueClasses(double[] y) {
        return Arrays.stream(y).distinct().sorted().toArray();
    }
    
    private double computeLoss(double[][] X, double[] y) {
        // Implement loss function
        return 0.0;
    }
    
    private void updateParameters(double[][] X, double[] y) {
        // Implement parameter update (gradient computation, etc.)
    }
    
    // Getters for inspecting trained model
    public double[] getWeights() {
        checkFitted();
        return Arrays.copyOf(weights, weights.length);
    }
    
    public double getBias() {
        checkFitted();
        return bias;
    }
    
    @Override
    public String toString() {
        return String.format("%s(parameter1=%.3f, parameter2=%d, maxIterations=%d)", 
            getClass().getSimpleName(), parameter1, parameter2, maxIterations);
    }
}
```

### 2. Algorithm Test Template

```java
package com.superml.linear_model;

import com.superml.datasets.TestDatasets;
import com.superml.metrics.Metrics;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

class MyNewAlgorithmTest {
    
    private MyNewAlgorithm algorithm;
    private double[][] X;
    private double[] y;
    
    @BeforeEach
    void setUp() {
        algorithm = new MyNewAlgorithm();
        var dataset = TestDatasets.makeClassification(100, 5, 2, 42);
        X = dataset.X;
        y = dataset.y;
    }
    
    @Test
    void testDefaultConstruction() {
        assertNotNull(algorithm);
        assertFalse(algorithm.isFitted());
    }
    
    @Test
    void testParameterManagement() {
        algorithm.setParameter1(2.0).setParameter2(200);
        
        Map<String, Object> params = algorithm.getParams();
        assertEquals(2.0, (Double) params.get("parameter1"), 1e-6);
        assertEquals(200, (Integer) params.get("parameter2"));
    }
    
    @Test
    void testFitAndPredict() {
        algorithm.fit(X, y);
        
        assertTrue(algorithm.isFitted());
        double[] predictions = algorithm.predict(X);
        assertEquals(X.length, predictions.length);
        
        // Check predictions are valid class labels
        double[] classes = algorithm.getClasses();
        for (double pred : predictions) {
            assertTrue(Arrays.stream(classes).anyMatch(c -> c == pred));
        }
    }
    
    @Test
    void testPredictProba() {
        algorithm.fit(X, y);
        double[][] probabilities = algorithm.predictProba(X);
        
        assertEquals(X.length, probabilities.length);
        assertEquals(2, probabilities[0].length);  // Binary classification
        
        // Check probabilities sum to 1
        for (double[] probs : probabilities) {
            double sum = Arrays.stream(probs).sum();
            assertEquals(1.0, sum, 1e-6);
            
            // Check all probabilities are valid
            for (double prob : probs) {
                assertTrue(prob >= 0.0 && prob <= 1.0);
            }
        }
    }
    
    @Test
    void testPerformanceOnSyntheticData() {
        algorithm.fit(X, y);
        double[] predictions = algorithm.predict(X);
        double accuracy = Metrics.accuracy(y, predictions);
        
        // Should achieve reasonable accuracy on synthetic data
        assertTrue(accuracy > 0.7, "Algorithm should achieve > 70% accuracy");
    }
    
    @Test
    void testParameterValidation() {
        assertThrows(IllegalArgumentException.class, 
            () -> algorithm.setParameter1(-1.0));
        assertThrows(IllegalArgumentException.class, 
            () -> algorithm.setParameter2(0));
    }
    
    @Test
    void testInputValidation() {
        // Test null inputs
        assertThrows(IllegalArgumentException.class, 
            () -> algorithm.fit(null, y));
        assertThrows(IllegalArgumentException.class, 
            () -> algorithm.fit(X, null));
        
        // Test mismatched dimensions
        double[] wrongY = new double[X.length + 1];
        assertThrows(IllegalArgumentException.class, 
            () -> algorithm.fit(X, wrongY));
    }
    
    @Test
    void testUnfittedModelThrows() {
        assertThrows(ModelNotFittedException.class, 
            () -> algorithm.predict(X));
        assertThrows(ModelNotFittedException.class, 
            () -> algorithm.predictProba(X));
        assertThrows(ModelNotFittedException.class, 
            () -> algorithm.getClasses());
    }
    
    @Test
    void testConvergence() {
        algorithm.setMaxIterations(10).setTolerance(1e-3);
        algorithm.fit(X, y);
        
        // Should still work with limited iterations
        double[] predictions = algorithm.predict(X);
        assertNotNull(predictions);
    }
}
```

## 📋 Pull Request Process

### 1. Before Submitting

- [ ] **Code compiles** without warnings
- [ ] **All tests pass** (`mvn test`)
- [ ] **New features have tests** with good coverage
- [ ] **Documentation is updated** for new features
- [ ] **Code follows style guidelines**
- [ ] **Commit messages are clear** and descriptive

### 2. Pull Request Template

```markdown
## Description
Brief description of the changes and their purpose.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All existing tests pass
- [ ] Manual testing completed

## Performance Impact
- [ ] No performance impact
- [ ] Performance improved
- [ ] Performance impact acceptable (explain below)

## Documentation
- [ ] JavaDoc updated
- [ ] README updated
- [ ] Wiki/docs updated
- [ ] Examples added/updated

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Code compiles without warnings
- [ ] Meaningful commit messages
```

### 3. Review Process

1. **Automated checks** must pass (build, tests, style)
2. **Code review** by at least one maintainer
3. **Documentation review** for user-facing changes
4. **Performance review** for algorithm changes
5. **Final approval** and merge

## 🎯 Development Best Practices

### 1. Algorithm Development

- **Start with tests**: Write failing tests first (TDD)
- **Use synthetic data**: Create reproducible test cases
- **Benchmark performance**: Compare against reference implementations
- **Document complexity**: Include time/space complexity in JavaDoc
- **Handle edge cases**: Empty data, single samples, etc.

### 2. Code Quality

- **Single responsibility**: Each class should have one purpose
- **Immutable when possible**: Prefer immutable data structures
- **Fail fast**: Validate inputs early and clearly
- **Defensive copying**: Protect internal state from modification
- **Resource management**: Use try-with-resources for I/O

### 3. Testing Philosophy

- **Test behavior, not implementation**: Focus on public API
- **Use meaningful test names**: Test names should describe the scenario
- **Arrange-Act-Assert**: Structure tests clearly
- **Test edge cases**: Null inputs, empty data, boundary conditions
- **Performance tests**: Ensure algorithms scale appropriately

## 🏆 Recognition

Contributors will be recognized in:

- **README**: List of contributors
- **Release notes**: Acknowledgment of contributions
- **Documentation**: Author attribution for major features
- **GitHub**: Contributor graphs and statistics

## 🤝 Community Guidelines

- **Be respectful**: Treat all contributors with respect
- **Be constructive**: Provide helpful feedback and suggestions
- **Be patient**: Reviews take time, especially for complex changes
- **Ask questions**: Don't hesitate to ask for clarification
- **Help others**: Review pull requests and answer questions

Thank you for contributing to SuperML Java! Your contributions help make machine learning more accessible to the Java community. 🚀
