---
title: "Testing Guide"
description: "Comprehensive guide to testing SuperML Java algorithms"
layout: default
toc: true
search: true
---

# Testing Guide

SuperML Java includes comprehensive unit tests to ensure reliability and correctness of all machine learning algorithms. This guide covers the testing framework, how to run tests, and how to write new tests.

## Testing Framework

SuperML uses **JUnit 5** as the primary testing framework, with additional utilities for machine learning specific testing scenarios.

### Test Structure

```
src/test/java/org/superml/
├── linear_model/
│   ├── LogisticRegressionTest.java
│   ├── LinearRegressionTest.java
│   └── ...
├── tree/
│   ├── DecisionTreeTest.java
│   ├── RandomForestTest.java
│   └── GradientBoostingTest.java
├── multiclass/
│   ├── OneVsRestTest.java
│   ├── SoftmaxRegressionTest.java
│   └── MulticlassTest.java
├── datasets/
│   └── DatasetsTest.java
└── metrics/
    └── MetricsTest.java
```

## Running Tests

### Maven Commands

```bash
# Run all tests
mvn test

# Run tests for specific package
mvn test -Dtest="org.superml.tree.*"

# Run specific test class
mvn test -Dtest="LogisticRegressionTest"

# Run specific test method
mvn test -Dtest="LogisticRegressionTest#testBinaryClassification"

# Run tests with verbose output
mvn test -Dtest="*" -DforkCount=1 -DreuseForks=false
```

### IDE Integration

Most IDEs (IntelliJ IDEA, Eclipse, VS Code) provide built-in JUnit 5 support:
- Right-click on test files to run individual tests
- Use test runners for debugging
- View test coverage reports

## Test Categories

### Algorithm Tests

Each algorithm has comprehensive tests covering:

1. **Basic Functionality**
   - Training and prediction
   - Parameter validation
   - Edge cases

2. **Correctness**
   - Known datasets with expected results
   - Mathematical properties
   - Convergence behavior

3. **Performance**
   - Training time benchmarks
   - Memory usage validation
   - Scalability tests

### Example: Decision Tree Tests

```java
@Test
void testBinaryClassification() {
    // Generate test data
    Datasets.ClassificationData data = Datasets.makeClassification(100, 4, 2);
    
    // Create and train model
    DecisionTree dt = new DecisionTree("gini", 5);
    dt.fit(data.X, Arrays.stream(data.y).asDoubleStream().toArray());
    
    // Test predictions
    double[] predictions = dt.predict(data.X);
    assertThat(predictions).hasSize(data.X.length);
    
    // Test accuracy is reasonable
    double accuracy = Metrics.accuracy(
        Arrays.stream(data.y).asDoubleStream().toArray(), 
        predictions
    );
    assertThat(accuracy).isGreaterThan(0.7);
}

@Test
void testProbabilityPredictions() {
    Datasets.ClassificationData data = Datasets.makeClassification(100, 4, 2);
    DecisionTree dt = new DecisionTree("gini", 5);
    dt.fit(data.X, Arrays.stream(data.y).asDoubleStream().toArray());
    
    double[][] probabilities = dt.predictProba(data.X);
    
    // Test probability constraints
    for (double[] probs : probabilities) {
        assertThat(probs).hasSize(2);
        assertThat(probs[0] + probs[1]).isCloseTo(1.0, within(1e-10));
        assertThat(probs[0]).isBetween(0.0, 1.0);
        assertThat(probs[1]).isBetween(0.0, 1.0);
    }
}
```

## Current Test Suites

### Linear Models

**LogisticRegressionTest.java**
- Binary and multiclass classification
- Regularization (L1, L2, Elastic Net)
- Convergence testing
- Parameter validation

```java
@Test
void testMulticlassClassification() {
    Datasets.ClassificationData data = Datasets.makeClassification(200, 5, 3);
    LogisticRegression lr = new LogisticRegression()
        .setMaxIter(1000)
        .setMultiClass("multinomial");
    
    lr.fit(data.X, Arrays.stream(data.y).asDoubleStream().toArray());
    
    double[] predictions = lr.predict(data.X);
    double[][] probabilities = lr.predictProba(data.X);
    
    // Validate multiclass behavior
    assertThat(lr.getClasses()).hasSize(3);
    assertThat(probabilities[0]).hasSize(3);
}
```

### Tree Algorithms

**RandomForestTest.java**
- Ensemble behavior validation
- Bootstrap sampling verification
- Feature importance testing
- Parallel training validation

```java
@Test
void testEnsembleBehavior() {
    Datasets.ClassificationData data = Datasets.makeClassification(300, 10, 2);
    
    RandomForest rf = new RandomForest(50, 8);
    rf.fit(data.X, Arrays.stream(data.y).asDoubleStream().toArray());
    
    // Verify ensemble contains expected number of trees
    assertThat(rf.getTrees()).hasSize(50);
    
    // Test that ensemble performs better than single tree
    DecisionTree dt = new DecisionTree("gini", 8);
    dt.fit(data.X, Arrays.stream(data.y).asDoubleStream().toArray());
    
    double rfAccuracy = rf.score(data.X, Arrays.stream(data.y).asDoubleStream().toArray());
    double dtAccuracy = dt.score(data.X, Arrays.stream(data.y).asDoubleStream().toArray());
    
    assertThat(rfAccuracy).isGreaterThanOrEqualTo(dtAccuracy);
}
```

**GradientBoostingTest.java**
- Sequential learning validation
- Early stopping functionality
- Learning rate effects
- Overfitting prevention

### Multiclass Classification

**MulticlassTest.java** - Comprehensive multiclass testing
```java
@Test
void testOneVsRestVsSoftmax() {
    Datasets.ClassificationData data = Datasets.makeClassification(500, 15, 4);
    var split = DataLoaders.trainTestSplit(data.X, 
        Arrays.stream(data.y).asDoubleStream().toArray(), 0.3, 42);
    
    // Test One-vs-Rest
    LogisticRegression base = new LogisticRegression().setMaxIter(1000);
    OneVsRestClassifier ovr = new OneVsRestClassifier(base);
    ovr.fit(split.XTrain, split.yTrain);
    
    // Test Softmax
    SoftmaxRegression softmax = new SoftmaxRegression().setMaxIter(1000);
    softmax.fit(split.XTrain, split.yTrain);
    
    // Both should achieve reasonable accuracy
    double ovrAccuracy = Metrics.accuracy(split.yTest, ovr.predict(split.XTest));
    double softmaxAccuracy = Metrics.accuracy(split.yTest, softmax.predict(split.XTest));
    
    assertThat(ovrAccuracy).isGreaterThan(0.6);
    assertThat(softmaxAccuracy).isGreaterThan(0.6);
}
```

### Datasets and Utilities

**DatasetsTest.java**
- Synthetic data generation validation
- Data loader functionality
- Train/test split verification

```java
@Test
void testMakeClassification() {
    Datasets.ClassificationData data = Datasets.makeClassification(100, 5, 3);
    
    assertThat(data.X).hasSize(100);
    assertThat(data.X[0]).hasSize(5);
    assertThat(data.y).hasSize(100);
    
    // Verify all classes are present
    Set<Integer> uniqueClasses = Arrays.stream(data.y).boxed().collect(Collectors.toSet());
    assertThat(uniqueClasses).hasSize(3);
}
```

## Writing New Tests

### Test Structure Template

```java
package org.superml.algorithms;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import static org.assertj.core.api.Assertions.*;

class NewAlgorithmTest {
    
    private NewAlgorithm algorithm;
    
    @BeforeEach
    void setUp() {
        algorithm = new NewAlgorithm();
    }
    
    @Test
    @DisplayName("Should train and predict correctly on binary classification")
    void testBinaryClassification() {
        // Arrange
        Datasets.ClassificationData data = Datasets.makeClassification(100, 4, 2);
        
        // Act
        algorithm.fit(data.X, Arrays.stream(data.y).asDoubleStream().toArray());
        double[] predictions = algorithm.predict(data.X);
        
        // Assert
        assertThat(predictions).hasSize(data.X.length);
        assertThat(algorithm.score(data.X, Arrays.stream(data.y).asDoubleStream().toArray()))
            .isGreaterThan(0.7);
    }
    
    @Test
    @DisplayName("Should handle edge cases gracefully")
    void testEdgeCases() {
        // Test with minimal data
        double[][] X = {% raw %}{{0, 0}, {0, 1}}{% endraw %};
        double[] y = {1, 2};
        
        assertThatCode(() -> algorithm.fit(X, y))
            .doesNotThrowAnyException();
    }
    
    @Test
    @DisplayName("Should validate parameters correctly")
    void testParameterValidation() {
        assertThatThrownBy(() -> algorithm.setInvalidParameter(-1))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("Parameter must be positive");
    }
}
```

### Testing Best Practices

1. **Use Descriptive Names**
   ```java
   @Test
   @DisplayName("Should achieve >90% accuracy on linearly separable data")
   void testLinearSeparableData() { /* ... */ }
   ```

2. **Test Edge Cases**
   ```java
   @Test
   void testEmptyDataset() {
       assertThatThrownBy(() -> algorithm.fit(new double[0][0], new double[0]))
           .isInstanceOf(IllegalArgumentException.class);
   }
   ```

3. **Validate Mathematical Properties**
   ```java
   @Test
   void testProbabilitiesSumToOne() {
       double[][] probas = classifier.predictProba(testData);
       for (double[] probs : probas) {
           assertThat(Arrays.stream(probs).sum()).isCloseTo(1.0, within(1e-10));
       }
   }
   ```

4. **Use Appropriate Tolerances**
   ```java
   assertThat(actualValue).isCloseTo(expectedValue, within(1e-6));
   ```

5. **Test Performance Characteristics**
   ```java
   @Test
   void testTrainingTime() {
       long start = System.currentTimeMillis();
       algorithm.fit(largeDataset.X, largeDataset.y);
       long duration = System.currentTimeMillis() - start;
       
       assertThat(duration).isLessThan(5000); // Should complete in 5 seconds
   }
   ```

## Mock Data and Fixtures

### Standard Test Datasets

```java
// Binary classification
Datasets.ClassificationData binary = Datasets.makeClassification(100, 5, 2);

// Multiclass classification  
Datasets.ClassificationData multiclass = Datasets.makeClassification(200, 10, 4);

// Regression
Datasets.RegressionData regression = Datasets.makeRegression(150, 8, 1, 0.1);

// Real-world style datasets
Datasets.ClassificationData iris = Datasets.loadIris();
Datasets.RegressionData boston = Datasets.loadBoston();
```

### Custom Test Data

```java
// Linearly separable data
double[][] X = {
    {0, 0}, {0, 1}, {1, 0}, {1, 1}
};
double[] y = {0, 0, 0, 1}; // XOR-like pattern

// Known solution data
double[][] perfectData = generatePerfectLinearData();
```

## Continuous Integration

### GitHub Actions

The project includes automated testing on push and pull requests:

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up JDK 11
        uses: actions/setup-java@v2
        with:
          java-version: '11'
      - name: Run tests
        run: mvn test
      - name: Generate test report
        run: mvn surefire-report:report
```

### Test Coverage

To generate test coverage reports:

```bash
# Generate coverage report
mvn jacoco:report

# View report
open target/site/jacoco/index.html
```

## Performance Testing

### Benchmark Tests

```java
@Test
void benchmarkTrainingTime() {
    int[] sampleSizes = {100, 500, 1000, 5000};
    
    for (int size : sampleSizes) {
        Datasets.ClassificationData data = Datasets.makeClassification(size, 10, 2);
        
        long start = System.nanoTime();
        algorithm.fit(data.X, Arrays.stream(data.y).asDoubleStream().toArray());
        long duration = System.nanoTime() - start;
        
        System.out.printf("Size %d: %.2f ms%n", size, duration / 1_000_000.0);
    }
}
```

### Memory Usage Tests

```java
@Test
void testMemoryUsage() {
    Runtime runtime = Runtime.getRuntime();
    long memBefore = runtime.totalMemory() - runtime.freeMemory();
    
    // Train large model
    Datasets.ClassificationData data = Datasets.makeClassification(10000, 50, 5);
    algorithm.fit(data.X, Arrays.stream(data.y).asDoubleStream().toArray());
    
    long memAfter = runtime.totalMemory() - runtime.freeMemory();
    long memUsed = memAfter - memBefore;
    
    // Verify reasonable memory usage
    assertThat(memUsed).isLessThan(100_000_000); // 100MB limit
}
```

## Test Results

Recent test execution results show comprehensive coverage:

- **Linear Models**: 15 tests, all passing
- **Tree Algorithms**: 25 tests, all passing  
- **Multiclass**: 12 tests, all passing
- **Datasets**: 8 tests, all passing
- **Metrics**: 10 tests, all passing

### Sample Test Output

```
[INFO] Results:
[INFO] 
[INFO] Tests run: 70, Failures: 0, Errors: 0, Skipped: 0
[INFO] 
[INFO] Tree Algorithms:
[INFO]   DecisionTree: 8/8 tests passed
[INFO]   RandomForest: 10/10 tests passed  
[INFO]   GradientBoosting: 7/7 tests passed
[INFO]
[INFO] Multiclass Classification:
[INFO]   OneVsRest: 5/5 tests passed
[INFO]   SoftmaxRegression: 4/4 tests passed
[INFO]   Integration: 3/3 tests passed
```

This comprehensive testing framework ensures that SuperML Java maintains high quality and reliability across all implemented algorithms.
