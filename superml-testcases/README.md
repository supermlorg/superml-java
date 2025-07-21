# SuperML Test Cases Module

This module contains **all test cases** for the SuperML Java Framework, providing a centralized testing environment for all algorithms and components.

## 🎯 **Purpose**

- **Centralized Testing**: All test cases in one location for easier maintenance
- **Cleaner Core Modules**: Algorithm implementations focus purely on functionality
- **Comprehensive Coverage**: Tests for all SuperML components in one place
- **Performance Benchmarks**: JMH-based performance testing capabilities
- **Integration Testing**: Cross-module integration test scenarios

## 📁 **Test Organization**

```
superml-testcases/
├── src/test/java/org/superml/testcases/
│   ├── SuperMLTestSuite.java           # Main test suite runner
│   ├── linear_model/                   # Linear model tests
│   │   ├── LinearRegressionTest.java
│   │   ├── LogisticRegressionTest.java
│   │   ├── RidgeTest.java
│   │   └── MulticlassTest.java
│   ├── tree/                          # Tree algorithm tests
│   │   ├── DecisionTreeTest.java
│   │   └── XGBoostTest.java
│   ├── neural/                        # Neural network tests
│   │   ├── MLPClassifierTest.java
│   │   ├── CNNClassifierTest.java
│   │   └── RNNClassifierTest.java
│   ├── transformers/                  # Transformer tests
│   │   ├── attention/
│   │   ├── layers/
│   │   └── metrics/
│   ├── clustering/                    # Clustering tests
│   ├── preprocessing/                 # Preprocessing tests
│   │   └── StandardScalerTest.java
│   ├── metrics/                       # Metrics tests
│   │   └── MetricsTest.java
│   ├── model_selection/               # Model selection tests
│   │   ├── CrossValidationTest.java
│   │   └── HyperparameterTuningTest.java
│   └── examples/                      # Examples integration tests
│       └── transformers/
└── pom.xml
```

## 🚀 **Running Tests**

### All Tests
```bash
cd superml-testcases
mvn test
```

### Specific Test Categories
```bash
# Linear model tests only
mvn test -Dtest="**/linear_model/**/*Test"

# Transformer tests only  
mvn test -Dtest="**/transformers/**/*Test"

# Neural network tests only
mvn test -Dtest="**/neural/**/*Test"

# Integration tests only
mvn test -Dtest="**/examples/**/*Test"
```

### Test Suite Runner
```bash
mvn test -Dtest=SuperMLTestSuite
```

### Performance Benchmarks
```bash
mvn test -P benchmark
```

## 🔧 **Test Dependencies**

The testcases module includes all SuperML modules as test-scoped dependencies:

- `superml-core` - Core interfaces and utilities
- `superml-linear-models` - Linear algorithms 
- `superml-tree-models` - Tree-based algorithms
- `superml-neural` - Neural networks
- `superml-transformers` - Transformer models
- `superml-clustering` - Clustering algorithms
- `superml-preprocessing` - Data preprocessing
- `superml-metrics` - Evaluation metrics
- `superml-model-selection` - Model selection utilities
- `superml-pipeline` - ML pipelines
- `superml-utils` - Common utilities
- `superml-visualization` - Visualization components

## 🧪 **Testing Framework**

- **JUnit 5**: Primary testing framework
- **JUnit 4**: Legacy test compatibility
- **Mockito**: Mocking and stubbing
- **JMH**: Java Microbenchmark Harness for performance testing
- **Dropwizard Metrics**: Performance monitoring during tests

## 📊 **Benefits of Centralized Testing**

### ✅ **For Algorithm Modules**
- **Cleaner POMs**: No test dependencies in core modules
- **Smaller JARs**: Production JARs without test overhead  
- **Focused Code**: Implementation modules focus purely on algorithms
- **Faster Builds**: Individual modules build faster without tests

### ✅ **For Testing**
- **Complete Coverage**: All components tested in one environment
- **Integration Testing**: Cross-module interactions easily tested
- **Consistent Environment**: Same test setup for all components
- **Performance Comparison**: Easy to benchmark different algorithms

### ✅ **For Maintenance**
- **Single Test Environment**: One place to update test frameworks
- **Dependency Management**: Test dependencies managed in one location
- **CI/CD Simplification**: Single test command for entire framework
- **Documentation**: All test documentation in one place

## 🎯 **Test Categories**

1. **Unit Tests**: Individual component functionality
2. **Integration Tests**: Cross-module interactions
3. **Performance Tests**: Algorithm performance benchmarks
4. **Regression Tests**: Prevent functionality regressions
5. **Example Tests**: Verify examples work correctly

## 🔍 **Test Coverage**

The testcases module provides comprehensive coverage for:

- ✅ All linear model algorithms
- ✅ All tree-based algorithms  
- ✅ All neural network models
- ✅ All transformer components
- ✅ Preprocessing pipelines
- ✅ Evaluation metrics
- ✅ Model selection utilities
- ✅ Integration workflows

## 🚀 **Usage in Development**

1. **During Development**: Run relevant test category
2. **Before Commits**: Run full test suite
3. **Performance Analysis**: Use JMH benchmarks
4. **Integration Testing**: Verify cross-module functionality

This centralized approach keeps the core algorithm implementations clean while providing comprehensive testing capabilities for the entire SuperML framework!
