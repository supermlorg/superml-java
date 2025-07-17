# SuperML Java Framework - AI Coding Assistant Instructions

## 🏗️ Framework Architecture Overview

SuperML Java is a **21-module machine learning framework** implementing **12+ algorithms** with scikit-learn-inspired design. The codebase follows a **modular, hierarchical architecture** where `superml-core` provides base interfaces (`BaseEstimator`, `Estimator`, `SupervisedLearner`) that all ML algorithms extend.

### Core Module Structure
```
superml-core/           → Base interfaces (BaseEstimator, Estimator, SupervisedLearner)
├─ superml-linear-models → LogisticRegression, LinearRegression, Ridge, Lasso  
├─ superml-tree-models  → DecisionTree, RandomForest, GradientBoosting
├─ superml-clustering   → KMeans with k-means++ initialization
├─ superml-neural       → MLPClassifier, CNNClassifier, RNNClassifier (WIP)
└─ superml-pipeline     → Pipeline system for chaining steps
```

## 🔧 Essential Development Patterns

### 1. Algorithm Implementation Pattern
All ML algorithms extend `BaseEstimator` and implement specific interfaces:
```java
public class LogisticRegression extends BaseEstimator implements Classifier {
    @Override
    public LogisticRegression fit(double[][] X, double[] y) { /* training logic */ }
    @Override  
    public double[] predict(double[][] X) { /* prediction logic */ }
}
```

### 2. Pipeline System Integration
The framework's **Pipeline** class chains preprocessing and models:
```java
Pipeline pipeline = new Pipeline()
    .addStep("scaler", new StandardScaler())
    .addStep("classifier", new LogisticRegression());
```

### 3. Cross-Cutting Functionality Pattern
Recent additions follow a **specialist pattern** with algorithm-specific optimizers:
- `AutoTrainer` (general AutoML) + `XGBoostAutoTrainer`, `LinearModelAutoTrainer` (specialists)
- `Metrics` (general) + `LinearRegressionMetrics` (specialized regression analysis)
- Algorithm-specific visualization, persistence, and optimization modules

## 🚀 Critical Build and Development Workflows

### Maven Build Commands
```bash
# Build all modules (21 modules, ~2-3 minutes)
mvn clean compile

# Build specific cross-cutting modules only
mvn compile -pl superml-metrics,superml-autotrainer,superml-visualization,superml-persistence

# Skip problematic modules (kaggle, some neural network integrations)
mvn compile -pl '!superml-kaggle,!superml-onnx'
```

### Example Execution Pattern
```bash
# Compile with explicit classpath for cross-module dependencies
javac -cp "superml-core/target/classes:superml-linear-models/target/classes:..." examples/src/main/java/org/superml/examples/LinearModelsIntegrationExample.java

# Run with full classpath
java -cp "examples/src/main/java:superml-core/target/classes:..." org.superml.examples.LinearModelsIntegrationExample
```

## 📁 Project-Specific Conventions

### 1. Module Dependencies
- **Core modules**: superml-core must be compiled first (foundation for all others)
- **Cross-cutting modules**: autotrainer, visualization, persistence depend on algorithm modules
- **Integration modules**: kaggle, onnx depend on multiple algorithm modules

### 2. Package Structure Convention
```
org.superml.{module_name}.{ClassName}
org.superml.linear_model.LogisticRegression
org.superml.autotrainer.LinearModelAutoTrainer  
org.superml.metrics.LinearRegressionMetrics
```

### 3. Algorithm Naming Patterns
- **Core algorithms**: Direct names (LogisticRegression, RandomForest)
- **Specialized trainers**: `{Algorithm}AutoTrainer` (XGBoostAutoTrainer)  
- **Specialized metrics**: `{Algorithm}Metrics` (LinearRegressionMetrics)
- **Visualization**: `{Algorithm}Visualization` (LinearModelVisualization)

## 🔗 Integration Points and Data Flow

### 1. Cross-Module Data Flow
```
Examples → Pipeline → Algorithm Modules → Cross-Cutting Modules
    ↓           ↓            ↓                   ↓
 User API → Preprocessing → Core Training → Analysis/Optimization
```

### 2. Parameter Management
All estimators use `BaseEstimator.setParam()` pattern with fluent interfaces:
```java
model.setMaxIter(1000).setLearningRate(0.01).setTolerance(1e-4)
```

### 3. Pipeline Integration Points
The `Pipeline` class uses reflection-like patterns to chain estimators:
```java
// All steps except last must implement transform()
if (estimator instanceof UnsupervisedLearner) {
    currentX = ((UnsupervisedLearner) estimator).transform(currentX);
}
```

## 🎯 Working Examples and Entry Points
All Examples should be created in the `superml-examples` module, demonstrating core functionality and cross-cutting features.

### Fully Functional Examples
- `superml-examples/src/main/java/org/superml/examples/LinearModelsIntegrationExample.java` - **400+ lines**, demonstrates complete cross-cutting functionality
- `superml-examples/src/main/java/org/superml/examples/PipelineExample.java` - Comprehensive pipeline workflows
- `superml-examples/src/main/java/org/superml/examples/BasicClassification.java` - Simple algorithm usage

### Current Implementation Status
✅ **Working**: Linear models, tree models, clustering, preprocessing, pipelines, cross-cutting functionality  
⚠️ **WIP**: Neural networks (interfaces exist, limited implementation)  
❌ **Incomplete**: Some Kaggle integrations, ONNX export modules

## 🧪 Testing and Validation Patterns

### Synthetic Data Generation
Examples use consistent data generation patterns:
```java
// Classification: generateClassificationData(nSamples, nFeatures)
// Regression: generateRegressionData(nSamples, nFeatures)  
// Time series: generateTimeSeriesDataset(nSequences, sequenceLength, nFeatures)
```

### Performance Validation
Cross-cutting functionality includes comprehensive metrics:
```java
LinearRegressionMetrics.LinearRegressionEvaluation eval = 
    LinearRegressionMetrics.evaluateModel(model, X, y);
// Returns: R², AIC/BIC, residual analysis, outlier detection
```

## 🚨 Common Issues and Solutions

### 1. ClassPath Issues
When referencing cross-cutting functionality, ensure **all dependent modules** are in classpath:
```bash
# Required for LinearModelsIntegrationExample
-cp "superml-core/target/classes:superml-linear-models/target/classes:superml-metrics/target/classes:superml-visualization/target/classes:superml-autotrainer/target/classes:superml-persistence/target/classes"
```

### 2. Module Build Order
Some modules have circular-like dependencies. Build in this order:
1. superml-core (foundation)
2. Algorithm modules (linear-models, tree-models, clustering)  
3. Cross-cutting modules (metrics, autotrainer, visualization, persistence)
4. Integration modules (examples, kaggle)

### 3. Neural Network Limitations  
The `superml-neural` module has interface definitions but limited implementations. For neural network examples, use placeholder patterns or focus on working algorithms.

This framework represents a **production-ready ML library** with sophisticated cross-cutting functionality that enhances core algorithms with specialized analytics, visualization, automation, and deployment capabilities.
