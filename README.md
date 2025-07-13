<p align="center">
  <img src="/docs/logo.png" alt="SuperML Logo" width="100" />
</p>

**By [SuperML.org](https://superml.org) with [SuperML.dev](https://superml.dev)**

# SuperML Java

A comprehensive Java Machine Learning Framework inspired by scikit-learn, developed by the SuperML community.

## Overview

SuperML Java is a comprehensive machine learning library for Java that provides:

- **Supervised Learning**: Classification and regression algorithms (Logistic Regression, Linear Regression, Ridge, Lasso)
- **Unsupervised Learning**: Clustering algorithms (K-Means with k-means++ initialization)
- **Data Preprocessing**: Feature scaling, normalization, and transformation utilities
- **Model Selection**: Cross-validation, train-test split, and automated hyperparameter tuning
- **Pipeline System**: Chain preprocessing and models like scikit-learn
- **Kaggle Integration**: One-line training on any Kaggle dataset with automated workflows
- **Inference Layer**: High-performance model inference with caching, monitoring, and batch processing
- **Metrics**: Comprehensive evaluation metrics for classification and regression
- **Professional Logging**: Configurable Logback/SLF4J logging framework
- **Production Ready**: Enterprise-grade error handling and validation

## 🚀 Quick Start

```java
import org.superml.datasets.Datasets;
import org.superml.linear_model.LogisticRegression;
import org.superml.pipeline.Pipeline;
import org.superml.preprocessing.StandardScaler;
import org.superml.model_selection.ModelSelection;

// Load data and create pipeline
Datasets.Dataset dataset = Datasets.loadIris();
Pipeline pipeline = new Pipeline()
    .addStep("scaler", new StandardScaler())
    .addStep("classifier", new LogisticRegression());

// Train and evaluate
ModelSelection.TrainTestSplit split = ModelSelection.trainTestSplit(dataset.X, dataset.y, 0.2, 42);
pipeline.fit(split.XTrain, split.yTrain);
double[] predictions = pipeline.predict(split.XTest);
```

## 🌐 Kaggle Integration

Train on any Kaggle dataset with one line:

```java
import org.superml.datasets.KaggleTrainingManager;
import org.superml.datasets.KaggleIntegration.KaggleCredentials;

KaggleCredentials credentials = KaggleCredentials.fromDefaultLocation();
KaggleTrainingManager trainer = new KaggleTrainingManager(credentials);

// Configure training with model saving
KaggleTrainingManager.TrainingConfig config = new KaggleTrainingManager.TrainingConfig()
    .setSaveModels(true)
    .setModelsDirectory("kaggle_models")
    .setAlgorithms("logistic", "ridge")
    .setGridSearch(true);

List<KaggleTrainingManager.TrainingResult> results = trainer.trainOnDataset("titanic", "titanic", "survived", config);
System.out.println("Best model: " + results.get(0).algorithm);
System.out.println("Model saved to: " + results.get(0).modelFilePath);
```

## 💾 Model Persistence

Save and load trained models with automatic training statistics capture:

```java
import org.superml.persistence.ModelPersistence;
import org.superml.persistence.ModelManager;

// Train a model
LogisticRegression model = new LogisticRegression().setMaxIter(1000);
model.fit(X_train, y_train);

// Save with automatic performance evaluation and statistics
ModelPersistence.saveWithStats(model, "my_model", 
                               "Production iris classifier", 
                               X_test, y_test);

// Load model with type safety
LogisticRegression loadedModel = ModelPersistence.load("my_model", LogisticRegression.class);
double[] predictions = loadedModel.predict(X_test);

// The framework automatically captures:
// - Performance metrics (accuracy, precision, recall, F1)
// - Dataset statistics and hyperparameters
// - System information and timestamps

// Manage multiple models with automatic statistics
ModelManager manager = new ModelManager("models");
String savedPath = manager.saveModel(model, "iris");
List<String> allModels = manager.listModels();
```

## 🎯 Features

### Algorithms
- **Linear Models**: Logistic Regression, Linear Regression, Ridge, Lasso with L1/L2 regularization
- **Clustering**: K-Means with k-means++ initialization and multiple restarts

### Data Processing
- **StandardScaler**: Feature standardization and normalization
- **DataLoaders**: CSV loading, synthetic data generation, and built-in datasets
- **Pipeline System**: Chain preprocessing steps and models seamlessly

### Model Selection & Evaluation
- **Cross-Validation**: K-fold validation with comprehensive metrics (accuracy, precision, recall, F1, MSE, MAE, R²)
- **Hyperparameter Tuning**: Grid Search and Random Search with parallel execution and custom configurations
- **Parameter Specifications**: Discrete, continuous, and integer parameter spaces for systematic optimization
- **Performance Metrics**: Complete evaluation suite with statistical analysis and confidence intervals

### Enterprise Features
- **Kaggle Integration**: Direct dataset download and automated training workflows
- **Model Persistence**: Save and load trained models with automatic training statistics capture and metadata
- **Professional Logging**: Structured logging with Logback and SLF4J
- **Error Handling**: Comprehensive validation and informative error messages
- **Thread Safety**: Safe concurrent prediction after model training
- **Parallel Processing**: Multi-threaded hyperparameter tuning and cross-validation

## 📦 Installation

### Maven Dependency

```xml
<dependency>
    <groupId>org.superml</groupId>
    <artifactId>superml-java</artifactId>
    <version>1.0.0</version>
</dependency>
```

### Build from Source

```bash
git clone https://github.com/superml/superml-java.git
mvn clean install
```

## 💻 Usage

### Basic Classification

```java
import org.superml.datasets.Datasets;
import org.superml.linear_model.LogisticRegression;
import org.superml.metrics.Metrics;
import org.superml.model_selection.ModelSelection;

// Load dataset
Datasets.Dataset dataset = Datasets.loadIris();
ModelSelection.TrainTestSplit split = ModelSelection.trainTestSplit(dataset.X, dataset.y, 0.2, 42);

// Train model
LogisticRegression model = new LogisticRegression()
    .setMaxIterations(1000)
    .setLearningRate(0.01);
model.fit(split.XTrain, split.yTrain);

// Evaluate
double[] predictions = model.predict(split.XTest);
double accuracy = Metrics.accuracy(split.yTest, predictions);
System.out.printf("Accuracy: %.3f\n", accuracy);
```

### Cross-Validation and Model Evaluation

```java
import org.superml.model_selection.CrossValidation;

// Basic cross-validation
LogisticRegression classifier = new LogisticRegression();
CrossValidation.CrossValidationResults results = 
    CrossValidation.crossValidate(classifier, X, y);

System.out.println("Accuracy: " + results.getMeanScore("accuracy") + 
                   " ± " + results.getStdScore("accuracy"));

// Custom cross-validation configuration
CrossValidation.CrossValidationConfig config = 
    new CrossValidation.CrossValidationConfig()
        .setFolds(10)
        .setShuffle(true)
        .setRandomSeed(42L)
        .setMetrics("accuracy", "precision", "recall", "f1");

CrossValidation.CrossValidationResults detailedResults = 
    CrossValidation.crossValidate(classifier, X, y, config);

// Regression cross-validation
Ridge regressor = new Ridge();
CrossValidation.CrossValidationResults regressionResults = 
    CrossValidation.crossValidateRegression(regressor, X, y, 
        new CrossValidation.CrossValidationConfig());
```

### Advanced Hyperparameter Tuning

```java
import org.superml.model_selection.HyperparameterTuning;

// Grid Search for Classification
HyperparameterTuning.TuningResults gridResults = HyperparameterTuning.gridSearch(
    new LogisticRegression(),
    X, y,
    HyperparameterTuning.ParameterSpec.discrete("learningRate", 0.01, 0.1, 0.5),
    HyperparameterTuning.ParameterSpec.discrete("maxIter", 500, 1000, 1500)
);

System.out.println("Best parameters: " + gridResults.getBestParameters());
System.out.println("Best score: " + gridResults.getBestScore());

// Grid Search for Regression
HyperparameterTuning.TuningResults regressionGrid = 
    HyperparameterTuning.gridSearchRegressor(
        new Ridge(),
        X, y,
        HyperparameterTuning.ParameterSpec.discrete("alpha", 0.1, 1.0, 10.0),
        HyperparameterTuning.ParameterSpec.continuous("tolerance", 1e-6, 1e-3, 5)
    );

// Random Search with Custom Configuration
HyperparameterTuning.TuningConfig advancedConfig = 
    new HyperparameterTuning.TuningConfig()
        .setScoringMetric("f1")
        .setCvFolds(5)
        .setParallel(true)
        .setVerbose(true)
        .setRandomSeed(123L);

HyperparameterTuning.TuningResults randomResults = 
    HyperparameterTuning.RandomSearch.search(
        new LogisticRegression(),
        X, y,
        Arrays.asList(
            HyperparameterTuning.ParameterSpec.discrete("learningRate", 0.001, 0.01, 0.1, 0.5),
            HyperparameterTuning.ParameterSpec.integer("maxIter", 100, 2000)
        ),
        advancedConfig
    );

// Parameter Specifications
// Discrete values
HyperparameterTuning.ParameterSpec.discrete("param", "A", "B", "C");

// Continuous range with specified steps
HyperparameterTuning.ParameterSpec.continuous("learning_rate", 0.001, 0.1, 10);

// Integer range
HyperparameterTuning.ParameterSpec.integer("max_depth", 1, 20);
```

### Model Persistence and Management

```java
import org.superml.persistence.ModelPersistence;
import org.superml.persistence.ModelManager;

// Train and save a pipeline
Pipeline pipeline = new Pipeline()
    .addStep("scaler", new StandardScaler())
    .addStep("classifier", new LogisticRegression());

pipeline.fit(X_train, y_train);

// Save with rich metadata
Map<String, Object> metadata = Map.of(
    "accuracy", Metrics.accuracy(y_test, pipeline.predict(X_test)),
    "features", X_train[0].length,
    "samples", X_train.length,
    "created_by", "SuperML_Demo"
);

ModelPersistence.save(pipeline, "production_model", "Main classification pipeline", metadata);

// Later, load and use the model
Pipeline loadedPipeline = ModelPersistence.load("production_model", Pipeline.class);
double[] predictions = loadedPipeline.predict(X_new);

// Model management
ModelManager manager = new ModelManager("models");
List<ModelManager.ModelInfo> models = manager.getModelsInfo();
for (ModelManager.ModelInfo info : models) {
    System.out.println(info); // Shows class, size, save time, etc.
}
```

### 🚀 Inference Layer

Deploy models in production with high-performance inference capabilities:

```java
import org.superml.inference.InferenceEngine;
import org.superml.inference.BatchInferenceProcessor;

// Create inference engine and load model
InferenceEngine engine = new InferenceEngine();
engine.loadModel("classifier", "models/trained_model.superml");

// Single prediction
double prediction = engine.predict("classifier", features);

// Batch prediction with monitoring
double[] batchPredictions = engine.predict("classifier", batchFeatures);

// Asynchronous inference
CompletableFuture<Double> future = engine.predictAsync("classifier", features);

// Performance metrics
InferenceMetrics metrics = engine.getMetrics("classifier");
System.out.println("Throughput: " + metrics.getThroughputSamplesPerSecond() + " samples/sec");

// Batch processing for large datasets
BatchInferenceProcessor processor = new BatchInferenceProcessor(engine);
BatchResult result = processor.processCSV("input.csv", "output.csv", "classifier");
```

## 📚 Documentation

- **[SuperML Java Framework Introduction](https://superml-java.superml.org/)** - SuperML Java Framework Introduction
- **[Quick Start Guide](https://superml-java.superml.org/quick-start)** - Get started in 5 minutes
- **[Model Persistence](https://superml-java.superml.org/model-persistence)** - Save and load trained models
- **[Kaggle Integration](https://superml-java.superml.org/kaggle-integration)** - Train on real datasets
- **[API Reference](https://superml-java.superml.org/api/core-classes)** - Complete API documentation
- **[Examples](https://superml-java.superml.org/examples/basic-examples)** - Comprehensive code examples
- **[Architecture](https://superml-java.superml.org/architecture)** - Framework design and patterns
- **[Contributing](https://superml-java.superml.org/contributing)** - Development guidelines
- **[Inference Guide](https://superml-java.superml.org/inference-guide)** - High-performance model inference and deployment

## 🤝 Contributing

We welcome contributions to SuperML Java! Please see our [Contributing Guide](https://superml-java.superml.org/contributing) for details.

### Ways to Contribute

- **Code**: Implement new algorithms, improve performance, fix bugs
- **Documentation**: Improve guides, add examples, write tutorials  
- **Testing**: Add test cases, improve coverage, performance testing
- **Community**: Help others, report issues, suggest features

### Development Setup

```bash
git clone https://github.com/superml/superml-java.git
mvn clean compile
mvn test
```

### Code Coverage

SuperML Java includes comprehensive code coverage analysis using JaCoCo:

```bash
# Run tests and generate coverage report
mvn clean test jacoco:report

# Use the provided coverage script for detailed analysis
./coverage.sh --summary    # Show coverage summary
./coverage.sh --open       # Open HTML report in browser
```

**Coverage Reports:**
- **HTML Report**: `target/site/jacoco/index.html` (visual coverage report)
- **Coverage Summary**: Use `./coverage.sh --summary` for quick overview
- **Detailed Analysis**: See [docs/CODE_COVERAGE_REPORT.md](docs/CODE_COVERAGE_REPORT.md)

**Current Status:**
- ✅ **Multiclass Classification**: 85%+ coverage (LogisticRegression, SoftmaxRegression)
- ⚠️ **Tree Algorithms**: 0% coverage (new v2.0 features needing tests)
- ⚠️ **Linear Models**: 0% coverage (LinearRegression, Ridge, Lasso need tests)

## 🌟 Community & Support

- **Website**: [superML.dev](https://superML.dev) - Main project website
- **Organization**: [superML.org](https://superml.org/community) - Community organization
- **Documentation**: [GitHub Wiki](https://superml-java.superml.org)
- **Issues**: [GitHub Issues](https://github.com/superml/superml-java/issues)
- **Discussions**: [GitHub Discussions](https://github.com/superml/superml-java/discussions)

## 🏆 Attribution

**SuperML Java** is developed and maintained by the **SuperML Community**:

- **Primary Website**: [superML.dev](https://superML.dev)
- **Community Organization**: [superML.org](https://superML.org)
- **Project Lead**: SuperML Development Team
- **Contributors**: See [CONTRIBUTORS.md](CONTRIBUTORS.md) for full list

This project is inspired by scikit-learn and aims to bring the same ease of use and comprehensive functionality to the Java ecosystem.

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### MIT License Summary
- ✅ **Commercial use** - Use in commercial projects
- ✅ **Modification** - Modify and distribute
- ✅ **Distribution** - Distribute original or modified
- ✅ **Private use** - Use for private projects
- ❗ **License and copyright notice** - Include in all copies
- ❌ **Liability** - No warranty provided
- ❌ **Trademark use** - SuperML trademarks not included

## 🎯 Project Status

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Java Version](https://img.shields.io/badge/java-11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-1.0--SNAPSHOT-orange)

**Current Version**: 1.0-SNAPSHOT  
**Stability**: Beta - Core features complete, API may change  
**Java Compatibility**: Java 11+  
**Dependencies**: Minimal - only essential libraries

---

Made with ❤️ by the [SuperML Community](https://superML.org) | Visit [superML.dev](https://superML.dev) for more projects

