<img src="docs/logo.png" alt="SuperML Logo" width="100" /> By SuperML.org with SuperML.dev

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
import com.superml.datasets.Datasets;
import com.superml.linear_model.LogisticRegression;
import com.superml.pipeline.Pipeline;
import com.superml.preprocessing.StandardScaler;
import com.superml.model_selection.ModelSelection;

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
import com.superml.datasets.KaggleTrainingManager;
import com.superml.datasets.KaggleIntegration.KaggleCredentials;

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
import com.superml.persistence.ModelPersistence;
import com.superml.persistence.ModelManager;

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
- **Grid Search**: Automated hyperparameter tuning with cross-validation

### Data Processing
- **StandardScaler**: Feature standardization and normalization
- **DataLoaders**: CSV loading, synthetic data generation, and built-in datasets
- **Pipeline System**: Chain preprocessing steps and models seamlessly

### Evaluation & Selection
- **Metrics**: Accuracy, Precision, Recall, F1-Score, MSE, MAE, R² and confusion matrices
- **Cross-Validation**: K-fold validation and train/test splitting
- **Model Comparison**: Automated algorithm benchmarking

### Enterprise Features
- **Kaggle Integration**: Direct dataset download and automated training workflows
- **Model Persistence**: Save and load trained models with automatic training statistics capture and metadata
- **Professional Logging**: Structured logging with Logback and SLF4J
- **Error Handling**: Comprehensive validation and informative error messages
- **Thread Safety**: Safe concurrent prediction after model training

## 📦 Installation

### Maven Dependency

```xml
<dependency>
    <groupId>com.superml</groupId>
    <artifactId>superml-java</artifactId>
    <version>1.0-SNAPSHOT</version>
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
import com.superml.datasets.Datasets;
import com.superml.linear_model.LogisticRegression;
import com.superml.metrics.Metrics;
import com.superml.model_selection.ModelSelection;

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

### Advanced Pipeline with Hyperparameter Tuning

```java
import com.superml.pipeline.Pipeline;
import com.superml.preprocessing.StandardScaler;
import com.superml.model_selection.GridSearchCV;

// Create pipeline
Pipeline pipeline = new Pipeline()
    .addStep("scaler", new StandardScaler())
    .addStep("classifier", new LogisticRegression());

// Grid search
Map<String, Object[]> paramGrid = Map.of(
    "classifier__maxIterations", new Object[]{500, 1000, 1500},
    "classifier__learningRate", new Object[]{0.001, 0.01, 0.1}
);

GridSearchCV gridSearch = new GridSearchCV(pipeline, paramGrid, 5);
gridSearch.fit(X, y);

System.out.println("Best score: " + gridSearch.getBestScore());
System.out.println("Best params: " + gridSearch.getBestParams());
```

### Model Persistence and Management

```java
import com.superml.persistence.ModelPersistence;
import com.superml.persistence.ModelManager;

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
import com.superml.inference.InferenceEngine;
import com.superml.inference.BatchInferenceProcessor;

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

- **[Quick Start Guide](docs/quick-start.md)** - Get started in 5 minutes
- **[Model Persistence](docs/model-persistence.md)** - Save and load trained models
- **[Kaggle Integration](docs/kaggle-integration.md)** - Train on real datasets
- **[API Reference](docs/api/core-classes.md)** - Complete API documentation
- **[Examples](docs/examples/basic-examples.md)** - Comprehensive code examples
- **[Architecture](docs/architecture.md)** - Framework design and patterns
- **[Contributing](docs/contributing.md)** - Development guidelines
- **[Inference Guide](docs/inference-guide.md)** - High-performance model inference and deployment

## 🤝 Contributing

We welcome contributions to SuperML Java! Please see our [Contributing Guide](docs/contributing.md) for details.

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

## 🌟 Community & Support

- **Website**: [superML.dev](https://superML.dev) - Main project website
- **Organization**: [superML.org](https://superML.org) - Community organization
- **Documentation**: [GitHub Wiki](https://github.com/superml/superml-java/wiki)
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

