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
var dataset = Datasets.loadIris();
var pipeline = new Pipeline()
    .addStep("scaler", new StandardScaler())
    .addStep("classifier", new LogisticRegression());

// Train and evaluate
var split = ModelSelection.trainTestSplit(dataset.X, dataset.y, 0.2, 42);
pipeline.fit(split.XTrain, split.yTrain);
double[] predictions = pipeline.predict(split.XTest);
```

## 🌐 Kaggle Integration

Train on any Kaggle dataset with one line:

```java
import com.superml.datasets.KaggleTrainingManager;
import com.superml.datasets.KaggleIntegration.KaggleCredentials;

var credentials = KaggleCredentials.fromDefaultLocation();
var trainer = new KaggleTrainingManager(credentials);
var results = trainer.trainOnDataset("titanic", "titanic", "survived");
System.out.println("Best model: " + results.get(0).algorithm);
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
cd superml-java/ml-framework
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
var dataset = Datasets.loadIris();
var split = ModelSelection.trainTestSplit(dataset.X, dataset.y, 0.2, 42);

// Train model
var model = new LogisticRegression()
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
var pipeline = new Pipeline()
    .addStep("scaler", new StandardScaler())
    .addStep("classifier", new LogisticRegression());

// Grid search
Map<String, Object[]> paramGrid = Map.of(
    "classifier__maxIterations", new Object[]{500, 1000, 1500},
    "classifier__learningRate", new Object[]{0.001, 0.01, 0.1}
);

var gridSearch = new GridSearchCV(pipeline, paramGrid, 5);
gridSearch.fit(X, y);

System.out.println("Best score: " + gridSearch.getBestScore());
System.out.println("Best params: " + gridSearch.getBestParams());
```

## 📚 Documentation

- **[Quick Start Guide](docs/quick-start.md)** - Get started in 5 minutes
- **[Kaggle Integration](docs/kaggle-integration.md)** - Train on real datasets
- **[API Reference](docs/api/core-classes.md)** - Complete API documentation
- **[Examples](docs/examples/basic-examples.md)** - Comprehensive code examples
- **[Architecture](docs/architecture.md)** - Framework design and patterns
- **[Contributing](docs/contributing.md)** - Development guidelines

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
cd superml-java/ml-framework
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

