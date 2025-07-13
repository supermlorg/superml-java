---
title: "SuperML Java Framework"
description: "A modular machine learning framework for Java"
layout: default
---

# SuperML Java Framework

A comprehensive machine learning library for Java, inspired by scikit-learn and designed for enterprise-grade applications.

## 🚀 Features

### Core Machine Learning
- **Linear Models**: Logistic Regression, Linear Regression, Ridge, Lasso
- **Tree-Based Models**: Decision Trees, Random Forest, Gradient Boosting
- **Support Vector Machines**: SVM for classification and regression *(coming soon)*
- **Nearest Neighbors**: k-NN for classification and regression *(coming soon)*
- **Clustering**: K-Means, DBSCAN, Hierarchical clustering *(coming soon)*

### Advanced Features  
- **Multiclass Classification**: One-vs-Rest and Softmax/Multinomial approaches
- **Model Selection**: Cross-validation, grid search, and hyperparameter tuning
- **Preprocessing**: Data scaling, normalization, and transformation
- **Pipeline Support**: Chain preprocessing and modeling steps
- **Inference Layer**: High-performance model deployment and inference
- **Model Persistence**: Save and load models with comprehensive metadata

### Integration & Tools
- **Kaggle Integration**: Seamless integration with Kaggle datasets and competitions
- **Datasets**: Synthetic data generation and real dataset loaders
- **Comprehensive Metrics**: Accuracy, precision, recall, F1, R², MSE, and more
- **Logging**: Structured logging with SLF4J integration

## 📚 Documentation

- [**Quick Start Guide**](quick-start.md) - Get started in 5 minutes
- [**Tree Algorithms Guide**](tree-algorithms.md) - Decision Trees, Random Forest, Gradient Boosting
- [**Multiclass Classification**](multiclass-guide.md) - One-vs-Rest and Softmax approaches
- [**Inference Guide**](inference-guide.md) - Production model deployment
- [**Model Persistence**](model-persistence.md) - Save and load trained models
- [**Kaggle Integration**](kaggle-integration.md) - Competition workflows and data loading
- [**API Reference**](api/core-classes.md) - Complete API documentation
- [**Examples**](examples/basic-examples.md) - Practical code examples
- [**Test Guide**](testing-guide.md) - Unit tests and validation
- [**Architecture**](architecture.md) - Framework design and internals
- [**Contributing**](contributing.md) - How to contribute to the project

## 🔗 Quick Links

- [GitHub Repository](https://github.com/supermlorg/superml-java)
- [Issue Tracker](https://github.com/supermlorg/superml-java/issues)
- [Discussions](https://github.com/supermlorg/superml-java/discussions)

## 🎯 Quick Example

```java
import org.superml.datasets.Datasets;
import org.superml.tree.RandomForest;
import org.superml.multiclass.OneVsRestClassifier;
import org.superml.linear_model.LogisticRegression;
import org.superml.metrics.Metrics;

// Load dataset
var dataset = Datasets.loadIris();
var split = DataLoaders.trainTestSplit(dataset.X, 
    Arrays.stream(dataset.y).asDoubleStream().toArray(), 0.2, 42);

// Train multiclass model
var base = new LogisticRegression();
var classifier = new OneVsRestClassifier(base);
classifier.fit(split.XTrain, split.yTrain);

// Or use tree-based model
var forest = new RandomForest(100, 10);
forest.fit(split.XTrain, split.yTrain);

// Make predictions
double[] predictions = forest.predict(split.XTest);
double[][] probabilities = forest.predictProba(split.XTest);

// Evaluate
double accuracy = Metrics.accuracy(split.yTest, predictions);
System.out.println("Accuracy: " + accuracy);
```

// Train model
var classifier = new LogisticRegression().setMaxIter(1000);
classifier.fit(split.XTrain, split.yTrain);

// Evaluate
double[] predictions = classifier.predict(split.XTest);
double accuracy = Metrics.accuracy(split.yTest, predictions);
System.out.printf("Accuracy: %.2f%%\n", accuracy * 100);
```

Start your machine learning journey with SuperML Java today! 🚀
