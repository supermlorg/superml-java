---
title: "SuperML Java Framework"
description: "A modular machine learning framework for Java"
layout: default
---

# SuperML Java Framework

A comprehensive machine learning library for Java, inspired by scikit-learn and designed for enterprise-grade applications.

## 🚀 Features

- **Supervised Learning**: Classification and regression algorithms
- **Unsupervised Learning**: Clustering and dimensionality reduction
- **Preprocessing**: Data scaling, normalization, and transformation
- **Model Selection**: Cross-validation, grid search, and hyperparameter tuning
- **Inference Layer**: High-performance model deployment and inference
- **Model Persistence**: Save and load models with comprehensive metadata
- **Pipeline Support**: Chain preprocessing and modeling steps
- **Kaggle Integration**: Seamless integration with Kaggle datasets and competitions

## 📚 Documentation

- [**Quick Start Guide**](quick-start.md) - Get started in 5 minutes
- [**Inference Guide**](inference-guide.md) - Production model deployment
- [**Model Persistence**](model-persistence.md) - Save and load trained models
- [**API Reference**](api/core-classes.md) - Complete API documentation
- [**Examples**](examples/basic-examples.md) - Practical code examples
- [**Architecture**](architecture.md) - Framework design and internals
- [**Contributing**](contributing.md) - How to contribute to the project

## 🔗 Quick Links

- [GitHub Repository](https://github.com/supermlorg/superml-java)
- [Issue Tracker](https://github.com/supermlorg/superml-java/issues)
- [Discussions](https://github.com/supermlorg/superml-java/discussions)

## 🎯 Quick Example

```java
import org.superml.datasets.Datasets;
import org.superml.linear_model.LogisticRegression;
import org.superml.model_selection.ModelSelection;
import org.superml.metrics.Metrics;

// Load dataset
var dataset = Datasets.loadIris();
var split = ModelSelection.trainTestSplit(dataset.data, dataset.target, 0.2, 42);

// Train model
var classifier = new LogisticRegression().setMaxIter(1000);
classifier.fit(split.XTrain, split.yTrain);

// Evaluate
double[] predictions = classifier.predict(split.XTest);
double accuracy = Metrics.accuracy(split.yTest, predictions);
System.out.printf("Accuracy: %.2f%%\n", accuracy * 100);
```

Start your machine learning journey with SuperML Java today! 🚀
