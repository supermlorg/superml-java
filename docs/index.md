---
title: "SuperML Java Framework"
description: "A comprehensive 21-module machine learning framework for Java"
layout: default
toc: true
search: true
---

# SuperML Java Framework

A comprehensive, modular machine learning library for Java, inspired by scikit-learn and designed for enterprise-grade applications. Version 2.0.0 features a sophisticated 21-module architecture.

## 🚀 Features

### Core Machine Learning (12+ Algorithms)
- **Linear Models** (6): Logistic Regression, Linear Regression, Ridge, Lasso, SGD Classifier/Regressor
- **Tree-Based Models** (5): Decision Trees, Random Forest (Classifier/Regressor), Gradient Boosting
- **Clustering** (1): K-Means with k-means++ initialization and advanced convergence
- **Preprocessing**: StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder

### Advanced Features  
- **AutoML Framework**: Automated algorithm selection and hyperparameter optimization
- **Dual-Mode Visualization**: Professional XChart GUI with ASCII terminal fallback
- **Model Selection**: Cross-validation, Grid/Random Search, advanced hyperparameter tuning
- **Pipeline System**: Seamless chaining of preprocessing and modeling steps
- **High-Performance Inference**: Microsecond predictions with caching and batch processing
- **Model Persistence**: Save/load models with automatic statistics and metadata capture

### Production & Enterprise
- **Cross-Platform Export**: ONNX and PMML support for enterprise deployment
- **Drift Detection**: Real-time model and data drift monitoring with statistical tests
- **Kaggle Integration**: One-line training on any Kaggle dataset with automated workflows
- **Professional Logging**: Structured logging with Logback and SLF4J
- **Comprehensive Metrics**: Complete evaluation suite for all ML tasks
- **Thread Safety**: Concurrent prediction capabilities after model training

## 📚 Documentation

### **Getting Started**
- [**Quick Start Guide**](quick-start.md) - Get started in 5 minutes with visualization examples
- [**Modular Architecture**](modular-architecture.md) - Complete 21-module system overview
- [**Architecture Overview**](architecture.md) - Framework design and internal workings

### **Algorithm Documentation**
- [**Algorithms Reference**](algorithms-reference.md) - Complete guide to all 12+ implemented algorithms
- [**Tree Algorithms Guide**](tree-algorithms.md) - Decision Trees, Random Forest, Gradient Boosting
- [**Multiclass Classification**](multiclass-guide.md) - Advanced classification strategies

### **Advanced Features**
- [**Implementation Status**](implementation-status.md) - Detailed status of all modules and features
- [**Inference Guide**](inference-guide.md) - Production model deployment and optimization
- [**Model Persistence**](model-persistence.md) - Advanced save/load with statistics capture
- [**Kaggle Integration**](kaggle-integration.md) - Competition workflows and automation

### **API & Examples**
- [**API Reference**](api/core-classes.md) - Complete API documentation for all modules
- [**Basic Examples**](examples/basic-examples.md) - Fundamental ML concepts and workflows
- [**Advanced Examples**](examples/advanced-examples.md) - XChart GUI, AutoML, and production patterns

### **Development**
- [**Testing Guide**](testing-guide.md) - Comprehensive unit tests and validation
- [**Logging Guide**](logging-guide.md) - Professional logging configuration
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

// Train model
var classifier = new LogisticRegression().setMaxIter(1000);
classifier.fit(split.XTrain, split.yTrain);

// Evaluate
double[] predictions = classifier.predict(split.XTest);
double accuracy = Metrics.accuracy(split.yTest, predictions);
System.out.printf("Accuracy: %.2f%%\n", accuracy * 100);
```

Start your machine learning journey with SuperML Java today! 🚀
