---
title: "Quick Start Guide"
description: "Get up and running with SuperML Java in minutes"
layout: default
---

# Quick Start Guide

Get up and running with SuperML Java in just a few minutes! This guide will walk you through setting up the framework and training your first machine learning model.

## 🚀 5-Minute Quickstart

### Step 1: Add Dependency

Add SuperML Java to your Maven project:

```xml
<dependency>
    <groupId>org.superml</groupId>
    <artifactId>superml-java</artifactId>
    <version>1.0.0</version>
</dependency>
```

### Step 2: Your First Model

```java
import org.superml.datasets.Datasets;
import org.superml.linear_model.LogisticRegression;
import org.superml.metrics.Metrics;
import org.superml.model_selection.ModelSelection;

public class QuickStart {
    public static void main(String[] args) {
        // 1. Load a dataset
        var dataset = Datasets.loadIris();
        
        // 2. Split into train/test
        var split = ModelSelection.trainTestSplit(
            dataset.X, dataset.y, 0.2, 42);
        
        // 3. Create and train a model
        var model = new LogisticRegression()
            .setMaxIterations(1000)
            .setLearningRate(0.01);
        
        model.fit(split.XTrain, split.yTrain);
        
        // 4. Make predictions
        double[] predictions = model.predict(split.XTest);
        
        // 5. Evaluate performance
        double accuracy = Metrics.accuracy(split.yTest, predictions);
        System.out.printf("Accuracy: %.2f%%\n", accuracy * 100);
    }
}
```

### Step 3: Run and See Results

```bash
mvn compile exec:java -Dexec.mainClass="QuickStart"
```

Expected output:
```
Accuracy: 97.37%
```

## 🔧 Core Concepts

### Estimators
All models implement the `Estimator` interface:
```java
// Training
model.fit(X, y);

// Prediction
double[] predictions = model.predict(X);

// Parameters
Map<String, Object> params = model.getParams();
model.setParams(params);
```

### Datasets
Built-in datasets for quick experimentation:
```java
// Classification datasets
var iris = Datasets.loadIris();
var wine = Datasets.loadWine();

// Regression datasets  
var boston = Datasets.loadBoston();
var diabetes = Datasets.loadDiabetes();

// Synthetic data
var classification = Datasets.makeClassification(1000, 20, 2);
var regression = Datasets.makeRegression(1000, 10);
```

### Model Selection
Split data and validate models:
```java
// Train/test split
var split = ModelSelection.trainTestSplit(X, y, 0.2, 42);

// Cross-validation
double[] scores = ModelSelection.crossValidate(model, X, y, 5);
double meanScore = Arrays.stream(scores).average().orElse(0.0);
```

## 🏗️ Building Pipelines

Chain preprocessing and models together:

```java
import org.superml.pipeline.Pipeline;
import org.superml.preprocessing.StandardScaler;

// Create a pipeline
var pipeline = new Pipeline()
    .addStep("scaler", new StandardScaler())
    .addStep("classifier", new LogisticRegression());

// Train the entire pipeline
pipeline.fit(X, y);

// Make predictions (automatically applies preprocessing)
double[] predictions = pipeline.predict(X);
```

## 📊 Model Evaluation

Comprehensive metrics for model evaluation:

```java
// Classification metrics
double accuracy = Metrics.accuracy(yTrue, yPred);
double precision = Metrics.precision(yTrue, yPred);
double recall = Metrics.recall(yTrue, yPred);
double f1 = Metrics.f1Score(yTrue, yPred);

// Confusion matrix
int[][] confMatrix = Metrics.confusionMatrix(yTrue, yPred);

// Regression metrics
double mse = Metrics.meanSquaredError(yTrue, yPred);
double mae = Metrics.meanAbsoluteError(yTrue, yPred);
double r2 = Metrics.r2Score(yTrue, yPred);
```

## 🔍 Hyperparameter Tuning

Automatically find the best parameters:

```java
import org.superml.model_selection.GridSearchCV;

// Define parameter grid
Map<String, Object[]> paramGrid = Map.of(
    "maxIterations", new Object[]{500, 1000, 1500},
    "learningRate", new Object[]{0.001, 0.01, 0.1}
);

// Create grid search
var gridSearch = new GridSearchCV(
    new LogisticRegression(), paramGrid, 5);

// Find best parameters
gridSearch.fit(X, y);

// Get results
System.out.println("Best score: " + gridSearch.getBestScore());
System.out.println("Best params: " + gridSearch.getBestParams());
```

## 🌐 Kaggle Integration

Train models on real Kaggle datasets with one line:

```java
import org.superml.datasets.KaggleTrainingManager;
import org.superml.datasets.KaggleIntegration.KaggleCredentials;

// Setup Kaggle credentials (see Kaggle Integration guide)
var credentials = KaggleCredentials.fromDefaultLocation();
var trainer = new KaggleTrainingManager(credentials);

// Train on any Kaggle dataset
var results = trainer.trainOnDataset("titanic", "titanic", "survived");

// Get best model
var bestResult = results.get(0);
System.out.println("Best algorithm: " + bestResult.algorithm);
System.out.println("Best score: " + bestResult.score);
```

## 📈 Available Algorithms

### Supervised Learning

**Classification:**
- `LogisticRegression` - Binary and multiclass classification
- `Ridge` - L2 regularized classification (when used with discrete targets)

**Regression:**
- `LinearRegression` - Ordinary least squares
- `Ridge` - L2 regularized regression
- `Lasso` - L1 regularized regression with feature selection

### Unsupervised Learning

**Clustering:**
- `KMeans` - K-means clustering with k-means++ initialization

### Preprocessing
- `StandardScaler` - Feature standardization (z-score normalization)

## 📁 Project Structure

```
src/main/java/com/superml/
├── core/                    # Base interfaces
├── linear_model/           # Linear algorithms
├── cluster/                # Clustering algorithms
├── preprocessing/          # Data preprocessing
├── metrics/               # Evaluation metrics
├── model_selection/       # Cross-validation & tuning
├── pipeline/              # ML pipelines
└── datasets/              # Data loading & Kaggle integration
```

## 🎯 Next Steps

1. **Try More Examples**: Check out [Basic Examples](examples/basic-examples.md)
2. **Learn Pipelines**: Read the [Pipeline System](pipelines.md) guide
3. **Explore Kaggle**: Try [Kaggle Integration](kaggle-integration.md)
4. **Optimize Models**: Learn [Hyperparameter Tuning](hyperparameter-tuning.md)
5. **Production Ready**: Study [Performance Optimization](performance.md)

## 💡 Tips for Success

- **Start Simple**: Begin with basic models before complex pipelines
- **Use Built-in Datasets**: Great for learning and testing
- **Validate Everything**: Always use cross-validation for model evaluation
- **Log Performance**: Use the logging framework to track training progress
- **Read the Examples**: Real code examples are in the `examples/` folder

Ready to build amazing ML applications? Let's go! 🚀
