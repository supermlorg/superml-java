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
import org.superml.tree.RandomForest;
import org.superml.metrics.Metrics;
import org.superml.datasets.DataLoaders;

public class QuickStart {
    public static void main(String[] args) {
        // 1. Load a dataset
        var dataset = Datasets.loadIris();
        
        // 2. Split into train/test
        var split = DataLoaders.trainTestSplit(dataset.X, 
            Arrays.stream(dataset.y).asDoubleStream().toArray(), 0.2, 42);
        
        // 3. Create and train a model
        var model = new RandomForest(100, 10)  // 100 trees, max depth 10
            .setRandomState(42);
        
        model.fit(split.XTrain, split.yTrain);
        
        // 4. Make predictions
        double[] predictions = model.predict(split.XTest);
        double[][] probabilities = model.predictProba(split.XTest);
        
        // 5. Evaluate performance
        double accuracy = Metrics.accuracy(split.yTest, predictions);
        System.out.printf("Accuracy: %.2f%%\n", accuracy * 100);
        
        // 6. Show probability predictions
        System.out.println("\nClass probabilities for first 3 samples:");
        for (int i = 0; i < 3; i++) {
            System.out.printf("Sample %d: [%.3f, %.3f, %.3f]\n", 
                i+1, probabilities[i][0], probabilities[i][1], probabilities[i][2]);
        }
    }
}
```

### Step 3: Run and See Results

```bash
mvn compile exec:java -Dexec.mainClass="QuickStart"
```

Expected output:
```
Accuracy: 100.00%

Class probabilities for first 3 samples:
Sample 1: [0.000, 0.020, 0.980]
Sample 2: [0.980, 0.020, 0.000]
Sample 3: [0.000, 1.000, 0.000]
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

## 🎯 Algorithm Quick Examples

### Tree-Based Algorithms

```java
// Decision Tree
DecisionTree dt = new DecisionTree("gini", 10);
dt.fit(XTrain, yTrain);
double[] predictions = dt.predict(XTest);

// Random Forest  
RandomForest rf = new RandomForest(100, 15);
rf.fit(XTrain, yTrain);
double[] rfPredictions = rf.predict(XTest);

// Gradient Boosting
GradientBoosting gb = new GradientBoosting(100, 0.1, 6);
gb.fit(XTrain, yTrain);
double[] gbPredictions = gb.predict(XTest);
```

### Multiclass Classification

```java
// One-vs-Rest with any binary classifier
LogisticRegression base = new LogisticRegression();
OneVsRestClassifier ovr = new OneVsRestClassifier(base);
ovr.fit(XTrain, yTrain);

// Direct multinomial approach
SoftmaxRegression softmax = new SoftmaxRegression();
softmax.fit(XTrain, yTrain);
double[][] probabilities = softmax.predictProba(XTest);

// Enhanced LogisticRegression (auto multiclass)
LogisticRegression lr = new LogisticRegression().setMultiClass("auto");
lr.fit(XTrain, yTrain);  // Automatically handles multiclass
```

### Linear Models

```java
// Logistic Regression
LogisticRegression lr = new LogisticRegression()
    .setMaxIter(1000)
    .setRegularization("l2")
    .setC(1.0);

// Ridge Regression
Ridge ridge = new Ridge()
    .setAlpha(1.0)
    .setNormalize(true);

// Lasso Regression
Lasso lasso = new Lasso()
    .setAlpha(0.1)
    .setMaxIter(1000);
```

## 🚀 30-Second Examples

### Binary Classification
```java
var data = Datasets.makeClassification(1000, 10, 2);
var split = DataLoaders.trainTestSplit(data.X, 
    Arrays.stream(data.y).asDoubleStream().toArray(), 0.2, 42);

RandomForest rf = new RandomForest(50, 10);
rf.fit(split.XTrain, split.yTrain);
System.out.println("Accuracy: " + rf.score(split.XTest, split.yTest));
```

### Multiclass Classification
```java
var data = Datasets.loadIris();  // 3-class problem
var split = DataLoaders.trainTestSplit(data.X, 
    Arrays.stream(data.y).asDoubleStream().toArray(), 0.3, 42);

SoftmaxRegression softmax = new SoftmaxRegression();
softmax.fit(split.XTrain, split.yTrain);
double[][] probas = softmax.predictProba(split.XTest);
```

### Regression
```java
var data = Datasets.makeRegression(800, 5, 1, 0.1);
var split = DataLoaders.trainTestSplit(data.X, data.y, 0.2, 42);

GradientBoosting gb = new GradientBoosting(100, 0.05, 6);
gb.fit(split.XTrain, split.yTrain);
System.out.println("R² Score: " + gb.score(split.XTest, split.yTest));
```
