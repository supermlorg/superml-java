---
title: "Tree-Based Algorithms Guide"
description: "Complete guide to Decision Trees, Random Forest, and Gradient Boosting"
layout: default
toc: true
search: true
---

# Tree-Based Algorithms Guide

SuperML Java provides comprehensive implementations of tree-based machine learning algorithms, including Decision Trees, Random Forest, and Gradient Boosting. These algorithms are among the most popular and effective machine learning methods for both classification and regression tasks.

## Overview

Tree-based algorithms work by recursively splitting the feature space into regions and making predictions based on the majority class (classification) or average value (regression) in each region.

### Available Algorithms

| Algorithm | Class | Best For | Key Features |
|-----------|-------|----------|--------------|
| Decision Tree | `DecisionTree` | Interpretable models, feature selection | Fast training, easy to understand |
| Random Forest | `RandomForest` | Robust predictions, feature importance | Ensemble method, handles overfitting |
| Gradient Boosting | `GradientBoosting` | High accuracy, competitions | Sequential learning, excellent performance |

## Decision Trees

Decision Trees are the foundation of tree-based algorithms. They create a model that predicts target values by learning simple decision rules inferred from data features.

### Basic Usage

```java
import org.superml.tree.DecisionTree;
import org.superml.datasets.Datasets;

// Create and configure decision tree
DecisionTree dt = new DecisionTree("gini", 10)  // criterion, max_depth
    .setMinSamplesSplit(5)
    .setMinSamplesLeaf(2)
    .setRandomState(42);

// Load data
var dataset = Datasets.makeClassification(1000, 20, 2);
var split = DataLoaders.trainTestSplit(dataset.X, 
    Arrays.stream(dataset.y).asDoubleStream().toArray(), 0.2, 42);

// Train
dt.fit(split.XTrain, split.yTrain);

// Predict
double[] predictions = dt.predict(split.XTest);
double[][] probabilities = dt.predictProba(split.XTest);
```

### Parameters

- **criterion**: Splitting criterion
  - Classification: `"gini"`, `"entropy"`
  - Regression: `"mse"`
- **max_depth**: Maximum tree depth (prevents overfitting)
- **min_samples_split**: Minimum samples required to split a node
- **min_samples_leaf**: Minimum samples required in a leaf node
- **max_features**: Number of features to consider for splits
- **random_state**: Random seed for reproducibility

### Regression Example

```java
DecisionTree regressor = new DecisionTree("mse", 15)
    .setMinSamplesSplit(10)
    .setMinSamplesLeaf(5);

var dataset = Datasets.makeRegression(1000, 10, 1, 0.1);
regressor.fit(dataset.X, dataset.y);

double[] predictions = regressor.predict(testX);
double r2Score = Metrics.r2Score(testY, predictions);
```

## Random Forest

Random Forest builds multiple decision trees and combines their predictions through voting (classification) or averaging (regression). It uses bootstrap sampling and random feature selection to create diverse trees.

### Basic Usage

```java
import org.superml.tree.RandomForest;

// Create Random Forest
RandomForest rf = new RandomForest(100, 10)  // n_estimators, max_depth
    .setMaxFeatures(5)
    .setBootstrap(true)
    .setRandomState(42);

// Train
rf.fit(XTrain, yTrain);

// Predict
double[] predictions = rf.predict(XTest);
double[][] probabilities = rf.predictProba(XTest);

// Get feature importances
double[] importances = rf.getFeatureImportances();
```

### Advanced Configuration

```java
RandomForest rf = new RandomForest()
    .setNEstimators(200)
    .setMaxDepth(15)
    .setMinSamplesSplit(5)
    .setMinSamplesLeaf(2)
    .setMaxFeatures(10)         // or -1 for auto
    .setBootstrap(true)
    .setMaxSamples(0.8)         // sample 80% for each tree
    .setNJobs(4)                // parallel training
    .setRandomState(42);
```

### Parallel Training

Random Forest supports parallel training for faster performance:

```java
// Use all available cores
RandomForest rf = new RandomForest().setNJobs(-1);

// Use specific number of threads
RandomForest rf = new RandomForest().setNJobs(4);
```

## Gradient Boosting

Gradient Boosting builds an ensemble of weak learners (typically shallow trees) sequentially, where each new tree corrects the errors of the previous ones.

### Basic Usage

```java
import org.superml.tree.GradientBoosting;

// Create Gradient Boosting model
GradientBoosting gb = new GradientBoosting(100, 0.1, 6)  // n_estimators, learning_rate, max_depth
    .setSubsample(0.8)
    .setRandomState(42);

// Train
gb.fit(XTrain, yTrain);

// Predict
double[] predictions = gb.predict(XTest);
double[] rawPredictions = gb.predictRaw(XTest);  // before sigmoid/softmax
```

### Early Stopping

Gradient Boosting supports early stopping to prevent overfitting:

```java
GradientBoosting gb = new GradientBoosting()
    .setNEstimators(1000)
    .setLearningRate(0.1)
    .setValidationFraction(0.1)    // 10% for validation
    .setNIterNoChange(10)          // stop after 10 rounds without improvement
    .setTol(1e-4);                 // tolerance for improvement

gb.fit(XTrain, yTrain);

// Get training history
List<Double> trainScores = gb.getTrainScores();
List<Double> validScores = gb.getValidationScores();
```

### Progressive Predictions

You can get predictions at any stage of boosting:

```java
// Predict using only first 50 estimators
double[] earlyPreds = gb.predictAtIteration(XTest, 50);

// Compare with full model
double[] fullPreds = gb.predict(XTest);
```

## Performance Comparison

Here's a typical performance comparison on a synthetic dataset:

```java
// Generate dataset
var dataset = Datasets.makeClassification(1000, 20, 2);
var split = DataLoaders.trainTestSplit(dataset.X, 
    Arrays.stream(dataset.y).asDoubleStream().toArray(), 0.2, 42);

// Decision Tree
DecisionTree dt = new DecisionTree("gini", 10);
dt.fit(split.XTrain, split.yTrain);
double dtAccuracy = Metrics.accuracy(split.yTest, dt.predict(split.XTest));

// Random Forest  
RandomForest rf = new RandomForest(100, 10);
rf.fit(split.XTrain, split.yTrain);
double rfAccuracy = Metrics.accuracy(split.yTest, rf.predict(split.XTest));

// Gradient Boosting
GradientBoosting gb = new GradientBoosting(100, 0.1, 6);
gb.fit(split.XTrain, split.yTrain);
double gbAccuracy = Metrics.accuracy(split.yTest, gb.predict(split.XTest));

System.out.println("Decision Tree: " + dtAccuracy);
System.out.println("Random Forest: " + rfAccuracy);
System.out.println("Gradient Boosting: " + gbAccuracy);
```

## Best Practices

### Decision Trees
- Start with moderate depth (5-15) to avoid overfitting
- Use `min_samples_split` and `min_samples_leaf` for regularization
- Consider pruning for better generalization

### Random Forest
- More trees generally improve performance (100-500 is common)
- Use bootstrap sampling (`bootstrap=true`)
- Set `max_features` to sqrt(n_features) for classification
- Use parallel training for large datasets

### Gradient Boosting
- Start with small learning rate (0.01-0.3)
- Use shallow trees (depth 3-8)
- Enable early stopping for optimal performance
- Consider subsample < 1.0 for regularization

### General Tips
- Always validate on a separate test set
- Use cross-validation for model selection
- Monitor for overfitting, especially with Decision Trees
- Random Forest and Gradient Boosting usually outperform single trees
- Gradient Boosting often achieves the best performance but takes longer to train

## Feature Importance

Tree-based models provide feature importance scores:

```java
// Random Forest feature importance
RandomForest rf = new RandomForest(100, 10);
rf.fit(XTrain, yTrain);
double[] importances = rf.getFeatureImportances();

// Print top features
for (int i = 0; i < importances.length; i++) {
    System.out.println("Feature " + i + ": " + importances[i]);
}
```

## Integration with Other Components

Tree algorithms work seamlessly with other SuperML components:

```java
// With preprocessing
StandardScaler scaler = new StandardScaler();
double[][] scaledX = scaler.fitTransform(XTrain);
RandomForest rf = new RandomForest().fit(scaledX, yTrain);

// With pipelines
Pipeline pipeline = new Pipeline()
    .addStep("scaler", new StandardScaler())
    .addStep("classifier", new RandomForest(100, 10));

// With multiclass
OneVsRestClassifier ovr = new OneVsRestClassifier(new DecisionTree());
ovr.fit(XTrain, yTrain);
```

## Complete Example

See [TreeAlgorithmsExample.java](../examples/TreeAlgorithmsExample.java) for a comprehensive demonstration of all tree-based algorithms with both classification and regression examples.
