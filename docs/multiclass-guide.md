# Multiclass Classification Guide

SuperML Java provides comprehensive support for multiclass classification through multiple strategies, including One-vs-Rest (OvR) and direct multinomial approaches. This guide covers all aspects of multiclass classification in the framework.

## Overview

While binary classification deals with two classes, multiclass classification handles problems with three or more classes. SuperML provides several approaches:

1. **One-vs-Rest (OvR)**: Trains one binary classifier per class
2. **Multinomial/Softmax**: Direct multiclass optimization
3. **Automatic Strategy Selection**: Algorithms choose the best approach

## Available Strategies

| Strategy | Class | Best For | Algorithms |
|----------|-------|----------|------------|
| One-vs-Rest | `OneVsRestClassifier` | Any binary classifier | All binary classifiers |
| Multinomial | `SoftmaxRegression` | Probabilistic classification | Logistic regression |
| Auto | `LogisticRegression` | Convenience | Logistic regression |
| Native | Tree algorithms | Tree-based models | Decision Trees, Random Forest, etc. |

## One-vs-Rest (OvR) Classification

One-vs-Rest trains one binary classifier for each class, treating it as a binary problem of "this class vs. all others."

### Basic Usage

```java
import org.superml.multiclass.OneVsRestClassifier;
import org.superml.linear_model.LogisticRegression;
import org.superml.datasets.Datasets;

// Create base binary classifier
LogisticRegression baseClassifier = new LogisticRegression()
    .setMaxIter(1000)
    .setRegularization("l2")
    .setC(1.0);

// Wrap with One-vs-Rest
OneVsRestClassifier ovr = new OneVsRestClassifier(baseClassifier);

// Load multiclass data
var dataset = Datasets.makeClassification(1000, 20, 3);  // 3 classes
var split = DataLoaders.trainTestSplit(dataset.X, 
    Arrays.stream(dataset.y).asDoubleStream().toArray(), 0.2, 42);

// Train
ovr.fit(split.XTrain, split.yTrain);

// Predict
double[] predictions = ovr.predict(split.XTest);
double[][] probabilities = ovr.predictProba(split.XTest);
```

### How OvR Works

For a 3-class problem (classes 0, 1, 2), OvR trains:
- Classifier 1: Class 0 vs (Class 1 + Class 2)
- Classifier 2: Class 1 vs (Class 0 + Class 2)  
- Classifier 3: Class 2 vs (Class 0 + Class 1)

During prediction, it runs all classifiers and selects the class with highest confidence.

### Advanced Configuration

```java
// Configure base classifier
LogisticRegression base = new LogisticRegression()
    .setMaxIter(2000)
    .setTol(1e-6)
    .setRegularization("l1")  // L1 for feature selection
    .setC(0.5);

OneVsRestClassifier ovr = new OneVsRestClassifier(base);

// You can also access individual classifiers after training
ovr.fit(XTrain, yTrain);
List<Classifier> binaryClassifiers = ovr.getClassifiers();
```

### With Different Base Classifiers

OvR works with any binary classifier:

```java
// With Decision Trees
import org.superml.tree.DecisionTree;
DecisionTree dt = new DecisionTree("gini", 10);
OneVsRestClassifier ovrTree = new OneVsRestClassifier(dt);

// With SVM (when available)
// SVM svm = new SVM("rbf").setC(1.0);
// OneVsRestClassifier ovrSVM = new OneVsRestClassifier(svm);
```

## Multinomial/Softmax Classification

Softmax regression directly optimizes for multiclass problems using the softmax function and cross-entropy loss.

### Basic Usage

```java
import org.superml.multiclass.SoftmaxRegression;

// Create softmax classifier
SoftmaxRegression softmax = new SoftmaxRegression()
    .setMaxIter(1000)
    .setLearningRate(0.01)
    .setRegularization("l2")
    .setC(1.0);

// Train on multiclass data
softmax.fit(XTrain, yTrain);

// Predict
double[] predictions = softmax.predict(XTest);
double[][] probabilities = softmax.predictProba(XTest);
```

### Key Features

- **Direct Optimization**: Optimizes multiclass objective directly
- **Probabilistic Output**: Natural probability interpretation
- **Regularization**: L1, L2, and Elastic Net support
- **Efficient**: Single model vs. multiple models in OvR

### Advanced Configuration

```java
SoftmaxRegression softmax = new SoftmaxRegression()
    .setMaxIter(2000)
    .setLearningRate(0.05)
    .setTol(1e-6)
    .setRegularization("elasticnet")
    .setC(0.1)
    .setL1Ratio(0.5)        // For elastic net
    .setRandomState(42);
```

## Enhanced Logistic Regression

The `LogisticRegression` class automatically handles multiclass problems:

### Automatic Strategy Selection

```java
import org.superml.linear_model.LogisticRegression;

LogisticRegression lr = new LogisticRegression()
    .setMaxIter(1000)
    .setMultiClass("auto");  // Chooses best strategy

// For binary: uses standard logistic regression
// For multiclass: uses multinomial by default
lr.fit(XTrain, yTrain);
```

### Manual Strategy Selection

```java
// Force One-vs-Rest
LogisticRegression lr = new LogisticRegression()
    .setMultiClass("ovr");

// Force Multinomial
LogisticRegression lr = new LogisticRegression()
    .setMultiClass("multinomial");
```

## Tree-Based Multiclass

Tree algorithms naturally handle multiclass problems:

### Decision Trees

```java
import org.superml.tree.DecisionTree;

// Decision trees handle multiclass natively
DecisionTree dt = new DecisionTree("gini", 10);
dt.fit(XTrain, yTrain);  // Works with any number of classes

double[] predictions = dt.predict(XTest);
double[][] probabilities = dt.predictProba(XTest);
```

### Random Forest

```java
import org.superml.tree.RandomForest;

RandomForest rf = new RandomForest(100, 10);
rf.fit(XTrain, yTrain);  // Handles multiclass automatically

// Get class predictions
double[] predictions = rf.predict(XTest);

// Get class probabilities  
double[][] probabilities = rf.predictProba(XTest);

// Access individual classes
double[] classes = rf.getClasses();
```

### Gradient Boosting

```java
import org.superml.tree.GradientBoosting;

// Note: Current implementation supports binary classification
// Multiclass support coming soon
GradientBoosting gb = new GradientBoosting(100, 0.1, 6);

// For now, use with OvR for multiclass
OneVsRestClassifier ovrGB = new OneVsRestClassifier(gb);
ovrGB.fit(XTrain, yTrain);
```

## Performance Comparison

Here's how different strategies compare:

```java
// Load 3-class dataset
var dataset = Datasets.makeClassification(1000, 20, 3);
var split = DataLoaders.trainTestSplit(dataset.X, 
    Arrays.stream(dataset.y).asDoubleStream().toArray(), 0.2, 42);

// Test different approaches
Map<String, Double> results = new HashMap<>();

// 1. One-vs-Rest with Logistic Regression
LogisticRegression base = new LogisticRegression().setMaxIter(1000);
OneVsRestClassifier ovr = new OneVsRestClassifier(base);
ovr.fit(split.XTrain, split.yTrain);
results.put("OvR", Metrics.accuracy(split.yTest, ovr.predict(split.XTest)));

// 2. Softmax Regression
SoftmaxRegression softmax = new SoftmaxRegression().setMaxIter(1000);
softmax.fit(split.XTrain, split.yTrain);
results.put("Softmax", Metrics.accuracy(split.yTest, softmax.predict(split.XTest)));

// 3. Enhanced Logistic Regression (auto)
LogisticRegression auto = new LogisticRegression().setMaxIter(1000);
auto.fit(split.XTrain, split.yTrain);
results.put("LR Auto", Metrics.accuracy(split.yTest, auto.predict(split.XTest)));

// 4. Random Forest
RandomForest rf = new RandomForest(100, 10);
rf.fit(split.XTrain, split.yTrain);
results.put("Random Forest", Metrics.accuracy(split.yTest, rf.predict(split.XTest)));

// Print results
results.forEach((method, accuracy) -> 
    System.out.println(method + ": " + String.format("%.4f", accuracy)));
```

## Evaluation Metrics

SuperML provides comprehensive metrics for multiclass evaluation:

### Basic Metrics

```java
import org.superml.metrics.Metrics;

double[] yTrue = {0, 1, 2, 0, 1, 2};
double[] yPred = {0, 1, 1, 0, 2, 2};

// Overall accuracy
double accuracy = Metrics.accuracy(yTrue, yPred);

// Per-class precision, recall, F1
double[] precision = Metrics.precisionScore(yTrue, yPred);
double[] recall = Metrics.recallScore(yTrue, yPred);
double[] f1 = Metrics.f1Score(yTrue, yPred);
```

### Confusion Matrix

```java
// Get confusion matrix
int[][] confMatrix = Metrics.confusionMatrix(yTrue, yPred, classes.length);

// Print confusion matrix
System.out.println("Confusion Matrix:");
for (int i = 0; i < confMatrix.length; i++) {
    System.out.println(Arrays.toString(confMatrix[i]));
}
```

### Classification Report

```java
// Detailed classification report
Map<String, Object> report = Metrics.classificationReport(yTrue, yPred);
System.out.println("Classification Report: " + report);
```

## Best Practices

### Choosing a Strategy

1. **Small datasets**: Use SoftmaxRegression for better convergence
2. **Large datasets**: OvR can be parallelized and may be faster  
3. **Tree algorithms**: Use native multiclass support
4. **Feature selection**: OvR with L1 regularization
5. **Probability calibration**: SoftmaxRegression gives better probabilities

### Data Preparation

```java
// Ensure labels start from 0 and are consecutive
double[] labels = {1, 3, 5, 1, 3, 5};  // Bad: non-consecutive
double[] goodLabels = {0, 1, 2, 0, 1, 2};  // Good: 0-indexed consecutive

// Handle class imbalance
// Consider stratified sampling or class weights (when available)
```

### Hyperparameter Tuning

```java
// Use cross-validation for hyperparameter selection
import org.superml.model_selection.GridSearchCV;

Map<String, Object[]> paramGrid = new HashMap<>();
paramGrid.put("C", new Object[]{0.1, 1.0, 10.0});
paramGrid.put("max_iter", new Object[]{1000, 2000});

GridSearchCV grid = new GridSearchCV(new SoftmaxRegression(), paramGrid, 5);
grid.fit(XTrain, yTrain);

SoftmaxRegression bestModel = (SoftmaxRegression) grid.getBestEstimator();
```

## Integration Examples

### With Preprocessing

```java
import org.superml.preprocessing.StandardScaler;
import org.superml.pipeline.Pipeline;

// Create preprocessing pipeline
Pipeline pipeline = new Pipeline()
    .addStep("scaler", new StandardScaler())
    .addStep("classifier", new SoftmaxRegression());

pipeline.fit(XTrain, yTrain);
double[] predictions = pipeline.predict(XTest);
```

### With Kaggle Integration

```java
import org.superml.datasets.KaggleTrainingManager;

// Setup training configuration
KaggleTrainingManager.TrainingConfig config = 
    new KaggleTrainingManager.TrainingConfig("multiclass-competition", "data.csv")
        .setValidationSplit(0.2)
        .setCrossValidation(true)
        .setCvFolds(5);

// Train multiclass model
SoftmaxRegression model = new SoftmaxRegression();
KaggleTrainingManager manager = new KaggleTrainingManager(credentials);
var result = manager.trainModel(model, config);
```

## Common Issues and Solutions

### Class Imbalance
```java
// Use stratified sampling in train/test split
// Consider class weights (feature coming soon)
// Evaluate with per-class metrics, not just accuracy
```

### Convergence Issues
```java
// Increase max_iter
SoftmaxRegression lr = new SoftmaxRegression().setMaxIter(5000);

// Decrease learning rate
SoftmaxRegression lr = new SoftmaxRegression().setLearningRate(0.001);

// Add regularization
SoftmaxRegression lr = new SoftmaxRegression().setC(0.1);
```

### Memory Issues with OvR
```java
// Use simpler base classifiers
// Consider reducing max_iter for base classifiers
// Use feature selection to reduce dimensionality
```

## Complete Examples

See the following example files for comprehensive demonstrations:
- [MulticlassExample.java](../examples/MulticlassExample.java) - Complete multiclass workflow
- [TreeAlgorithmsExample.java](../examples/TreeAlgorithmsExample.java) - Tree-based multiclass
- [MulticlassTest.java](../src/test/java/org/superml/multiclass/MulticlassTest.java) - Unit tests
