---
title: "Quick Start Guide"
description: "Get up and running with SuperML Java 2.1.0 in minutes with AutoML, neural networks, and visualization"
layout: default
toc: true
search: true
---

# Quick Start Guide

Get up and running with SuperML Java 2.1.0 in just a few minutes! This guide will walk you through setting up the framework, training your first model with AutoML, deep learning capabilities, and creating professional visualizations.

## 🚀 5-Minute Quickstart with AutoML & Visualization

### Step 1: Add Dependency

#### Complete Framework (Recommended)
```xml
<dependency>
    <groupId>org.superml</groupId>
    <artifactId>superml-bundle-all</artifactId>
    <version>2.1.0</version>
</dependency>
```

#### Modular Installation (Advanced)
```xml
<!-- Core + Linear Models + Neural Networks + Visualization -->
<dependency>
    <groupId>org.superml</groupId>
    <artifactId>superml-core</artifactId>
    <version>2.1.0</version>
</dependency>
<dependency>
    <groupId>org.superml</groupId>
    <artifactId>superml-linear-models</artifactId>
    <version>2.1.0</version>
</dependency>
<dependency>
    <groupId>org.superml</groupId>
    <artifactId>superml-neural</artifactId>
    <version>2.1.0</version>
</dependency>
<dependency>
    <groupId>org.superml</groupId>
    <artifactId>superml-visualization</artifactId>
    <version>2.1.0</version>
</dependency>
```

### Step 2: AutoML - Your First Model (One Line!)

```java
import org.superml.datasets.Datasets;
import org.superml.autotrainer.AutoTrainer;
import org.superml.visualization.VisualizationFactory;

public class QuickStartAutoML {
    public static void main(String[] args) {
        // 1. Load a dataset
        var dataset = Datasets.loadIris();
        
        // 2. AutoML - One line training!
        var result = AutoTrainer.autoML(dataset.X, dataset.y, "classification");
        
        System.out.println("🎯 Best Algorithm: " + result.getBestAlgorithm());
        System.out.println("📊 Best Score: " + result.getBestScore());
        System.out.println("⚙️ Best Parameters: " + result.getBestParams());
        
        // 3. Professional visualization (GUI + ASCII fallback)
        VisualizationFactory.createDualModeConfusionMatrix(
            dataset.y, 
            result.getBestModel().predict(dataset.X),
            new String[]{"Setosa", "Versicolor", "Virginica"}
        ).display();
    }
}
```

### Step 3: Traditional ML Pipeline with Visualization

```java
import org.superml.datasets.Datasets;
import org.superml.linear_model.LogisticRegression;
import org.superml.preprocessing.StandardScaler;
import org.superml.pipeline.Pipeline;
import org.superml.model_selection.ModelSelection;
import org.superml.visualization.VisualizationFactory;

public class QuickStartPipeline {
    public static void main(String[] args) {
        // 1. Load and split data
        var dataset = Datasets.loadIris();
        var split = ModelSelection.trainTestSplit(dataset.X, dataset.y, 0.2, 42);
        
        // 2. Create ML pipeline
        var pipeline = new Pipeline()
            .addStep("scaler", new StandardScaler())
            .addStep("classifier", new LogisticRegression().setMaxIter(1000));
        
        // 3. Train the pipeline
        pipeline.fit(split.XTrain, split.yTrain);
        
        // 4. Make predictions
        double[] predictions = pipeline.predict(split.XTest);
        double[] predictions = model.predict(split.XTest);
        double[][] probabilities = model.predictProba(split.XTest);
        
        // 5. Evaluate performance
        var metrics = Metrics.classificationReport(split.yTest, predictions);
        System.out.println("📈 Accuracy: " + String.format("%.3f", metrics.accuracy));
        System.out.println("📊 F1-Score: " + String.format("%.3f", metrics.f1Score));
        
        // 6. Create professional confusion matrix (GUI or ASCII)
        VisualizationFactory.createDualModeConfusionMatrix(
            split.yTest, 
            predictions,
            new String[]{"Setosa", "Versicolor", "Virginica"}
        ).display();
        
        // 7. Feature importance visualization
        VisualizationFactory.createRegressionPlot(
            split.yTest, 
            predictions,
            "Iris Classification Results"
        ).display();
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

## 🎯 Advanced Features Showcase

### AutoML with Hyperparameter Optimization

```java
import org.superml.datasets.Datasets;
import org.superml.autotrainer.AutoTrainer;
import org.superml.model_selection.GridSearchCV;

public class AdvancedAutoML {
    public static void main(String[] args) {
        // Load dataset
        var dataset = Datasets.makeClassification(1000, 20, 5, 42);
        
        // Advanced AutoML with custom configuration
        var config = new AutoTrainer.Config()
            .setAlgorithms("logistic", "randomforest", "gradientboosting")
            .setSearchStrategy("random")  // or "grid", "bayesian"
            .setCrossValidationFolds(5)
            .setMaxEvaluationTime(300)  // 5 minutes max
            .setEnsembleMethods(true);
        
        var result = AutoTrainer.autoMLWithConfig(dataset.X, dataset.y, config);
        
        System.out.println("🏆 Best Model Performance:");
        System.out.println("   Algorithm: " + result.getBestAlgorithm());
        System.out.println("   CV Score: " + String.format("%.4f", result.getBestScore()));
        System.out.println("   Parameters: " + result.getBestParams());
        
        // Get ensemble if available
        if (result.hasEnsemble()) {
            System.out.println("🤖 Ensemble Performance: " + 
                String.format("%.4f", result.getEnsembleScore()));
        }
    }
}
```

### Production Inference with Monitoring

```java
import org.superml.inference.InferenceEngine;
import org.superml.persistence.ModelPersistence;
import org.superml.drift.DriftDetector;

public class ProductionInference {
    public static void main(String[] args) {
        // Load trained model
        var model = ModelPersistence.load("my_iris_model.json");
        
        // Setup inference engine
        var engine = new InferenceEngine()
            .setModelCache(true)
            .setPerformanceMonitoring(true)
            .setBatchSize(100);
        
        // Register model
        engine.registerModel("iris_classifier", model);
        
        // Setup drift monitoring
        var driftDetector = new DriftDetector("iris_classifier")
            .setThreshold(0.05)
            .setAlertCallback(alert -> {
                System.out.println("🚨 Drift detected: " + alert.getMessage());
            });
        
        // Make predictions with monitoring
        double[][] newData = { {5.1, 3.5, 1.4, 0.2} };
        double[] predictions = engine.predict("iris_classifier", newData);
        
        // Monitor for drift
        driftDetector.checkDrift(newData, predictions);
        
        System.out.println("🎯 Prediction: " + predictions[0]);
        System.out.println("⚡ Inference time: " + engine.getLastInferenceTime() + "μs");
    }
}
```

### Kaggle Competition Integration

```java
import org.superml.kaggle.KaggleTrainingManager;
import org.superml.kaggle.KaggleIntegration.KaggleCredentials;

public class KaggleCompetition {
    public static void main(String[] args) {
        // Setup Kaggle credentials
        var credentials = KaggleCredentials.fromDefaultLocation();
        var manager = new KaggleTrainingManager(credentials);
        
        // One-line training on any Kaggle dataset
        var config = new KaggleTrainingManager.TrainingConfig()
            .setAlgorithms("logistic", "randomforest", "xgboost")
            .setGridSearch(true)
            .setSaveModels(true)
            .setSubmissionFormat(true);
        
        var results = manager.trainOnDataset(
            "titanic",           // competition name
            "titanic",           // dataset name  
            "survived",          // target column
            config
        );
        
        // Best model results
        var bestResult = results.get(0);
        System.out.println("🏆 Best Model: " + bestResult.algorithm);
        System.out.println("📊 CV Score: " + String.format("%.4f", bestResult.cvScore));
        System.out.println("💾 Model saved: " + bestResult.modelFilePath);
        System.out.println("📤 Submission: " + bestResult.submissionFilePath);
    }
}
```

## 📊 Visualization Examples

### Professional GUI Charts

```java
import org.superml.visualization.VisualizationFactory;
import org.superml.datasets.Datasets;

public class VisualizationShowcase {
    public static void main(String[] args) {
        var dataset = Datasets.loadIris();
        
        // 1. Interactive Confusion Matrix (XChart GUI)
        VisualizationFactory.createXChartConfusionMatrix(
            dataset.y,
            someModel.predict(dataset.X),
            new String[]{"Setosa", "Versicolor", "Virginica"}
        ).display();
        
        // 2. Feature Scatter Plot with Clusters
        VisualizationFactory.createXChartScatterPlot(
            dataset.X,
            dataset.y,
            "Iris Dataset Features",
            "Sepal Length", "Sepal Width"
        ).display();
        
        // 3. Model Performance Comparison
        VisualizationFactory.createModelComparisonChart(
            Arrays.asList("LogisticRegression", "RandomForest", "SVM"),
            Arrays.asList(0.95, 0.97, 0.94),
            "Model Performance Comparison"
        ).display();
        
        // 4. Automatic fallback to ASCII if no GUI
        VisualizationFactory.createDualModeConfusionMatrix(dataset.y, predictions)
            .setAsciiMode(true)  // Force ASCII mode
            .display();
    }
}
```

## 🔧 Module Selection Guide

### Minimal Setup (Core ML only)
```xml
<!-- Just core algorithms -->
<dependency>
    <groupId>org.superml</groupId>
    <artifactId>superml-core</artifactId>
    <version>2.1.0</version>
</dependency>
<dependency>
    <groupId>org.superml</groupId>
    <artifactId>superml-linear-models</artifactId>
    <version>2.1.0</version>
</dependency>
```

### Standard ML Pipeline
```xml
<!-- Core + preprocessing + model selection -->
<dependency>
    <groupId>org.superml</groupId>
    <artifactId>superml-core</artifactId>
    <version>2.1.0</version>
</dependency>
<dependency>
    <groupId>org.superml</groupId>
    <artifactId>superml-linear-models</artifactId>
    <version>2.1.0</version>
</dependency>
<dependency>
    <groupId>org.superml</groupId>
    <artifactId>superml-preprocessing</artifactId>
    <version>2.1.0</version>
</dependency>
<dependency>
    <groupId>org.superml</groupId>
    <artifactId>superml-model-selection</artifactId>
    <version>2.1.0</version>
</dependency>
```

### AutoML & Visualization
```xml
<!-- Add AutoML and professional visualization -->
<dependency>
    <groupId>org.superml</groupId>
    <artifactId>superml-autotrainer</artifactId>
    <version>2.1.0</version>
</dependency>
<dependency>
    <groupId>org.superml</groupId>
    <artifactId>superml-visualization</artifactId>
    <version>2.1.0</version>
</dependency>
```

### Production Deployment
```xml
<!-- Add inference engine and model persistence -->
<dependency>
    <groupId>org.superml</groupId>
    <artifactId>superml-inference</artifactId>
    <version>2.1.0</version>
</dependency>
<dependency>
    <groupId>org.superml</groupId>
    <artifactId>superml-persistence</artifactId>
    <version>2.1.0</version>
</dependency>
<dependency>
    <groupId>org.superml</groupId>
    <artifactId>superml-drift</artifactId>
    <version>2.1.0</version>
</dependency>
```

### Everything (Recommended for Development)
```xml
<!-- Complete framework -->
<dependency>
    <groupId>org.superml</groupId>
    <artifactId>superml-bundle-all</artifactId>
    <version>2.1.0</version>
</dependency>
```

## 🎓 Next Steps

### Learning Path
1. **Start Here**: Run the AutoML example above
2. **Core Concepts**: Try the pipeline example  
3. **Advanced Features**: Experiment with visualization
4. **Production**: Explore inference and persistence
5. **Competitions**: Try Kaggle integration
6. **Custom Solutions**: Build your own ML applications

### Essential Documentation
- [**Modular Architecture**](modular-architecture.md) - Understanding the 21-module system
- [**Algorithm Reference**](algorithms-reference.md) - Complete guide to all 15+ algorithms
- [**Examples Collection**](examples/basic-examples.md) - 11 comprehensive examples
- [**API Reference**](api/core-classes.md) - Complete API documentation
- [**Production Guide**](inference-guide.md) - Deployment and monitoring

### Code Examples
All code examples are available in the `superml-examples` module:
- `BasicClassification.java` - Fundamental concepts
- `AutoMLExample.java` - Automated machine learning
- `XChartVisualizationExample.java` - Professional GUI charts
- `ProductionInferenceExample.java` - High-performance serving
- `KaggleIntegrationExample.java` - Competition workflows

---

**Ready to build amazing ML applications with SuperML Java 2.1.0!** 🚀

Start with AutoML for instant results, then dive deeper into the modular architecture for custom solutions.
