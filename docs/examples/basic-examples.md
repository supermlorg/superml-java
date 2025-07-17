---
title: "Basic Examples"
description: "Simple, focused examples to get started with SuperML Java 2.1.0"
layout: default
toc: true
search: true
---

# Basic Examples

This guide provides simple, focused examples to help you get started with SuperML Java 2.1.0. Each example demonstrates core concepts with the 21-module architecture and can be run independently.

## 🚀 Quick Start Examples

### Example 1: AutoML - One Line Classification

```java
import org.superml.datasets.Datasets;
import org.superml.autotrainer.AutoTrainer;
import org.superml.visualization.VisualizationFactory;

public class AutoMLExample {
    public static void main(String[] args) {
        System.out.println("🤖 SuperML Java 2.1.0 - AutoML Example");
        
        // 1. Load classic dataset
        var dataset = Datasets.loadIris();
        System.out.println("✓ Loaded Iris dataset: " + dataset.X.length + " samples, " + dataset.X[0].length + " features");
        
        // 2. AutoML - One line training!
        System.out.println("🔄 Running AutoML...");
        var result = AutoTrainer.autoML(dataset.X, dataset.y, "classification");
        
        // 3. Display results
        System.out.println("\n🏆 AutoML Results:");
        System.out.println("   Best Algorithm: " + result.getBestAlgorithm());
        System.out.println("   Best Score: " + String.format("%.3f", result.getBestScore()));
        System.out.println("   Best Parameters: " + result.getBestParams());
        
        // 4. Professional visualization (GUI + ASCII fallback)
        System.out.println("\n📊 Creating confusion matrix visualization...");
        VisualizationFactory.createDualModeConfusionMatrix(
            dataset.y, 
            result.getBestModel().predict(dataset.X),
            new String[]{"Setosa", "Versicolor", "Virginica"}
        ).display();
        
        System.out.println("✅ AutoML example completed!");
    }
}
```

### Example 2: Traditional Pipeline with Visualization

```java
import org.superml.datasets.Datasets;
import org.superml.linear_model.LogisticRegression;
import org.superml.preprocessing.StandardScaler;
import org.superml.pipeline.Pipeline;
import org.superml.model_selection.ModelSelection;
import org.superml.visualization.VisualizationFactory;
import org.superml.metrics.Metrics;

public class PipelineWithVisualizationExample {
    public static void main(String[] args) {
        System.out.println("🔧 SuperML Java 2.1.0 - Pipeline + Visualization Example");
        
        // 1. Load and prepare data
        var dataset = Datasets.loadIris();
        var split = ModelSelection.trainTestSplit(dataset.X, dataset.y, 0.2, 42);
        System.out.println("✓ Data split: " + split.XTrain.length + " train, " + split.XTest.length + " test");
        
        // 2. Create sophisticated ML pipeline
        var pipeline = new Pipeline()
            .addStep("scaler", new StandardScaler())
            .addStep("classifier", new LogisticRegression()
                .setMaxIter(1000)
                .setC(1.0)
                .setPenalty("l2"));
        
        System.out.println("🔄 Training pipeline...");
        
        // 3. Train the pipeline
        pipeline.fit(split.XTrain, split.yTrain);
        System.out.println("✓ Pipeline trained successfully");
        
        // 4. Make predictions
        double[] predictions = pipeline.predict(split.XTest);
        
        // 5. Evaluate performance
        var metrics = Metrics.classificationReport(split.yTest, predictions);
        System.out.println("\n📈 Performance Metrics:");
        System.out.println("   Accuracy: " + String.format("%.3f", metrics.accuracy));
        System.out.println("   F1-Score: " + String.format("%.3f", metrics.f1Score));
        System.out.println("   Precision: " + String.format("%.3f", metrics.precision));
        System.out.println("   Recall: " + String.format("%.3f", metrics.recall));
        
        // 6. Professional GUI visualization
        System.out.println("\n📊 Creating visualizations...");
        
        // Confusion Matrix
        VisualizationFactory.createDualModeConfusionMatrix(
            split.yTest, 
            predictions,
            new String[]{"Setosa", "Versicolor", "Virginica"}
        ).display();
        
        // Feature scatter plot
        VisualizationFactory.createXChartScatterPlot(
            dataset.X, 
            dataset.y,
            "Iris Dataset Feature Space",
            "Sepal Length", "Sepal Width"
        ).display();
        
        System.out.println("✅ Pipeline example completed!");
    }
}
```

## 🎯 Classification Examples

### Example 3: Multiple Algorithm Comparison

```java
import org.superml.datasets.Datasets;
import org.superml.linear_model.LogisticRegression;
import org.superml.tree.RandomForestClassifier;
import org.superml.tree.DecisionTreeClassifier;
import org.superml.model_selection.ModelSelection;
import org.superml.metrics.Metrics;
import org.superml.visualization.VisualizationFactory;

import java.util.*;

public class AlgorithmComparisonExample {
    public static void main(String[] args) {
        System.out.println("🏆 SuperML Java 2.1.0 - Algorithm Comparison Example");
        
        // 1. Generate challenging dataset
        var dataset = Datasets.makeClassification(1000, 20, 5, 42);
        var split = ModelSelection.trainTestSplit(dataset.X, dataset.y, 0.2, 42);
        System.out.println("✓ Generated dataset: " + dataset.X.length + " samples, " + dataset.X[0].length + " features, 5 classes");
        
        // 2. Define algorithms to compare
        Map<String, org.superml.core.Classifier> algorithms = new HashMap<>();
        algorithms.put("LogisticRegression", new LogisticRegression().setMaxIter(1000));
        algorithms.put("DecisionTree", new DecisionTreeClassifier().setMaxDepth(10));
        algorithms.put("RandomForest", new RandomForestClassifier().setNEstimators(100));
        
        // 3. Train and evaluate each algorithm
        Map<String, Double> scores = new HashMap<>();
        System.out.println("\n🔄 Training and evaluating algorithms...");
        
        for (Map.Entry<String, org.superml.core.Classifier> entry : algorithms.entrySet()) {
            String name = entry.getKey();
            org.superml.core.Classifier algorithm = entry.getValue();
            
            System.out.println("   Training " + name + "...");
            algorithm.fit(split.XTrain, split.yTrain);
            
            double[] predictions = algorithm.predict(split.XTest);
            var metrics = Metrics.classificationReport(split.yTest, predictions);
            scores.put(name, metrics.accuracy);
            
            System.out.println("   ✓ " + name + " accuracy: " + String.format("%.3f", metrics.accuracy));
        }
        
        // 4. Display results
        System.out.println("\n🏆 Algorithm Comparison Results:");
        scores.entrySet().stream()
            .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
            .forEach(entry -> System.out.println("   " + entry.getKey() + ": " + String.format("%.3f", entry.getValue())));
        
        // 5. Visualize comparison
        System.out.println("\n📊 Creating comparison visualization...");
        VisualizationFactory.createModelComparisonChart(
            new ArrayList<>(scores.keySet()),
            new ArrayList<>(scores.values()),
            "Algorithm Performance Comparison"
        ).display();
        
        System.out.println("✅ Algorithm comparison completed!");
    }
}
```

### Example 4: Cross-Validation and Hyperparameter Tuning

```java
import org.superml.datasets.Datasets;
import org.superml.linear_model.LogisticRegression;
import org.superml.model_selection.GridSearchCV;
import org.superml.model_selection.CrossValidation;
import org.superml.metrics.Metrics;

import java.util.*;

public class CrossValidationExample {
    public static void main(String[] args) {
        System.out.println("🔍 SuperML Java 2.1.0 - Cross-Validation and Hyperparameter Tuning");
        
        // 1. Load dataset
        var dataset = Datasets.loadIris();
        System.out.println("✓ Loaded dataset: " + dataset.X.length + " samples");
        
        // 2. Simple Cross-Validation
        System.out.println("\n📊 Performing 5-fold cross-validation...");
        var lr = new LogisticRegression().setMaxIter(1000);
        var cvResults = CrossValidation.crossValidateScore(lr, dataset.X, dataset.y, 5);
        
        System.out.println("   CV Scores: " + Arrays.toString(cvResults.scores));
        System.out.println("   Mean CV Score: " + String.format("%.3f ± %.3f", cvResults.meanScore, cvResults.stdScore));
        
        // 3. Grid Search for Hyperparameter Tuning
        System.out.println("\n🔧 Performing grid search for hyperparameter tuning...");
        
        // Define parameter grid
        Map<String, Object[]> paramGrid = new HashMap<>();
        paramGrid.put("C", new Double[]{0.1, 1.0, 10.0});
        paramGrid.put("penalty", new String[]{"l1", "l2"});
        paramGrid.put("maxIter", new Integer[]{500, 1000});
        
        // Perform grid search
        var gridSearch = new GridSearchCV(new LogisticRegression(), paramGrid, 5);
        gridSearch.fit(dataset.X, dataset.y);
        
        // Display results
        System.out.println("   Best Parameters: " + gridSearch.getBestParams());
        System.out.println("   Best CV Score: " + String.format("%.3f", gridSearch.getBestScore()));
        System.out.println("   Best Estimator: " + gridSearch.getBestEstimator().getClass().getSimpleName());
        
        // 4. Final model evaluation
        var split = org.superml.model_selection.ModelSelection.trainTestSplit(dataset.X, dataset.y, 0.2, 42);
        var bestModel = gridSearch.getBestEstimator();
        bestModel.fit(split.XTrain, split.yTrain);
        
        double[] predictions = bestModel.predict(split.XTest);
        var metrics = Metrics.classificationReport(split.yTest, predictions);
        
        System.out.println("\n🎯 Final Model Performance:");
        System.out.println("   Test Accuracy: " + String.format("%.3f", metrics.accuracy));
        System.out.println("   Test F1-Score: " + String.format("%.3f", metrics.f1Score));
        
        System.out.println("✅ Cross-validation example completed!");
    }
}
```

## 📈 Regression Examples

### Example 5: Regression with Multiple Algorithms

```java
import org.superml.datasets.Datasets;
import org.superml.linear_model.LinearRegression;
import org.superml.linear_model.Ridge;
import org.superml.linear_model.Lasso;
import org.superml.tree.DecisionTreeRegressor;
import org.superml.model_selection.ModelSelection;
import org.superml.metrics.Metrics;
import org.superml.visualization.VisualizationFactory;

import java.util.*;

public class RegressionComparisonExample {
    public static void main(String[] args) {
        System.out.println("📈 SuperML Java 2.1.0 - Regression Comparison Example");
        
        // 1. Generate regression dataset
        var dataset = Datasets.makeRegression(1000, 10, 0.1, 42);
        var split = ModelSelection.trainTestSplit(dataset.X, dataset.y, 0.2, 42);
        System.out.println("✓ Generated regression dataset: " + dataset.X.length + " samples, " + dataset.X[0].length + " features");
        
        // 2. Define regression algorithms
        Map<String, org.superml.core.Regressor> algorithms = new HashMap<>();
        algorithms.put("LinearRegression", new LinearRegression());
        algorithms.put("Ridge", new Ridge().setAlpha(1.0));
        algorithms.put("Lasso", new Lasso().setAlpha(0.1));
        algorithms.put("DecisionTree", new DecisionTreeRegressor().setMaxDepth(10));
        
        // 3. Train and evaluate each algorithm
        Map<String, Double> r2Scores = new HashMap<>();
        Map<String, Double> mseScores = new HashMap<>();
        System.out.println("\n🔄 Training and evaluating regression algorithms...");
        
        for (Map.Entry<String, org.superml.core.Regressor> entry : algorithms.entrySet()) {
            String name = entry.getKey();
            org.superml.core.Regressor algorithm = entry.getValue();
            
            System.out.println("   Training " + name + "...");
            algorithm.fit(split.XTrain, split.yTrain);
            
            double[] predictions = algorithm.predict(split.XTest);
            var metrics = Metrics.regressionReport(split.yTest, predictions);
            
            r2Scores.put(name, metrics.r2Score);
            mseScores.put(name, metrics.mse);
            
            System.out.println("   ✓ " + name + " - R²: " + String.format("%.3f", metrics.r2Score) + 
                             ", MSE: " + String.format("%.3f", metrics.mse));
        }
        
        // 4. Display results
        System.out.println("\n🏆 Regression Comparison Results (by R² Score):");
        r2Scores.entrySet().stream()
            .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
            .forEach(entry -> System.out.println("   " + entry.getKey() + ": " + String.format("%.3f", entry.getValue())));
        
        // 5. Visualize best model predictions
        String bestModel = r2Scores.entrySet().stream()
            .max(Map.Entry.comparingByValue())
            .map(Map.Entry::getKey)
            .orElse("LinearRegression");
        
        System.out.println("\n📊 Creating prediction visualization for best model: " + bestModel);
        var bestAlgorithm = algorithms.get(bestModel);
        double[] bestPredictions = bestAlgorithm.predict(split.XTest);
        
        VisualizationFactory.createRegressionPlot(
            split.yTest,
            bestPredictions,
            "Regression Results: " + bestModel
        ).display();
        
        System.out.println("✅ Regression comparison completed!");
    }
}
```

## 🔗 Clustering Examples

### Example 6: K-Means Clustering with Visualization

```java
import org.superml.datasets.Datasets;
import org.superml.cluster.KMeans;
import org.superml.visualization.VisualizationFactory;
import org.superml.metrics.Metrics;

public class ClusteringExample {
    public static void main(String[] args) {
        System.out.println("🔗 SuperML Java 2.1.0 - K-Means Clustering Example");
        
        // 1. Generate clustering dataset
        var dataset = Datasets.makeBlobs(300, 2, 4, 42); // 300 samples, 2 features, 4 clusters
        System.out.println("✓ Generated clustering dataset: " + dataset.X.length + " samples, " + dataset.X[0].length + " features");
        
        // 2. Apply K-Means clustering
        System.out.println("\n🔄 Applying K-Means clustering...");
        var kmeans = new KMeans(4)
            .setInit("k-means++")
            .setNInit(10)
            .setMaxIter(300)
            .setTol(1e-4);
        
        kmeans.fit(dataset.X);
        int[] clusterLabels = kmeans.predict(dataset.X);
        
        // 3. Display clustering results
        System.out.println("✓ Clustering completed:");
        System.out.println("   Final inertia: " + String.format("%.2f", kmeans.getInertia()));
        System.out.println("   Iterations to convergence: " + kmeans.getNumIter());
        System.out.println("   Cluster centers shape: " + kmeans.getClusterCenters().length + "x" + kmeans.getClusterCenters()[0].length);
        
        // 4. Calculate clustering metrics
        double silhouetteScore = Metrics.silhouetteScore(dataset.X, clusterLabels);
        System.out.println("   Silhouette score: " + String.format("%.3f", silhouetteScore));
        
        // 5. Visualize clustering results
        System.out.println("\n📊 Creating cluster visualization...");
        
        // Convert int[] to double[] for visualization
        double[] clusterLabelsDouble = Arrays.stream(clusterLabels).asDoubleStream().toArray();
        
        VisualizationFactory.createXChartScatterPlot(
            dataset.X,
            clusterLabelsDouble,
            "K-Means Clustering Results (k=4)",
            "Feature 1", "Feature 2"
        ).display();
        
        // 6. Show cluster centers
        System.out.println("\n🎯 Cluster Centers:");
        double[][] centers = kmeans.getClusterCenters();
        for (int i = 0; i < centers.length; i++) {
            System.out.println("   Cluster " + i + ": [" + 
                String.format("%.2f", centers[i][0]) + ", " + 
                String.format("%.2f", centers[i][1]) + "]");
        }
        
        System.out.println("✅ Clustering example completed!");
    }
}
```

## ⚡ Production Examples

### Example 7: High-Performance Inference

```java
import org.superml.datasets.Datasets;
import org.superml.linear_model.LogisticRegression;
import org.superml.inference.InferenceEngine;
import org.superml.persistence.ModelPersistence;

import java.util.concurrent.CompletableFuture;

public class ProductionInferenceExample {
    public static void main(String[] args) throws Exception {
        System.out.println("⚡ SuperML Java 2.1.0 - Production Inference Example");
        
        // 1. Train and save a model
        var dataset = Datasets.loadIris();
        var model = new LogisticRegression().setMaxIter(1000);
        model.fit(dataset.X, dataset.y);
        
        String modelPath = ModelPersistence.saveWithStats(
            model, "iris_classifier", 
            "Production Iris classifier",
            dataset.X, dataset.y
        );
        System.out.println("✓ Model saved: " + modelPath);
        
        // 2. Setup production inference engine
        var engine = new InferenceEngine()
            .setModelCache(true)
            .setPerformanceMonitoring(true)
            .setBatchSize(100);
        
        // 3. Load and register model
        var loadedModel = ModelPersistence.load(modelPath);
        engine.registerModel("iris_classifier", loadedModel);
        System.out.println("✓ Model registered for inference");
        
        // 4. Single prediction
        System.out.println("\n🎯 Single prediction:");
        double[][] singleSample = { {5.1, 3.5, 1.4, 0.2} };
        double[] singlePrediction = engine.predict("iris_classifier", singleSample);
        System.out.println("   Prediction: " + singlePrediction[0]);
        System.out.println("   Inference time: " + engine.getLastInferenceTime() + "μs");
        
        // 5. Batch prediction
        System.out.println("\n📦 Batch prediction:");
        double[][] batchData = {
            {5.1, 3.5, 1.4, 0.2},
            {6.7, 3.1, 4.4, 1.4}, 
            {6.3, 3.3, 6.0, 2.5}
        };
        double[] batchPredictions = engine.predict("iris_classifier", batchData);
        System.out.println("   Batch predictions: " + java.util.Arrays.toString(batchPredictions));
        System.out.println("   Batch inference time: " + engine.getLastInferenceTime() + "μs");
        
        // 6. Asynchronous prediction
        System.out.println("\n🚀 Asynchronous prediction:");
        CompletableFuture<Double> asyncPrediction = engine.predictAsync("iris_classifier", singleSample[0]);
        double asyncResult = asyncPrediction.get();
        System.out.println("   Async prediction: " + asyncResult);
        
        // 7. Performance metrics
        var metrics = engine.getMetrics("iris_classifier");
        System.out.println("\n📊 Performance Metrics:");
        System.out.println("   Total predictions: " + metrics.getTotalPredictions());
        System.out.println("   Average inference time: " + String.format("%.1f", metrics.getAverageInferenceTime()) + "μs");
        System.out.println("   Predictions per second: " + String.format("%.0f", metrics.getPredictionsPerSecond()));
        
        System.out.println("✅ Production inference example completed!");
    }
}
```

### Example 8: Professional XChart Visualization Showcase

```java
import org.superml.datasets.Datasets;
import org.superml.linear_model.LogisticRegression;
import org.superml.tree.RandomForestClassifier;
import org.superml.model_selection.ModelSelection;
import org.superml.visualization.VisualizationFactory;
import org.superml.metrics.Metrics;

import java.util.*;

public class XChartVisualizationShowcase {
    public static void main(String[] args) {
        System.out.println("📊 SuperML Java 2.1.0 - XChart Visualization Showcase");
        
        // 1. Prepare data and models
        var dataset = Datasets.loadIris();
        var split = ModelSelection.trainTestSplit(dataset.X, dataset.y, 0.2, 42);
        
        // Train multiple models
        var lr = new LogisticRegression().setMaxIter(1000);
        var rf = new RandomForestClassifier().setNEstimators(100);
        
        lr.fit(split.XTrain, split.yTrain);
        rf.fit(split.XTrain, split.yTrain);
        
        // Get predictions
        double[] lrPredictions = lr.predict(split.XTest);
        double[] rfPredictions = rf.predict(split.XTest);
        
        System.out.println("✓ Models trained and predictions generated");
        
        // 2. Professional Confusion Matrix
        System.out.println("\n📊 Creating professional confusion matrix...");
        VisualizationFactory.createXChartConfusionMatrix(
            split.yTest, 
            lrPredictions,
            new String[]{"Setosa", "Versicolor", "Virginica"}
        ).display();
        
        // 3. Feature Scatter Plot with Classes
        System.out.println("📊 Creating feature scatter plot...");
        VisualizationFactory.createXChartScatterPlot(
            dataset.X,
            dataset.y,
            "Iris Dataset Feature Distribution",
            "Sepal Length", "Sepal Width"
        ).display();
        
        // 4. Model Performance Comparison Chart
        System.out.println("📊 Creating model comparison chart...");
        var lrMetrics = Metrics.classificationReport(split.yTest, lrPredictions);
        var rfMetrics = Metrics.classificationReport(split.yTest, rfPredictions);
        
        VisualizationFactory.createModelComparisonChart(
            Arrays.asList("Logistic Regression", "Random Forest"),
            Arrays.asList(lrMetrics.accuracy, rfMetrics.accuracy),
            "Model Accuracy Comparison"
        ).display();
        
        // 5. Dual-Mode Visualization (Auto-detects GUI/ASCII)
        System.out.println("📊 Creating dual-mode visualization...");
        VisualizationFactory.createDualModeConfusionMatrix(
            split.yTest,
            rfPredictions,
            new String[]{"Setosa", "Versicolor", "Virginica"}
        ).display();
        
        System.out.println("✅ XChart visualization showcase completed!");
        System.out.println("💡 Tip: XChart GUI windows can be saved as PNG/JPEG images!");
    }
}
```

## 🎓 Learning Path

### Recommended Order for Beginners:

1. **AutoML Example** - Start here for instant results and framework overview
2. **Pipeline with Visualization** - Learn core concepts and see professional charts  
3. **Algorithm Comparison** - Understand different algorithms and their performance
4. **Cross-Validation Example** - Learn proper model evaluation and hyperparameter tuning
5. **Regression Comparison** - Explore regression tasks and metrics
6. **Clustering Example** - Understand unsupervised learning
7. **Production Inference** - Learn deployment and performance optimization
8. **XChart Visualization** - Master professional chart creation

### Next Steps:
- Explore the [Advanced Examples](advanced-examples.md) for complex workflows
- Check out the [API Reference](../api/core-classes.md) for comprehensive documentation
- Try the complete example suite in the `superml-examples` module
- Build your own custom ML applications using the modular architecture

---

**Ready to build amazing ML applications!** 🚀 Each example showcases different aspects of SuperML Java 2.1.0's powerful 21-module architecture.
