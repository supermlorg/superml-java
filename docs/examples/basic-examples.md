# Basic Examples

This guide provides simple, focused examples to help you get started with SuperML Java. Each example demonstrates one core concept and can be run independently.

## 🎯 Classification Examples

### Example 1: Basic Binary Classification

```java
import org.superml.datasets.Datasets;
import org.superml.linear_model.LogisticRegression;
import org.superml.metrics.Metrics;
import org.superml.datasets.DataLoaders;

public class BasicClassification {
    public static void main(String[] args) {
        // 1. Generate synthetic binary classification data
        var dataset = Datasets.makeClassification(1000, 10, 2);
        
        // 2. Split into train/test sets
        var split = DataLoaders.trainTestSplit(dataset.X, 
            Arrays.stream(dataset.y).asDoubleStream().toArray(), 0.2, 42);
        
        // 3. Create and train classifier
        var classifier = new LogisticRegression()
            .setMaxIter(1000)
            .setLearningRate(0.01);
        
        classifier.fit(split.XTrain, split.yTrain);
        
        // 4. Make predictions
        double[] predictions = classifier.predict(split.XTest);
        
        // 5. Evaluate performance
        double accuracy = Metrics.accuracy(split.yTest, predictions);
        double precision = Metrics.precision(split.yTest, predictions);
        double recall = Metrics.recall(split.yTest, predictions);
        double f1 = Metrics.f1Score(split.yTest, predictions);
        
        // 6. Print results
        System.out.printf("Accuracy:  %.3f\n", accuracy);
        System.out.printf("Precision: %.3f\n", precision);
        System.out.printf("Recall:    %.3f\n", recall);
        System.out.printf("F1-Score:  %.3f\n", f1);
        
        // 7. Show confusion matrix
        int[][] confMatrix = Metrics.confusionMatrix(split.yTest, predictions);
        System.out.println("\nConfusion Matrix:");
        for (int[] row : confMatrix) {
            System.out.println(Arrays.toString(row));
        }
    }
}
```

Expected Output:
```
Accuracy:  0.955
Precision: 0.952
Recall:    0.950
F1-Score:  0.951

Confusion Matrix:
[95, 5]
[4, 96]
```

### Example 2: Multiclass Classification with One-vs-Rest

```java
import org.superml.datasets.Datasets;
import org.superml.multiclass.OneVsRestClassifier;
import org.superml.linear_model.LogisticRegression;
import org.superml.metrics.Metrics;

public class MulticlassExample {
    public static void main(String[] args) {
        // 1. Generate 3-class dataset
        var dataset = Datasets.makeClassification(1000, 15, 3);
        var split = DataLoaders.trainTestSplit(dataset.X, 
            Arrays.stream(dataset.y).asDoubleStream().toArray(), 0.2, 42);
        
        // 2. Create One-vs-Rest classifier
        LogisticRegression base = new LogisticRegression().setMaxIter(1000);
        OneVsRestClassifier ovr = new OneVsRestClassifier(base);
        
        // 3. Train and predict
        ovr.fit(split.XTrain, split.yTrain);
        double[] predictions = ovr.predict(split.XTest);
        double[][] probabilities = ovr.predictProba(split.XTest);
        
        // 4. Evaluate
        double accuracy = Metrics.accuracy(split.yTest, predictions);
        System.out.printf("Multiclass Accuracy: %.3f\n", accuracy);
        
        // 5. Show probability predictions for first few samples
        for (int i = 0; i < 3; i++) {
            System.out.printf("Sample %d: Predicted=%.0f, Probabilities=[%.3f, %.3f, %.3f]\n",
                i+1, predictions[i], probabilities[i][0], probabilities[i][1], probabilities[i][2]);
        }
    }
}
```

### Example 3: Decision Tree Classification

```java
import org.superml.datasets.Datasets;
import org.superml.tree.DecisionTree;
import org.superml.metrics.Metrics;

public class DecisionTreeExample {
    public static void main(String[] args) {
        // 1. Load dataset
        var dataset = Datasets.loadIris();  // 3-class iris dataset
        var split = DataLoaders.trainTestSplit(dataset.X, 
            Arrays.stream(dataset.y).asDoubleStream().toArray(), 0.3, 42);
        
        // 2. Create and configure decision tree
        DecisionTree dt = new DecisionTree("gini", 5)
            .setMinSamplesSplit(5)
            .setMinSamplesLeaf(2)
            .setRandomState(42);
        
        // 3. Train
        dt.fit(split.XTrain, split.yTrain);
        
        // 4. Predict
        double[] predictions = dt.predict(split.XTest);
        double[][] probabilities = dt.predictProba(split.XTest);
        
        // 5. Evaluate
        double accuracy = dt.score(split.XTest, split.yTest);
        System.out.printf("Decision Tree Accuracy: %.3f\n", accuracy);
        
        // 6. Show tree characteristics
        System.out.println("Tree trained successfully");
        System.out.println("Classes: " + Arrays.toString(dt.getClasses()));
    }
}
```

### Example 4: Random Forest vs Single Tree

```java
import org.superml.datasets.Datasets;
import org.superml.tree.DecisionTree;
import org.superml.tree.RandomForest;
import org.superml.metrics.Metrics;

public class ForestComparisonExample {
    public static void main(String[] args) {
        // 1. Generate challenging dataset
        var dataset = Datasets.makeClassification(1000, 20, 2);
        var split = DataLoaders.trainTestSplit(dataset.X, 
            Arrays.stream(dataset.y).asDoubleStream().toArray(), 0.2, 42);
        
        // 2. Train single decision tree
        DecisionTree dt = new DecisionTree("gini", 10);
        long start = System.currentTimeMillis();
        dt.fit(split.XTrain, split.yTrain);
        long dtTime = System.currentTimeMillis() - start;
        
        double dtAccuracy = dt.score(split.XTest, split.yTest);
        
        // 3. Train random forest
        RandomForest rf = new RandomForest(100, 10);
        start = System.currentTimeMillis();
        rf.fit(split.XTrain, split.yTrain);
        long rfTime = System.currentTimeMillis() - start;
        
        double rfAccuracy = rf.score(split.XTest, split.yTest);
        
        // 4. Compare results
        System.out.println("=== Comparison Results ===");
        System.out.printf("Decision Tree: %.3f accuracy, %d ms\n", dtAccuracy, dtTime);
        System.out.printf("Random Forest: %.3f accuracy, %d ms\n", rfAccuracy, rfTime);
        System.out.printf("Improvement: %.3f\n", rfAccuracy - dtAccuracy);
        System.out.printf("Forest has %d trees\n", rf.getTrees().size());
    }
}
```

## 📈 Regression Examples

### Example 5: Linear Regression

```java
import org.superml.datasets.Datasets;
import org.superml.linear_model.LinearRegression;
import org.superml.metrics.Metrics;
import org.superml.model_selection.ModelSelection;

public class BasicRegression {
    public static void main(String[] args) {
        // 1. Generate synthetic regression data
        var dataset = Datasets.makeRegression(
            500,     // 500 samples
            5,       // 5 features
            0.1,     // noise level
            42       // random seed
        );
        
        // 2. Split data
        var split = ModelSelection.trainTestSplit(
            dataset.X, dataset.y, 0.2, 42);
        
        // 3. Train linear regression
        var regressor = new LinearRegression();
        regressor.fit(split.XTrain, split.yTrain);
        
        // 4. Make predictions
        double[] predictions = regressor.predict(split.XTest);
        
        // 5. Calculate regression metrics
        double mse = Metrics.meanSquaredError(split.yTest, predictions);
        double mae = Metrics.meanAbsoluteError(split.yTest, predictions);
        double r2 = Metrics.r2Score(split.yTest, predictions);
        double rmse = Math.sqrt(mse);
        
        System.out.println("Regression Results:");
        System.out.printf("R² Score: %.4f\n", r2);
        System.out.printf("RMSE:     %.4f\n", rmse);
        System.out.printf("MAE:      %.4f\n", mae);
        System.out.printf("MSE:      %.4f\n", mse);
        
        // 6. Show some predictions vs actual
        System.out.println("\nSample Predictions vs Actual:");
        System.out.println("Actual    | Predicted | Error");
        System.out.println("----------|-----------|-------");
        for (int i = 0; i < Math.min(10, predictions.length); i++) {
            double actual = split.yTest[i];
            double predicted = predictions[i];
            double error = Math.abs(actual - predicted);
            System.out.printf("%8.3f | %9.3f | %6.3f\n", actual, predicted, error);
        }
    }
}
```

### Example 6: Regularized Regression Comparison

```java
import org.superml.datasets.Datasets;
import org.superml.linear_model.*;
import org.superml.metrics.Metrics;
import org.superml.model_selection.ModelSelection;

public class RegularizedRegression {
    public static void main(String[] args) {
        // 1. Generate data with some irrelevant features
        var dataset = Datasets.makeRegression(
            300,     // samples
            20,      // features (some will be irrelevant)
            0.1,     // noise
            42       // seed
        );
        
        // 2. Split data
        var split = ModelSelection.trainTestSplit(
            dataset.X, dataset.y, 0.25, 42);
        
        // 3. Train different regression models
        var linear = new LinearRegression();
        var ridge = new Ridge().setAlpha(1.0);
        var lasso = new Lasso().setAlpha(0.1);
        
        // Train all models
        linear.fit(split.XTrain, split.yTrain);
        ridge.fit(split.XTrain, split.yTrain);
        lasso.fit(split.XTrain, split.yTrain);
        
        // 4. Compare performance
        System.out.println("Model Comparison:");
        System.out.println("================");
        
        String[] modelNames = {"Linear", "Ridge", "Lasso"};
        var models = new Regressor[]{linear, ridge, lasso};
        
        for (int i = 0; i < models.length; i++) {
            double[] pred = models[i].predict(split.XTest);
            double r2 = Metrics.r2Score(split.yTest, pred);
            double rmse = Math.sqrt(Metrics.meanSquaredError(split.yTest, pred));
            
            System.out.printf("%-8s: R²=%.4f, RMSE=%.4f\n", 
                modelNames[i], r2, rmse);
        }
        
        // 5. Show feature selection with Lasso
        System.out.println("\nLasso Feature Selection:");
        double[] lassoCoefs = lasso.getCoefficients();
        int selectedFeatures = 0;
        for (int i = 0; i < lassoCoefs.length; i++) {
            if (Math.abs(lassoCoefs[i]) > 1e-6) {
                selectedFeatures++;
                System.out.printf("Feature %d: %.4f\n", i, lassoCoefs[i]);
            }
        }
        System.out.printf("Selected %d out of %d features\n", 
            selectedFeatures, lassoCoefs.length);
    }
}
```

## 🔄 Pipeline Examples

### Example 7: Classification Pipeline

```java
import org.superml.datasets.Datasets;
import org.superml.linear_model.LogisticRegression;
import org.superml.pipeline.Pipeline;
import org.superml.preprocessing.StandardScaler;
import org.superml.metrics.Metrics;
import org.superml.model_selection.ModelSelection;

public class ClassificationPipeline {
    public static void main(String[] args) {
        // 1. Load dataset
        var dataset = Datasets.makeClassification(800, 15, 2, 42);
        var split = ModelSelection.trainTestSplit(
            dataset.X, dataset.y, 0.2, 42);
        
        // 2. Create pipeline: Scaling → Classification
        var pipeline = new Pipeline()
            .addStep("scaler", new StandardScaler())
            .addStep("classifier", new LogisticRegression()
                .setMaxIterations(1000)
                .setLearningRate(0.01));
        
        System.out.println("Pipeline: " + pipeline);
        
        // 3. Train pipeline
        pipeline.fit(split.XTrain, split.yTrain);
        
        // 4. Make predictions
        double[] predictions = pipeline.predict(split.XTest);
        
        // 5. Evaluate
        double accuracy = Metrics.accuracy(split.yTest, predictions);
        System.out.printf("Pipeline Accuracy: %.4f\n", accuracy);
        
        // 6. Compare with non-scaled version
        var nonScaledClassifier = new LogisticRegression()
            .setMaxIterations(1000)
            .setLearningRate(0.01);
        
        nonScaledClassifier.fit(split.XTrain, split.yTrain);
        double[] nonScaledPred = nonScaledClassifier.predict(split.XTest);
        double nonScaledAccuracy = Metrics.accuracy(split.yTest, nonScaledPred);
        
        System.out.printf("Non-scaled Accuracy: %.4f\n", nonScaledAccuracy);
        System.out.printf("Improvement: %.4f\n", accuracy - nonScaledAccuracy);
    }
}
```

### Example 8: Regression Pipeline

```java
import org.superml.datasets.Datasets;
import org.superml.linear_model.Ridge;
import org.superml.pipeline.Pipeline;
import org.superml.preprocessing.StandardScaler;
import org.superml.metrics.Metrics;
import org.superml.model_selection.ModelSelection;

public class RegressionPipeline {
    public static void main(String[] args) {
        // 1. Generate data with different feature scales
        var dataset = Datasets.makeRegression(400, 8, 0.15, 42);
        
        // Artificially scale some features to show scaler effect
        for (int i = 0; i < dataset.X.length; i++) {
            dataset.X[i][0] *= 1000;    // Scale first feature
            dataset.X[i][1] *= 0.001;   // Scale second feature
        }
        
        var split = ModelSelection.trainTestSplit(
            dataset.X, dataset.y, 0.2, 42);
        
        // 2. Create pipeline
        var pipeline = new Pipeline()
            .addStep("scaler", new StandardScaler())
            .addStep("regressor", new Ridge().setAlpha(1.0));
        
        // 3. Train and evaluate pipeline
        pipeline.fit(split.XTrain, split.yTrain);
        double[] pipelinePred = pipeline.predict(split.XTest);
        double pipelineR2 = Metrics.r2Score(split.yTest, pipelinePred);
        
        // 4. Compare with non-scaled version
        var nonScaledRegressor = new Ridge().setAlpha(1.0);
        nonScaledRegressor.fit(split.XTrain, split.yTrain);
        double[] nonScaledPred = nonScaledRegressor.predict(split.XTest);
        double nonScaledR2 = Metrics.r2Score(split.yTest, nonScaledPred);
        
        System.out.println("Regression Pipeline Comparison:");
        System.out.printf("With Scaling:    R² = %.4f\n", pipelineR2);
        System.out.printf("Without Scaling: R² = %.4f\n", nonScaledR2);
        System.out.printf("Improvement:     R² = %.4f\n", pipelineR2 - nonScaledR2);
        
        // 5. Show feature scaling effect
        var scaler = (StandardScaler) pipeline.getStep("scaler");
        double[] means = scaler.getMean();
        double[] stds = scaler.getStd();
        
        System.out.println("\nFeature Scaling Summary:");
        for (int i = 0; i < Math.min(5, means.length); i++) {
            System.out.printf("Feature %d: mean=%.3f, std=%.3f\n", i, means[i], stds[i]);
        }
    }
}
```

## 🧮 Clustering Examples

### Example 9: K-Means Clustering

```java
import org.superml.cluster.KMeans;
import org.superml.datasets.Datasets;

public class BasicClustering {
    public static void main(String[] args) {
        // 1. Generate clustered data
        var dataset = Datasets.makeBlobs(
            300,     // samples
            2,       // features (for easy visualization)
            4,       // centers (true clusters)
            1.5,     // cluster_std
            42       // random seed
        );
        
        // 2. Apply K-means clustering
        var kmeans = new KMeans()
            .setNClusters(4)
            .setMaxIterations(300)
            .setRandomState(42);
        
        // 3. Fit and predict clusters
        kmeans.fit(dataset.X);
        int[] clusterLabels = kmeans.predict(dataset.X);
        
        // 4. Analyze results
        System.out.println("K-Means Clustering Results:");
        System.out.println("Number of clusters: " + kmeans.getNClusters());
        System.out.printf("Inertia: %.4f\n", kmeans.getInertia());
        
        // 5. Show cluster distribution
        Map<Integer, Integer> clusterCounts = new HashMap<>();
        for (int label : clusterLabels) {
            clusterCounts.merge(label, 1, Integer::sum);
        }
        
        System.out.println("\nCluster Distribution:");
        for (Map.Entry<Integer, Integer> entry : clusterCounts.entrySet()) {
            System.out.printf("Cluster %d: %d points\n", 
                entry.getKey(), entry.getValue());
        }
        
        // 6. Show cluster centers
        double[][] centers = kmeans.getClusterCenters();
        System.out.println("\nCluster Centers:");
        for (int i = 0; i < centers.length; i++) {
            System.out.printf("Center %d: [%.3f, %.3f]\n", 
                i, centers[i][0], centers[i][1]);
        }
        
        // 7. Sample cluster assignments
        System.out.println("\nSample Points and Clusters:");
        for (int i = 0; i < Math.min(10, dataset.X.length); i++) {
            System.out.printf("Point [%.2f, %.2f] → Cluster %d\n",
                dataset.X[i][0], dataset.X[i][1], clusterLabels[i]);
        }
    }
}
```

## 📊 Cross-Validation Examples

### Example 10: Model Validation

```java
import org.superml.datasets.Datasets;
import org.superml.linear_model.LogisticRegression;
import org.superml.model_selection.ModelSelection;

public class CrossValidation {
    public static void main(String[] args) {
        // 1. Load dataset
        var dataset = Datasets.loadIris();
        
        // 2. Create model
        var classifier = new LogisticRegression()
            .setMaxIterations(1000);
        
        // 3. Perform cross-validation
        System.out.println("Performing 5-fold Cross-Validation...");
        double[] cvScores = ModelSelection.crossValidate(
            classifier, dataset.X, dataset.y, 5);
        
        // 4. Analyze results
        double meanScore = Arrays.stream(cvScores).average().orElse(0.0);
        double stdScore = calculateStandardDeviation(cvScores, meanScore);
        
        System.out.println("Cross-Validation Results:");
        System.out.println("Fold Scores: " + Arrays.toString(cvScores));
        System.out.printf("Mean Score: %.4f (±%.4f)\n", meanScore, stdScore);
        
        // 5. Compare with simple train/test split
        var split = ModelSelection.trainTestSplit(
            dataset.X, dataset.y, 0.2, 42);
        
        classifier.fit(split.XTrain, split.yTrain);
        double[] predictions = classifier.predict(split.XTest);
        double testScore = Metrics.accuracy(split.yTest, predictions);
        
        System.out.printf("Single Train/Test Score: %.4f\n", testScore);
        
        // 6. Model stability analysis
        if (stdScore < 0.05) {
            System.out.println("✓ Model is stable across folds");
        } else {
            System.out.println("⚠ Model shows high variance across folds");
        }
    }
    
    private static double calculateStandardDeviation(double[] values, double mean) {
        double sumSquaredDiffs = 0.0;
        for (double value : values) {
            sumSquaredDiffs += Math.pow(value - mean, 2);
        }
        return Math.sqrt(sumSquaredDiffs / values.length);
    }
}
```

## 🎛️ Parameter Management Examples

### Example 11: Model Parameters

```java
import org.superml.linear_model.LogisticRegression;
import java.util.Map;

public class ParameterManagement {
    public static void main(String[] args) {
        // 1. Create model with default parameters
        var model = new LogisticRegression();
        
        // 2. View current parameters
        Map<String, Object> params = model.getParams();
        System.out.println("Default Parameters:");
        params.forEach((key, value) -> 
            System.out.printf("  %s: %s\n", key, value));
        
        // 3. Modify parameters fluently
        model.setMaxIterations(2000)
             .setLearningRate(0.001)
             .setTolerance(1e-8);
        
        System.out.println("\nModified Parameters:");
        model.getParams().forEach((key, value) -> 
            System.out.printf("  %s: %s\n", key, value));
        
        // 4. Set parameters from map
        Map<String, Object> newParams = Map.of(
            "maxIterations", 1500,
            "learningRate", 0.01,
            "tolerance", 1e-6
        );
        
        model.setParams(newParams);
        
        System.out.println("\nParameters from Map:");
        model.getParams().forEach((key, value) -> 
            System.out.printf("  %s: %s\n", key, value));
        
        // 5. Create identical model from parameters
        var clonedModel = new LogisticRegression();
        clonedModel.setParams(model.getParams());
        
        System.out.println("\nCloned model has same parameters: " + 
            clonedModel.getParams().equals(model.getParams()));
    }
}
```

## 🚀 Running the Examples

### Compile and Run

```bash
# Compile all examples
javac -cp "target/classes:lib/*" examples/*.java

# Run a specific example
java -cp ".:target/classes:lib/*" BasicClassification
```

### Maven Execution

```bash
# Place examples in src/main/java/examples/
mvn compile exec:java -Dexec.mainClass="examples.BasicClassification"
```

## 💡 Tips for Examples

1. **Start Simple**: Begin with basic classification/regression examples
2. **Understand Metrics**: Learn what each evaluation metric means
3. **Experiment**: Change parameters and see how results change
4. **Compare Models**: Always compare multiple approaches
5. **Use Pipelines**: Real-world ML almost always needs preprocessing
6. **Validate Properly**: Use cross-validation for reliable estimates

Each example is self-contained and demonstrates a specific concept. Try modifying parameters, datasets, or algorithms to see how they affect results! 🎯

## 🏭 Production Inference Examples

### Example 12: Production Inference

```java
import org.superml.inference.InferenceEngine;
import org.superml.inference.BatchInferenceProcessor;
import org.superml.inference.InferenceMetrics;
import java.util.concurrent.CompletableFuture;

public class ProductionInference {
    public static void main(String[] args) {
        // Create inference engine
        InferenceEngine engine = new InferenceEngine();
        
        try {
            // Load pre-trained model
            var modelInfo = engine.loadModel("production_classifier", "models/classifier.superml");
            System.out.println("Loaded: " + modelInfo);
            
            // Warm up for optimal performance
            engine.warmUp("production_classifier", 1000);
            
            // Single prediction
            double[] features = {5.1, 3.5, 1.4, 0.2};
            double prediction = engine.predict("production_classifier", features);
            System.out.printf("Single prediction: %.0f\n", prediction);
            
            // Batch prediction
            double[][] batchFeatures = {
                {5.1, 3.5, 1.4, 0.2},
                {4.9, 3.0, 1.4, 0.2},
                {6.2, 3.4, 5.4, 2.3},
                {5.9, 3.0, 5.1, 1.8}
            };
            
            double[] batchPredictions = engine.predict("production_classifier", batchFeatures);
            System.out.println("Batch predictions: " + Arrays.toString(batchPredictions));
            
            // Probability predictions
            double[][] probabilities = engine.predictProba("production_classifier", batchFeatures);
            System.out.println("\nClass Probabilities:");
            for (int i = 0; i < probabilities.length; i++) {
                System.out.printf("Sample %d: [%.3f, %.3f, %.3f]\n", 
                                i, probabilities[i][0], probabilities[i][1], probabilities[i][2]);
            }
            
            // Asynchronous prediction
            CompletableFuture<Double> asyncPrediction = 
                engine.predictAsync("production_classifier", features);
            
            asyncPrediction.thenAccept(result -> 
                System.out.printf("Async prediction: %.0f\n", result));
            
            // Wait for async completion
            asyncPrediction.get();
            
            // Performance monitoring
            InferenceMetrics metrics = engine.getMetrics("production_classifier");
            System.out.println("\nPerformance Metrics:");
            System.out.printf("Total inferences: %d\n", metrics.getTotalInferences());
            System.out.printf("Average time: %.2f ms\n", metrics.getAverageInferenceTimeMs());
            System.out.printf("Throughput: %.1f samples/sec\n", metrics.getThroughputSamplesPerSecond());
            System.out.printf("Error rate: %.2f%%\n", metrics.getErrorRate());
            
            // Batch file processing
            BatchInferenceProcessor processor = new BatchInferenceProcessor(engine);
            
            // Configure batch processing
            var batchConfig = new BatchInferenceProcessor.BatchConfig()
                .setBatchSize(1000)
                .setShowProgress(true)
                .setPredictionColumnName("iris_class");
            
            // Process large CSV file (commented out - requires actual file)
            // BatchInferenceProcessor.BatchResult result = 
            //     processor.processCSV("large_dataset.csv", "predictions.csv", 
            //                         "production_classifier", batchConfig);
            // System.out.println("Batch result: " + result.getSummary());
            
        } catch (Exception e) {
            System.err.println("Inference failed: " + e.getMessage());
        } finally {
            engine.shutdown();
        }
    }
}
```

Expected Output:
```
Loaded: ModelInfo{id='production_classifier', class='LogisticRegression', description='Iris classifier'}
Single prediction: 0
Batch predictions: [0.0, 0.0, 2.0, 2.0]

Class Probabilities:
Sample 0: [0.967, 0.033, 0.000]
Sample 1: [0.952, 0.048, 0.000]
Sample 2: [0.000, 0.023, 0.977]
Sample 3: [0.000, 0.089, 0.911]

Async prediction: 0

Performance Metrics:
Total inferences: 4
Average time: 0.25 ms
Throughput: 16000.0 samples/sec
Error rate: 0.00%
```
