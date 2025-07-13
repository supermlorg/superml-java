# Advanced Examples

This guide demonstrates advanced usage patterns, complex workflows, and real-world scenarios using SuperML Java.

## 🚀 Advanced Tree Algorithms

### Gradient Boosting with Early Stopping

```java
import org.superml.datasets.Datasets;
import org.superml.tree.GradientBoosting;
import org.superml.metrics.Metrics;

public class GradientBoostingAdvanced {
    public static void main(String[] args) {
        // Generate large dataset
        var dataset = Datasets.makeClassification(2000, 25, 2);
        var split = DataLoaders.trainTestSplit(dataset.X, 
            Arrays.stream(dataset.y).asDoubleStream().toArray(), 0.2, 42);
        
        // Configure with early stopping
        GradientBoosting gb = new GradientBoosting()
            .setNEstimators(1000)           // High number
            .setLearningRate(0.05)          // Conservative learning rate
            .setMaxDepth(6)
            .setSubsample(0.8)              // Use 80% of data per tree
            .setValidationFraction(0.1)     // 10% for validation
            .setNIterNoChange(20)           // Stop after 20 rounds without improvement
            .setTol(1e-4)                   // Improvement tolerance
            .setRandomState(42);
        
        // Train with monitoring
        System.out.println("Training Gradient Boosting with early stopping...");
        long start = System.currentTimeMillis();
        gb.fit(split.XTrain, split.yTrain);
        long trainTime = System.currentTimeMillis() - start;
        
        // Analyze training progress
        List<Double> trainScores = gb.getTrainScores();
        List<Double> validScores = gb.getValidationScores();
        
        System.out.printf("Training completed in %d ms\n", trainTime);
        System.out.printf("Actual estimators used: %d/%d\n", 
            gb.getTrees().size(), gb.getNEstimators());
        
        // Show training progress
        System.out.println("\nTraining Progress (last 10 iterations):");
        int start_idx = Math.max(0, trainScores.size() - 10);
        for (int i = start_idx; i < trainScores.size(); i++) {
            System.out.printf("Iteration %d: Train=%.4f, Valid=%.4f\n", 
                i+1, trainScores.get(i), validScores.get(i));
        }
        
        // Test performance
        double accuracy = gb.score(split.XTest, split.yTest);
        System.out.printf("\nFinal Test Accuracy: %.4f\n", accuracy);
        
        // Compare with different iteration counts
        System.out.println("\nPerformance at different stages:");
        int[] testIterations = {10, 25, 50, 100, gb.getTrees().size()};
        for (int iters : testIterations) {
            if (iters <= gb.getTrees().size()) {
                double[] preds = gb.predictAtIteration(split.XTest, iters);
                double acc = Metrics.accuracy(split.yTest, preds);
                System.out.printf("  %d trees: %.4f\n", iters, acc);
            }
        }
    }
}
```

### Hyperparameter Tuning with Grid Search

```java
import org.superml.tree.RandomForest;
import org.superml.model_selection.GridSearchCV;

public class HyperparameterTuning {
    public static void main(String[] args) {
        // Load dataset
        var dataset = Datasets.makeClassification(1500, 20, 3);
        var split = DataLoaders.trainTestSplit(dataset.X, 
            Arrays.stream(dataset.y).asDoubleStream().toArray(), 0.2, 42);
        
        // Define parameter grid
        Map<String, Object[]> paramGrid = new HashMap<>();
        paramGrid.put("n_estimators", new Object[]{50, 100, 200});
        paramGrid.put("max_depth", new Object[]{5, 10, 15, 20});
        paramGrid.put("min_samples_split", new Object[]{2, 5, 10});
        paramGrid.put("max_features", new Object[]{5, 10, -1});  // -1 = auto
        
        // Create grid search
        RandomForest baseModel = new RandomForest();
        GridSearchCV gridSearch = new GridSearchCV(baseModel, paramGrid, 3); // 3-fold CV
        
        System.out.println("Starting hyperparameter tuning...");
        System.out.printf("Total combinations: %d\n", 
            Arrays.stream(paramGrid.values())
                .mapToInt(arr -> arr.length)
                .reduce(1, (a, b) -> a * b));
        
        // Perform grid search
        long start = System.currentTimeMillis();
        gridSearch.fit(split.XTrain, split.yTrain);
        long searchTime = System.currentTimeMillis() - start;
        
        // Get best results
        RandomForest bestModel = (RandomForest) gridSearch.getBestEstimator();
        Map<String, Object> bestParams = gridSearch.getBestParams();
        double bestScore = gridSearch.getBestScore();
        
        System.out.printf("Grid search completed in %.2f seconds\n", searchTime / 1000.0);
        System.out.printf("Best CV score: %.4f\n", bestScore);
        System.out.println("Best parameters:");
        bestParams.forEach((key, value) -> 
            System.out.printf("  %s: %s\n", key, value));
        
        // Evaluate on test set
        double testScore = bestModel.score(split.XTest, split.yTest);
        System.out.printf("Test accuracy with best params: %.4f\n", testScore);
        
        // Compare with default parameters
        RandomForest defaultModel = new RandomForest();
        defaultModel.fit(split.XTrain, split.yTrain);
        double defaultScore = defaultModel.score(split.XTest, split.yTest);
        
        System.out.printf("Default model test accuracy: %.4f\n", defaultScore);
        System.out.printf("Improvement: %.4f\n", testScore - defaultScore);
    }
}
```

## 🎯 Advanced Multiclass Scenarios

### Comparing Multiclass Strategies

```java
import org.superml.multiclass.*;
import org.superml.linear_model.LogisticRegression;
import org.superml.tree.RandomForest;

public class MulticlassStrategyComparison {
    public static void main(String[] args) {
        // Generate challenging multiclass dataset
        var dataset = Datasets.makeClassification(2000, 30, 5); // 5 classes
        var split = DataLoaders.trainTestSplit(dataset.X, 
            Arrays.stream(dataset.y).asDoubleStream().toArray(), 0.2, 42);
        
        System.out.println("Comparing Multiclass Strategies on 5-class problem");
        System.out.printf("Training samples: %d, Features: %d\n", 
            split.XTrain.length, split.XTrain[0].length);
        
        Map<String, Double> results = new LinkedHashMap<>();
        Map<String, Long> timings = new LinkedHashMap<>();
        
        // 1. One-vs-Rest with Logistic Regression
        System.out.println("\n1. Training One-vs-Rest...");
        LogisticRegression lr = new LogisticRegression().setMaxIter(1000);
        OneVsRestClassifier ovr = new OneVsRestClassifier(lr);
        
        long start = System.currentTimeMillis();
        ovr.fit(split.XTrain, split.yTrain);
        long time = System.currentTimeMillis() - start;
        
        double ovrScore = ovr.score(split.XTest, split.yTest);
        results.put("One-vs-Rest (LR)", ovrScore);
        timings.put("One-vs-Rest (LR)", time);
        
        // 2. Softmax Regression
        System.out.println("2. Training Softmax Regression...");
        SoftmaxRegression softmax = new SoftmaxRegression()
            .setMaxIter(1000)
            .setLearningRate(0.01);
        
        start = System.currentTimeMillis();
        softmax.fit(split.XTrain, split.yTrain);
        time = System.currentTimeMillis() - start;
        
        double softmaxScore = softmax.score(split.XTest, split.yTest);
        results.put("Softmax Regression", softmaxScore);
        timings.put("Softmax Regression", time);
        
        // 3. Enhanced Logistic Regression (auto)
        System.out.println("3. Training Enhanced LR (auto)...");
        LogisticRegression autoLR = new LogisticRegression()
            .setMaxIter(1000)
            .setMultiClass("auto");
        
        start = System.currentTimeMillis();
        autoLR.fit(split.XTrain, split.yTrain);
        time = System.currentTimeMillis() - start;
        
        double autoScore = autoLR.score(split.XTest, split.yTest);
        results.put("LR Auto", autoScore);
        timings.put("LR Auto", time);
        
        // 4. One-vs-Rest with Random Forest
        System.out.println("4. Training OvR with Random Forest...");
        RandomForest rf = new RandomForest(100, 10);
        OneVsRestClassifier ovrRF = new OneVsRestClassifier(rf);
        
        start = System.currentTimeMillis();
        ovrRF.fit(split.XTrain, split.yTrain);
        time = System.currentTimeMillis() - start;
        
        double ovrRFScore = ovrRF.score(split.XTest, split.yTest);
        results.put("One-vs-Rest (RF)", ovrRFScore);
        timings.put("One-vs-Rest (RF)", time);
        
        // 5. Native Random Forest
        System.out.println("5. Training Native Random Forest...");
        RandomForest nativeRF = new RandomForest(100, 10);
        
        start = System.currentTimeMillis();
        nativeRF.fit(split.XTrain, split.yTrain);
        time = System.currentTimeMillis() - start;
        
        double nativeRFScore = nativeRF.score(split.XTest, split.yTest);
        results.put("Native Random Forest", nativeRFScore);
        timings.put("Native Random Forest", time);
        
        // Print comparison results
        System.out.println("\n=== COMPARISON RESULTS ===");
        System.out.printf("%-25s %10s %12s\n", "Method", "Accuracy", "Time (ms)");
        System.out.println("-".repeat(50));
        
        results.forEach((method, score) -> {
            System.out.printf("%-25s %10.4f %12d\n", 
                method, score, timings.get(method));
        });
        
        // Find best method
        String bestMethod = results.entrySet().stream()
            .max(Map.Entry.comparingByValue())
            .map(Map.Entry::getKey)
            .orElse("None");
        
        System.out.printf("\nBest performing method: %s (%.4f)\n", 
            bestMethod, results.get(bestMethod));
            
        // Analyze probability predictions for a few samples
        System.out.println("\n=== PROBABILITY ANALYSIS ===");
        analyzeClassProbabilities(ovr, softmax, nativeRF, split.XTest, split.yTest);
    }
    
    private static void analyzeClassProbabilities(OneVsRestClassifier ovr, 
                                                SoftmaxRegression softmax,
                                                RandomForest rf,
                                                double[][] XTest, double[] yTest) {
        // Get probabilities for first 3 test samples
        double[][] ovrProbs = ovr.predictProba(Arrays.copyOfRange(XTest, 0, 3));
        double[][] softmaxProbs = softmax.predictProba(Arrays.copyOfRange(XTest, 0, 3));
        double[][] rfProbs = rf.predictProba(Arrays.copyOfRange(XTest, 0, 3));
        
        for (int i = 0; i < 3; i++) {
            System.out.printf("Sample %d (True class: %.0f):\n", i+1, yTest[i]);
            System.out.printf("  OvR:     %s\n", formatProbabilities(ovrProbs[i]));
            System.out.printf("  Softmax: %s\n", formatProbabilities(softmaxProbs[i]));
            System.out.printf("  RF:      %s\n", formatProbabilities(rfProbs[i]));
            System.out.println();
        }
    }
    
    private static String formatProbabilities(double[] probs) {
        return Arrays.stream(probs)
            .mapToObj(p -> String.format("%.3f", p))
            .collect(Collectors.joining(", ", "[", "]"));
    }
}
```

## 🔄 Complex Workflow Examples

### Kaggle Competition Workflow

```java
import org.superml.datasets.KaggleTrainingManager;
import org.superml.preprocessing.StandardScaler;
import org.superml.pipeline.Pipeline;

public class KaggleWorkflowExample {
    public static void main(String[] args) {
        // Setup Kaggle credentials (normally from environment)
        KaggleIntegration.KaggleCredentials credentials = 
            new KaggleIntegration.KaggleCredentials("username", "token");
        
        KaggleTrainingManager manager = new KaggleTrainingManager(credentials);
        
        // Configure training for competition
        KaggleTrainingManager.TrainingConfig config = 
            new KaggleTrainingManager.TrainingConfig("titanic", "data/train.csv")
                .setTargetColumn("Survived")
                .setValidationSplit(0.2)
                .setCrossValidation(true)
                .setCvFolds(5)
                .setRandomState(42);
        
        // Test multiple models
        List<SupervisedLearner> models = Arrays.asList(
            new LogisticRegression().setMaxIter(1000),
            new RandomForest(200, 15),
            new GradientBoosting(100, 0.1, 6)
        );
        
        List<KaggleTrainingManager.TrainingResult> results = new ArrayList<>();
        
        System.out.println("=== KAGGLE COMPETITION TRAINING ===");
        
        for (SupervisedLearner model : models) {
            System.out.printf("\nTraining %s...\n", model.getClass().getSimpleName());
            
            try {
                // Train with cross-validation
                var result = manager.crossValidateModel(model, config);
                results.add(result);
                
                System.out.printf("CV Score: %.4f (±%.4f)\n", 
                    result.getScore(), 
                    (Double) result.getMetrics().get("cv_std") * 2);
                
            } catch (Exception e) {
                System.err.printf("Failed to train %s: %s\n", 
                    model.getClass().getSimpleName(), e.getMessage());
            }
        }
        
        // Find best model
        KaggleTrainingManager.TrainingResult bestResult = results.stream()
            .max(Comparator.comparing(KaggleTrainingManager.TrainingResult::getScore))
            .orElse(null);
        
        if (bestResult != null) {
            System.out.println("\n=== BEST MODEL ===");
            System.out.println(bestResult);
            
            // Train final model on full dataset for submission
            System.out.println("\nTraining final model on full dataset...");
            var finalConfig = new KaggleTrainingManager.TrainingConfig("titanic", "data/train.csv")
                .setTargetColumn("Survived")
                .setValidationSplit(0.0)  // Use all data
                .setRandomState(42);
            
            var finalResult = manager.trainModel(bestResult.getTrainedModel(), finalConfig);
            System.out.println("Final model ready for predictions");
        }
    }
}
```

### Feature Importance Analysis

```java
import org.superml.tree.RandomForest;
import org.superml.tree.GradientBoosting;

public class FeatureImportanceAnalysis {
    public static void main(String[] args) {
        // Generate dataset with known feature relationships
        var dataset = Datasets.makeClassification(1000, 20, 2);
        String[] featureNames = generateFeatureNames(20);
        
        var split = DataLoaders.trainTestSplit(dataset.X, 
            Arrays.stream(dataset.y).asDoubleStream().toArray(), 0.2, 42);
        
        // Train different models
        RandomForest rf = new RandomForest(200, 15);
        rf.fit(split.XTrain, split.yTrain);
        
        GradientBoosting gb = new GradientBoosting(100, 0.1, 6);
        gb.fit(split.XTrain, split.yTrain);
        
        // Get feature importances
        double[] rfImportances = rf.getFeatureImportances();
        double[] gbImportances = gb.getFeatureImportances();
        
        // Create importance rankings
        List<FeatureImportance> rfRanking = createRanking(featureNames, rfImportances);
        List<FeatureImportance> gbRanking = createRanking(featureNames, gbImportances);
        
        // Sort by importance
        rfRanking.sort((a, b) -> Double.compare(b.importance, a.importance));
        gbRanking.sort((a, b) -> Double.compare(b.importance, a.importance));
        
        // Display results
        System.out.println("=== FEATURE IMPORTANCE ANALYSIS ===\n");
        
        System.out.println("Random Forest Top 10 Features:");
        printTopFeatures(rfRanking, 10);
        
        System.out.println("\nGradient Boosting Top 10 Features:");
        printTopFeatures(gbRanking, 10);
        
        // Compare model agreement
        System.out.println("\n=== MODEL AGREEMENT ===");
        analyzeFeatureAgreement(rfRanking, gbRanking);
        
        // Performance comparison
        double rfAccuracy = rf.score(split.XTest, split.yTest);
        double gbAccuracy = gb.score(split.XTest, split.yTest);
        
        System.out.printf("\nModel Performance:\n");
        System.out.printf("Random Forest: %.4f\n", rfAccuracy);
        System.out.printf("Gradient Boosting: %.4f\n", gbAccuracy);
    }
    
    private static class FeatureImportance {
        String name;
        double importance;
        int rank;
        
        FeatureImportance(String name, double importance) {
            this.name = name;
            this.importance = importance;
        }
    }
    
    private static List<FeatureImportance> createRanking(String[] names, double[] importances) {
        List<FeatureImportance> ranking = new ArrayList<>();
        for (int i = 0; i < names.length; i++) {
            ranking.add(new FeatureImportance(names[i], importances[i]));
        }
        return ranking;
    }
    
    private static void printTopFeatures(List<FeatureImportance> ranking, int top) {
        for (int i = 0; i < Math.min(top, ranking.size()); i++) {
            FeatureImportance fi = ranking.get(i);
            System.out.printf("%2d. %-12s: %.4f\n", i+1, fi.name, fi.importance);
        }
    }
    
    private static void analyzeFeatureAgreement(List<FeatureImportance> rf, List<FeatureImportance> gb) {
        Set<String> rfTop5 = rf.stream().limit(5).map(f -> f.name).collect(Collectors.toSet());
        Set<String> gbTop5 = gb.stream().limit(5).map(f -> f.name).collect(Collectors.toSet());
        
        Set<String> agreement = new HashSet<>(rfTop5);
        agreement.retainAll(gbTop5);
        
        System.out.printf("Top 5 feature agreement: %d/5 features\n", agreement.size());
        System.out.println("Agreed features: " + agreement);
    }
    
    private static String[] generateFeatureNames(int count) {
        String[] names = new String[count];
        for (int i = 0; i < count; i++) {
            names[i] = "feature_" + i;
        }
        return names;
    }
}
```

## 📊 Model Comparison and Ensemble

### Comprehensive Model Comparison

```java
public class ModelComparisonSuite {
    public static void main(String[] args) {
        // Load multiple datasets for robust comparison
        List<TestDataset> datasets = Arrays.asList(
            new TestDataset("Binary", Datasets.makeClassification(1000, 10, 2)),
            new TestDataset("Multiclass", Datasets.makeClassification(1000, 15, 4)),
            new TestDataset("High-dim", Datasets.makeClassification(500, 50, 2))
        );
        
        // Define models to compare
        List<SupervisedLearner> models = Arrays.asList(
            new LogisticRegression().setMaxIter(1000),
            new DecisionTree("gini", 10),
            new RandomForest(100, 10),
            new GradientBoosting(100, 0.1, 6)
        );
        
        System.out.println("=== COMPREHENSIVE MODEL COMPARISON ===\n");
        
        // Results table
        Map<String, Map<String, Double>> results = new LinkedHashMap<>();
        
        for (TestDataset dataset : datasets) {
            System.out.printf("Testing on %s dataset...\n", dataset.name);
            
            var split = DataLoaders.trainTestSplit(dataset.data.X, 
                Arrays.stream(dataset.data.y).asDoubleStream().toArray(), 0.2, 42);
            
            Map<String, Double> datasetResults = new LinkedHashMap<>();
            
            for (SupervisedLearner model : models) {
                String modelName = model.getClass().getSimpleName();
                
                try {
                    // Train and test
                    long start = System.nanoTime();
                    model.fit(split.XTrain, split.yTrain);
                    long trainTime = System.nanoTime() - start;
                    
                    start = System.nanoTime();
                    double accuracy = model.score(split.XTest, split.yTest);
                    long testTime = System.nanoTime() - start;
                    
                    datasetResults.put(modelName, accuracy);
                    
                    System.out.printf("  %s: %.4f (train: %.1fms, test: %.1fms)\n", 
                        modelName, accuracy, trainTime/1e6, testTime/1e6);
                        
                } catch (Exception e) {
                    System.err.printf("  %s: FAILED (%s)\n", modelName, e.getMessage());
                    datasetResults.put(modelName, 0.0);
                }
            }
            
            results.put(dataset.name, datasetResults);
            System.out.println();
        }
        
        // Print summary table
        printComparisonTable(results, models);
        
        // Statistical analysis
        performStatisticalAnalysis(results, models);
    }
    
    private static class TestDataset {
        String name;
        Datasets.ClassificationData data;
        
        TestDataset(String name, Datasets.ClassificationData data) {
            this.name = name;
            this.data = data;
        }
    }
    
    private static void printComparisonTable(Map<String, Map<String, Double>> results, 
                                           List<SupervisedLearner> models) {
        System.out.println("=== SUMMARY TABLE ===");
        System.out.printf("%-15s", "Dataset");
        for (SupervisedLearner model : models) {
            System.out.printf("%15s", model.getClass().getSimpleName());
        }
        System.out.println();
        System.out.println("-".repeat(15 + 15 * models.size()));
        
        for (Map.Entry<String, Map<String, Double>> entry : results.entrySet()) {
            System.out.printf("%-15s", entry.getKey());
            for (SupervisedLearner model : models) {
                String modelName = model.getClass().getSimpleName();
                double score = entry.getValue().get(modelName);
                System.out.printf("%15.4f", score);
            }
            System.out.println();
        }
    }
    
    private static void performStatisticalAnalysis(Map<String, Map<String, Double>> results,
                                                 List<SupervisedLearner> models) {
        System.out.println("\n=== STATISTICAL ANALYSIS ===");
        
        // Calculate averages
        Map<String, Double> averages = new LinkedHashMap<>();
        for (SupervisedLearner model : models) {
            String modelName = model.getClass().getSimpleName();
            double avg = results.values().stream()
                .mapToDouble(map -> map.get(modelName))
                .average()
                .orElse(0.0);
            averages.put(modelName, avg);
        }
        
        // Find best performing model
        String bestModel = averages.entrySet().stream()
            .max(Map.Entry.comparingByValue())
            .map(Map.Entry::getKey)
            .orElse("None");
        
        System.out.println("Average Performance:");
        averages.forEach((model, avg) -> 
            System.out.printf("  %s: %.4f%s\n", model, avg, 
                model.equals(bestModel) ? " (BEST)" : ""));
    }
}
```

These advanced examples demonstrate sophisticated usage patterns, comprehensive evaluation strategies, and real-world workflow implementations using SuperML Java's rich feature set.
