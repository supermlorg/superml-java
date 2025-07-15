---
title: "Advanced Examples"
description: "Advanced usage patterns, complex workflows, and real-world scenarios with SuperML Java 2.0.0"
layout: default
toc: true
search: true
---

# Advanced Examples

This guide demonstrates advanced usage patterns, complex workflows, and real-world scenarios using SuperML Java 2.0.0's comprehensive 21-module architecture. These examples showcase production-ready features and sophisticated ML workflows.

## 🚀 Advanced AutoML Workflows

### Enterprise AutoML with Custom Metrics

```java
import org.superml.datasets.Datasets;
import org.superml.autotrainer.AutoTrainer;
import org.superml.autotrainer.AutoMLConfig;
import org.superml.metrics.CustomMetrics;
import org.superml.visualization.VisualizationFactory;
import org.superml.persistence.ModelPersistence;

import java.util.*;

public class EnterpriseAutoMLExample {
    public static void main(String[] args) {
        System.out.println("🏢 SuperML Java 2.0.0 - Enterprise AutoML Example");
        
        // 1. Generate complex multi-class dataset
        var dataset = Datasets.makeClassification(2000, 25, 5, 42); // 5 classes, challenging
        System.out.println("✓ Generated enterprise dataset: " + dataset.X.length + " samples, " + dataset.X[0].length + " features, 5 classes");
        
        // 2. Configure advanced AutoML
        var config = new AutoMLConfig()
            .setTimeLimit(300)              // 5 minutes max
            .setMemoryLimit(2048)           // 2GB memory limit
            .setEnsembleMethods(true)       // Enable ensemble methods
            .setFeatureSelection(true)      // Enable feature selection
            .setHyperparameterOptimization(true)
            .setScoringMetric("f1_macro")   // Use macro F1 for imbalanced classes
            .setValidationSplit(0.2)        // Hold out 20% for validation
            .setRandomState(42)
            .setVerbose(true);
        
        // 3. Custom metric for domain-specific evaluation
        var customMetric = new CustomMetrics.BusinessMetric() {
            @Override
            public double evaluate(double[] yTrue, double[] yPred) {
                // Custom business logic: penalize false positives in class 0 heavily
                double score = 0.0;
                for (int i = 0; i < yTrue.length; i++) {
                    if (yTrue[i] == 0 && yPred[i] != 0) {
                        score -= 2.0; // Heavy penalty for false positive on critical class
                    } else if (yTrue[i] == yPred[i]) {
                        score += 1.0; // Reward correct predictions
                    }
                }
                return score / yTrue.length;
            }
        };
        config.addCustomMetric("business_score", customMetric);
        
        // 4. Run enterprise AutoML
        System.out.println("\n🔄 Running enterprise AutoML with advanced configuration...");
        var autoMLResult = AutoTrainer.autoMLAdvanced(dataset.X, dataset.y, "classification", config);
        
        // 5. Comprehensive results analysis
        System.out.println("\n📊 AutoML Results Analysis:");
        System.out.println("   Best Algorithm: " + autoMLResult.getBestAlgorithm());
        System.out.println("   Best Score (F1-Macro): " + String.format("%.4f", autoMLResult.getBestScore()));
        System.out.println("   Custom Business Score: " + String.format("%.4f", autoMLResult.getCustomScore("business_score")));
        System.out.println("   Feature Selection: " + autoMLResult.getSelectedFeatures().length + "/" + dataset.X[0].length + " features selected");
        System.out.println("   Training Time: " + autoMLResult.getTrainingTime() + "ms");
        System.out.println("   Memory Usage: " + autoMLResult.getMemoryUsage() + "MB");
        
        // 6. Model ensemble analysis
        if (autoMLResult.isEnsembleUsed()) {
            System.out.println("\n🔧 Ensemble Analysis:");
            var ensembleWeights = autoMLResult.getEnsembleWeights();
            var ensembleModels = autoMLResult.getEnsembleModels();
            
            for (int i = 0; i < ensembleModels.size(); i++) {
                System.out.println("   Model " + (i+1) + ": " + ensembleModels.get(i).getClass().getSimpleName() + 
                                 " (weight: " + String.format("%.3f", ensembleWeights.get(i)) + ")");
            }
        }
        
        // 7. Advanced visualization suite
        System.out.println("\n📈 Creating advanced visualizations...");
        
        // Model comparison heatmap
        VisualizationFactory.createModelPerformanceHeatmap(
            autoMLResult.getAllModelResults(),
            "AutoML Model Performance Matrix"
        ).display();
        
        // Feature importance plot
        VisualizationFactory.createFeatureImportancePlot(
            autoMLResult.getFeatureImportances(),
            autoMLResult.getFeatureNames(),
            "Feature Importance Analysis"
        ).display();
        
        // Learning curves
        VisualizationFactory.createLearningCurves(
            autoMLResult.getLearningCurveData(),
            "AutoML Learning Curves"
        ).display();
        
        // 8. Save enterprise model with metadata
        String modelPath = ModelPersistence.saveWithMetadata(
            autoMLResult.getBestModel(),
            "enterprise_automl_model",
            Map.of(
                "dataset_size", dataset.X.length,
                "num_features", dataset.X[0].length,
                "num_classes", 5,
                "automl_config", config.toString(),
                "performance_metrics", autoMLResult.getAllMetrics(),
                "business_score", autoMLResult.getCustomScore("business_score")
            )
        );
        
        System.out.println("\n💾 Enterprise model saved: " + modelPath);
        System.out.println("✅ Enterprise AutoML example completed!");
    }
}
```

## 🌲 Advanced Tree Algorithms

### Gradient Boosting with Early Stopping and Custom Loss

```java
import org.superml.datasets.Datasets;
import org.superml.tree.GradientBoostingClassifier;
import org.superml.tree.GradientBoostingRegressor;
import org.superml.tree.losses.CustomLoss;
import org.superml.model_selection.ModelSelection;
import org.superml.metrics.Metrics;
import org.superml.visualization.VisualizationFactory;

import java.util.*;

public class AdvancedGradientBoostingExample {
    public static void main(String[] args) {
        System.out.println("🌲 SuperML Java 2.0.0 - Advanced Gradient Boosting Example");
        
        // 1. Generate challenging dataset
        var dataset = Datasets.makeClassification(2000, 30, 3, 42, 0.15); // 15% noise
        var split = ModelSelection.trainTestSplit(dataset.X, dataset.y, 0.2, 42);
        System.out.println("✓ Generated noisy dataset: " + dataset.X.length + " samples, " + dataset.X[0].length + " features");
        
        // 2. Custom loss function for robust learning
        var robustLoss = new CustomLoss() {
            @Override
            public double computeLoss(double yTrue, double yPred) {
                // Huber loss for robustness to outliers
                double error = Math.abs(yTrue - yPred);
                double delta = 1.0;
                return error <= delta ? 0.5 * error * error : delta * error - 0.5 * delta * delta;
            }
            
            @Override
            public double computeGradient(double yTrue, double yPred) {
                double error = yTrue - yPred;
                double delta = 1.0;
                return Math.abs(error) <= delta ? error : delta * Math.signum(error);
            }
        };
        
        // 3. Configure advanced gradient boosting
        var gb = new GradientBoostingClassifier()
            .setNEstimators(500)
            .setLearningRate(0.05)
            .setMaxDepth(6)
            .setSubsample(0.8)              // Stochastic gradient boosting
            .setColsampleBytree(0.8)        // Random feature sampling
            .setMinSamplesLeaf(5)
            .setMinSamplesSplit(10)
            .setValidationFraction(0.15)    // Early stopping validation
            .setNIterNoChange(25)           // Patience for early stopping
            .setTol(1e-4)
            .setLossFunction(robustLoss)    // Custom robust loss
            .setRandomState(42)
            .setVerbose(true);
        
        // 4. Train with advanced monitoring
        System.out.println("\n🔄 Training gradient boosting with early stopping...");
        long startTime = System.currentTimeMillis();
        
        gb.fit(split.XTrain, split.yTrain);
        
        long trainingTime = System.currentTimeMillis() - startTime;
        System.out.println("✓ Training completed in " + trainingTime + "ms");
        
        // 5. Analyze training progress
        var trainingHistory = gb.getTrainingHistory();
        System.out.println("\n📊 Training Analysis:");
        System.out.println("   Estimators used: " + gb.getNEstimatorsUsed() + "/" + gb.getNEstimators());
        System.out.println("   Early stopping triggered: " + gb.isEarlyStoppingStopped());
        System.out.println("   Best iteration: " + gb.getBestIteration());
        System.out.println("   Best validation score: " + String.format("%.4f", gb.getBestValidationScore()));
        
        // 6. Performance at different stages
        System.out.println("\n🎯 Performance Evolution:");
        int[] stages = {10, 25, 50, 100, gb.getNEstimatorsUsed()};
        for (int stage : stages) {
            if (stage <= gb.getNEstimatorsUsed()) {
                double[] predictions = gb.predictAtStage(split.XTest, stage);
                var metrics = Metrics.classificationReport(split.yTest, predictions);
                System.out.println("   Stage " + stage + ": Accuracy=" + String.format("%.4f", metrics.accuracy) + 
                                 ", F1=" + String.format("%.4f", metrics.f1Score));
            }
        }
        
        // 7. Feature importance analysis
        double[] importance = gb.getFeatureImportances();
        System.out.println("\n🔍 Top 10 Most Important Features:");
        var sortedFeatures = IntStream.range(0, importance.length)
            .boxed()
            .sorted((i, j) -> Double.compare(importance[j], importance[i]))
            .limit(10)
            .collect(Collectors.toList());
        
        for (int i = 0; i < sortedFeatures.size(); i++) {
            int featureIdx = sortedFeatures.get(i);
            System.out.println("   Feature " + featureIdx + ": " + String.format("%.4f", importance[featureIdx]));
        }
        
        // 8. Advanced visualizations
        System.out.println("\n📈 Creating advanced visualizations...");
        
        // Training progress
        VisualizationFactory.createTrainingProgressPlot(
            trainingHistory,
            "Gradient Boosting Training Progress"
        ).display();
        
        // Feature importance
        VisualizationFactory.createFeatureImportancePlot(
            importance,
            IntStream.range(0, importance.length).mapToObj(i -> "Feature_" + i).toArray(String[]::new),
            "Gradient Boosting Feature Importance"
        ).display();
        
        // Prediction distribution
        double[] finalPredictions = gb.predict(split.XTest);
        VisualizationFactory.createPredictionDistributionPlot(
            split.yTest,
            finalPredictions,
            "Prediction Distribution Analysis"
        ).display();
        
        System.out.println("✅ Advanced gradient boosting example completed!");
    }
}
```

### Hyperparameter Optimization with Bayesian Search

```java
import org.superml.datasets.Datasets;
import org.superml.tree.RandomForestClassifier;
import org.superml.model_selection.BayesianSearchCV;
import org.superml.model_selection.CrossValidation;
import org.superml.optimization.BayesianOptimizer;
import org.superml.visualization.VisualizationFactory;

import java.util.*;

public class BayesianHyperparameterOptimization {
    public static void main(String[] args) {
        System.out.println("🔬 SuperML Java 2.0.0 - Bayesian Hyperparameter Optimization");
        
        // 1. Load dataset
        var dataset = Datasets.makeClassification(1500, 20, 3, 42);
        System.out.println("✓ Generated dataset: " + dataset.X.length + " samples, " + dataset.X[0].length + " features");
        
        // 2. Define hyperparameter space with distributions
        var paramSpace = new BayesianOptimizer.ParameterSpace()
            .addIntegerParameter("nEstimators", 50, 500)
            .addIntegerParameter("maxDepth", 3, 20)
            .addIntegerParameter("minSamplesLeaf", 1, 20)
            .addRealParameter("maxFeatures", 0.1, 1.0)
            .addCategoricalParameter("criterion", Arrays.asList("gini", "entropy"))
            .addBooleanParameter("bootstrap", Arrays.asList(true, false));
        
        // 3. Custom scoring function with multiple objectives
        var multiObjectiveScorer = new BayesianSearchCV.MultiObjectiveScorer() {
            @Override
            public Map<String, Double> score(Object model, double[][] X, double[] y) {
                var rf = (RandomForestClassifier) model;
                var cvResult = CrossValidation.crossValidateScore(rf, X, y, 3);
                
                Map<String, Double> scores = new HashMap<>();
                scores.put("accuracy", cvResult.meanScore);
                scores.put("model_complexity", -rf.getNEstimators() * rf.getMaxDepth() / 1000.0); // Prefer simpler models
                scores.put("training_time", -cvResult.meanFitTime / 1000.0); // Prefer faster training
                
                return scores;
            }
        };
        
        // 4. Configure Bayesian optimization
        var bayesianSearch = new BayesianSearchCV(
            new RandomForestClassifier(),
            paramSpace,
            multiObjectiveScorer
        )
            .setNIter(50)                   // 50 optimization iterations
            .setAcquisitionFunction("ei")   // Expected improvement
            .setKappa(2.576)               // Exploration-exploitation balance
            .setRandomState(42)
            .setNJobs(-1)                  // Use all cores
            .setVerbose(true);
        
        // 5. Run Bayesian optimization
        System.out.println("\n🔄 Running Bayesian hyperparameter optimization...");
        long startTime = System.currentTimeMillis();
        
        bayesianSearch.fit(dataset.X, dataset.y);
        
        long optimizationTime = System.currentTimeMillis() - startTime;
        System.out.println("✓ Optimization completed in " + optimizationTime + "ms");
        
        // 6. Analyze optimization results
        System.out.println("\n🏆 Optimization Results:");
        var bestParams = bayesianSearch.getBestParams();
        var bestScores = bayesianSearch.getBestScores();
        
        System.out.println("   Best Parameters:");
        bestParams.forEach((param, value) -> 
            System.out.println("     " + param + ": " + value));
        
        System.out.println("   Best Scores:");
        bestScores.forEach((metric, score) -> 
            System.out.println("     " + metric + ": " + String.format("%.4f", score)));
        
        // 7. Pareto frontier analysis for multi-objective optimization
        var paretoFrontier = bayesianSearch.getParetoFrontier();
        System.out.println("\n📊 Pareto Frontier Analysis:");
        System.out.println("   Found " + paretoFrontier.size() + " non-dominated solutions");
        
        for (int i = 0; i < Math.min(5, paretoFrontier.size()); i++) {
            var solution = paretoFrontier.get(i);
            System.out.println("   Solution " + (i+1) + ":");
            System.out.println("     Accuracy: " + String.format("%.4f", solution.getScore("accuracy")));
            System.out.println("     Complexity: " + String.format("%.4f", -solution.getScore("model_complexity")));
            System.out.println("     Training Time: " + String.format("%.2f", -solution.getScore("training_time")) + "s");
        }
        
        // 8. Optimization convergence analysis
        var convergenceHistory = bayesianSearch.getConvergenceHistory();
        System.out.println("\n📈 Convergence Analysis:");
        System.out.println("   Iterations to best solution: " + bayesianSearch.getBestIteration());
        System.out.println("   Improvement over random search: " + 
                         String.format("%.2f%%", bayesianSearch.getImprovementOverRandom() * 100));
        
        // 9. Advanced visualizations
        System.out.println("\n📊 Creating optimization visualizations...");
        
        // Optimization progress
        VisualizationFactory.createOptimizationProgressPlot(
            convergenceHistory,
            "Bayesian Optimization Convergence"
        ).display();
        
        // Parameter importance
        var paramImportance = bayesianSearch.getParameterImportance();
        VisualizationFactory.createParameterImportancePlot(
            paramImportance,
            "Hyperparameter Importance Analysis"
        ).display();
        
        // Pareto frontier
        VisualizationFactory.createParetoFrontierPlot(
            paretoFrontier,
            "Multi-Objective Optimization Results"
        ).display();
        
        // 10. Final model training and evaluation
        System.out.println("\n🎯 Training final optimized model...");
        var finalModel = bayesianSearch.getBestEstimator();
        var split = ModelSelection.trainTestSplit(dataset.X, dataset.y, 0.2, 42);
        
        finalModel.fit(split.XTrain, split.yTrain);
        double[] predictions = finalModel.predict(split.XTest);
        var finalMetrics = Metrics.classificationReport(split.yTest, predictions);
        
        System.out.println("   Final Test Performance:");
        System.out.println("     Accuracy: " + String.format("%.4f", finalMetrics.accuracy));
        System.out.println("     F1-Score: " + String.format("%.4f", finalMetrics.f1Score));
        System.out.println("     Precision: " + String.format("%.4f", finalMetrics.precision));
        System.out.println("     Recall: " + String.format("%.4f", finalMetrics.recall));
        
        System.out.println("✅ Bayesian hyperparameter optimization completed!");
    }
}
```

## 🧠 Deep Learning Integration

### Neural Network Ensemble with Transfer Learning

```java
import org.superml.neural.MLPClassifier;
import org.superml.neural.transferlearning.TransferLearning;
import org.superml.ensemble.VotingClassifier;
import org.superml.datasets.Datasets;
import org.superml.preprocessing.StandardScaler;
import org.superml.visualization.VisualizationFactory;

import java.util.*;

public class NeuralNetworkEnsembleExample {
    public static void main(String[] args) {
        System.out.println("🧠 SuperML Java 2.0.0 - Neural Network Ensemble with Transfer Learning");
        
        // 1. Create complex dataset
        var dataset = Datasets.makeClassification(2000, 50, 10, 42); // 10 classes
        var split = ModelSelection.trainTestSplit(dataset.X, dataset.y, 0.2, 42);
        System.out.println("✓ Generated complex dataset: " + dataset.X.length + " samples, " + dataset.X[0].length + " features, 10 classes");
        
        // 2. Preprocessing pipeline
        var scaler = new StandardScaler();
        double[][] XTrainScaled = scaler.fitTransform(split.XTrain);
        double[][] XTestScaled = scaler.transform(split.XTest);
        
        // 3. Pre-trained feature extractor (simulated transfer learning)
        var featureExtractor = new TransferLearning.FeatureExtractor()
            .loadPretrainedWeights("imagenet_features")  // Simulated pre-trained features
            .setFreezeWeights(true)                       // Freeze pre-trained layers
            .setOutputDim(128);                          // Extract 128 features
        
        double[][] extractedFeatures = featureExtractor.transform(XTrainScaled);
        double[][] extractedFeaturesTest = featureExtractor.transform(XTestScaled);
        
        System.out.println("✓ Extracted features: " + extractedFeatures[0].length + " dimensions");
        
        // 4. Create diverse neural network ensemble
        List<org.superml.core.Classifier> baseModels = new ArrayList<>();
        
        // Deep narrow network
        baseModels.add(new MLPClassifier()
            .setHiddenLayerSizes(new int[]{256, 128, 64, 32})
            .setActivation("relu")
            .setDropoutRate(0.3)
            .setLearningRate(0.001)
            .setMaxIter(200)
            .setBatchSize(64)
            .setEarlyStopping(true)
            .setValidationFraction(0.1));
        
        // Wide shallow network
        baseModels.add(new MLPClassifier()
            .setHiddenLayerSizes(new int[]{512, 256})
            .setActivation("tanh")
            .setDropoutRate(0.2)
            .setLearningRate(0.005)
            .setMaxIter(150)
            .setBatchSize(32)
            .setRegularization("l2")
            .setAlpha(0.001));
        
        // Regularized network
        baseModels.add(new MLPClassifier()
            .setHiddenLayerSizes(new int[]{128, 128, 128})
            .setActivation("relu")
            .setDropoutRate(0.5)
            .setLearningRate(0.002)
            .setMaxIter(300)
            .setBatchSize(128)
            .setRegularization("l1")
            .setAlpha(0.01));
        
        // 5. Create voting ensemble
        var ensemble = new VotingClassifier(baseModels)
            .setVotingType("soft")                    // Use probability voting
            .setWeights(new double[]{0.4, 0.3, 0.3}) // Weighted voting
            .setParallelTraining(true)               // Train models in parallel
            .setCrossValidationFolds(3);             // CV for weight optimization
        
        // 6. Train ensemble with progress monitoring
        System.out.println("\n🔄 Training neural network ensemble...");
        long startTime = System.currentTimeMillis();
        
        ensemble.fit(extractedFeatures, split.yTrain);
        
        long trainingTime = System.currentTimeMillis() - startTime;
        System.out.println("✓ Ensemble training completed in " + trainingTime + "ms");
        
        // 7. Evaluate individual models and ensemble
        System.out.println("\n📊 Model Performance Comparison:");
        Map<String, Double> modelScores = new HashMap<>();
        
        for (int i = 0; i < baseModels.size(); i++) {
            double[] predictions = baseModels.get(i).predict(extractedFeaturesTest);
            var metrics = Metrics.classificationReport(split.yTest, predictions);
            String modelName = "Neural_Net_" + (i + 1);
            modelScores.put(modelName, metrics.accuracy);
            System.out.println("   " + modelName + ": " + String.format("%.4f", metrics.accuracy));
        }
        
        // Ensemble performance
        double[] ensemblePredictions = ensemble.predict(extractedFeaturesTest);
        var ensembleMetrics = Metrics.classificationReport(split.yTest, ensemblePredictions);
        modelScores.put("Ensemble", ensembleMetrics.accuracy);
        System.out.println("   Ensemble: " + String.format("%.4f", ensembleMetrics.accuracy));
        
        // 8. Ensemble analysis
        System.out.println("\n🔍 Ensemble Analysis:");
        var ensembleWeights = ensemble.getOptimizedWeights();
        for (int i = 0; i < ensembleWeights.length; i++) {
            System.out.println("   Model " + (i+1) + " weight: " + String.format("%.3f", ensembleWeights[i]));
        }
        
        // Prediction confidence analysis
        double[][] predictionsProba = ensemble.predictProba(extractedFeaturesTest);
        double avgConfidence = Arrays.stream(predictionsProba)
            .mapToDouble(proba -> Arrays.stream(proba).max().orElse(0.0))
            .average().orElse(0.0);
        System.out.println("   Average prediction confidence: " + String.format("%.3f", avgConfidence));
        
        // 9. Advanced visualizations
        System.out.println("\n📈 Creating ensemble visualizations...");
        
        // Model comparison
        VisualizationFactory.createModelComparisonChart(
            new ArrayList<>(modelScores.keySet()),
            new ArrayList<>(modelScores.values()),
            "Neural Network Ensemble Performance"
        ).display();
        
        // Confusion matrix
        VisualizationFactory.createDualModeConfusionMatrix(
            split.yTest,
            ensemblePredictions,
            IntStream.range(0, 10).mapToObj(i -> "Class_" + i).toArray(String[]::new)
        ).display();
        
        // Training convergence for each model
        for (int i = 0; i < baseModels.size(); i++) {
            var model = (MLPClassifier) baseModels.get(i);
            VisualizationFactory.createTrainingProgressPlot(
                model.getLossHistory(),
                "Neural Network " + (i+1) + " Training Progress"
            ).display();
        }
        
        System.out.println("✅ Neural network ensemble example completed!");
    }
}
```

## 🎛️ Custom Pipeline Development

### Advanced Feature Engineering Pipeline

```java
import org.superml.pipeline.Pipeline;
import org.superml.preprocessing.*;
import org.superml.feature_selection.*;
import org.superml.feature_engineering.*;
import org.superml.datasets.Datasets;
import org.superml.linear_model.LogisticRegression;
import org.superml.visualization.VisualizationFactory;

public class AdvancedFeatureEngineeringPipeline {
    public static void main(String[] args) {
        System.out.println("🎛️ SuperML Java 2.0.0 - Advanced Feature Engineering Pipeline");
        
        // 1. Load mixed-type dataset
        var dataset = Datasets.loadTitanic(); // Mixed numerical/categorical data
        System.out.println("✓ Loaded Titanic dataset: " + dataset.X.length + " samples");
        
        // 2. Create comprehensive preprocessing pipeline
        var featurePipeline = new Pipeline()
            // Data cleaning
            .addStep("missing_indicator", new MissingIndicator())
            .addStep("imputer", new SimpleImputer().setStrategy("median"))
            
            // Categorical encoding
            .addStep("onehot_encoder", new OneHotEncoder()
                .setDropFirst(true)
                .setHandleUnknown("ignore"))
            
            // Feature engineering
            .addStep("polynomial_features", new PolynomialFeatures(2)
                .setInteractionOnly(false)
                .setIncludeBias(false))
            
            // Feature scaling
            .addStep("robust_scaler", new RobustScaler())
            
            // Feature selection
            .addStep("variance_threshold", new VarianceThreshold(0.01))
            .addStep("univariate_selection", new SelectKBest(50)
                .setScoreFunction("f_classif"))
            
            // Advanced feature selection
            .addStep("recursive_feature_elimination", new RFE(
                new LogisticRegression(), 30))
            
            // Final normalization
            .addStep("standard_scaler", new StandardScaler());
        
        // 3. Fit the entire pipeline
        System.out.println("\n🔄 Fitting advanced feature engineering pipeline...");
        long startTime = System.currentTimeMillis();
        
        featurePipeline.fit(dataset.X, dataset.y);
        
        long pipelineTime = System.currentTimeMillis() - startTime;
        System.out.println("✓ Pipeline fitted in " + pipelineTime + "ms");
        
        // 4. Transform data through pipeline
        double[][] XTransformed = featurePipeline.transform(dataset.X);
        System.out.println("✓ Feature transformation completed");
        System.out.println("   Original features: " + dataset.X[0].length);
        System.out.println("   Final features: " + XTransformed[0].length);
        
        // 5. Analyze feature engineering impact
        System.out.println("\n📊 Feature Engineering Analysis:");
        
        // Get intermediate transformations
        var stepOutputs = featurePipeline.getStepOutputs();
        for (Map.Entry<String, double[][]> entry : stepOutputs.entrySet()) {
            System.out.println("   After " + entry.getKey() + ": " + entry.getValue()[0].length + " features");
        }
        
        // Feature importance from RFE
        var rfe = (RFE) featurePipeline.getStep("recursive_feature_elimination");
        boolean[] selectedFeatures = rfe.getSupport();
        double[] featureRanking = rfe.getRanking();
        
        System.out.println("\n🔍 Feature Selection Results:");
        System.out.println("   Selected features: " + Arrays.stream(selectedFeatures).mapToInt(b -> b ? 1 : 0).sum() + "/" + selectedFeatures.length);
        
        // Top 10 features by ranking
        System.out.println("   Top 10 selected features by importance:");
        IntStream.range(0, featureRanking.length)
            .filter(i -> selectedFeatures[i])
            .boxed()
            .sorted(Comparator.comparingDouble(i -> featureRanking[i]))
            .limit(10)
            .forEach(i -> System.out.println("     Feature " + i + " (rank: " + featureRanking[i] + ")"));
        
        // 6. Compare models with/without feature engineering
        var split = ModelSelection.trainTestSplit(dataset.X, dataset.y, 0.2, 42);
        var splitTransformed = ModelSelection.trainTestSplit(XTransformed, dataset.y, 0.2, 42);
        
        // Model without feature engineering
        var modelBasic = new LogisticRegression().setMaxIter(1000);
        modelBasic.fit(split.XTrain, split.yTrain);
        double[] predictionsBasic = modelBasic.predict(split.XTest);
        var metricsBasic = Metrics.classificationReport(split.yTest, predictionsBasic);
        
        // Model with feature engineering
        var modelAdvanced = new LogisticRegression().setMaxIter(1000);
        modelAdvanced.fit(splitTransformed.XTrain, splitTransformed.yTrain);
        double[] predictionsAdvanced = modelAdvanced.predict(splitTransformed.XTest);
        var metricsAdvanced = Metrics.classificationReport(splitTransformed.yTest, predictionsAdvanced);
        
        // 7. Performance comparison
        System.out.println("\n🏆 Performance Comparison:");
        System.out.println("   Basic Model (no feature engineering):");
        System.out.println("     Accuracy: " + String.format("%.4f", metricsBasic.accuracy));
        System.out.println("     F1-Score: " + String.format("%.4f", metricsBasic.f1Score));
        
        System.out.println("   Advanced Model (with feature engineering):");
        System.out.println("     Accuracy: " + String.format("%.4f", metricsAdvanced.accuracy));
        System.out.println("     F1-Score: " + String.format("%.4f", metricsAdvanced.f1Score));
        
        double improvement = (metricsAdvanced.accuracy - metricsBasic.accuracy) / metricsBasic.accuracy * 100;
        System.out.println("   Improvement: " + String.format("%.2f%%", improvement));
        
        // 8. Advanced visualizations
        System.out.println("\n📈 Creating feature engineering visualizations...");
        
        // Feature transformation pipeline
        VisualizationFactory.createPipelineVisualization(
            featurePipeline,
            "Advanced Feature Engineering Pipeline"
        ).display();
        
        // Feature importance comparison
        VisualizationFactory.createFeatureImportanceComparison(
            modelBasic.getFeatureImportances(),
            modelAdvanced.getFeatureImportances(),
            "Feature Importance: Basic vs Advanced"
        ).display();
        
        // Performance comparison
        VisualizationFactory.createModelComparisonChart(
            Arrays.asList("Basic Model", "Advanced Model"),
            Arrays.asList(metricsBasic.accuracy, metricsAdvanced.accuracy),
            "Feature Engineering Impact"
        ).display();
        
        System.out.println("✅ Advanced feature engineering pipeline completed!");
    }
}
```

## 🚀 Production Deployment

### High-Performance Inference Service

```java
import org.superml.inference.InferenceEngine;
import org.superml.inference.ModelServer;
import org.superml.monitoring.PerformanceMonitor;
import org.superml.monitoring.DriftDetector;
import org.superml.persistence.ModelPersistence;
import org.superml.datasets.Datasets;

import java.util.concurrent.*;

public class ProductionInferenceService {
    public static void main(String[] args) throws Exception {
        System.out.println("🚀 SuperML Java 2.0.0 - Production Inference Service");
        
        // 1. Setup production environment
        var dataset = Datasets.loadIris();
        var model = new LogisticRegression().setMaxIter(1000);
        model.fit(dataset.X, dataset.y);
        
        // Save model with production metadata
        String modelPath = ModelPersistence.saveForProduction(
            model, "iris_production_v1.0",
            Map.of(
                "version", "1.0",
                "training_date", System.currentTimeMillis(),
                "model_type", "classification",
                "performance_benchmark", "0.97_accuracy"
            )
        );
        
        // 2. Initialize high-performance inference engine
        var inferenceEngine = new InferenceEngine()
            .setModelCache(true)
            .setCacheSize(100)
            .setBatchOptimization(true)
            .setMaxBatchSize(1000)
            .setBatchTimeout(50)              // 50ms batch timeout
            .setThreadPoolSize(8)             // 8 inference threads
            .setPerformanceMonitoring(true)
            .setMemoryOptimization(true);
        
        // 3. Setup model server
        var modelServer = new ModelServer()
            .setPort(8080)
            .setInferenceEngine(inferenceEngine)
            .setMaxConnections(1000)
            .setRequestTimeout(5000)
            .setHealthCheckEnabled(true)
            .setMetricsEndpoint("/metrics")
            .setModelEndpoint("/predict")
            .setSecurityEnabled(false);       // Simplified for example
        
        // 4. Load and register production model
        var productionModel = ModelPersistence.load(modelPath);
        inferenceEngine.registerModel("iris_v1", productionModel);
        System.out.println("✓ Model registered for production inference");
        
        // 5. Setup monitoring and drift detection
        var performanceMonitor = new PerformanceMonitor()
            .setMetricsRetention(24)          // 24 hours
            .setSamplingRate(0.1)             // Sample 10% of requests
            .setAlertThresholds(Map.of(
                "latency_p95", 100.0,         // 100ms P95 latency
                "throughput_min", 100.0,      // Min 100 RPS
                "error_rate_max", 0.01        // Max 1% error rate
            ));
        
        var driftDetector = new DriftDetector()
            .setReferenceData(dataset.X)
            .setDetectionMethod("ks_test")
            .setSignificanceLevel(0.05)
            .setWindowSize(1000)
            .setUpdateFrequency(100);
        
        inferenceEngine.addMonitor(performanceMonitor);
        inferenceEngine.addDriftDetector(driftDetector);
        
        // 6. Start model server (non-blocking)
        System.out.println("\n🌐 Starting production model server...");
        modelServer.start();
        System.out.println("✓ Model server started on port 8080");
        
        // 7. Simulate production load
        System.out.println("\n⚡ Simulating production inference load...");
        
        // Create thread pool for load simulation
        ExecutorService executor = Executors.newFixedThreadPool(20);
        CountDownLatch latch = new CountDownLatch(1000);
        
        // Simulate 1000 concurrent requests
        long loadTestStart = System.currentTimeMillis();
        
        for (int i = 0; i < 1000; i++) {
            final int requestId = i;
            executor.submit(() -> {
                try {
                    // Generate random sample similar to iris data
                    double[] sample = {
                        4.0 + Math.random() * 4.0,  // Sepal length
                        2.0 + Math.random() * 3.0,  // Sepal width  
                        1.0 + Math.random() * 6.0,  // Petal length
                        0.1 + Math.random() * 2.5   // Petal width
                    };
                    
                    // Make prediction
                    double prediction = inferenceEngine.predict("iris_v1", sample);
                    
                    // Simulate some drift after 500 requests
                    if (requestId > 500) {
                        driftDetector.checkForDrift(sample);
                    }
                    
                } catch (Exception e) {
                    System.err.println("Request " + requestId + " failed: " + e.getMessage());
                } finally {
                    latch.countDown();
                }
            });
        }
        
        // Wait for all requests to complete
        latch.await(30, TimeUnit.SECONDS);
        long loadTestDuration = System.currentTimeMillis() - loadTestStart;
        
        // 8. Analyze production performance
        System.out.println("\n📊 Production Performance Analysis:");
        var metrics = performanceMonitor.getMetrics("iris_v1");
        
        System.out.println("   Load Test Duration: " + loadTestDuration + "ms");
        System.out.println("   Total Requests: " + metrics.getTotalRequests());
        System.out.println("   Successful Requests: " + metrics.getSuccessfulRequests());
        System.out.println("   Error Rate: " + String.format("%.2f%%", metrics.getErrorRate() * 100));
        System.out.println("   Average Latency: " + String.format("%.2f", metrics.getAverageLatency()) + "ms");
        System.out.println("   P95 Latency: " + String.format("%.2f", metrics.getP95Latency()) + "ms");
        System.out.println("   P99 Latency: " + String.format("%.2f", metrics.getP99Latency()) + "ms");
        System.out.println("   Throughput: " + String.format("%.0f", metrics.getThroughput()) + " RPS");
        
        // Memory usage
        System.out.println("   Memory Usage: " + String.format("%.2f", metrics.getMemoryUsage()) + "MB");
        System.out.println("   Cache Hit Rate: " + String.format("%.2f%%", metrics.getCacheHitRate() * 100));
        
        // 9. Drift detection results
        System.out.println("\n🔍 Drift Detection Analysis:");
        if (driftDetector.isDriftDetected()) {
            System.out.println("   ⚠️  Data drift detected!");
            System.out.println("   Drift Score: " + String.format("%.4f", driftDetector.getDriftScore()));
            System.out.println("   Detection Point: Request #" + driftDetector.getDriftDetectionPoint());
        } else {
            System.out.println("   ✓ No significant data drift detected");
        }
        
        // 10. Production health check
        System.out.println("\n🏥 Production Health Check:");
        var healthStatus = modelServer.getHealthStatus();
        System.out.println("   Server Status: " + healthStatus.getStatus());
        System.out.println("   Uptime: " + healthStatus.getUptime() + "ms");
        System.out.println("   Memory Available: " + healthStatus.getAvailableMemory() + "MB");
        System.out.println("   CPU Usage: " + String.format("%.2f%%", healthStatus.getCpuUsage() * 100));
        
        // 11. Cleanup
        executor.shutdown();
        modelServer.stop();
        
        System.out.println("✅ Production inference service example completed!");
    }
}
```

## 📊 Advanced Analytics and Reporting

### Comprehensive ML Experiment Tracking

```java
import org.superml.experiments.ExperimentTracker;
import org.superml.experiments.MLExperiment;
import org.superml.autotrainer.AutoTrainer;
import org.superml.datasets.Datasets;
import org.superml.visualization.VisualizationFactory;

import java.util.*;

public class MLExperimentTrackingExample {
    public static void main(String[] args) {
        System.out.println("📊 SuperML Java 2.0.0 - ML Experiment Tracking");
        
        // 1. Initialize experiment tracker
        var experimentTracker = new ExperimentTracker()
            .setProjectName("SuperML_Advanced_Classification")
            .setExperimentName("Algorithm_Comparison_Study")
            .setStorageBackend("local")
            .setResultsPath("./ml_experiments")
            .setAutoSave(true)
            .setVerbose(true);
        
        // 2. Define datasets for comprehensive evaluation
        List<Map<String, Object>> datasets = Arrays.asList(
            Map.of("name", "Iris", "data", Datasets.loadIris()),
            Map.of("name", "Wine", "data", Datasets.loadWine()),
            Map.of("name", "Digits", "data", Datasets.loadDigits()),
            Map.of("name", "Synthetic_Easy", "data", Datasets.makeClassification(1000, 20, 3, 42, 0.0)),
            Map.of("name", "Synthetic_Hard", "data", Datasets.makeClassification(1000, 50, 5, 42, 0.2))
        );
        
        // 3. Define algorithms for comparison
        List<String> algorithms = Arrays.asList(
            "LogisticRegression", "RandomForest", "GradientBoosting", 
            "SVM", "NeuralNetwork", "AutoML"
        );
        
        System.out.println("✓ Initialized experiment: " + datasets.size() + " datasets, " + algorithms.size() + " algorithms");
        
        // 4. Run comprehensive experiments
        System.out.println("\n🔄 Running comprehensive ML experiments...");
        Map<String, List<MLExperiment>> allExperiments = new HashMap<>();
        
        for (Map<String, Object> datasetInfo : datasets) {
            String datasetName = (String) datasetInfo.get("name");
            var dataset = datasetInfo.get("data");
            
            System.out.println("\n📋 Running experiments on " + datasetName + " dataset...");
            List<MLExperiment> datasetExperiments = new ArrayList<>();
            
            for (String algorithm : algorithms) {
                System.out.println("   Testing " + algorithm + "...");
                
                // Create experiment
                var experiment = new MLExperiment(datasetName + "_" + algorithm)
                    .setDataset(datasetName, dataset)
                    .setAlgorithm(algorithm)
                    .setRandomState(42)
                    .setValidationStrategy("stratified_kfold", 5)
                    .setMetrics(Arrays.asList("accuracy", "f1_macro", "precision_macro", "recall_macro"))
                    .setHyperparameterOptimization(true)
                    .setFeatureEngineering(true)
                    .setTimeLimit(120); // 2 minutes per experiment
                
                // Run experiment
                long startTime = System.currentTimeMillis();
                var results = experiment.run();
                long duration = System.currentTimeMillis() - startTime;
                
                // Log results
                experimentTracker.logExperiment(experiment, results);
                datasetExperiments.add(experiment);
                
                System.out.println("     ✓ " + algorithm + " completed in " + duration + "ms");
                System.out.println("       Accuracy: " + String.format("%.4f", results.getMetric("accuracy")));
                System.out.println("       F1-Score: " + String.format("%.4f", results.getMetric("f1_macro")));
            }
            
            allExperiments.put(datasetName, datasetExperiments);
        }
        
        // 5. Comprehensive analysis
        System.out.println("\n📊 Comprehensive Experiment Analysis:");
        
        // Overall best performers
        var overallResults = experimentTracker.getOverallResults();
        System.out.println("\n🏆 Overall Best Performers:");
        
        overallResults.getBestByMetric("accuracy")
            .entrySet().stream()
            .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
            .limit(5)
            .forEach(entry -> System.out.println("   " + entry.getKey() + ": " + String.format("%.4f", entry.getValue())));
        
        // Algorithm ranking across datasets
        System.out.println("\n📈 Algorithm Ranking (by average accuracy):");
        var algorithmRanking = experimentTracker.getAlgorithmRanking("accuracy");
        algorithmRanking.forEach((algorithm, avgScore) -> 
            System.out.println("   " + algorithm + ": " + String.format("%.4f", avgScore)));
        
        // Dataset difficulty analysis
        System.out.println("\n🎯 Dataset Difficulty Analysis:");
        var datasetDifficulty = experimentTracker.getDatasetDifficulty();
        datasetDifficulty.entrySet().stream()
            .sorted(Map.Entry.comparingByValue())
            .forEach(entry -> System.out.println("   " + entry.getKey() + ": " + String.format("%.4f", entry.getValue()) + " (lower = harder)"));
        
        // 6. Statistical significance testing
        System.out.println("\n🔬 Statistical Significance Analysis:");
        var significanceResults = experimentTracker.performSignificanceTests("accuracy", 0.05);
        
        significanceResults.forEach((comparison, result) -> {
            if (result.isSignificant()) {
                System.out.println("   " + comparison + ": Significant difference (p=" + String.format("%.4f", result.getPValue()) + ")");
            }
        });
        
        // 7. Advanced visualizations
        System.out.println("\n📊 Creating comprehensive experiment visualizations...");
        
        // Algorithm performance heatmap
        VisualizationFactory.createExperimentHeatmap(
            allExperiments,
            "accuracy",
            "Algorithm Performance Across Datasets"
        ).display();
        
        // Performance distribution box plots
        VisualizationFactory.createPerformanceDistributionPlot(
            allExperiments,
            "Algorithm Performance Distribution"
        ).display();
        
        // Training time vs accuracy scatter
        VisualizationFactory.createTrainingTimeVsAccuracyPlot(
            allExperiments,
            "Training Time vs Accuracy Trade-off"
        ).display();
        
        // Statistical significance visualization
        VisualizationFactory.createSignificanceMatrix(
            significanceResults,
            "Statistical Significance Matrix"
        ).display();
        
        // 8. Generate comprehensive report
        System.out.println("\n📄 Generating comprehensive experiment report...");
        var report = experimentTracker.generateReport()
            .includeDatasetSummary(true)
            .includeAlgorithmComparison(true)
            .includeStatisticalTests(true)
            .includeVisualizationSummary(true)
            .includeRecommendations(true)
            .setFormat("html");
        
        String reportPath = report.save("./ml_experiments/comprehensive_report.html");
        System.out.println("✓ Comprehensive report saved: " + reportPath);
        
        // 9. Export results for further analysis
        experimentTracker.exportResults("./ml_experiments/experiment_results.csv", "csv");
        experimentTracker.exportResults("./ml_experiments/experiment_results.json", "json");
        
        System.out.println("✓ Experiment results exported");
        System.out.println("✅ ML experiment tracking example completed!");
        
        // 10. Print final summary
        System.out.println("\n📋 Final Experiment Summary:");
        System.out.println("   Total Experiments: " + experimentTracker.getTotalExperiments());
        System.out.println("   Total Runtime: " + experimentTracker.getTotalRuntime() + "ms");
        System.out.println("   Best Overall Algorithm: " + algorithmRanking.entrySet().iterator().next().getKey());
        System.out.println("   Most Challenging Dataset: " + datasetDifficulty.entrySet().stream()
            .min(Map.Entry.comparingByValue()).map(Map.Entry::getKey).orElse("Unknown"));
    }
}
```

---

## 🎯 Advanced Learning Path

### For Advanced Users:

1. **Enterprise AutoML** - Start with comprehensive AutoML workflows and custom metrics
2. **Advanced Tree Algorithms** - Master gradient boosting with early stopping and custom loss functions  
3. **Bayesian Optimization** - Learn sophisticated hyperparameter optimization techniques
4. **Neural Network Ensembles** - Explore deep learning integration and transfer learning
5. **Feature Engineering Pipelines** - Build sophisticated preprocessing workflows
6. **Production Deployment** - Master high-performance inference and monitoring
7. **Experiment Tracking** - Implement comprehensive ML experiment management

### Advanced Topics Covered:

- **AutoML with Custom Metrics**: Business-specific evaluation functions
- **Multi-Objective Optimization**: Pareto frontier analysis and trade-offs
- **Transfer Learning**: Pre-trained feature extractors and domain adaptation
- **Ensemble Methods**: Advanced voting and stacking strategies
- **Production Monitoring**: Drift detection and performance monitoring
- **Statistical Analysis**: Significance testing and confidence intervals
- **Scalable Inference**: Batch optimization and async prediction
- **Comprehensive Reporting**: Automated experiment documentation

### Next Steps:
- Explore the complete [API Reference](../api/core-classes.md) for all available methods
- Check out specialized modules in the SuperML ecosystem
- Build custom algorithms using the foundation interfaces
- Integrate with cloud platforms and big data systems

---

**Master the full power of SuperML Java 2.0.0!** 🚀 These advanced examples showcase the framework's sophisticated capabilities for enterprise-grade machine learning applications.
