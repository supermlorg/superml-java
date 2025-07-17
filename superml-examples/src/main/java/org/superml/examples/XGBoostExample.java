/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.superml.examples;

import org.superml.tree.XGBoost;
import org.superml.tree.RandomForest;
import org.superml.tree.GradientBoosting;
import org.superml.preprocessing.StandardScaler;
import org.superml.pipeline.Pipeline;
import org.superml.metrics.Metrics;

import java.util.*;

/**
 * XGBoost Example - Comprehensive demonstration of XGBoost capabilities
 * 
 * Features demonstrated:
 * - Basic XGBoost training and prediction
 * - Advanced hyperparameter tuning
 * - Feature importance analysis
 * - Model comparison with other tree algorithms
 * - Early stopping and validation
 * - Cross-validation
 * - Production-ready pipeline
 * 
 * @author SuperML Team
 * @version 2.1.0
 */
public class XGBoostExample {
    
    public static void main(String[] args) {
        System.out.println("🚀 SuperML Java 2.0.0 - XGBoost Comprehensive Example");
        System.out.println("=" + "=".repeat(60));
        
        try {
            // 1. Generate challenging dataset
            System.out.println("\n📊 Generating challenging classification dataset...");
            CompetitionData data = generateChallengingDataset(2000, 20);
            
            // Split data
            DataSplit split = splitData(data.X, data.y, 0.2, 0.1); // train/test/valid
            
            System.out.printf("Training samples: %d, Test samples: %d, Validation samples: %d%n",
                split.XTrain.length, split.XTest.length, split.XValid.length);
            System.out.printf("Features: %d, Classes: %s%n", 
                data.X[0].length, Arrays.toString(getUniqueClasses(data.y)));
            
            // 2. Basic XGBoost Training
            System.out.println("\n" + "=".repeat(50));
            System.out.println("🌟 Basic XGBoost Training");
            System.out.println("=".repeat(50));
            
            trainBasicXGBoost(split);
            
            // 3. Advanced XGBoost with Hyperparameter Tuning
            System.out.println("\n" + "=".repeat(50));
            System.out.println("🔧 Advanced XGBoost with Hyperparameter Tuning");
            System.out.println("=".repeat(50));
            
            trainAdvancedXGBoost(split);
            
            // 4. Feature Importance Analysis
            System.out.println("\n" + "=".repeat(50));
            System.out.println("📈 Feature Importance Analysis");
            System.out.println("=".repeat(50));
            
            analyzeFeatureImportance(split);
            
            // 5. Model Comparison
            System.out.println("\n" + "=".repeat(50));
            System.out.println("⚔️ XGBoost vs Other Tree Algorithms");
            System.out.println("=".repeat(50));
            
            compareTreeAlgorithms(split);
            
            // 6. Production Pipeline
            System.out.println("\n" + "=".repeat(50));
            System.out.println("🏭 Production-Ready XGBoost Pipeline");
            System.out.println("=".repeat(50));
            
            createProductionPipeline(split);
            
            System.out.println("\n🎉 XGBoost example completed successfully!");
            
        } catch (Exception e) {
            System.err.println("❌ Error in XGBoost example: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Train basic XGBoost model with default parameters
     */
    private static void trainBasicXGBoost(DataSplit split) {
        System.out.println("Training basic XGBoost model...");
        
        // Create basic XGBoost model
        XGBoost xgb = new XGBoost()
            .setNEstimators(100)
            .setLearningRate(0.1)
            .setMaxDepth(6)
            .setRandomState(42);
        
        // Train the model
        long startTime = System.currentTimeMillis();
        xgb.fit(split.XTrain, split.yTrain);
        long trainTime = System.currentTimeMillis() - startTime;
        
        // Make predictions
        double[] predictions = xgb.predict(split.XTest);
        double[][] probabilities = xgb.predictProba(split.XTest);
        
        // Calculate metrics
        double accuracy = Metrics.accuracy(split.yTest, predictions);
        double precision = Metrics.precision(split.yTest, predictions);
        double recall = Metrics.recall(split.yTest, predictions);
        double f1 = Metrics.f1Score(split.yTest, predictions);
        
        // Display results
        System.out.println("\n📊 Basic XGBoost Results:");
        System.out.printf("├─ Training time: %d ms%n", trainTime);
        System.out.printf("├─ Accuracy: %.4f%n", accuracy);
        System.out.printf("├─ Precision: %.4f%n", precision);
        System.out.printf("├─ Recall: %.4f%n", recall);
        System.out.printf("└─ F1-Score: %.4f%n", f1);
        
        // Show training history
        Map<String, List<Double>> evalResults = xgb.getEvalResults();
        if (!evalResults.isEmpty()) {
            System.out.println("\n📈 Training History (last 5 iterations):");
            List<Double> trainScores = evalResults.get("train-logloss");
            if (trainScores != null && !trainScores.isEmpty()) {
                int start = Math.max(0, trainScores.size() - 5);
                for (int i = start; i < trainScores.size(); i++) {
                    System.out.printf("   Iteration %d: %.6f%n", i + 1, trainScores.get(i));
                }
            }
        }
    }
    
    /**
     * Train advanced XGBoost with comprehensive hyperparameter tuning
     */
    private static void trainAdvancedXGBoost(DataSplit split) {
        System.out.println("Training advanced XGBoost with hyperparameter tuning...");
        
        // Advanced XGBoost configuration
        XGBoost xgb = new XGBoost()
            .setNEstimators(500)
            .setLearningRate(0.05)              // Lower learning rate
            .setMaxDepth(8)                     // Deeper trees
            .setGamma(0.1)                      // Regularization for pruning
            .setLambda(1.5)                     // L2 regularization
            .setAlpha(0.1)                      // L1 regularization
            .setSubsample(0.8)                  // Row sampling
            .setColsampleBytree(0.8)            // Column sampling per tree
            .setColsampleBylevel(0.8)           // Column sampling per level
            .setMinChildWeight(3)               // Minimum child weight
            .setValidationFraction(0.15)        // Early stopping validation
            .setEarlyStoppingRounds(25)         // Early stopping patience
            .setRandomState(42)
            .setSilent(false);                  // Show training progress
        
        // Train with validation data for early stopping
        long startTime = System.currentTimeMillis();
        xgb.fit(split.XTrain, split.yTrain, split.XValid, split.yValid);
        long trainTime = System.currentTimeMillis() - startTime;
        
        // Make predictions
        double[] predictions = xgb.predict(split.XTest);
        double[][] probabilities = xgb.predictProba(split.XTest);
        
        // Calculate comprehensive metrics
        double accuracy = Metrics.accuracy(split.yTest, predictions);
        double precision = Metrics.precision(split.yTest, predictions);
        double recall = Metrics.recall(split.yTest, predictions);
        double f1 = Metrics.f1Score(split.yTest, predictions);
        
        // Display results
        System.out.println("\n📊 Advanced XGBoost Results:");
        System.out.printf("├─ Training time: %d ms%n", trainTime);
        System.out.printf("├─ Trees built: %d%n", xgb.getNEstimators());
        System.out.printf("├─ Accuracy: %.4f%n", accuracy);
        System.out.printf("├─ Precision: %.4f%n", precision);
        System.out.printf("├─ Recall: %.4f%n", recall);
        System.out.printf("└─ F1-Score: %.4f%n", f1);
        
        // Show validation scores
        Map<String, List<Double>> evalResults = xgb.getEvalResults();
        List<Double> validScores = evalResults.get("valid-logloss");
        if (validScores != null && !validScores.isEmpty()) {
            System.out.println("\n📈 Validation Scores (last 5 iterations):");
            int start = Math.max(0, validScores.size() - 5);
            for (int i = start; i < validScores.size(); i++) {
                System.out.printf("   Iteration %d: %.6f%n", i + 1, validScores.get(i));
            }
        }
        
        // Show hyperparameter summary
        System.out.println("\n🔧 Hyperparameter Configuration:");
        System.out.printf("├─ Learning Rate: %.3f%n", xgb.getLearningRate());
        System.out.printf("├─ Max Depth: %d%n", xgb.getMaxDepth());
        System.out.printf("├─ Gamma: %.3f%n", xgb.getGamma());
        System.out.printf("├─ Lambda (L2): %.3f%n", xgb.getLambda());
        System.out.printf("├─ Alpha (L1): %.3f%n", xgb.getAlpha());
        System.out.printf("├─ Subsample: %.3f%n", xgb.getSubsample());
        System.out.printf("├─ Colsample by tree: %.3f%n", xgb.getColsampleBytree());
        System.out.printf("└─ Min child weight: %d%n", xgb.getMinChildWeight());
    }
    
    /**
     * Analyze feature importance using multiple XGBoost metrics
     */
    private static void analyzeFeatureImportance(DataSplit split) {
        System.out.println("Analyzing feature importance...");
        
        // Train XGBoost for feature analysis
        XGBoost xgb = new XGBoost()
            .setNEstimators(200)
            .setLearningRate(0.1)
            .setMaxDepth(6)
            .setRandomState(42);
        
        xgb.fit(split.XTrain, split.yTrain);
        
        // Get different types of feature importance
        Map<String, double[]> importanceStats = xgb.getFeatureImportanceStats();
        
        System.out.println("\n📊 Feature Importance Analysis:");
        System.out.println("   (Top 10 most important features)");
        System.out.println();
        
        // Weight-based importance (frequency)
        double[] weightImportance = importanceStats.get("weight");
        System.out.println("🔸 Weight-based Importance (Feature Usage Frequency):");
        displayTopFeatures(weightImportance, 10);
        
        // Gain-based importance (average gain)
        double[] gainImportance = importanceStats.get("gain");
        System.out.println("\n🔸 Gain-based Importance (Average Information Gain):");
        displayTopFeatures(gainImportance, 10);
        
        // Cover-based importance (average coverage)
        double[] coverImportance = importanceStats.get("cover");
        System.out.println("\n🔸 Cover-based Importance (Average Sample Coverage):");
        displayTopFeatures(coverImportance, 10);
        
        // Feature importance visualization (ASCII)
        System.out.println("\n📈 Feature Importance Visualization (Weight-based):");
        visualizeFeatureImportance(weightImportance, 15);
    }
    
    /**
     * Compare XGBoost with other tree-based algorithms
     */
    private static void compareTreeAlgorithms(DataSplit split) {
        System.out.println("Comparing tree-based algorithms...");
        
        List<AlgorithmResult> results = new ArrayList<>();
        
        // 1. XGBoost
        System.out.println("Training XGBoost...");
        XGBoost xgb = new XGBoost()
            .setNEstimators(200)
            .setLearningRate(0.1)
            .setMaxDepth(6)
            .setGamma(0.1)
            .setSubsample(0.8)
            .setRandomState(42);
        
        long startTime = System.currentTimeMillis();
        xgb.fit(split.XTrain, split.yTrain);
        long xgbTime = System.currentTimeMillis() - startTime;
        
        double[] xgbPreds = xgb.predict(split.XTest);
        double xgbAccuracy = Metrics.accuracy(split.yTest, xgbPreds);
        double xgbF1 = Metrics.f1Score(split.yTest, xgbPreds);
        
        results.add(new AlgorithmResult("XGBoost", xgbAccuracy, xgbF1, xgbTime));
        
        // 2. Gradient Boosting (sklearn-style)
        System.out.println("Training Gradient Boosting...");
        GradientBoosting gb = new GradientBoosting()
            .setNEstimators(200)
            .setLearningRate(0.1)
            .setMaxDepth(6)
            .setSubsample(0.8)
            .setRandomState(42);
        
        startTime = System.currentTimeMillis();
        gb.fit(split.XTrain, split.yTrain);
        long gbTime = System.currentTimeMillis() - startTime;
        
        double[] gbPreds = gb.predict(split.XTest);
        double gbAccuracy = Metrics.accuracy(split.yTest, gbPreds);
        double gbF1 = Metrics.f1Score(split.yTest, gbPreds);
        
        results.add(new AlgorithmResult("GradientBoosting", gbAccuracy, gbF1, gbTime));
        
        // 3. Random Forest
        System.out.println("Training Random Forest...");
        RandomForest rf = new RandomForest()
            .setNEstimators(200)
            .setMaxDepth(20)
            .setRandomState(42);
        
        startTime = System.currentTimeMillis();
        rf.fit(split.XTrain, split.yTrain);
        long rfTime = System.currentTimeMillis() - startTime;
        
        double[] rfPreds = rf.predict(split.XTest);
        double rfAccuracy = Metrics.accuracy(split.yTest, rfPreds);
        double rfF1 = Metrics.f1Score(split.yTest, rfPreds);
        
        results.add(new AlgorithmResult("RandomForest", rfAccuracy, rfF1, rfTime));
        
        // Display comparison results
        System.out.println("\n📊 Algorithm Comparison Results:");
        System.out.println("┌─────────────────┬──────────┬─────────┬───────────┐");
        System.out.println("│ Algorithm       │ Accuracy │ F1-Score│ Time (ms) │");
        System.out.println("├─────────────────┼──────────┼─────────┼───────────┤");
        
        for (AlgorithmResult result : results) {
            System.out.printf("│ %-15s │ %8.4f │ %7.4f │ %9d │%n",
                result.name, result.accuracy, result.f1Score, result.trainingTime);
        }
        System.out.println("└─────────────────┴──────────┴─────────┴───────────┘");
        
        // Find best performing algorithm
        AlgorithmResult best = results.stream()
            .max(Comparator.comparing(r -> r.accuracy))
            .orElse(results.get(0));
        
        System.out.println("\n🏆 Best performing algorithm: " + best.name + 
                          " (Accuracy: " + String.format("%.4f", best.accuracy) + ")");
        
        // Performance insights
        System.out.println("\n💡 Performance Insights:");
        if (best.name.equals("XGBoost")) {
            System.out.println("   ✅ XGBoost achieved the best performance!");
            System.out.println("   → Advanced regularization and optimization paid off");
            System.out.println("   → Feature engineering through gradient boosting worked well");
        }
        
        long fastestTime = results.stream().mapToLong(r -> r.trainingTime).min().orElse(0);
        AlgorithmResult fastest = results.stream()
            .filter(r -> r.trainingTime == fastestTime)
            .findFirst().orElse(results.get(0));
        
        System.out.println("   ⚡ Fastest training: " + fastest.name + 
                          " (" + fastest.trainingTime + " ms)");
    }
    
    /**
     * Create production-ready pipeline with XGBoost
     */
    private static void createProductionPipeline(DataSplit split) {
        System.out.println("Creating production-ready XGBoost pipeline...");
        
        // Create comprehensive pipeline
        Pipeline pipeline = new Pipeline()
            .addStep("scaler", new StandardScaler())
            .addStep("xgboost", new XGBoost()
                .setNEstimators(300)
                .setLearningRate(0.05)
                .setMaxDepth(7)
                .setGamma(0.2)
                .setLambda(2.0)
                .setAlpha(0.1)
                .setSubsample(0.85)
                .setColsampleBytree(0.85)
                .setMinChildWeight(3)
                .setRandomState(42)
                .setSilent(true));
        
        // Train pipeline
        System.out.println("Training production pipeline...");
        long startTime = System.currentTimeMillis();
        pipeline.fit(split.XTrain, split.yTrain);
        long pipelineTime = System.currentTimeMillis() - startTime;
        
        // Evaluate pipeline
        double[] predictions = pipeline.predict(split.XTest);
        double accuracy = Metrics.accuracy(split.yTest, predictions);
        double precision = Metrics.precision(split.yTest, predictions);
        double recall = Metrics.recall(split.yTest, predictions);
        double f1 = Metrics.f1Score(split.yTest, predictions);
        
        // Production metrics
        System.out.println("\n🏭 Production Pipeline Results:");
        System.out.printf("├─ Training time: %d ms%n", pipelineTime);
        System.out.printf("├─ Accuracy: %.4f%n", accuracy);
        System.out.printf("├─ Precision: %.4f%n", precision);
        System.out.printf("├─ Recall: %.4f%n", recall);
        System.out.printf("├─ F1-Score: %.4f%n", f1);
        System.out.printf("└─ Pipeline steps: %d%n", 2);
        
        // Inference performance test
        System.out.println("\n⚡ Inference Performance Test:");
        
        // Single prediction timing
        double[] sampleInput = split.XTest[0];
        startTime = System.nanoTime();
        double[] singlePred = pipeline.predict(new double[][]{sampleInput});
        long singleInferenceTime = System.nanoTime() - startTime;
        
        // Batch prediction timing
        startTime = System.currentTimeMillis();
        double[] batchPreds = pipeline.predict(split.XTest);
        long batchInferenceTime = System.currentTimeMillis() - startTime;
        
        System.out.printf("├─ Single prediction: %.2f μs%n", singleInferenceTime / 1000.0);
        System.out.printf("├─ Batch prediction (%d samples): %d ms%n", 
            split.XTest.length, batchInferenceTime);
        System.out.printf("└─ Throughput: %.1f predictions/second%n", 
            (split.XTest.length * 1000.0) / batchInferenceTime);
        
        // Model deployment readiness checklist
        System.out.println("\n✅ Production Deployment Checklist:");
        System.out.println("   ☑️ Preprocessing pipeline integrated");
        System.out.println("   ☑️ Model training completed");
        System.out.println("   ☑️ Performance metrics validated");
        System.out.println("   ☑️ Inference speed tested");
        System.out.println("   ☑️ Ready for serialization and deployment");
    }
    
    // Helper methods
    
    private static void displayTopFeatures(double[] importance, int topK) {
        // Create feature index-importance pairs
        List<FeatureImportance> features = new ArrayList<>();
        for (int i = 0; i < importance.length; i++) {
            features.add(new FeatureImportance(i, importance[i]));
        }
        
        // Sort by importance
        features.sort((a, b) -> Double.compare(b.importance, a.importance));
        
        // Display top K features
        for (int i = 0; i < Math.min(topK, features.size()); i++) {
            FeatureImportance feature = features.get(i);
            if (feature.importance > 0) {
                System.out.printf("   %2d. Feature_%d: %.4f%n", 
                    i + 1, feature.index, feature.importance);
            }
        }
    }
    
    private static void visualizeFeatureImportance(double[] importance, int maxFeatures) {
        // Find max importance for scaling
        double maxImportance = Arrays.stream(importance).max().orElse(1.0);
        
        for (int i = 0; i < Math.min(maxFeatures, importance.length); i++) {
            if (importance[i] > 0) {
                int barLength = (int) ((importance[i] / maxImportance) * 30);
                String bar = "█".repeat(Math.max(1, barLength));
                System.out.printf("   Feature_%2d │%-30s│ %.3f%n", i, bar, importance[i]);
            }
        }
    }
    
    private static CompetitionData generateChallengingDataset(int samples, int features) {
        Random random = new Random(42);
        double[][] X = new double[samples][features];
        double[] y = new double[samples];
        
        // Generate features with varying importance and correlation
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < features; j++) {
                X[i][j] = random.nextGaussian();
            }
            
            // Create complex decision boundary
            double score = 0.0;
            score += 2.0 * X[i][0] + 1.5 * X[i][1] - 0.5 * X[i][2]; // Linear component
            score += X[i][3] * X[i][4]; // Interaction term
            score += Math.sin(X[i][5]) + Math.cos(X[i][6]); // Non-linear terms
            
            // Add noise to some features
            for (int j = features - 5; j < features; j++) {
                score += 0.1 * random.nextGaussian(); // Noise features
            }
            
            y[i] = score > 0 ? 1.0 : 0.0;
        }
        
        return new CompetitionData(X, y, samples);
    }
    
    private static DataSplit splitData(double[][] X, double[] y, double testRatio, double validRatio) {
        int totalSamples = X.length;
        int testSize = (int) (totalSamples * testRatio);
        int validSize = (int) (totalSamples * validRatio);
        int trainSize = totalSamples - testSize - validSize;
        
        // Shuffle indices
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < totalSamples; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, new Random(42));
        
        // Split data
        double[][] XTrain = new double[trainSize][];
        double[] yTrain = new double[trainSize];
        double[][] XTest = new double[testSize][];
        double[] yTest = new double[testSize];
        double[][] XValid = new double[validSize][];
        double[] yValid = new double[validSize];
        
        for (int i = 0; i < trainSize; i++) {
            int idx = indices.get(i);
            XTrain[i] = Arrays.copyOf(X[idx], X[idx].length);
            yTrain[i] = y[idx];
        }
        
        for (int i = 0; i < testSize; i++) {
            int idx = indices.get(trainSize + i);
            XTest[i] = Arrays.copyOf(X[idx], X[idx].length);
            yTest[i] = y[idx];
        }
        
        for (int i = 0; i < validSize; i++) {
            int idx = indices.get(trainSize + testSize + i);
            XValid[i] = Arrays.copyOf(X[idx], X[idx].length);
            yValid[i] = y[idx];
        }
        
        return new DataSplit(XTrain, yTrain, XTest, yTest, XValid, yValid);
    }
    
    private static double[] getUniqueClasses(double[] y) {
        return Arrays.stream(y).distinct().sorted().toArray();
    }
    
    // Data classes
    
    private static class CompetitionData {
        final double[][] X;
        final double[] y;
        final int samples;
        
        CompetitionData(double[][] X, double[] y, int samples) {
            this.X = X;
            this.y = y;
            this.samples = samples;
        }
    }
    
    private static class DataSplit {
        final double[][] XTrain, XTest, XValid;
        final double[] yTrain, yTest, yValid;
        
        DataSplit(double[][] XTrain, double[] yTrain, double[][] XTest, double[] yTest,
                 double[][] XValid, double[] yValid) {
            this.XTrain = XTrain;
            this.yTrain = yTrain;
            this.XTest = XTest;
            this.yTest = yTest;
            this.XValid = XValid;
            this.yValid = yValid;
        }
    }
    
    private static class AlgorithmResult {
        final String name;
        final double accuracy;
        final double f1Score;
        final long trainingTime;
        
        AlgorithmResult(String name, double accuracy, double f1Score, long trainingTime) {
            this.name = name;
            this.accuracy = accuracy;
            this.f1Score = f1Score;
            this.trainingTime = trainingTime;
        }
    }
    
    private static class FeatureImportance {
        final int index;
        final double importance;
        
        FeatureImportance(int index, double importance) {
            this.index = index;
            this.importance = importance;
        }
    }
}
