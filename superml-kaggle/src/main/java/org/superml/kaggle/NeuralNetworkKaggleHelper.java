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

package org.superml.kaggle;

import org.superml.neural.MLPClassifier;
import org.superml.neural.CNNClassifier;
import org.superml.neural.RNNClassifier;
import org.superml.pipeline.Pipeline;
import org.superml.pipeline.NeuralNetworkPipelineFactory;
import org.superml.preprocessing.NeuralNetworkPreprocessor;
import org.superml.model_selection.GridSearchCV;
import org.superml.metrics.NeuralNetworkMetrics;

import java.util.*;
import java.io.PrintWriter;
import java.io.FileWriter;
import java.io.IOException;

/**
 * Kaggle competition utilities for neural networks.
 * Provides tools for neural network model selection, ensemble creation, and submission generation.
 */
public class NeuralNetworkKaggleHelper {
    
    /**
     * Competition configuration
     */
    public static class CompetitionConfig {
        public final String competitionName;
        public final String dataType; // "tabular", "image", "sequence"
        public final Map<String, Object> dataProperties;
        public final String taskType; // "binary_classification", "multiclass", "regression"
        public final String metric; // "accuracy", "auc", "f1", "rmse"
        
        public CompetitionConfig(String competitionName, String dataType, String taskType, String metric) {
            this.competitionName = competitionName;
            this.dataType = dataType;
            this.taskType = taskType;
            this.metric = metric;
            this.dataProperties = new HashMap<>();
        }
        
        public CompetitionConfig setDataProperty(String key, Object value) {
            dataProperties.put(key, value);
            return this;
        }
    }
    
    /**
     * Model results for competition analysis
     */
    public static class ModelResult {
        public final String modelName;
        public final Object model;
        public final Map<String, Double> metrics;
        public final double[] predictions;
        public final long trainingTime;
        
        public ModelResult(String modelName, Object model, Map<String, Double> metrics, 
                          double[] predictions, long trainingTime) {
            this.modelName = modelName;
            this.model = model;
            this.metrics = metrics;
            this.predictions = predictions;
            this.trainingTime = trainingTime;
        }
    }
    
    /**
     * Train and evaluate multiple neural network models for competition
     */
    public static List<ModelResult> trainCompetitionModels(double[][] XTrain, double[] yTrain,
                                                          double[][] XVal, double[] yVal,
                                                          CompetitionConfig config) {
        List<ModelResult> results = new ArrayList<>();
        
        // Train MLP
        results.add(trainMLPModel(XTrain, yTrain, XVal, yVal, config));
        
        // Train CNN (if image data)
        if ("image".equals(config.dataType)) {
            results.add(trainCNNModel(XTrain, yTrain, XVal, yVal, config));
        }
        
        // Train RNN (if sequence data)
        if ("sequence".equals(config.dataType) || "time_series".equals(config.dataType)) {
            results.add(trainRNNModel(XTrain, yTrain, XVal, yVal, config));
        }
        
        // Sort by primary metric
        results.sort((a, b) -> Double.compare(
            b.metrics.getOrDefault(config.metric, 0.0),
            a.metrics.getOrDefault(config.metric, 0.0)
        ));
        
        return results;
    }
    
    /**
     * Train MLP model for competition
     */
    private static ModelResult trainMLPModel(double[][] XTrain, double[] yTrain,
                                           double[][] XVal, double[] yVal,
                                           CompetitionConfig config) {
        long startTime = System.currentTimeMillis();
        
        // Create MLP pipeline
        Pipeline pipeline = NeuralNetworkPipelineFactory.createMLPPipeline();
        
        // Train model
        pipeline.fit(XTrain, yTrain);
        
        // Make predictions
        double[] predictions = pipeline.predict(XVal);
        
        // Calculate metrics
        Map<String, Double> metrics = NeuralNetworkMetrics.comprehensiveMetrics(
            yVal, predictions, config.taskType);
        
        long trainingTime = System.currentTimeMillis() - startTime;
        
        return new ModelResult("MLP", pipeline, metrics, predictions, trainingTime);
    }
    
    /**
     * Train CNN model for competition
     */
    private static ModelResult trainCNNModel(double[][] XTrain, double[] yTrain,
                                           double[][] XVal, double[] yVal,
                                           CompetitionConfig config) {
        long startTime = System.currentTimeMillis();
        
        // Get image dimensions
        int height = (Integer) config.dataProperties.getOrDefault("height", 32);
        int width = (Integer) config.dataProperties.getOrDefault("width", 32);
        int channels = (Integer) config.dataProperties.getOrDefault("channels", 1);
        
        // Create CNN pipeline
        Pipeline pipeline = NeuralNetworkPipelineFactory.createCNNPipeline(height, width, channels);
        
        // Train model
        pipeline.fit(XTrain, yTrain);
        
        // Make predictions
        double[] predictions = pipeline.predict(XVal);
        
        // Calculate metrics
        Map<String, Double> metrics = NeuralNetworkMetrics.comprehensiveMetrics(
            yVal, predictions, config.taskType);
        
        long trainingTime = System.currentTimeMillis() - startTime;
        
        return new ModelResult("CNN", pipeline, metrics, predictions, trainingTime);
    }
    
    /**
     * Train RNN model for competition
     */
    private static ModelResult trainRNNModel(double[][] XTrain, double[] yTrain,
                                           double[][] XVal, double[] yVal,
                                           CompetitionConfig config) {
        long startTime = System.currentTimeMillis();
        
        // Get sequence dimensions
        int seqLength = (Integer) config.dataProperties.getOrDefault("sequence_length", 30);
        int features = (Integer) config.dataProperties.getOrDefault("features_per_timestep", 1);
        
        // Create RNN pipeline
        Pipeline pipeline = NeuralNetworkPipelineFactory.createRNNPipeline(seqLength, features);
        
        // Train model
        pipeline.fit(XTrain, yTrain);
        
        // Make predictions
        double[] predictions = pipeline.predict(XVal);
        
        // Calculate metrics
        Map<String, Double> metrics = NeuralNetworkMetrics.comprehensiveMetrics(
            yVal, predictions, config.taskType);
        
        long trainingTime = System.currentTimeMillis() - startTime;
        
        return new ModelResult("RNN", pipeline, metrics, predictions, trainingTime);
    }
    
    /**
     * Create ensemble predictions from multiple models
     */
    public static double[] createEnsemble(List<ModelResult> models, String method) {
        if (models.isEmpty()) {
            throw new IllegalArgumentException("No models provided for ensemble");
        }
        
        int numPredictions = models.get(0).predictions.length;
        double[] ensemblePredictions = new double[numPredictions];
        
        switch (method.toLowerCase()) {
            case "average":
                // Simple averaging
                for (int i = 0; i < numPredictions; i++) {
                    double sum = 0.0;
                    for (ModelResult model : models) {
                        sum += model.predictions[i];
                    }
                    ensemblePredictions[i] = sum / models.size();
                }
                break;
                
            case "weighted":
                // Weight by model performance
                double totalWeight = 0.0;
                for (ModelResult model : models) {
                    totalWeight += model.metrics.getOrDefault("accuracy", 0.0);
                }
                
                for (int i = 0; i < numPredictions; i++) {
                    double weightedSum = 0.0;
                    for (ModelResult model : models) {
                        double weight = model.metrics.getOrDefault("accuracy", 0.0) / totalWeight;
                        weightedSum += weight * model.predictions[i];
                    }
                    ensemblePredictions[i] = weightedSum;
                }
                break;
                
            default:
                throw new IllegalArgumentException("Unknown ensemble method: " + method);
        }
        
        return ensemblePredictions;
    }
    
    /**
     * Generate Kaggle submission file
     */
    public static void generateSubmission(double[] predictions, String filename) throws IOException {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            writer.println("Id,Prediction");
            for (int i = 0; i < predictions.length; i++) {
                writer.printf("%d,%.6f%n", i + 1, predictions[i]);
            }
        }
    }
    
    /**
     * Generate detailed competition report
     */
    public static void generateReport(List<ModelResult> results, CompetitionConfig config, 
                                    String filename) throws IOException {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            writer.println("=".repeat(60));
            writer.println("Neural Network Competition Report");
            writer.println("=".repeat(60));
            writer.println("Competition: " + config.competitionName);
            writer.println("Data Type: " + config.dataType);
            writer.println("Task Type: " + config.taskType);
            writer.println("Primary Metric: " + config.metric);
            writer.println();
            
            writer.println("Model Performance Comparison:");
            writer.println("-".repeat(60));
            writer.printf("%-15s %-12s %-12s %-12s %-10s%n", 
                "Model", "Accuracy", "Precision", "Recall", "Time(ms)");
            writer.println("-".repeat(60));
            
            for (ModelResult result : results) {
                writer.printf("%-15s %-12.4f %-12.4f %-12.4f %-10d%n",
                    result.modelName,
                    result.metrics.getOrDefault("accuracy", 0.0),
                    result.metrics.getOrDefault("precision", 0.0),
                    result.metrics.getOrDefault("recall", 0.0),
                    result.trainingTime);
            }
            
            writer.println();
            writer.println("Best Model: " + results.get(0).modelName);
            writer.println("Best " + config.metric + ": " + 
                String.format("%.4f", results.get(0).metrics.getOrDefault(config.metric, 0.0)));
            
            writer.println();
            writer.println("Detailed Metrics:");
            writer.println("-".repeat(30));
            for (ModelResult result : results) {
                writer.println(result.modelName + ":");
                for (Map.Entry<String, Double> entry : result.metrics.entrySet()) {
                    writer.printf("  %s: %.4f%n", entry.getKey(), entry.getValue());
                }
                writer.println();
            }
        }
    }
    
    /**
     * Hyperparameter tuning for neural networks
     */
    public static ModelResult hyperparameterTuning(double[][] XTrain, double[] yTrain,
                                                  double[][] XVal, double[] yVal,
                                                  String modelType, CompetitionConfig config) {
        // This would integrate with GridSearchCV for hyperparameter optimization
        // For now, return the basic model
        
        switch (modelType.toLowerCase()) {
            case "mlp":
                return trainMLPModel(XTrain, yTrain, XVal, yVal, config);
            case "cnn":
                return trainCNNModel(XTrain, yTrain, XVal, yVal, config);
            case "rnn":
                return trainRNNModel(XTrain, yTrain, XVal, yVal, config);
            default:
                throw new IllegalArgumentException("Unknown model type: " + modelType);
        }
    }
    
    /**
     * Cross-validation for neural networks
     */
    public static Map<String, Double> crossValidate(double[][] X, double[] y, String modelType,
                                                   CompetitionConfig config, int cv) {
        // Implement k-fold cross-validation for neural networks
        Map<String, Double> cvScores = new HashMap<>();
        
        int foldSize = X.length / cv;
        double[] scores = new double[cv];
        
        for (int fold = 0; fold < cv; fold++) {
            int start = fold * foldSize;
            int end = (fold == cv - 1) ? X.length : start + foldSize;
            
            // Create train/val splits
            List<double[]> trainX = new ArrayList<>();
            List<Double> trainY = new ArrayList<>();
            List<double[]> valX = new ArrayList<>();
            List<Double> valY = new ArrayList<>();
            
            for (int i = 0; i < X.length; i++) {
                if (i >= start && i < end) {
                    valX.add(X[i]);
                    valY.add(y[i]);
                } else {
                    trainX.add(X[i]);
                    trainY.add(y[i]);
                }
            }
            
            // Convert to arrays
            double[][] XTrain = trainX.toArray(new double[0][]);
            double[] yTrain = trainY.stream().mapToDouble(Double::doubleValue).toArray();
            double[][] XVal = valX.toArray(new double[0][]);
            double[] yVal = valY.stream().mapToDouble(Double::doubleValue).toArray();
            
            // Train and evaluate
            ModelResult result = hyperparameterTuning(XTrain, yTrain, XVal, yVal, modelType, config);
            scores[fold] = result.metrics.getOrDefault(config.metric, 0.0);
        }
        
        // Calculate statistics
        double mean = Arrays.stream(scores).average().orElse(0.0);
        double std = Math.sqrt(Arrays.stream(scores)
            .map(x -> Math.pow(x - mean, 2))
            .average().orElse(0.0));
        
        cvScores.put("mean", mean);
        cvScores.put("std", std);
        cvScores.put("min", Arrays.stream(scores).min().orElse(0.0));
        cvScores.put("max", Arrays.stream(scores).max().orElse(0.0));
        
        return cvScores;
    }
}
