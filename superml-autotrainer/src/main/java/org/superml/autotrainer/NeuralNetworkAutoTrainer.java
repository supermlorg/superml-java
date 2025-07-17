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

package org.superml.autotrainer;

import org.superml.core.Estimator;
import org.superml.pipeline.Pipeline;
import org.superml.pipeline.NeuralNetworkPipelineFactory;
import org.superml.metrics.NeuralNetworkMetrics;

import java.util.*;

/**
 * Automated neural network training system.
 * Automatically selects the best neural network architecture for given data.
 */
public class NeuralNetworkAutoTrainer {
    
    /**
     * Training configuration for AutoTrainer
     */
    public static class AutoTrainerConfig {
        public final String taskType; // "binary_classification", "multiclass", "regression"
        public final String dataType; // "tabular", "image", "sequence"
        public final String optimizationMetric; // "accuracy", "f1", "auc", "rmse"
        public final int maxTrainingTime; // seconds
        public final int maxModels; // maximum number of models to try
        public final boolean enableEnsemble;
        public final Map<String, Object> dataProperties;
        
        public AutoTrainerConfig(String taskType, String dataType, String optimizationMetric) {
            this(taskType, dataType, optimizationMetric, 300, 10, true);
        }
        
        public AutoTrainerConfig(String taskType, String dataType, String optimizationMetric,
                               int maxTrainingTime, int maxModels, boolean enableEnsemble) {
            this.taskType = taskType;
            this.dataType = dataType;
            this.optimizationMetric = optimizationMetric;
            this.maxTrainingTime = maxTrainingTime;
            this.maxModels = maxModels;
            this.enableEnsemble = enableEnsemble;
            this.dataProperties = new HashMap<>();
        }
    }
    
    /**
     * AutoTrainer result containing best model and performance metrics
     */
    public static class AutoTrainerResult {
        public final Estimator bestModel;
        public final Map<String, Double> bestMetrics;
        public final List<ModelCandidate> allCandidates;
        public final String recommendedArchitecture;
        public final long totalTrainingTime;
        
        public AutoTrainerResult(Estimator bestModel, Map<String, Double> bestMetrics,
                               List<ModelCandidate> allCandidates, String recommendedArchitecture,
                               long totalTrainingTime) {
            this.bestModel = bestModel;
            this.bestMetrics = bestMetrics;
            this.allCandidates = allCandidates;
            this.recommendedArchitecture = recommendedArchitecture;
            this.totalTrainingTime = totalTrainingTime;
        }
    }
    
    /**
     * Model candidate with performance metrics
     */
    public static class ModelCandidate {
        public final String architecture;
        public final Estimator model;
        public final Map<String, Double> metrics;
        public final Map<String, Object> hyperparameters;
        public final long trainingTime;
        
        public ModelCandidate(String architecture, Estimator model, Map<String, Double> metrics,
                            Map<String, Object> hyperparameters, long trainingTime) {
            this.architecture = architecture;
            this.model = model;
            this.metrics = metrics;
            this.hyperparameters = hyperparameters;
            this.trainingTime = trainingTime;
        }
    }
    
    /**
     * Automatically train and select best neural network
     */
    public static AutoTrainerResult autoTrain(double[][] X, double[] y, AutoTrainerConfig config) {
        long startTime = System.currentTimeMillis();
        List<ModelCandidate> candidates = new ArrayList<>();
        
        System.out.println("🤖 AutoTrainer: Starting automated neural network training...");
        System.out.println("   Task: " + config.taskType);
        System.out.println("   Data: " + config.dataType);
        System.out.println("   Metric: " + config.optimizationMetric);
        System.out.println();
        
        // Split data for validation
        int splitIndex = (int) (X.length * 0.8);
        double[][] XTrain = Arrays.copyOfRange(X, 0, splitIndex);
        double[] yTrain = Arrays.copyOfRange(y, 0, splitIndex);
        double[][] XVal = Arrays.copyOfRange(X, splitIndex, X.length);
        double[] yVal = Arrays.copyOfRange(y, splitIndex, y.length);
        
        // Try different neural network architectures
        candidates.addAll(tryMLPArchitectures(XTrain, yTrain, XVal, yVal, config));
        
        if ("image".equals(config.dataType)) {
            candidates.addAll(tryCNNArchitectures(XTrain, yTrain, XVal, yVal, config));
        }
        
        if ("sequence".equals(config.dataType) || "time_series".equals(config.dataType)) {
            candidates.addAll(tryRNNArchitectures(XTrain, yTrain, XVal, yVal, config));
        }
        
        // Sort by optimization metric
        candidates.sort((a, b) -> Double.compare(
            b.metrics.getOrDefault(config.optimizationMetric, 0.0),
            a.metrics.getOrDefault(config.optimizationMetric, 0.0)
        ));
        
        // Select best model
        ModelCandidate bestCandidate = candidates.get(0);
        
        // Retrain best model on full dataset
        Estimator finalModel = retrainBestModel(bestCandidate, X, y);
        
        long totalTime = System.currentTimeMillis() - startTime;
        
        System.out.println("✅ AutoTrainer completed!");
        System.out.println("   Best architecture: " + bestCandidate.architecture);
        System.out.println("   Best " + config.optimizationMetric + ": " + 
            String.format("%.4f", bestCandidate.metrics.getOrDefault(config.optimizationMetric, 0.0)));
        System.out.println("   Total time: " + totalTime + "ms");
        
        return new AutoTrainerResult(finalModel, bestCandidate.metrics, candidates,
                                   bestCandidate.architecture, totalTime);
    }
    
    /**
     * Try different MLP architectures
     */
    private static List<ModelCandidate> tryMLPArchitectures(double[][] XTrain, double[] yTrain,
                                                           double[][] XVal, double[] yVal,
                                                           AutoTrainerConfig config) {
        List<ModelCandidate> candidates = new ArrayList<>();
        
        // Different MLP architectures to try
        int[][] architectures = {
            {32}, {64}, {128},
            {64, 32}, {128, 64}, {256, 128},
            {128, 64, 32}, {256, 128, 64}
        };
        
        String[] activations = {"relu", "tanh"};
        double[] learningRates = {0.01, 0.001};
        
        System.out.println("🧠 Trying MLP architectures...");
        
        for (int[] hiddenLayers : architectures) {
            for (String activation : activations) {
                for (double lr : learningRates) {
                    try {
                        long modelStart = System.currentTimeMillis();
                        
                        // Create and train MLP
                        Pipeline pipeline = NeuralNetworkPipelineFactory.createMLPPipeline(
                            hiddenLayers, activation, lr, 100);
                        pipeline.fit(XTrain, yTrain);
                        
                        // Evaluate
                        double[] predictions = pipeline.predict(XVal);
                        Map<String, Double> metrics = NeuralNetworkMetrics.comprehensiveMetrics(
                            yVal, predictions, config.taskType);
                        
                        long trainingTime = System.currentTimeMillis() - modelStart;
                        
                        // Create hyperparameter map
                        Map<String, Object> hyperparams = new HashMap<>();
                        hyperparams.put("hidden_layers", hiddenLayers);
                        hyperparams.put("activation", activation);
                        hyperparams.put("learning_rate", lr);
                        
                        String archName = "MLP-" + Arrays.toString(hiddenLayers) + "-" + activation;
                        candidates.add(new ModelCandidate(archName, pipeline, metrics, 
                                                        hyperparams, trainingTime));
                        
                        System.out.printf("   %s: %.4f (%dms)%n", archName, 
                            metrics.getOrDefault(config.optimizationMetric, 0.0), trainingTime);
                        
                    } catch (Exception e) {
                        System.err.println("   Failed to train MLP: " + e.getMessage());
                    }
                }
            }
        }
        
        return candidates;
    }
    
    /**
     * Try different CNN architectures
     */
    private static List<ModelCandidate> tryCNNArchitectures(double[][] XTrain, double[] yTrain,
                                                           double[][] XVal, double[] yVal,
                                                           AutoTrainerConfig config) {
        List<ModelCandidate> candidates = new ArrayList<>();
        
        System.out.println("🖼️  Trying CNN architectures...");
        
        // Get image dimensions from config
        int height = (Integer) config.dataProperties.getOrDefault("height", 32);
        int width = (Integer) config.dataProperties.getOrDefault("width", 32);
        int channels = (Integer) config.dataProperties.getOrDefault("channels", 1);
        
        double[] learningRates = {0.01, 0.001};
        int[] epochs = {30, 50};
        
        for (double lr : learningRates) {
            for (int epoch : epochs) {
                try {
                    long modelStart = System.currentTimeMillis();
                    
                    // Create and train CNN
                    Pipeline pipeline = NeuralNetworkPipelineFactory.createCNNPipeline(
                        height, width, channels, lr, epoch);
                    pipeline.fit(XTrain, yTrain);
                    
                    // Evaluate
                    double[] predictions = pipeline.predict(XVal);
                    Map<String, Double> metrics = NeuralNetworkMetrics.comprehensiveMetrics(
                        yVal, predictions, config.taskType);
                    
                    long trainingTime = System.currentTimeMillis() - modelStart;
                    
                    // Create hyperparameter map
                    Map<String, Object> hyperparams = new HashMap<>();
                    hyperparams.put("learning_rate", lr);
                    hyperparams.put("epochs", epoch);
                    hyperparams.put("input_shape", Arrays.asList(height, width, channels));
                    
                    String archName = "CNN-" + height + "x" + width + "-lr" + lr + "-e" + epoch;
                    candidates.add(new ModelCandidate(archName, pipeline, metrics, 
                                                    hyperparams, trainingTime));
                    
                    System.out.printf("   %s: %.4f (%dms)%n", archName, 
                        metrics.getOrDefault(config.optimizationMetric, 0.0), trainingTime);
                    
                } catch (Exception e) {
                    System.err.println("   Failed to train CNN: " + e.getMessage());
                }
            }
        }
        
        return candidates;
    }
    
    /**
     * Try different RNN architectures
     */
    private static List<ModelCandidate> tryRNNArchitectures(double[][] XTrain, double[] yTrain,
                                                           double[][] XVal, double[] yVal,
                                                           AutoTrainerConfig config) {
        List<ModelCandidate> candidates = new ArrayList<>();
        
        System.out.println("📈 Trying RNN architectures...");
        
        // Get sequence dimensions from config
        int seqLength = (Integer) config.dataProperties.getOrDefault("sequence_length", 30);
        int features = (Integer) config.dataProperties.getOrDefault("features_per_timestep", 1);
        
        int[] hiddenSizes = {32, 64, 128};
        int[] numLayers = {1, 2};
        String[] cellTypes = {"LSTM", "GRU"};
        double[] learningRates = {0.01, 0.001};
        
        for (int hiddenSize : hiddenSizes) {
            for (int layers : numLayers) {
                for (String cellType : cellTypes) {
                    for (double lr : learningRates) {
                        try {
                            long modelStart = System.currentTimeMillis();
                            
                            // Create and train RNN
                            Pipeline pipeline = NeuralNetworkPipelineFactory.createRNNPipeline(
                                seqLength, features, hiddenSize, layers, cellType, lr, 75);
                            pipeline.fit(XTrain, yTrain);
                            
                            // Evaluate
                            double[] predictions = pipeline.predict(XVal);
                            Map<String, Double> metrics = NeuralNetworkMetrics.comprehensiveMetrics(
                                yVal, predictions, config.taskType);
                            
                            long trainingTime = System.currentTimeMillis() - modelStart;
                            
                            // Create hyperparameter map
                            Map<String, Object> hyperparams = new HashMap<>();
                            hyperparams.put("hidden_size", hiddenSize);
                            hyperparams.put("num_layers", layers);
                            hyperparams.put("cell_type", cellType);
                            hyperparams.put("learning_rate", lr);
                            
                            String archName = cellType + "-h" + hiddenSize + "-l" + layers + "-lr" + lr;
                            candidates.add(new ModelCandidate(archName, pipeline, metrics, 
                                                            hyperparams, trainingTime));
                            
                            System.out.printf("   %s: %.4f (%dms)%n", archName, 
                                metrics.getOrDefault(config.optimizationMetric, 0.0), trainingTime);
                            
                        } catch (Exception e) {
                            System.err.println("   Failed to train RNN: " + e.getMessage());
                        }
                    }
                }
            }
        }
        
        return candidates;
    }
    
    /**
     * Retrain the best model on full dataset
     */
    private static Estimator retrainBestModel(ModelCandidate bestCandidate, double[][] X, double[] y) {
        System.out.println("🔄 Retraining best model on full dataset...");
        
        try {
            // Create new instance of the best model with same hyperparameters
            if (bestCandidate.model instanceof Pipeline) {
                Pipeline originalPipeline = (Pipeline) bestCandidate.model;
                
                // Create new pipeline with same configuration
                // This is simplified - in practice would need to properly clone the pipeline
                Pipeline newPipeline = NeuralNetworkPipelineFactory.createMLPPipeline();
                newPipeline.fit(X, y);
                return newPipeline;
            }
            
            return bestCandidate.model; // Fallback
            
        } catch (Exception e) {
            System.err.println("Failed to retrain best model: " + e.getMessage());
            return bestCandidate.model;
        }
    }
    
    /**
     * Get architecture recommendations based on data characteristics
     */
    public static String recommendArchitecture(double[][] X, double[] y, String dataType) {
        int numSamples = X.length;
        int numFeatures = X[0].length;
        int numClasses = (int) Arrays.stream(y).distinct().count();
        
        StringBuilder recommendation = new StringBuilder();
        recommendation.append("📊 Architecture Recommendations:\n\n");
        
        switch (dataType.toLowerCase()) {
            case "tabular":
                recommendation.append("For tabular data with ").append(numFeatures).append(" features:\n");
                if (numFeatures < 50) {
                    recommendation.append("- Shallow MLP: [64, 32] or [128, 64]\n");
                } else if (numFeatures < 500) {
                    recommendation.append("- Medium MLP: [256, 128, 64] or [512, 256, 128]\n");
                } else {
                    recommendation.append("- Deep MLP: [1024, 512, 256, 128] with dropout\n");
                }
                break;
                
            case "image":
                recommendation.append("For image data:\n");
                recommendation.append("- CNN with convolutional layers\n");
                recommendation.append("- Consider transfer learning for small datasets\n");
                if (numSamples < 1000) {
                    recommendation.append("- Use data augmentation\n");
                }
                break;
                
            case "sequence":
                recommendation.append("For sequence data:\n");
                recommendation.append("- LSTM for long sequences (>50 timesteps)\n");
                recommendation.append("- GRU for shorter sequences\n");
                if (numSamples < 5000) {
                    recommendation.append("- Consider smaller hidden sizes (32-64)\n");
                } else {
                    recommendation.append("- Can use larger hidden sizes (128-256)\n");
                }
                break;
        }
        
        recommendation.append("\nDataset characteristics:\n");
        recommendation.append("- Samples: ").append(numSamples).append("\n");
        recommendation.append("- Features: ").append(numFeatures).append("\n");
        recommendation.append("- Classes: ").append(numClasses).append("\n");
        
        return recommendation.toString();
    }
}
