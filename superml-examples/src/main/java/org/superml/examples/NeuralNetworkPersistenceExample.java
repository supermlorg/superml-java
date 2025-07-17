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

import org.superml.neural.MLPClassifier;
import org.superml.neural.CNNClassifier;
import org.superml.neural.RNNClassifier;
import org.superml.persistence.ModelPersistence;
import org.superml.persistence.ModelManager;
import org.superml.inference.InferenceEngine;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Comprehensive example demonstrating neural network model persistence and inference
 * using SuperML's integrated persistence and inference architecture.
 * 
 * This example shows:
 * 1. Training neural network models (MLP, CNN, RNN)
 * 2. Saving trained models with metadata using ModelPersistence
 * 3. Loading and caching models using InferenceEngine
 * 4. Making predictions on new data
 * 5. Managing model collections with ModelManager
 * 
 * @author SuperML Java Team
 */
public class NeuralNetworkPersistenceExample {
    
    private static final Logger logger = LoggerFactory.getLogger(NeuralNetworkPersistenceExample.class);
    private static final Random random = new Random(42);
    
    public static void main(String[] args) {
        System.out.println("=== Neural Network Persistence & Inference Example ===\n");
        
        try {
            // 1. Train and save models
            System.out.println("Phase 1: Training and Persisting Models");
            System.out.println("=====================================");
            trainAndSaveModels();
            
            // 2. Load and infer with models
            System.out.println("\nPhase 2: Loading and Inference");
            System.out.println("=============================");
            loadAndInferModels();
            
            // 3. Demonstrate model management
            System.out.println("\nPhase 3: Model Management");
            System.out.println("========================");
            demonstrateModelManagement();
            
        } catch (Exception e) {
            logger.error("Example execution failed", e);
            e.printStackTrace();
        }
    }
    
    /**
     * Train neural network models and save them using ModelPersistence
     */
    private static void trainAndSaveModels() {
        // Generate datasets for different model types
        DatasetGenerator generator = new DatasetGenerator();
        
        // 1. Train and save MLP Classifier
        trainAndSaveMLP(generator);
        
        // 2. Train and save CNN Classifier  
        trainAndSaveCNN(generator);
        
        // 3. Train and save RNN Classifier
        trainAndSaveRNN(generator);
    }
    
    private static void trainAndSaveMLP(DatasetGenerator generator) {
        System.out.println("\n1. Training MLP Classifier");
        System.out.println("--------------------------");
        
        // Generate classification dataset
        double[][] X = generator.generateClassificationData(1000, 20);
        double[] y = generator.generateBinaryLabels(1000);
        
        // Create and configure MLP
        MLPClassifier mlp = new MLPClassifier()
            .setHiddenLayerSizes(64, 32, 16)
            .setActivation("relu")
            .setLearningRate(0.001)
            .setMaxIter(200)
            .setBatchSize(32)
            .setEarlyStopping(true)
            .setValidationFraction(0.2);
        
        // Train the model
        System.out.println("Training MLP with " + X.length + " samples, " + X[0].length + " features...");
        mlp.fit(X, y);
        
        // Evaluate performance
        double[] predictions = mlp.predict(X);
        double accuracy = calculateAccuracy(y, predictions);
        System.out.printf("Training accuracy: %.4f%n", accuracy);
        
        // Save model with metadata
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("dataset_size", X.length);
        metadata.put("features", X[0].length);
        metadata.put("training_accuracy", accuracy);
        metadata.put("architecture", Arrays.toString(mlp.getHiddenLayerSizes()));
        metadata.put("activation", "relu");
        
        String description = String.format("MLP Classifier trained on %d samples with %.4f accuracy", 
                                         X.length, accuracy);
        
        ModelPersistence.save(mlp, "models/mlp_classifier.superml", description, metadata);
        System.out.println("✓ MLP model saved to models/mlp_classifier.superml");
    }
    
    private static void trainAndSaveCNN(DatasetGenerator generator) {
        System.out.println("\n2. Training CNN Classifier");
        System.out.println("--------------------------");
        
        // Generate image-like dataset (flattened)
        double[][] X = generator.generateImageData(500, 28, 28);
        double[] y = generator.generateBinaryLabels(500);
        
        // Create and configure CNN
        CNNClassifier cnn = new CNNClassifier()
            .setInputShape(28, 28, 1)
            .setLearningRate(0.001)
            .setMaxEpochs(100)
            .setBatchSize(32);
        
        // Train the model
        System.out.println("Training CNN with " + X.length + " samples, 28x28 images...");
        cnn.fit(X, y);
        
        // Evaluate performance
        double[] predictions = cnn.predict(X);
        double accuracy = calculateAccuracy(y, predictions);
        System.out.printf("Training accuracy: %.4f%n", accuracy);
        
        // Save model with metadata
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("dataset_size", X.length);
        metadata.put("image_shape", "28x28x1");
        metadata.put("training_accuracy", accuracy);
        metadata.put("conv_layers", "32x3, 64x3");
        metadata.put("dense_units", "128");
        metadata.put("input_shape", Arrays.toString(cnn.getInputShape()));
        
        String description = String.format("CNN Classifier trained on %d images with %.4f accuracy", 
                                         X.length, accuracy);
        
        ModelPersistence.save(cnn, "models/cnn_classifier.superml", description, metadata);
        System.out.println("✓ CNN model saved to models/cnn_classifier.superml");
    }
    
    private static void trainAndSaveRNN(DatasetGenerator generator) {
        System.out.println("\n3. Training RNN Classifier");
        System.out.println("--------------------------");
        
        // Generate time series dataset
        double[][] X = generator.generateTimeSeriesData(800, 50, 10);
        double[] y = generator.generateBinaryLabels(800);
        
        // Create and configure RNN
        RNNClassifier rnn = new RNNClassifier()
            .setHiddenSize(64)
            .setNumLayers(2)
            .setCellType("LSTM")
            .setLearningRate(0.001)
            .setMaxEpochs(150)
            .setBatchSize(32);
        
        // Train the model
        System.out.println("Training RNN with " + X.length + " sequences, length 50...");
        rnn.fit(X, y);
        
        // Evaluate performance
        double[] predictions = rnn.predict(X);
        double accuracy = calculateAccuracy(y, predictions);
        System.out.printf("Training accuracy: %.4f%n", accuracy);
        
        // Save model with metadata
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("dataset_size", X.length);
        metadata.put("sequence_length", 50);
        metadata.put("input_size", 10);
        metadata.put("training_accuracy", accuracy);
        metadata.put("hidden_size", rnn.getHiddenSize());
        metadata.put("cell_type", rnn.getCellType());
        
        String description = String.format("RNN Classifier trained on %d sequences with %.4f accuracy", 
                                         X.length, accuracy);
        
        ModelPersistence.save(rnn, "models/rnn_classifier.superml", description, metadata);
        System.out.println("✓ RNN model saved to models/rnn_classifier.superml");
    }
    
    /**
     * Load saved models and perform inference using InferenceEngine
     */
    private static void loadAndInferModels() {
        // Create inference engine
        InferenceEngine engine = new InferenceEngine();
        
        try {
            // Load models into inference engine
            System.out.println("\nLoading models into inference engine...");
            
            var mlpInfo = engine.loadModel("mlp_model", "models/mlp_classifier.superml");
            System.out.println("✓ Loaded: " + mlpInfo);
            
            var cnnInfo = engine.loadModel("cnn_model", "models/cnn_classifier.superml");
            System.out.println("✓ Loaded: " + cnnInfo);
            
            var rnnInfo = engine.loadModel("rnn_model", "models/rnn_classifier.superml");
            System.out.println("✓ Loaded: " + rnnInfo);
            
            // Generate test data and make predictions
            DatasetGenerator generator = new DatasetGenerator();
            
            // Test MLP
            System.out.println("\nTesting MLP predictions:");
            double[] mlpFeatures = generator.generateSample(20);
            double mlpPrediction = engine.predict("mlp_model", mlpFeatures);
            System.out.printf("MLP prediction: %.4f%n", mlpPrediction);
            
            // Test CNN
            System.out.println("\nTesting CNN predictions:");
            double[] cnnFeatures = generator.generateImageSample(28, 28);
            double cnnPrediction = engine.predict("cnn_model", cnnFeatures);
            System.out.printf("CNN prediction: %.4f%n", cnnPrediction);
            
            // Test RNN
            System.out.println("\nTesting RNN predictions:");
            double[] rnnFeatures = generator.generateTimeSeriesSample(50, 10);
            double rnnPrediction = engine.predict("rnn_model", rnnFeatures);
            System.out.printf("RNN prediction: %.4f%n", rnnPrediction);
            
            // Batch predictions
            System.out.println("\nTesting batch predictions:");
            double[][] batchFeatures = {
                generator.generateSample(20),
                generator.generateSample(20),
                generator.generateSample(20)
            };
            double[] batchPredictions = engine.predict("mlp_model", batchFeatures);
            System.out.println("Batch predictions: " + Arrays.toString(batchPredictions));
            
            // Display inference metrics
            System.out.println("\nInference Engine Status:");
            System.out.println("Loaded models: " + engine.getLoadedModels());
            
        } catch (Exception e) {
            logger.error("Inference failed", e);
            e.printStackTrace();
        }
    }
    
    /**
     * Demonstrate model management capabilities
     */
    private static void demonstrateModelManagement() {
        // Create model manager
        ModelManager manager = new ModelManager("models");
        
        // List saved models
        System.out.println("\nListing saved models:");
        try {
            var modelFiles = manager.listModels();
            for (String modelFile : modelFiles) {
                System.out.println("- " + modelFile);
                
                // Show model metadata
                var metadata = ModelPersistence.getMetadata(modelFile);
                System.out.println("  Class: " + metadata.modelClass);
                System.out.println("  Saved: " + metadata.savedAt);
                System.out.println("  Description: " + metadata.description);
                System.out.println();
            }
        } catch (Exception e) {
            logger.error("Model listing failed", e);
        }
        
        System.out.println("Model management demonstration completed!");
    }
    
    /**
     * Calculate classification accuracy
     */
    private static double calculateAccuracy(double[] actual, double[] predicted) {
        int correct = 0;
        for (int i = 0; i < actual.length; i++) {
            if (Math.round(predicted[i]) == Math.round(actual[i])) {
                correct++;
            }
        }
        return (double) correct / actual.length;
    }
    
    /**
     * Helper class for generating synthetic datasets
     */
    private static class DatasetGenerator {
        
        public double[][] generateClassificationData(int samples, int features) {
            double[][] data = new double[samples][features];
            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < features; j++) {
                    data[i][j] = random.nextGaussian();
                }
            }
            return data;
        }
        
        public double[][] generateImageData(int samples, int height, int width) {
            double[][] data = new double[samples][height * width];
            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < height * width; j++) {
                    data[i][j] = random.nextDouble();
                }
            }
            return data;
        }
        
        public double[][] generateTimeSeriesData(int samples, int sequenceLength, int features) {
            double[][] data = new double[samples][sequenceLength * features];
            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < sequenceLength * features; j++) {
                    data[i][j] = Math.sin(j * 0.1) + random.nextGaussian() * 0.1;
                }
            }
            return data;
        }
        
        public double[] generateBinaryLabels(int samples) {
            double[] labels = new double[samples];
            for (int i = 0; i < samples; i++) {
                labels[i] = random.nextDouble() > 0.5 ? 1.0 : 0.0;
            }
            return labels;
        }
        
        public double[] generateSample(int features) {
            double[] sample = new double[features];
            for (int i = 0; i < features; i++) {
                sample[i] = random.nextGaussian();
            }
            return sample;
        }
        
        public double[] generateImageSample(int height, int width) {
            double[] sample = new double[height * width];
            for (int i = 0; i < height * width; i++) {
                sample[i] = random.nextDouble();
            }
            return sample;
        }
        
        public double[] generateTimeSeriesSample(int sequenceLength, int features) {
            double[] sample = new double[sequenceLength * features];
            for (int i = 0; i < sequenceLength * features; i++) {
                sample[i] = Math.sin(i * 0.1) + random.nextGaussian() * 0.1;
            }
            return sample;
        }
    }
}
