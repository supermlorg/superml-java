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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Complete example demonstrating neural network model persistence workflow
 * using SuperML's integrated persistence architecture.
 * 
 * This example shows:
 * 1. Training neural network models (MLP, CNN, RNN)
 * 2. Saving trained models with metadata using ModelPersistence
 * 3. Loading saved models for inference
 * 4. Making predictions on new data
 * 5. Managing model collections with ModelManager
 * 
 * @author SuperML Java Team
 */
public class NeuralNetworkModelPersistenceExample {
    
    private static final Logger logger = LoggerFactory.getLogger(NeuralNetworkModelPersistenceExample.class);
    private static final Random random = new Random(42);
    
    public static void main(String[] args) {
        System.out.println("=== Neural Network Model Persistence Example ===\n");
        
        try {
            // 1. Train and save models
            System.out.println("Phase 1: Training and Persisting Models");
            System.out.println("=====================================");
            Map<String, String> savedModels = trainAndSaveModels();
            
            // 2. Load and test models
            System.out.println("\nPhase 2: Loading and Testing Models");
            System.out.println("===================================");
            testSavedModels(savedModels);
            
            // 3. Demonstrate model management
            System.out.println("\nPhase 3: Model Management");
            System.out.println("========================");
            demonstrateModelManagement();
            
            System.out.println("\n✅ Complete neural network persistence workflow demonstrated successfully!");
            
        } catch (Exception e) {
            logger.error("Example execution failed", e);
            e.printStackTrace();
        }
    }
    
    /**
     * Train neural network models and save them using ModelPersistence
     */
    private static Map<String, String> trainAndSaveModels() {
        Map<String, String> savedModels = new HashMap<>();
        
        // 1. Train and save MLP Classifier
        String mlpPath = trainAndSaveMLP();
        savedModels.put("mlp", mlpPath);
        
        // 2. Train and save CNN Classifier  
        String cnnPath = trainAndSaveCNN();
        savedModels.put("cnn", cnnPath);
        
        // 3. Train and save RNN Classifier
        String rnnPath = trainAndSaveRNN();
        savedModels.put("rnn", rnnPath);
        
        return savedModels;
    }
    
    private static String trainAndSaveMLP() {
        System.out.println("\n1. Training MLP Classifier");
        System.out.println("--------------------------");
        
        // Generate classification dataset
        double[][] X = generateClassificationData(800, 15);
        double[] y = generateBinaryLabels(800);
        
        // Create and configure MLP
        MLPClassifier mlp = new MLPClassifier()
            .setHiddenLayerSizes(32, 16)
            .setActivation("relu")
            .setLearningRate(0.01)
            .setMaxIter(100)
            .setBatchSize(32);
        
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
        
        String modelPath = "models/mlp_classifier.superml";
        ModelPersistence.save(mlp, modelPath, description, metadata);
        System.out.println("✓ MLP model saved to " + modelPath);
        
        return modelPath;
    }
    
    private static String trainAndSaveCNN() {
        System.out.println("\n2. Training CNN Classifier");
        System.out.println("--------------------------");
        
        // Generate image-like dataset (flattened 16x16 images)
        double[][] X = generateImageData(400, 16, 16);
        double[] y = generateBinaryLabels(400);
        
        // Create and configure CNN
        CNNClassifier cnn = new CNNClassifier()
            .setInputShape(16, 16, 1)
            .setLearningRate(0.01)
            .setMaxEpochs(50)
            .setBatchSize(32);
        
        // Train the model
        System.out.println("Training CNN with " + X.length + " samples, 16x16 images...");
        cnn.fit(X, y);
        
        // Evaluate performance
        double[] predictions = cnn.predict(X);
        double accuracy = calculateAccuracy(y, predictions);
        System.out.printf("Training accuracy: %.4f%n", accuracy);
        
        // Save model with metadata
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("dataset_size", X.length);
        metadata.put("image_shape", "16x16x1");
        metadata.put("training_accuracy", accuracy);
        metadata.put("input_shape", Arrays.toString(cnn.getInputShape()));
        
        String description = String.format("CNN Classifier trained on %d images with %.4f accuracy", 
                                         X.length, accuracy);
        
        String modelPath = "models/cnn_classifier.superml";
        ModelPersistence.save(cnn, modelPath, description, metadata);
        System.out.println("✓ CNN model saved to " + modelPath);
        
        return modelPath;
    }
    
    private static String trainAndSaveRNN() {
        System.out.println("\n3. Training RNN Classifier");
        System.out.println("--------------------------");
        
        // Generate time series dataset (sequences of length 20 with 5 features each)
        double[][] X = generateTimeSeriesData(600, 20, 5);
        double[] y = generateBinaryLabels(600);
        
        // Create and configure RNN
        RNNClassifier rnn = new RNNClassifier()
            .setHiddenSize(32)
            .setNumLayers(2)
            .setCellType("LSTM")
            .setLearningRate(0.01)
            .setMaxEpochs(75)
            .setBatchSize(32);
        
        // Train the model
        System.out.println("Training RNN with " + X.length + " sequences, length 20...");
        rnn.fit(X, y);
        
        // Evaluate performance
        double[] predictions = rnn.predict(X);
        double accuracy = calculateAccuracy(y, predictions);
        System.out.printf("Training accuracy: %.4f%n", accuracy);
        
        // Save model with metadata
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("dataset_size", X.length);
        metadata.put("sequence_length", 20);
        metadata.put("input_features", 5);
        metadata.put("training_accuracy", accuracy);
        metadata.put("hidden_size", rnn.getHiddenSize());
        metadata.put("cell_type", rnn.getCellType());
        
        String description = String.format("RNN Classifier trained on %d sequences with %.4f accuracy", 
                                         X.length, accuracy);
        
        String modelPath = "models/rnn_classifier.superml";
        ModelPersistence.save(rnn, modelPath, description, metadata);
        System.out.println("✓ RNN model saved to " + modelPath);
        
        return modelPath;
    }
    
    /**
     * Load saved models and test inference
     */
    private static void testSavedModels(Map<String, String> savedModels) {
        
        // Test MLP
        testMLP(savedModels.get("mlp"));
        
        // Test CNN
        testCNN(savedModels.get("cnn"));
        
        // Test RNN
        testRNN(savedModels.get("rnn"));
    }
    
    private static void testMLP(String modelPath) {
        System.out.println("\nTesting MLP Model:");
        System.out.println("-----------------");
        
        try {
            // Load model
            MLPClassifier loadedMlp = ModelPersistence.load(modelPath, MLPClassifier.class);
            System.out.println("✓ MLP model loaded successfully");
            
            // Generate test data
            double[] testSample = generateSample(15);
            System.out.println("Test sample features: " + testSample.length);
            
            // Make prediction
            double prediction = loadedMlp.predict(new double[][]{testSample})[0];
            System.out.printf("MLP prediction: %.4f%n", prediction);
            
            // Test batch prediction
            double[][] batchSamples = {
                generateSample(15),
                generateSample(15),
                generateSample(15)
            };
            double[] batchPredictions = loadedMlp.predict(batchSamples);
            System.out.println("Batch predictions: " + Arrays.toString(batchPredictions));
            
        } catch (Exception e) {
            System.err.println("❌ MLP test failed: " + e.getMessage());
        }
    }
    
    private static void testCNN(String modelPath) {
        System.out.println("\nTesting CNN Model:");
        System.out.println("-----------------");
        
        try {
            // Load model
            CNNClassifier loadedCnn = ModelPersistence.load(modelPath, CNNClassifier.class);
            System.out.println("✓ CNN model loaded successfully");
            
            // Generate test data (16x16 image)
            double[] testImage = generateImageSample(16, 16);
            System.out.println("Test image pixels: " + testImage.length);
            
            // Make prediction
            double prediction = loadedCnn.predict(new double[][]{testImage})[0];
            System.out.printf("CNN prediction: %.4f%n", prediction);
            
        } catch (Exception e) {
            System.err.println("❌ CNN test failed: " + e.getMessage());
        }
    }
    
    private static void testRNN(String modelPath) {
        System.out.println("\nTesting RNN Model:");
        System.out.println("-----------------");
        
        try {
            // Load model
            RNNClassifier loadedRnn = ModelPersistence.load(modelPath, RNNClassifier.class);
            System.out.println("✓ RNN model loaded successfully");
            
            // Generate test data (sequence of length 20 with 5 features)
            double[] testSequence = generateTimeSeriesSample(20, 5);
            System.out.println("Test sequence length: " + testSequence.length);
            
            // Make prediction
            double prediction = loadedRnn.predict(new double[][]{testSequence})[0];
            System.out.printf("RNN prediction: %.4f%n", prediction);
            
        } catch (Exception e) {
            System.err.println("❌ RNN test failed: " + e.getMessage());
        }
    }
    
    /**
     * Demonstrate model management capabilities
     */
    private static void demonstrateModelManagement() {
        // Create model manager
        ModelManager manager = new ModelManager("models");
        
        // List saved models
        System.out.println("\nSaved Models:");
        try {
            var modelFiles = manager.listModels();
            for (String modelFile : modelFiles) {
                System.out.println("📁 " + modelFile);
                
                // Show model metadata
                var metadata = ModelPersistence.getMetadata(modelFile);
                System.out.println("   Class: " + metadata.modelClass);
                System.out.println("   Saved: " + metadata.savedAt);
                System.out.println("   Description: " + metadata.description);
                System.out.println();
            }
        } catch (Exception e) {
            System.err.println("Model listing failed: " + e.getMessage());
        }
    }
    
    // Helper methods for data generation
    
    private static double[][] generateClassificationData(int samples, int features) {
        double[][] data = new double[samples][features];
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < features; j++) {
                data[i][j] = random.nextGaussian();
            }
        }
        return data;
    }
    
    private static double[][] generateImageData(int samples, int height, int width) {
        double[][] data = new double[samples][height * width];
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < height * width; j++) {
                data[i][j] = random.nextDouble();
            }
        }
        return data;
    }
    
    private static double[][] generateTimeSeriesData(int samples, int sequenceLength, int features) {
        double[][] data = new double[samples][sequenceLength * features];
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < sequenceLength * features; j++) {
                data[i][j] = Math.sin(j * 0.1) + random.nextGaussian() * 0.1;
            }
        }
        return data;
    }
    
    private static double[] generateBinaryLabels(int samples) {
        double[] labels = new double[samples];
        for (int i = 0; i < samples; i++) {
            labels[i] = random.nextDouble() > 0.5 ? 1.0 : 0.0;
        }
        return labels;
    }
    
    private static double[] generateSample(int features) {
        double[] sample = new double[features];
        for (int i = 0; i < features; i++) {
            sample[i] = random.nextGaussian();
        }
        return sample;
    }
    
    private static double[] generateImageSample(int height, int width) {
        double[] sample = new double[height * width];
        for (int i = 0; i < height * width; i++) {
            sample[i] = random.nextDouble();
        }
        return sample;
    }
    
    private static double[] generateTimeSeriesSample(int sequenceLength, int features) {
        double[] sample = new double[sequenceLength * features];
        for (int i = 0; i < sequenceLength * features; i++) {
            sample[i] = Math.sin(i * 0.1) + random.nextGaussian() * 0.1;
        }
        return sample;
    }
    
    private static double calculateAccuracy(double[] actual, double[] predicted) {
        int correct = 0;
        for (int i = 0; i < actual.length; i++) {
            if (Math.round(predicted[i]) == Math.round(actual[i])) {
                correct++;
            }
        }
        return (double) correct / actual.length;
    }
}
