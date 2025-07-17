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

import org.superml.linear_model.LogisticRegression;
import org.superml.neural.MLPClassifier;
import org.superml.neural.CNNClassifier;
import org.superml.neural.RNNClassifier;
import org.superml.persistence.ModelPersistence;
import org.superml.preprocessing.NeuralNetworkPreprocessor;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Advanced Kaggle-style competition example showcasing neural networks
 * Demonstrates ML workflow with MLP, CNN, RNN models for competition submissions.
 * 
 * Features:
 * - Multiple model types (Logistic, MLP, CNN, RNN)
 * - Specialized neural network preprocessing
 * - Model persistence and loading
 * - Performance comparison with confusion matrices
 * - Ensemble predictions
 * - Competition-ready submission format
 */
public class SimpleKaggleExample {
    
    public static void main(String[] args) {
        System.out.println("=".repeat(70));
        System.out.println("    SuperML Java - Advanced Kaggle Competition with Neural Networks");
        System.out.println("=".repeat(70));
        
        try {
            // 1. Generate competition dataset with different data types
            System.out.println("🏆 Kaggle Competition: Multi-Modal Data Challenge");
            System.out.println("\nGenerating competition datasets...");
            
            CompetitionData tabularData = generateTabularData(800, 20);
            CompetitionData imageData = generateImageData(400, 16, 16);
            CompetitionData sequenceData = generateSequenceData(600, 30, 8);
            
            System.out.printf("📊 Tabular data: %d samples, %d features\n", 
                tabularData.samples, tabularData.features);
            System.out.printf("🖼️  Image data: %d samples, %dx%d images\n", 
                imageData.samples, 16, 16);
            System.out.printf("📈 Sequence data: %d samples, %d timesteps\n", 
                sequenceData.samples, 30);
            
            // 2. Train multiple models for ensemble
            System.out.println("\n" + "=".repeat(50));
            System.out.println("Training Multiple Models for Ensemble");
            System.out.println("=".repeat(50));
            
            ModelResults logisticResults = trainLogisticModel(tabularData);
            ModelResults mlpResults = trainMLPModel(tabularData);
            ModelResults cnnResults = trainCNNModel(imageData);
            ModelResults rnnResults = trainRNNModel(sequenceData);
            
            // 3. Compare model performances
            System.out.println("\n" + "=".repeat(50));
            System.out.println("Model Performance Comparison");
            System.out.println("=".repeat(50));
            
            displayModelComparison(logisticResults, mlpResults, cnnResults, rnnResults);
            
            // 3.5. Display confusion matrices for neural networks
            System.out.println("\n" + "=".repeat(50));
            System.out.println("Neural Network Confusion Matrices");
            System.out.println("=".repeat(50));
            
            displayConfusionMatrices(mlpResults, cnnResults, rnnResults);
            
            // 4. Create ensemble predictions
            System.out.println("\n" + "=".repeat(50));
            System.out.println("Ensemble Predictions");
            System.out.println("=".repeat(50));
            
            generateEnsemblePredictions(logisticResults, mlpResults, cnnResults, rnnResults);
            
            // 5. Save models for future use
            System.out.println("\n" + "=".repeat(50));
            System.out.println("Model Persistence for Deployment");
            System.out.println("=".repeat(50));
            
            saveModelsForDeployment(logisticResults, mlpResults, cnnResults, rnnResults);
            
            System.out.println("\n🎉 Kaggle competition workflow completed successfully!");
            System.out.println("🚀 Models trained, evaluated, and ready for submission!");
            
        } catch (Exception e) {
            System.err.println("❌ Error in competition workflow: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    // ==================== Data Classes ====================
    
    /**
     * Container for competition dataset
     */
    private static class CompetitionData {
        final double[][] X;
        final double[] y;
        final double[][] XTrain;
        final double[][] XVal;
        final double[] yTrain;
        final double[] yVal;
        final int samples;
        final int features;
        
        CompetitionData(double[][] X, double[] y) {
            this.X = X;
            this.y = y;
            this.samples = X.length;
            this.features = X[0].length;
            
            // Split into train/validation
            int trainSize = (int)(samples * 0.8);
            this.XTrain = new double[trainSize][];
            this.XVal = new double[samples - trainSize][];
            this.yTrain = new double[trainSize];
            this.yVal = new double[samples - trainSize];
            
            System.arraycopy(X, 0, XTrain, 0, trainSize);
            System.arraycopy(X, trainSize, XVal, 0, samples - trainSize);
            System.arraycopy(y, 0, yTrain, 0, trainSize);
            System.arraycopy(y, trainSize, yVal, 0, samples - trainSize);
        }
    }
    
    /**
     * Container for model training results
     */
    private static class ModelResults {
        final String modelName;
        final Object model;
        final double accuracy;
        final double precision;
        final double recall;
        final double f1Score;
        final long trainingTime;
        final double[] predictions;
        final double[] actualLabels;
        final String modelPath;
        
        ModelResults(String modelName, Object model, double accuracy, double precision, 
                    double recall, double f1Score, long trainingTime, double[] predictions, 
                    double[] actualLabels, String modelPath) {
            this.modelName = modelName;
            this.model = model;
            this.accuracy = accuracy;
            this.precision = precision;
            this.recall = recall;
            this.f1Score = f1Score;
            this.trainingTime = trainingTime;
            this.predictions = predictions;
            this.actualLabels = actualLabels;
            this.modelPath = modelPath;
        }
    }
    
    // ==================== Data Generation ====================
    
    /**
     * Generate tabular data for traditional ML models
     */
    private static CompetitionData generateTabularData(int samples, int features) {
        double[][] X = new double[samples][features];
        double[] y = new double[samples];
        java.util.Random random = new java.util.Random(42);
        
        System.out.println("📊 Generating tabular dataset...");
        for (int i = 0; i < samples; i++) {
            double sum = 0;
            for (int j = 0; j < features; j++) {
                X[i][j] = random.nextGaussian() + j * 0.1;
                sum += X[i][j] * (j % 2 == 0 ? 1 : -1);
            }
            y[i] = sum > 0 ? 1.0 : 0.0;
        }
        
        return new CompetitionData(X, y);
    }
    
    /**
     * Generate image-like data for CNN
     */
    private static CompetitionData generateImageData(int samples, int height, int width) {
        double[][] X = new double[samples][height * width];
        double[] y = new double[samples];
        java.util.Random random = new java.util.Random(42);
        
        System.out.println("🖼️  Generating image dataset...");
        for (int i = 0; i < samples; i++) {
            double centerMass = 0;
            for (int j = 0; j < height * width; j++) {
                X[i][j] = random.nextDouble();
                // Create pattern: center pixels influence classification
                int row = j / width;
                int col = j % width;
                if (row >= height/3 && row <= 2*height/3 && col >= width/3 && col <= 2*width/3) {
                    centerMass += X[i][j];
                }
            }
            y[i] = centerMass > height * width / 18.0 ? 1.0 : 0.0;
        }
        
        return new CompetitionData(X, y);
    }
    
    /**
     * Generate sequence data for RNN
     */
    private static CompetitionData generateSequenceData(int samples, int sequenceLength, int features) {
        double[][] X = new double[samples][sequenceLength * features];
        double[] y = new double[samples];
        java.util.Random random = new java.util.Random(42);
        
        System.out.println("📈 Generating sequence dataset...");
        for (int i = 0; i < samples; i++) {
            double trend = 0;
            for (int t = 0; t < sequenceLength; t++) {
                for (int f = 0; f < features; f++) {
                    int idx = t * features + f;
                    X[i][idx] = Math.sin(t * 0.1 + f) + random.nextGaussian() * 0.1;
                    if (t > sequenceLength / 2) trend += X[i][idx];
                }
            }
            y[i] = trend > 0 ? 1.0 : 0.0;
        }
        
        return new CompetitionData(X, y);
    }
    
    // ==================== Model Training ====================
    
    /**
     * Train Logistic Regression baseline
     */
    private static ModelResults trainLogisticModel(CompetitionData data) {
        System.out.println("\n🔵 Training Logistic Regression (Baseline)");
        
        long startTime = System.currentTimeMillis();
        
        LogisticRegression model = new LogisticRegression();
        model.fit(data.XTrain, data.yTrain);
        
        long trainingTime = System.currentTimeMillis() - startTime;
        
        double[] predictions = model.predict(data.XVal);
        double accuracy = calculateAccuracy(data.yVal, predictions);
        double precision = calculatePrecision(data.yVal, predictions);
        double recall = calculateRecall(data.yVal, predictions);
        double f1Score = 2 * (precision * recall) / (precision + recall);
        
        System.out.printf("   ⏱️  Training time: %d ms\n", trainingTime);
        System.out.printf("   🎯 Accuracy: %.4f\n", accuracy);
        
        return new ModelResults("Logistic Regression", model, accuracy, precision, 
                              recall, f1Score, trainingTime, predictions, data.yVal, null);
    }
    
    /**
     * Train MLP Classifier with preprocessing
     */
    private static ModelResults trainMLPModel(CompetitionData data) {
        System.out.println("\n🧠 Training MLP Neural Network");
        
        long startTime = System.currentTimeMillis();
        
        // Apply neural network preprocessing for MLP
        NeuralNetworkPreprocessor preprocessor = new NeuralNetworkPreprocessor(
            NeuralNetworkPreprocessor.NetworkType.MLP).configureMLP();
        
        double[][] XTrainProcessed = preprocessor.preprocessMLP(data.XTrain);
        double[][] XValProcessed = preprocessor.preprocessMLP(data.XVal);
        
        System.out.println("   📊 Applied MLP preprocessing: standardization + outlier clipping");
        
        MLPClassifier model = new MLPClassifier()
            .setHiddenLayerSizes(64, 32, 16)
            .setActivation("relu")
            .setLearningRate(0.01)
            .setMaxIter(100)
            .setBatchSize(32);
        
        model.fit(XTrainProcessed, data.yTrain);
        
        long trainingTime = System.currentTimeMillis() - startTime;
        
        double[] predictions = model.predict(XValProcessed);
        double accuracy = calculateAccuracy(data.yVal, predictions);
        double precision = calculatePrecision(data.yVal, predictions);
        double recall = calculateRecall(data.yVal, predictions);
        double f1Score = 2 * (precision * recall) / (precision + recall);
        
        System.out.printf("   ⏱️  Training time: %d ms\n", trainingTime);
        System.out.printf("   🎯 Accuracy: %.4f\n", accuracy);
        
        // Save model for competition deployment
        String modelPath = "models/kaggle_mlp.superml";
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("competition", "kaggle_advanced");
        metadata.put("accuracy", accuracy);
        metadata.put("architecture", Arrays.toString(model.getHiddenLayerSizes()));
        metadata.put("preprocessing", "standardization + outlier_clipping");
        
        try {
            ModelPersistence.save(model, modelPath, "Kaggle competition MLP", metadata);
        } catch (Exception e) {
            System.err.println("   ⚠️  Model save failed: " + e.getMessage());
            modelPath = null;
        }
        
        return new ModelResults("MLP Neural Network", model, accuracy, precision, 
                              recall, f1Score, trainingTime, predictions, data.yVal, modelPath);
    }
    
    /**
     * Train CNN Classifier with preprocessing
     */
    private static ModelResults trainCNNModel(CompetitionData data) {
        System.out.println("\n🖼️  Training CNN for Image Data");
        
        long startTime = System.currentTimeMillis();
        
        // Apply neural network preprocessing for CNN
        NeuralNetworkPreprocessor preprocessor = new NeuralNetworkPreprocessor(
            NeuralNetworkPreprocessor.NetworkType.CNN).configureCNN(16, 16, 1);
        
        double[][] XTrainProcessed = preprocessor.preprocessCNN(data.XTrain);
        double[][] XValProcessed = preprocessor.preprocessCNN(data.XVal);
        
        System.out.println("   📊 Applied CNN preprocessing: pixel normalization to [-1,1]");
        
        CNNClassifier model = new CNNClassifier()
            .setInputShape(16, 16, 1)
            .setLearningRate(0.01)
            .setMaxEpochs(50)
            .setBatchSize(32);
        
        try {
            model.fit(XTrainProcessed, data.yTrain);
            
            long trainingTime = System.currentTimeMillis() - startTime;
            
            double[] predictions = model.predict(XValProcessed);
            double accuracy = calculateAccuracy(data.yVal, predictions);
            double precision = calculatePrecision(data.yVal, predictions);
            double recall = calculateRecall(data.yVal, predictions);
            double f1Score = 2 * (precision * recall) / (precision + recall);
            
            System.out.printf("   ⏱️  Training time: %d ms\n", trainingTime);
            System.out.printf("   🎯 Accuracy: %.4f\n", accuracy);
            
            return new ModelResults("CNN", model, accuracy, precision, 
                                  recall, f1Score, trainingTime, predictions, data.yVal, null);
                                  
        } catch (Exception e) {
            System.err.println("   ❌ CNN training failed: " + e.getMessage());
            // Return dummy results
            double[] dummyPredictions = new double[data.XVal.length];
            Arrays.fill(dummyPredictions, 0.5);
            return new ModelResults("CNN (Failed)", null, 0.5, 0.5, 0.5, 0.5, 0, dummyPredictions, data.yVal, null);
        }
    }
    
    /**
     * Train RNN Classifier with preprocessing
     */
    private static ModelResults trainRNNModel(CompetitionData data) {
        System.out.println("\n📈 Training RNN for Sequence Data");
        
        long startTime = System.currentTimeMillis();
        
        // Apply neural network preprocessing for RNN
        NeuralNetworkPreprocessor preprocessor = new NeuralNetworkPreprocessor(
            NeuralNetworkPreprocessor.NetworkType.RNN).configureRNN(30, 8, false);
        
        double[][] XTrainProcessed = preprocessor.preprocessRNN(data.XTrain);
        double[][] XValProcessed = preprocessor.preprocessRNN(data.XVal);
        
        System.out.println("   📊 Applied RNN preprocessing: global scaling + temporal smoothing");
        
        RNNClassifier model = new RNNClassifier()
            .setHiddenSize(32)
            .setNumLayers(2)
            .setCellType("LSTM")
            .setLearningRate(0.01)
            .setMaxEpochs(75)
            .setBatchSize(32);
        
        try {
            model.fit(XTrainProcessed, data.yTrain);
            
            long trainingTime = System.currentTimeMillis() - startTime;
            
            double[] predictions = model.predict(XValProcessed);
            double accuracy = calculateAccuracy(data.yVal, predictions);
            double precision = calculatePrecision(data.yVal, predictions);
            double recall = calculateRecall(data.yVal, predictions);
            double f1Score = 2 * (precision * recall) / (precision + recall);
            
            System.out.printf("   ⏱️  Training time: %d ms\n", trainingTime);
            System.out.printf("   🎯 Accuracy: %.4f\n", accuracy);
            
            String modelPath = "models/kaggle_rnn.superml";
            Map<String, Object> metadata = new HashMap<>();
            metadata.put("competition", "kaggle_advanced");
            metadata.put("accuracy", accuracy);
            metadata.put("cell_type", model.getCellType());
            metadata.put("preprocessing", "global_scaling + temporal_smoothing");
            
            try {
                ModelPersistence.save(model, modelPath, "Kaggle competition RNN", metadata);
            } catch (Exception e) {
                System.err.println("   ⚠️  Model save failed: " + e.getMessage());
                modelPath = null;
            }
            
            return new ModelResults("RNN LSTM", model, accuracy, precision, 
                                  recall, f1Score, trainingTime, predictions, data.yVal, modelPath);
                                  
        } catch (Exception e) {
            System.err.println("   ❌ RNN training failed: " + e.getMessage());
            // Return dummy results
            double[] dummyPredictions = new double[data.XVal.length];
            Arrays.fill(dummyPredictions, 0.5);
            return new ModelResults("RNN (Failed)", null, 0.5, 0.5, 0.5, 0.5, 0, dummyPredictions, data.yVal, null);
        }
    }
    
    // ==================== Analysis and Comparison ====================
    
    /**
     * Display comprehensive model comparison
     */
    private static void displayModelComparison(ModelResults... results) {
        System.out.println("\n📊 Model Performance Leaderboard:");
        System.out.println("┌─────────────────────┬──────────┬───────────┬────────┬────────┬───────────┐");
        System.out.println("│ Model               │ Accuracy │ Precision │ Recall │ F1     │ Time (ms) │");
        System.out.println("├─────────────────────┼──────────┼───────────┼────────┼────────┼───────────┤");
        
        for (ModelResults result : results) {
            System.out.printf("│ %-19s │ %8.4f │ %9.4f │ %6.4f │ %6.4f │ %9d │%n",
                result.modelName, result.accuracy, result.precision, 
                result.recall, result.f1Score, result.trainingTime);
        }
        System.out.println("└─────────────────────┴──────────┴───────────┴────────┴────────┴───────────┘");
        
        // Find best model
        ModelResults best = results[0];
        for (ModelResults result : results) {
            if (result.accuracy > best.accuracy) {
                best = result;
            }
        }
        System.out.println("\n🏆 Best performing model: " + best.modelName + 
                          " (Accuracy: " + String.format("%.4f", best.accuracy) + ")");
    }
    
    /**
     * Display confusion matrices for neural network models
     */
    private static void displayConfusionMatrices(ModelResults... results) {
        for (ModelResults result : results) {
            if (result.model != null && result.predictions != null && result.actualLabels != null) {
                System.out.println("\n🧠 " + result.modelName + " - Confusion Matrix:");
                displaySingleConfusionMatrix(result.actualLabels, result.predictions, result.modelName);
            }
        }
    }
    
    /**
     * Display confusion matrix for a single model
     */
    private static void displaySingleConfusionMatrix(double[] actual, double[] predicted, String modelName) {
        // Convert to binary predictions
        int[] actualBinary = new int[actual.length];
        int[] predictedBinary = new int[predicted.length];
        
        for (int i = 0; i < actual.length; i++) {
            actualBinary[i] = (int) Math.round(actual[i]);
            predictedBinary[i] = (int) Math.round(predicted[i]);
        }
        
        // Calculate confusion matrix components
        int truePositive = 0, trueNegative = 0, falsePositive = 0, falseNegative = 0;
        
        for (int i = 0; i < actual.length; i++) {
            if (actualBinary[i] == 1 && predictedBinary[i] == 1) truePositive++;
            else if (actualBinary[i] == 0 && predictedBinary[i] == 0) trueNegative++;
            else if (actualBinary[i] == 0 && predictedBinary[i] == 1) falsePositive++;
            else if (actualBinary[i] == 1 && predictedBinary[i] == 0) falseNegative++;
        }
        
        // Display confusion matrix
        System.out.println("┌─────────────────┬─────────────────────────┐");
        System.out.println("│                 │      Predicted          │");
        System.out.println("├─────────────────┼───────────┬─────────────┤");
        System.out.println("│                 │     0     │      1      │");
        System.out.println("├─────────────────┼───────────┼─────────────┤");
        System.out.printf("│ Actual    0     │   %4d    │    %4d     │%n", trueNegative, falsePositive);
        System.out.printf("│           1     │   %4d    │    %4d     │%n", falseNegative, truePositive);
        System.out.println("└─────────────────┴───────────┴─────────────┘");
        
        // Calculate additional metrics
        double sensitivity = truePositive + falseNegative > 0 ? 
            (double) truePositive / (truePositive + falseNegative) : 0.0;
        double specificity = trueNegative + falsePositive > 0 ? 
            (double) trueNegative / (trueNegative + falsePositive) : 0.0;
        double ppv = truePositive + falsePositive > 0 ? 
            (double) truePositive / (truePositive + falsePositive) : 0.0;
        double npv = trueNegative + falseNegative > 0 ? 
            (double) trueNegative / (trueNegative + falseNegative) : 0.0;
        
        System.out.println("\n📊 Detailed Classification Metrics:");
        System.out.printf("   • True Positives:  %4d    • False Positives: %4d%n", truePositive, falsePositive);
        System.out.printf("   • True Negatives:  %4d    • False Negatives: %4d%n", trueNegative, falseNegative);
        System.out.printf("   • Sensitivity (Recall):    %.4f    • Specificity:        %.4f%n", sensitivity, specificity);
        System.out.printf("   • PPV (Precision):         %.4f    • NPV:                %.4f%n", ppv, npv);
        
        // Sample size info
        System.out.printf("   • Total Samples: %d    • Positive: %d (%.1f%%)    • Negative: %d (%.1f%%)%n", 
            actual.length, 
            truePositive + falseNegative, 
            100.0 * (truePositive + falseNegative) / actual.length,
            trueNegative + falsePositive,
            100.0 * (trueNegative + falsePositive) / actual.length);
    }
    
    /**
     * Generate ensemble predictions combining all models
     */
    private static void generateEnsemblePredictions(ModelResults... results) {
        if (results.length == 0) return;
        
        // Find minimum prediction length across all models
        int minPredictions = Integer.MAX_VALUE;
        for (ModelResults result : results) {
            if (result.model != null && result.predictions.length < minPredictions) {
                minPredictions = result.predictions.length;
            }
        }
        
        if (minPredictions == Integer.MAX_VALUE) {
            System.out.println("⚠️  No valid models for ensemble");
            return;
        }
        
        double[] ensemblePredictions = new double[minPredictions];
        
        // Simple averaging ensemble
        System.out.println("🔗 Creating ensemble predictions (simple averaging):");
        for (int i = 0; i < minPredictions; i++) {
            double sum = 0;
            int validModels = 0;
            for (ModelResults result : results) {
                if (result.model != null && i < result.predictions.length) {
                    sum += result.predictions[i];
                    validModels++;
                }
            }
            ensemblePredictions[i] = validModels > 0 ? sum / validModels : 0.5;
        }
        
        // Display sample ensemble predictions
        System.out.println("\n📋 Sample Ensemble Submission (first 10 predictions):");
        System.out.println("ID,Prediction");
        for (int i = 0; i < Math.min(10, minPredictions); i++) {
            System.out.printf("%d,%.6f%n", i + 1, ensemblePredictions[i]);
        }
        if (minPredictions > 10) {
            System.out.println("... (" + (minPredictions - 10) + " more predictions)");
        }
        
        System.out.println("\n💡 Ensemble combines predictions from " + results.length + " models");
        System.out.println("📤 Ready for Kaggle submission!");
    }
    
    /**
     * Save trained models for future deployment
     */
    private static void saveModelsForDeployment(ModelResults... results) {
        System.out.println("💾 Saving models for deployment:");
        
        for (ModelResults result : results) {
            if (result.modelPath != null) {
                System.out.println("✅ " + result.modelName + " → " + result.modelPath);
            } else if (result.model != null) {
                System.out.println("⚠️  " + result.modelName + " → Save failed (serialization issue)");
            } else {
                System.out.println("❌ " + result.modelName + " → Model training failed");
            }
        }
        
        System.out.println("\n🚀 Deployment ready models can be loaded with:");
        System.out.println("   ModelPersistence.load(\"models/kaggle_mlp.superml\", MLPClassifier.class)");
        System.out.println("   ModelPersistence.load(\"models/kaggle_rnn.superml\", RNNClassifier.class)");
    }
    
    // ==================== Utility Methods ====================
    
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
     * Calculate precision metric
     */
    private static double calculatePrecision(double[] actual, double[] predicted) {
        int truePositive = 0, falsePositive = 0;
        for (int i = 0; i < actual.length; i++) {
            if (Math.round(predicted[i]) == 1) {
                if (Math.round(actual[i]) == 1) truePositive++;
                else falsePositive++;
            }
        }
        return truePositive + falsePositive > 0 ? 
            (double) truePositive / (truePositive + falsePositive) : 0.0;
    }
    
    /**
     * Calculate recall metric
     */
    private static double calculateRecall(double[] actual, double[] predicted) {
        int truePositive = 0, falseNegative = 0;
        for (int i = 0; i < actual.length; i++) {
            if (Math.round(actual[i]) == 1) {
                if (Math.round(predicted[i]) == 1) truePositive++;
                else falseNegative++;
            }
        }
        return truePositive + falseNegative > 0 ? 
            (double) truePositive / (truePositive + falseNegative) : 0.0;
    }
}
