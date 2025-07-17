package org.superml.examples;

/**
 * Deep Learning Examples for SuperML Java Framework
 * 
 * This example demonstrates how to use the neural network classifiers:
 * - MLPClassifier (Multi-Layer Perceptron)
 * - CNNClassifier (Convolutional Neural Network)
 * - RNNClassifier (Recurrent Neural Network)
 * 
 * @author SuperML Java Team
 */

import org.superml.neural.*;
import java.util.Arrays;
import java.util.Random;

public class DeepLearningExample {
    private static final Random random = new Random(42);
    
    public static void main(String[] args) {
        System.out.println("=== SuperML Deep Learning Examples ===\n");
        
        // 1. Multi-Layer Perceptron (MLP) Example
        runMLPExample();
        
        // 2. Convolutional Neural Network (CNN) Example
        runCNNExample();
        
        // 3. Recurrent Neural Network (RNN) Example
        runRNNExample();
        
        // 4. Comparison of all three models
        compareModels();
    }
    
    /**
     * Demonstrates MLPClassifier usage
     */
    private static void runMLPExample() {
        System.out.println("1. Multi-Layer Perceptron (MLP) Example");
        System.out.println("----------------------------------------");
        
        // Generate simple XOR dataset
        double[][] X = {
            {0, 0}, {0, 1}, {1, 0}, {1, 1},
            {0.1, 0.1}, {0.1, 0.9}, {0.9, 0.1}, {0.9, 0.9}
        };
        double[] y = {0, 1, 1, 0, 0, 1, 1, 0};
        
        // Create and configure MLP classifier
        MLPClassifier mlp = new MLPClassifier(10, 5)  // Hidden layers: 10, 5 neurons
            .setActivation("relu")
            .setSolver("adam")
            .setLearningRate(0.01)
            .setMaxIter(100)
            .setTolerance(1e-4);
        
        System.out.println("Training MLP with architecture: [2] -> [10, 5] -> [2]");
        
        // Train the model
        mlp.fit(X, y);
        
        // Make predictions
        double[] predictions = mlp.predict(X);
        double[][] probabilities = mlp.predictProba(X);
        
        System.out.println("Predictions: " + Arrays.toString(predictions));
        System.out.println("Accuracy: " + calculateAccuracy(y, predictions));
        System.out.println("Model trained: " + mlp.isFitted());
        System.out.println();
    }
    
    /**
     * Demonstrates CNNClassifier usage for image-like data
     */
    private static void runCNNExample() {
        System.out.println("2. Convolutional Neural Network (CNN) Example");
        System.out.println("----------------------------------------------");
        
        // Generate simple image-like dataset (8x8 "images")
        double[][] X = generateImageDataset(20, 8, 8);
        double[] y = generateImageLabels(20);
        
        // Create and configure CNN classifier
        CNNClassifier cnn = new CNNClassifier(8, 8, 1)  // 8x8 grayscale images
            .setLearningRate(0.001)
            .setMaxEpochs(50)
            .setBatchSize(4);
        
        System.out.println("Training CNN with image data (8x8 pixels)");
        
        // Train the model
        cnn.fit(X, y);
        
        // Make predictions
        double[] predictions = cnn.predict(X);
        
        System.out.println("CNN Accuracy: " + calculateAccuracy(y, predictions));
        System.out.println();
        System.out.println();
    }
    
    /**
     * Demonstrates RNNClassifier usage for sequence data
     */
    private static void runRNNExample() {
        System.out.println("3. Recurrent Neural Network (RNN) Example");
        System.out.println("------------------------------------------");
        
        // Generate simple sequence dataset
        double[][] X = generateSequenceDataset(30, 10);  // 30 sequences of length 10
        double[] y = generateSequenceLabels(30);
        
        // Create and configure RNN classifier
        RNNClassifier rnn = new RNNClassifier(10, 16, "LSTM")  // input_size=10, hidden_size=16, cell_type=LSTM
            .setNumLayers(2)
            .setBidirectional(true)
            .setLearningRate(0.01)
            .setMaxEpochs(100);
        
        System.out.println("Training RNN with sequence data (length=10)");
        
        // Train the model
        rnn.fit(X, y);
        
        // Make predictions
        double[] predictions = rnn.predict(X);
        
        System.out.println("RNN Accuracy: " + calculateAccuracy(y, predictions));
        System.out.println();
        System.out.println();
    }
    
    /**
     * Compares all three neural network models
     */
    private static void compareModels() {
        System.out.println("4. Model Comparison");
        System.out.println("-------------------");
        
        // Generate common dataset for comparison
        double[][] X = generateCommonDataset(100, 20);
        double[] y = generateCommonLabels(100);
        
        // Split into train/test
        int trainSize = 80;
        double[][] XTrain = Arrays.copyOfRange(X, 0, trainSize);
        double[] yTrain = Arrays.copyOfRange(y, 0, trainSize);
        double[][] XTest = Arrays.copyOfRange(X, trainSize, X.length);
        double[] yTest = Arrays.copyOfRange(y, trainSize, y.length);
        
        // Train and evaluate MLP
        MLPClassifier mlp = new MLPClassifier(32, 16)
            .setActivation("relu")
            .setSolver("adam")
            .setMaxIter(100);
        
        long mlpStartTime = System.currentTimeMillis();
        mlp.fit(XTrain, yTrain);
        long mlpTrainTime = System.currentTimeMillis() - mlpStartTime;
        
        double[] mlpPredictions = mlp.predict(XTest);
        double mlpAccuracy = calculateAccuracy(yTest, mlpPredictions);
        
        // Train and evaluate CNN (reshape data for CNN)
        double[][] XTrainCNN = reshapeForCNN(XTrain, 4, 5);  // 4x5 "images"
        double[][] XTestCNN = reshapeForCNN(XTest, 4, 5);
        
        CNNClassifier cnn = new CNNClassifier(4, 5, 1)
            .setMaxEpochs(50);
        
        long cnnStartTime = System.currentTimeMillis();
        cnn.fit(XTrainCNN, yTrain);
        long cnnTrainTime = System.currentTimeMillis() - cnnStartTime;
        
        double[] cnnPredictions = cnn.predict(XTestCNN);
        double cnnAccuracy = calculateAccuracy(yTest, cnnPredictions);
        
        // Train and evaluate RNN
        RNNClassifier rnn = new RNNClassifier(20, 16, "LSTM")
            .setMaxEpochs(100);
        
        long rnnStartTime = System.currentTimeMillis();
        rnn.fit(XTrain, yTrain);
        long rnnTrainTime = System.currentTimeMillis() - rnnStartTime;
        
        double[] rnnPredictions = rnn.predict(XTest);
        double rnnAccuracy = calculateAccuracy(yTest, rnnPredictions);
        
        // Print comparison results
        System.out.println("Model Comparison Results:");
        System.out.println("========================");
        System.out.printf("MLP: Accuracy=%.3f, Training Time=%dms%n", mlpAccuracy, mlpTrainTime);
        System.out.printf("CNN: Accuracy=%.3f, Training Time=%dms%n", cnnAccuracy, cnnTrainTime);
        System.out.printf("RNN: Accuracy=%.3f, Training Time=%dms%n", rnnAccuracy, rnnTrainTime);
        System.out.println();
        
        // Model characteristics
        System.out.println("Model Characteristics:");
        System.out.println("======================");
        System.out.println("MLP: Best for tabular data, fully connected layers");
        System.out.println("CNN: Best for image data, spatial feature extraction");
        System.out.println("RNN: Best for sequential data, temporal dependencies");
        System.out.println();
        
        System.out.println("Example completed successfully!");
    }
    
    // Helper methods for data generation
    
    private static double[][] generateImageDataset(int numSamples, int height, int width) {
        double[][] dataset = new double[numSamples][height * width];
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < height * width; j++) {
                dataset[i][j] = random.nextDouble();
            }
        }
        return dataset;
    }
    
    private static double[] generateImageLabels(int numSamples) {
        double[] labels = new double[numSamples];
        for (int i = 0; i < numSamples; i++) {
            labels[i] = random.nextInt(2);  // Binary classification
        }
        return labels;
    }
    
    private static double[][] generateSequenceDataset(int numSequences, int sequenceLength) {
        double[][] dataset = new double[numSequences][sequenceLength];
        for (int i = 0; i < numSequences; i++) {
            for (int j = 0; j < sequenceLength; j++) {
                dataset[i][j] = Math.sin(j * 0.1 + i * 0.05) + random.nextGaussian() * 0.1;
            }
        }
        return dataset;
    }
    
    private static double[] generateSequenceLabels(int numSequences) {
        double[] labels = new double[numSequences];
        for (int i = 0; i < numSequences; i++) {
            // Label based on whether the sequence has an increasing trend
            labels[i] = (i % 3 == 0) ? 1 : 0;
        }
        return labels;
    }
    
    private static double[][] generateCommonDataset(int numSamples, int numFeatures) {
        double[][] dataset = new double[numSamples][numFeatures];
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numFeatures; j++) {
                dataset[i][j] = random.nextGaussian();
            }
        }
        return dataset;
    }
    
    private static double[] generateCommonLabels(int numSamples) {
        double[] labels = new double[numSamples];
        for (int i = 0; i < numSamples; i++) {
            labels[i] = random.nextInt(3);  // Multi-class classification (3 classes)
        }
        return labels;
    }
    
    private static double[][] reshapeForCNN(double[][] data, int height, int width) {
        // For this example, we'll just return the original data
        // In a real scenario, you'd reshape the data appropriately
        return data;
    }
    
    private static double calculateAccuracy(double[] actual, double[] predicted) {
        if (actual.length != predicted.length) {
            return 0.0;
        }
        
        int correct = 0;
        for (int i = 0; i < actual.length; i++) {
            if (Math.abs(actual[i] - predicted[i]) < 0.5) {
                correct++;
            }
        }
        return (double) correct / actual.length;
    }
}
