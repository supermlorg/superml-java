package org.superml.neural.examples;

import org.superml.neural.MLPClassifier;
import org.superml.neural.CNNClassifier;
import org.superml.neural.RNNClassifier;

import java.util.Random;
import java.util.Arrays;

/**
 * Comprehensive examples demonstrating deep learning capabilities in SuperML
 * 
 * This class showcases:
 * - Multi-Layer Perceptron (MLP) for tabular data
 * - Convolutional Neural Network (CNN) for image classification
 * - Recurrent Neural Network (RNN) for sequence classification
 * - Advanced architectures and configurations
 * 
 * @author SuperML Team
 * @version 2.0.0
 */
public class DeepLearningExamples {
    
    public static void main(String[] args) {
        System.out.println("=== SuperML Deep Learning Examples ===\n");
        
        // Run all examples
        runMLPExample();
        runCNNExample();
        runRNNExample();
        runAdvancedArchitecturesExample();
        runTransferLearningExample();
        
        System.out.println("\n=== All Deep Learning Examples Completed ===");
    }
    
    /**
     * Example 1: Multi-Layer Perceptron for tabular data classification
     */
    public static void runMLPExample() {
        System.out.println("1. Multi-Layer Perceptron (MLP) Example");
        System.out.println("========================================");
        
        // Generate synthetic tabular dataset (iris-like)
        DatasetResult dataset = generateTabularDataset(500, 4, 3);
        
        // Create and configure MLP
        MLPClassifier mlp = new MLPClassifier()
            .setHiddenLayerSizes(new int[]{100, 50, 25})
            .setActivation("relu")
            .setSolver("adam")
            .setLearningRate(0.001)
            .setMaxEpochs(200)
            .setBatchSize(32)
            .setAlpha(0.0001)
            .setEarlyStopping(true)
            .setValidationFraction(0.2)
            .setTolerance(1e-4)
            .setRandomState(42);
        
        // Train the model
        System.out.println("Training MLP...");
        long startTime = System.currentTimeMillis();
        mlp.fit(dataset.trainX, dataset.trainY);
        long trainingTime = System.currentTimeMillis() - startTime;
        
        // Make predictions
        double[] predictions = mlp.predict(dataset.testX);
        double[][] probabilities = mlp.predictProba(dataset.testX);
        
        // Calculate accuracy
        double accuracy = calculateAccuracy(predictions, dataset.testY);
        
        // Display results
        System.out.printf("Training time: %d ms%n", trainingTime);
        System.out.printf("Test accuracy: %.3f%n", accuracy);
        System.out.printf("Number of epochs: %d%n", mlp.getLossHistory().size());
        System.out.printf("Final loss: %.6f%n", mlp.getLossHistory().get(mlp.getLossHistory().size() - 1));
        
        // Show first few predictions vs actual
        System.out.println("\nFirst 10 predictions vs actual:");
        for (int i = 0; i < Math.min(10, predictions.length); i++) {
            System.out.printf("Predicted: %.0f, Actual: %.0f, Confidence: %.3f%n", 
                predictions[i], dataset.testY[i], getMaxProbability(probabilities[i]));
        }
        
        System.out.println();
    }
    
    /**
     * Example 2: Convolutional Neural Network for image classification
     */
    public static void runCNNExample() {
        System.out.println("2. Convolutional Neural Network (CNN) Example");
        System.out.println("==============================================");
        
        // Generate synthetic image dataset (MNIST-like)
        ImageDatasetResult imageDataset = generateImageDataset(200, 28, 28, 1, 3);
        
        // Create and configure CNN with custom architecture
        CNNClassifier cnn = new CNNClassifier(28, 28, 1)
            // First convolutional block
            .addConvolutionalLayer(32, 3, 3, 1, "relu")
            .addBatchNormalization()
            .addMaxPooling(2, 2)
            .addDropout(0.25)
            
            // Second convolutional block
            .addConvolutionalLayer(64, 3, 3, 1, "relu")
            .addBatchNormalization()
            .addMaxPooling(2, 2)
            .addDropout(0.25)
            
            // Third convolutional block
            .addConvolutionalLayer(128, 3, 3, 1, "relu")
            .addBatchNormalization()
            .addGlobalAveragePooling()
            .addDropout(0.5)
            
            // Dense layers
            .addDenseLayer(256, "relu")
            .addDropout(0.5)
            .addDenseLayer(3, "linear") // 3 classes
            
            // Configure training parameters
            .setLearningRate(0.001)
            .setMaxEpochs(50)
            .setBatchSize(16)
            .setOptimizer("adam")
            .setRegularization(0.0001);
        
        // Train the model
        System.out.println("Training CNN...");
        long startTime = System.currentTimeMillis();
        cnn.fit(imageDataset.trainX, imageDataset.trainY);
        long trainingTime = System.currentTimeMillis() - startTime;
        
        // Make predictions
        double[] predictions = cnn.predict(imageDataset.testX);
        double[][] probabilities = cnn.predictProba(imageDataset.testX);
        
        // Calculate accuracy
        double accuracy = calculateAccuracy(predictions, imageDataset.testY);
        
        // Display results
        System.out.printf("Training time: %d ms%n", trainingTime);
        System.out.printf("Test accuracy: %.3f%n", accuracy);
        System.out.printf("Number of epochs: %d%n", cnn.getTrainLossHistory().size());
        
        // Show first few predictions
        System.out.println("\nFirst 5 predictions vs actual:");
        for (int i = 0; i < Math.min(5, predictions.length); i++) {
            System.out.printf("Predicted: %.0f, Actual: %.0f, Confidence: %.3f%n", 
                predictions[i], imageDataset.testY[i], getMaxProbability(probabilities[i]));
        }
        
        // Test with different pre-defined architectures
        System.out.println("\nTesting LeNet architecture:");
        CNNClassifier leNet = new CNNClassifier(28, 28, 1)
            .useLeNetArchitecture()
            .setMaxEpochs(20)
            .setLearningRate(0.01)
            .setBatchSize(16);
        
        leNet.fit(imageDataset.trainX, imageDataset.trainY);
        double leNetAccuracy = calculateAccuracy(leNet.predict(imageDataset.testX), imageDataset.testY);
        System.out.printf("LeNet accuracy: %.3f%n", leNetAccuracy);
        
        System.out.println();
    }
    
    /**
     * Example 3: Recurrent Neural Network for sequence classification
     */
    public static void runRNNExample() {
        System.out.println("3. Recurrent Neural Network (RNN) Example");
        System.out.println("==========================================");
        
        // Generate synthetic sequence dataset (sentiment analysis-like)
        SequenceDatasetResult seqDataset = generateSequenceDataset(300, 15, 100, 2);
        
        // Create and configure LSTM
        RNNClassifier lstm = new RNNClassifier()
            .setCellType("LSTM")
            .setHiddenSize(64)
            .setNumLayers(2)
            .setBidirectional(true)
            .setDropoutRate(0.3)
            .setRecurrentDropout(0.2)
            .setLearningRate(0.001)
            .setMaxEpochs(50)
            .setBatchSize(16);
        
        // Train the model
        System.out.println("Training LSTM...");
        long startTime = System.currentTimeMillis();
        lstm.fit(seqDataset.trainX, seqDataset.trainY);
        long trainingTime = System.currentTimeMillis() - startTime;
        
        // Make predictions
        double[] lstmPredictions = lstm.predict(seqDataset.testX);
        double lstmAccuracy = calculateAccuracy(lstmPredictions, seqDataset.testY);
        
        System.out.printf("LSTM Training time: %d ms%n", trainingTime);
        System.out.printf("LSTM Test accuracy: %.3f%n", lstmAccuracy);
        
        // Compare with GRU
        System.out.println("\nComparing with GRU:");
        RNNClassifier gru = new RNNClassifier()
            .useGRUArchitecture()
            .setMaxEpochs(50)
            .setLearningRate(0.001)
            .setBatchSize(16);
        
        startTime = System.currentTimeMillis();
        gru.fit(seqDataset.trainX, seqDataset.trainY);
        long gruTime = System.currentTimeMillis() - startTime;
        
        double[] gruPredictions = gru.predict(seqDataset.testX);
        double gruAccuracy = calculateAccuracy(gruPredictions, seqDataset.testY);
        
        System.out.printf("GRU Training time: %d ms%n", gruTime);
        System.out.printf("GRU Test accuracy: %.3f%n", gruAccuracy);
        
        // Test with Attention mechanism
        System.out.println("\nTesting with Attention mechanism:");
        RNNClassifier attentionLSTM = new RNNClassifier()
            .useAttentionLSTM()
            .setMaxEpochs(50)
            .setLearningRate(0.001)
            .setBatchSize(16);
        
        startTime = System.currentTimeMillis();
        attentionLSTM.fit(seqDataset.trainX, seqDataset.trainY);
        long attentionTime = System.currentTimeMillis() - startTime;
        
        double[] attentionPredictions = attentionLSTM.predict(seqDataset.testX);
        double attentionAccuracy = calculateAccuracy(attentionPredictions, seqDataset.testY);
        
        System.out.printf("Attention LSTM Training time: %d ms%n", attentionTime);
        System.out.printf("Attention LSTM Test accuracy: %.3f%n", attentionAccuracy);
        
        System.out.println();
    }
    
    /**
     * Example 4: Advanced architectures and ensemble methods
     */
    public static void runAdvancedArchitecturesExample() {
        System.out.println("4. Advanced Architectures and Ensemble Example");
        System.out.println("===============================================");
        
        // Generate a challenging dataset
        DatasetResult complexDataset = generateComplexDataset(400, 20, 4);
        
        // Create ensemble of different neural networks
        System.out.println("Creating neural network ensemble...");
        
        // Deep MLP
        MLPClassifier deepMLP = new MLPClassifier()
            .setHiddenLayerSizes(new int[]{256, 128, 64, 32})
            .setActivation("relu")
            .setSolver("adam")
            .setLearningRate(0.0005)
            .setMaxEpochs(100)
            .setBatchSize(32)
            .setAlpha(0.001)
            .setEarlyStopping(true);
        
        // Wide MLP
        MLPClassifier wideMLP = new MLPClassifier()
            .setHiddenLayerSizes(new int[]{512, 256})
            .setActivation("leaky_relu")
            .setSolver("adam")
            .setLearningRate(0.001)
            .setMaxEpochs(100)
            .setBatchSize(32)
            .setAlpha(0.001)
            .setEarlyStopping(true);
        
        // Regularized MLP
        MLPClassifier regMLP = new MLPClassifier()
            .setHiddenLayerSizes(new int[]{128, 64})
            .setActivation("elu")
            .setSolver("adam")
            .setLearningRate(0.001)
            .setMaxEpochs(100)
            .setBatchSize(32)
            .setAlpha(0.01) // Higher regularization
            .setEarlyStopping(true);
        
        // Train ensemble models
        long ensembleStart = System.currentTimeMillis();
        
        deepMLP.fit(complexDataset.trainX, complexDataset.trainY);
        wideMLP.fit(complexDataset.trainX, complexDataset.trainY);
        regMLP.fit(complexDataset.trainX, complexDataset.trainY);
        
        long ensembleTime = System.currentTimeMillis() - ensembleStart;
        
        // Get individual predictions
        double[] deepPred = deepMLP.predict(complexDataset.testX);
        double[] widePred = wideMLP.predict(complexDataset.testX);
        double[] regPred = regMLP.predict(complexDataset.testX);
        
        // Ensemble prediction (majority voting)
        double[] ensemblePred = ensemblePredict(deepPred, widePred, regPred);
        
        // Calculate accuracies
        double deepAcc = calculateAccuracy(deepPred, complexDataset.testY);
        double wideAcc = calculateAccuracy(widePred, complexDataset.testY);
        double regAcc = calculateAccuracy(regPred, complexDataset.testY);
        double ensembleAcc = calculateAccuracy(ensemblePred, complexDataset.testY);
        
        System.out.printf("Ensemble training time: %d ms%n", ensembleTime);
        System.out.printf("Deep MLP accuracy: %.3f%n", deepAcc);
        System.out.printf("Wide MLP accuracy: %.3f%n", wideAcc);
        System.out.printf("Regularized MLP accuracy: %.3f%n", regAcc);
        System.out.printf("Ensemble accuracy: %.3f%n", ensembleAcc);
        
        System.out.println();
    }
    
    /**
     * Example 5: Transfer learning and fine-tuning simulation
     */
    public static void runTransferLearningExample() {
        System.out.println("5. Transfer Learning Example");
        System.out.println("============================");
        
        // Generate base dataset (source domain)
        DatasetResult baseDataset = generateTabularDataset(1000, 8, 3);
        
        // Generate target dataset (smaller, related domain)
        DatasetResult targetDataset = generateTabularDataset(100, 8, 3);
        
        // Pre-train on large dataset
        System.out.println("Pre-training on source domain...");
        MLPClassifier preTrainedModel = new MLPClassifier()
            .setHiddenLayerSizes(new int[]{128, 64, 32})
            .setActivation("relu")
            .setSolver("adam")
            .setLearningRate(0.001)
            .setMaxEpochs(100)
            .setBatchSize(32)
            .setAlpha(0.0001)
            .setEarlyStopping(true);
        
        long preTrainStart = System.currentTimeMillis();
        preTrainedModel.fit(baseDataset.trainX, baseDataset.trainY);
        long preTrainTime = System.currentTimeMillis() - preTrainStart;
        
        // Fine-tune on target dataset (simulated by training with lower learning rate)
        System.out.println("Fine-tuning on target domain...");
        MLPClassifier fineTunedModel = new MLPClassifier()
            .setHiddenLayerSizes(new int[]{128, 64, 32})
            .setActivation("relu")
            .setSolver("adam")
            .setLearningRate(0.0001) // Lower learning rate for fine-tuning
            .setMaxEpochs(50)
            .setBatchSize(16)
            .setAlpha(0.001)
            .setEarlyStopping(true);
        
        long fineTuneStart = System.currentTimeMillis();
        fineTunedModel.fit(targetDataset.trainX, targetDataset.trainY);
        long fineTuneTime = System.currentTimeMillis() - fineTuneStart;
        
        // Train from scratch on target dataset for comparison
        System.out.println("Training from scratch on target domain...");
        MLPClassifier scratchModel = new MLPClassifier()
            .setHiddenLayerSizes(new int[]{128, 64, 32})
            .setActivation("relu")
            .setSolver("adam")
            .setLearningRate(0.001)
            .setMaxEpochs(100)
            .setBatchSize(16)
            .setAlpha(0.0001)
            .setEarlyStopping(true);
        
        long scratchStart = System.currentTimeMillis();
        scratchModel.fit(targetDataset.trainX, targetDataset.trainY);
        long scratchTime = System.currentTimeMillis() - scratchStart;
        
        // Compare results
        double preTrainAcc = calculateAccuracy(
            preTrainedModel.predict(baseDataset.testX), baseDataset.testY);
        double fineTuneAcc = calculateAccuracy(
            fineTunedModel.predict(targetDataset.testX), targetDataset.testY);
        double scratchAcc = calculateAccuracy(
            scratchModel.predict(targetDataset.testX), targetDataset.testY);
        
        System.out.printf("Pre-training time: %d ms%n", preTrainTime);
        System.out.printf("Fine-tuning time: %d ms%n", fineTuneTime);
        System.out.printf("From-scratch time: %d ms%n", scratchTime);
        System.out.printf("Pre-trained model accuracy (source): %.3f%n", preTrainAcc);
        System.out.printf("Fine-tuned model accuracy (target): %.3f%n", fineTuneAcc);
        System.out.printf("From-scratch model accuracy (target): %.3f%n", scratchAcc);
        
        System.out.println();
    }
    
    // Utility methods for data generation and evaluation
    
    private static DatasetResult generateTabularDataset(int numSamples, int numFeatures, int numClasses) {
        Random random = new Random(42);
        
        double[][] X = new double[numSamples][numFeatures];
        double[] y = new double[numSamples];
        
        for (int i = 0; i < numSamples; i++) {
            int classLabel = i % numClasses;
            
            for (int j = 0; j < numFeatures; j++) {
                X[i][j] = random.nextGaussian() + classLabel * 2.0; // Class separation
            }
            
            y[i] = classLabel;
        }
        
        return splitDataset(X, y, 0.8);
    }
    
    private static ImageDatasetResult generateImageDataset(int numSamples, int height, int width, 
                                                         int channels, int numClasses) {
        Random random = new Random(42);
        int imageSize = height * width * channels;
        
        double[][] X = new double[numSamples][imageSize];
        double[] y = new double[numSamples];
        
        for (int i = 0; i < numSamples; i++) {
            int classLabel = i % numClasses;
            
            // Generate base noise
            for (int j = 0; j < imageSize; j++) {
                X[i][j] = random.nextGaussian() * 0.1;
            }
            
            // Add class-specific patterns
            addImagePattern(X[i], height, width, classLabel);
            y[i] = classLabel;
        }
        
        DatasetResult split = splitDataset(X, y, 0.8);
        return new ImageDatasetResult(split.trainX, split.trainY, split.testX, split.testY);
    }
    
    private static void addImagePattern(double[] image, int height, int width, int classLabel) {
        switch (classLabel) {
            case 0: // Horizontal lines
                for (int row = height / 4; row < 3 * height / 4; row += 4) {
                    for (int col = 0; col < width; col++) {
                        image[row * width + col] += 0.8;
                    }
                }
                break;
            case 1: // Vertical lines
                for (int col = width / 4; col < 3 * width / 4; col += 4) {
                    for (int row = 0; row < height; row++) {
                        image[row * width + col] += 0.8;
                    }
                }
                break;
            case 2: // Diagonal pattern
                for (int i = 0; i < Math.min(height, width); i++) {
                    image[i * width + i] += 0.8;
                    if (i < width - 1) {
                        image[i * width + (width - 1 - i)] += 0.8;
                    }
                }
                break;
        }
    }
    
    private static SequenceDatasetResult generateSequenceDataset(int numSamples, int seqLength, 
                                                               int vocabSize, int numClasses) {
        Random random = new Random(42);
        
        double[][] X = new double[numSamples][seqLength * vocabSize];
        double[] y = new double[numSamples];
        
        for (int i = 0; i < numSamples; i++) {
            int classLabel = i % numClasses;
            
            // Create one-hot encoded sequence
            for (int t = 0; t < seqLength; t++) {
                int wordIdx;
                
                // Class-specific word distributions
                if (classLabel == 0) {
                    wordIdx = random.nextInt(vocabSize / 2); // First half of vocabulary
                } else {
                    wordIdx = vocabSize / 2 + random.nextInt(vocabSize / 2); // Second half
                }
                
                X[i][t * vocabSize + wordIdx] = 1.0;
            }
            
            y[i] = classLabel;
        }
        
        DatasetResult split = splitDataset(X, y, 0.8);
        return new SequenceDatasetResult(split.trainX, split.trainY, split.testX, split.testY);
    }
    
    private static DatasetResult generateComplexDataset(int numSamples, int numFeatures, int numClasses) {
        Random random = new Random(42);
        
        double[][] X = new double[numSamples][numFeatures];
        double[] y = new double[numSamples];
        
        for (int i = 0; i < numSamples; i++) {
            int classLabel = random.nextInt(numClasses);
            
            // Create non-linear relationships
            for (int j = 0; j < numFeatures; j++) {
                double base = random.nextGaussian();
                
                // Add non-linear transformations based on class
                switch (classLabel) {
                    case 0:
                        X[i][j] = base + Math.sin(base * 2);
                        break;
                    case 1:
                        X[i][j] = base + Math.cos(base * 2);
                        break;
                    case 2:
                        X[i][j] = base + base * base * 0.5;
                        break;
                    case 3:
                        X[i][j] = base + Math.log(Math.abs(base) + 1);
                        break;
                }
            }
            
            y[i] = classLabel;
        }
        
        return splitDataset(X, y, 0.8);
    }
    
    private static DatasetResult splitDataset(double[][] X, double[] y, double trainRatio) {
        int trainSize = (int) (X.length * trainRatio);
        
        double[][] trainX = Arrays.copyOfRange(X, 0, trainSize);
        double[] trainY = Arrays.copyOfRange(y, 0, trainSize);
        double[][] testX = Arrays.copyOfRange(X, trainSize, X.length);
        double[] testY = Arrays.copyOfRange(y, trainSize, y.length);
        
        return new DatasetResult(trainX, trainY, testX, testY);
    }
    
    private static double calculateAccuracy(double[] predictions, double[] actual) {
        int correct = 0;
        for (int i = 0; i < predictions.length; i++) {
            if (predictions[i] == actual[i]) {
                correct++;
            }
        }
        return (double) correct / predictions.length;
    }
    
    private static double getMaxProbability(double[] probabilities) {
        double max = probabilities[0];
        for (int i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > max) {
                max = probabilities[i];
            }
        }
        return max;
    }
    
    private static double[] ensemblePredict(double[]... predictions) {
        int numSamples = predictions[0].length;
        int numModels = predictions.length;
        double[] ensemble = new double[numSamples];
        
        for (int i = 0; i < numSamples; i++) {
            // Majority voting
            double sum = 0;
            for (int j = 0; j < numModels; j++) {
                sum += predictions[j][i];
            }
            ensemble[i] = Math.round(sum / numModels);
        }
        
        return ensemble;
    }
    
    // Data classes
    private static class DatasetResult {
        final double[][] trainX, testX;
        final double[] trainY, testY;
        
        DatasetResult(double[][] trainX, double[] trainY, double[][] testX, double[] testY) {
            this.trainX = trainX;
            this.trainY = trainY;
            this.testX = testX;
            this.testY = testY;
        }
    }
    
    private static class ImageDatasetResult extends DatasetResult {
        ImageDatasetResult(double[][] trainX, double[] trainY, double[][] testX, double[] testY) {
            super(trainX, trainY, testX, testY);
        }
    }
    
    private static class SequenceDatasetResult extends DatasetResult {
        SequenceDatasetResult(double[][] trainX, double[] trainY, double[][] testX, double[] testY) {
            super(trainX, trainY, testX, testY);
        }
    }
}
