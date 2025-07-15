package org.superml.neural;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;
import java.util.List;
import java.util.Set;
import java.util.HashSet;
import java.util.Random;

/**
 * Comprehensive test suite for CNNClassifier
 * 
 * @author SuperML Team
 * @version 2.0.0
 */
public class CNNClassifierTest {
    
    private CNNClassifier classifier;
    private double[][] trainX;
    private double[] trainY;
    private double[][] testX;
    private double[] testY;
    
    @BeforeEach
    void setUp() {
        classifier = new CNNClassifier(28, 28, 1); // MNIST-like dimensions
        setupTestData();
    }
    
    private void setupTestData() {
        // Create synthetic image data (flattened)
        int imageSize = 28 * 28; // 784 pixels
        int numSamples = 100;
        
        trainX = new double[numSamples][imageSize];
        trainY = new double[numSamples];
        
        Random random = new Random(42);
        
        // Generate synthetic image patterns
        for (int i = 0; i < numSamples; i++) {
            // Create simple patterns for classification
            for (int j = 0; j < imageSize; j++) {
                trainX[i][j] = random.nextGaussian() * 0.1;
            }
            
            // Add pattern-specific features
            if (i % 2 == 0) {
                // Pattern 0: bright top-left quadrant
                for (int row = 0; row < 14; row++) {
                    for (int col = 0; col < 14; col++) {
                        trainX[i][row * 28 + col] += 0.8;
                    }
                }
                trainY[i] = 0.0;
            } else {
                // Pattern 1: bright bottom-right quadrant
                for (int row = 14; row < 28; row++) {
                    for (int col = 14; col < 28; col++) {
                        trainX[i][row * 28 + col] += 0.8;
                    }
                }
                trainY[i] = 1.0;
            }
        }
        
        // Create test data
        testX = new double[20][imageSize];
        testY = new double[20];
        for (int i = 0; i < 20; i++) {
            System.arraycopy(trainX[i], 0, testX[i], 0, imageSize);
            testY[i] = trainY[i];
        }
    }
    
    @Test
    @DisplayName("Test CNN constructor and input shape")
    void testConstructorAndInputShape() {
        CNNClassifier cnn = new CNNClassifier(32, 32, 3);
        assertNotNull(cnn);
        assertArrayEquals(new int[]{32, 32, 3}, cnn.getInputShape());
        
        CNNClassifier defaultCnn = new CNNClassifier();
        assertNotNull(defaultCnn);
    }
    
    @Test
    @DisplayName("Test CNN fit method")
    void testFit() {
        assertDoesNotThrow(() -> {
            classifier.setMaxEpochs(5) // Reduced for testing
                     .setLearningRate(0.01)
                     .setBatchSize(16)
                     .fit(trainX, trainY);
        });
    }
    
    @Test
    @DisplayName("Test CNN predict method")
    void testPredict() {
        classifier.setMaxEpochs(10)
                 .setLearningRate(0.01)
                 .setBatchSize(16)
                 .fit(trainX, trainY);
        
        double[] predictions = classifier.predict(testX);
        assertNotNull(predictions);
        assertEquals(testX.length, predictions.length);
        
        // Check that predictions are valid class labels
        for (double prediction : predictions) {
            assertTrue(prediction == 0.0 || prediction == 1.0);
        }
    }
    
    @Test
    @DisplayName("Test CNN predict_proba method")
    void testPredictProba() {
        classifier.setMaxEpochs(10)
                 .setLearningRate(0.01)
                 .setBatchSize(16)
                 .fit(trainX, trainY);
        
        double[][] probabilities = classifier.predictProba(testX);
        assertNotNull(probabilities);
        assertEquals(testX.length, probabilities.length);
        assertEquals(2, probabilities[0].length); // Binary classification
        
        // Check probability constraints
        for (double[] probs : probabilities) {
            double sum = 0.0;
            for (double prob : probs) {
                assertTrue(prob >= 0.0 && prob <= 1.0, "Probability out of range");
                sum += prob;
            }
            assertEquals(1.0, sum, 0.001, "Probabilities should sum to 1");
        }
    }
    
    @Test
    @DisplayName("Test CNN layer builder pattern")
    void testLayerBuilderPattern() {
        CNNClassifier cnn = new CNNClassifier(32, 32, 3)
            .addConvolutionalLayer(32, 3, 3, 1, "relu")
            .addBatchNormalization()
            .addMaxPooling(2, 2)
            .addDropout(0.25)
            .addConvolutionalLayer(64, 3, 3, 1, "relu")
            .addBatchNormalization()
            .addMaxPooling(2, 2)
            .addDropout(0.25)
            .addFlatten()
            .addDenseLayer(128, "relu")
            .addDropout(0.5)
            .addDenseLayer(2, "linear");
        
        assertNotNull(cnn);
        assertDoesNotThrow(() -> {
            cnn.setMaxEpochs(5)
               .setLearningRate(0.001)
               .setBatchSize(8)
               .fit(trainX, trainY);
        });
    }
    
    @Test
    @DisplayName("Test pre-defined CNN architectures")
    void testPreDefinedArchitectures() {
        // Test LeNet architecture
        CNNClassifier leNet = new CNNClassifier(28, 28, 1)
            .useLeNetArchitecture();
        
        assertNotNull(leNet);
        assertDoesNotThrow(() -> {
            leNet.setMaxEpochs(5)
                 .setLearningRate(0.01)
                 .setBatchSize(16)
                 .fit(trainX, trainY);
        });
        
        // Test AlexNet-style architecture
        CNNClassifier alexNet = new CNNClassifier(224, 224, 3)
            .useAlexNetStyleArchitecture();
        
        assertNotNull(alexNet);
    }
    
    @Test
    @DisplayName("Test different pooling layers")
    void testPoolingLayers() {
        CNNClassifier cnn = new CNNClassifier(28, 28, 1)
            .addConvolutionalLayer(16, 5, 5, 1, "relu")
            .addMaxPooling(2, 2)
            .addConvolutionalLayer(32, 3, 3, 1, "relu")
            .addAveragePooling(2, 2)
            .addGlobalAveragePooling()
            .addDenseLayer(2, "linear");
        
        assertNotNull(cnn);
        assertDoesNotThrow(() -> {
            cnn.setMaxEpochs(5)
               .setLearningRate(0.01)
               .setBatchSize(16)
               .fit(trainX, trainY);
        });
    }
    
    @Test
    @DisplayName("Test CNN with batch normalization")
    void testBatchNormalization() {
        classifier.addConvolutionalLayer(32, 3, 3, 1, "relu")
                 .addBatchNormalization()
                 .addMaxPooling(2, 2)
                 .addConvolutionalLayer(64, 3, 3, 1, "relu")
                 .addBatchNormalization()
                 .addMaxPooling(2, 2)
                 .addFlatten()
                 .addDenseLayer(128, "relu")
                 .addDenseLayer(2, "linear")
                 .setUseBatchNorm(true)
                 .setMaxEpochs(10)
                 .setLearningRate(0.001)
                 .setBatchSize(16);
        
        assertDoesNotThrow(() -> classifier.fit(trainX, trainY));
        
        double[] predictions = classifier.predict(testX);
        assertNotNull(predictions);
        assertEquals(testX.length, predictions.length);
    }
    
    @Test
    @DisplayName("Test CNN with dropout regularization")
    void testDropoutRegularization() {
        classifier.addConvolutionalLayer(32, 3, 3, 1, "relu")
                 .addMaxPooling(2, 2)
                 .addDropout(0.25)
                 .addConvolutionalLayer(64, 3, 3, 1, "relu")
                 .addMaxPooling(2, 2)
                 .addDropout(0.25)
                 .addFlatten()
                 .addDenseLayer(128, "relu")
                 .addDropout(0.5)
                 .addDenseLayer(2, "linear")
                 .setDropoutRate(0.5)
                 .setMaxEpochs(10)
                 .setLearningRate(0.001)
                 .setBatchSize(16);
        
        assertDoesNotThrow(() -> classifier.fit(trainX, trainY));
        
        double[] predictions = classifier.predict(testX);
        assertNotNull(predictions);
    }
    
    @Test
    @DisplayName("Test CNN configuration parameters")
    void testConfigurationParameters() {
        CNNClassifier cnn = new CNNClassifier()
            .setInputShape(64, 64, 3)
            .setLearningRate(0.0001)
            .setMaxEpochs(50)
            .setBatchSize(32)
            .setOptimizer("adam")
            .setRegularization(0.001)
            .setUseBatchNorm(true)
            .setDropoutRate(0.3);
        
        assertNotNull(cnn);
        assertArrayEquals(new int[]{64, 64, 3}, cnn.getInputShape());
    }
    
    @Test
    @DisplayName("Test CNN loss history tracking")
    void testLossHistoryTracking() {
        classifier.addConvolutionalLayer(16, 3, 3, 1, "relu")
                 .addMaxPooling(2, 2)
                 .addFlatten()
                 .addDenseLayer(32, "relu")
                 .addDenseLayer(2, "linear")
                 .setMaxEpochs(15)
                 .setLearningRate(0.01)
                 .setBatchSize(16)
                 .fit(trainX, trainY);
        
        List<Double> trainLoss = classifier.getTrainLossHistory();
        assertNotNull(trainLoss);
        assertTrue(trainLoss.size() > 0);
        
        // Loss should generally decrease
        if (trainLoss.size() > 1) {
            double firstLoss = trainLoss.get(0);
            double lastLoss = trainLoss.get(trainLoss.size() - 1);
            assertTrue(lastLoss <= firstLoss, "Loss should not increase significantly");
        }
    }
    
    @Test
    @DisplayName("Test multi-class CNN classification")
    void testMultiClassClassification() {
        // Create 3-class synthetic image data
        int numClasses = 3;
        int samplesPerClass = 30;
        int totalSamples = numClasses * samplesPerClass;
        int imageSize = 28 * 28;
        
        double[][] multiX = new double[totalSamples][imageSize];
        double[] multiY = new double[totalSamples];
        
        Random random = new Random(42);
        
        for (int cls = 0; cls < numClasses; cls++) {
            for (int i = 0; i < samplesPerClass; i++) {
                int idx = cls * samplesPerClass + i;
                
                // Base noise
                for (int j = 0; j < imageSize; j++) {
                    multiX[idx][j] = random.nextGaussian() * 0.1;
                }
                
                // Class-specific patterns
                switch (cls) {
                    case 0: // Top pattern
                        for (int row = 0; row < 10; row++) {
                            for (int col = 9; col < 19; col++) {
                                multiX[idx][row * 28 + col] += 0.8;
                            }
                        }
                        break;
                    case 1: // Middle pattern
                        for (int row = 9; row < 19; row++) {
                            for (int col = 9; col < 19; col++) {
                                multiX[idx][row * 28 + col] += 0.8;
                            }
                        }
                        break;
                    case 2: // Bottom pattern
                        for (int row = 18; row < 28; row++) {
                            for (int col = 9; col < 19; col++) {
                                multiX[idx][row * 28 + col] += 0.8;
                            }
                        }
                        break;
                }
                
                multiY[idx] = cls;
            }
        }
        
        CNNClassifier multiClassifier = new CNNClassifier(28, 28, 1)
            .addConvolutionalLayer(16, 5, 5, 1, "relu")
            .addMaxPooling(2, 2)
            .addConvolutionalLayer(32, 3, 3, 1, "relu")
            .addMaxPooling(2, 2)
            .addFlatten()
            .addDenseLayer(64, "relu")
            .addDenseLayer(numClasses, "linear")
            .setMaxEpochs(20)
            .setLearningRate(0.01)
            .setBatchSize(16);
        
        assertDoesNotThrow(() -> multiClassifier.fit(multiX, multiY));
        
        double[] predictions = multiClassifier.predict(multiX);
        assertNotNull(predictions);
        assertEquals(multiX.length, predictions.length);
        
        // Check that predictions contain expected class labels
        Set<Double> uniquePredictions = new HashSet<>();
        for (double pred : predictions) {
            uniquePredictions.add(pred);
        }
        assertTrue(uniquePredictions.size() <= numClasses);
    }
    
    @Test
    @DisplayName("Test CNN invalid input handling")
    void testInvalidInputHandling() {
        // Test empty data
        assertThrows(IllegalArgumentException.class, () -> {
            classifier.fit(new double[0][0], new double[0]);
        });
        
        // Test mismatched dimensions
        assertThrows(IllegalArgumentException.class, () -> {
            classifier.fit(trainX, new double[trainX.length + 1]);
        });
        
        // Test predict before fit
        assertThrows(IllegalStateException.class, () -> {
            new CNNClassifier().predict(testX);
        });
    }
    
    @Test
    @DisplayName("Test CNN performance metrics")
    void testPerformanceMetrics() {
        classifier.addConvolutionalLayer(16, 3, 3, 1, "relu")
                 .addMaxPooling(2, 2)
                 .addFlatten()
                 .addDenseLayer(32, "relu")
                 .addDenseLayer(2, "linear")
                 .setMaxEpochs(20)
                 .setLearningRate(0.01)
                 .setBatchSize(16);
        
        long startTime = System.currentTimeMillis();
        classifier.fit(trainX, trainY);
        long endTime = System.currentTimeMillis();
        
        System.out.printf("CNN training time: %d ms%n", endTime - startTime);
        
        double[] predictions = classifier.predict(testX);
        double accuracy = calculateAccuracy(predictions, testY);
        
        System.out.printf("CNN accuracy: %.3f%n", accuracy);
        assertTrue(accuracy > 0.5, "Accuracy should be better than random: " + accuracy);
    }
    
    @Test
    @DisplayName("Test CNN memory efficiency")
    void testMemoryEfficiency() {
        // Test with smaller batches to ensure memory efficiency
        CNNClassifier memoryEfficientCnn = new CNNClassifier(28, 28, 1)
            .addConvolutionalLayer(8, 3, 3, 1, "relu")
            .addMaxPooling(2, 2)
            .addConvolutionalLayer(16, 3, 3, 1, "relu")
            .addMaxPooling(2, 2)
            .addFlatten()
            .addDenseLayer(32, "relu")
            .addDenseLayer(2, "linear")
            .setBatchSize(4) // Very small batch size
            .setMaxEpochs(10)
            .setLearningRate(0.01);
        
        assertDoesNotThrow(() -> memoryEfficientCnn.fit(trainX, trainY));
        
        double[] predictions = memoryEfficientCnn.predict(testX);
        assertNotNull(predictions);
        assertEquals(testX.length, predictions.length);
    }
    
    private double calculateAccuracy(double[] predictions, double[] actual) {
        int correct = 0;
        for (int i = 0; i < predictions.length; i++) {
            if (predictions[i] == actual[i]) {
                correct++;
            }
        }
        return (double) correct / predictions.length;
    }
}
