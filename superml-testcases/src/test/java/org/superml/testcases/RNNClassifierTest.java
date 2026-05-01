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
 * Comprehensive test suite for RNNClassifier
 * 
 * @author SuperML Team
 * @version 2.1.0
 */
public class RNNClassifierTest {
    
    private RNNClassifier classifier;
    private double[][] trainX;
    private double[] trainY;
    private double[][] testX;
    private double[] testY;
    
    @BeforeEach
    void setUp() {
        classifier = new RNNClassifier();
        setupTestData();
    }
    
    private void setupTestData() {
        // Create synthetic sequence data for sentiment analysis-like task
        int sequenceLength = 10;
        int vocabSize = 50;
        int numSamples = 200;
        
        Random random = new Random(42);
        
        // Flatten sequences for input (each sample is a flattened sequence)
        trainX = new double[numSamples][sequenceLength * vocabSize];
        trainY = new double[numSamples];
        
        for (int i = 0; i < numSamples; i++) {
            // Create one-hot encoded sequences
            double[] sequence = new double[sequenceLength * vocabSize];
            
            // Generate sequence with pattern-based sentiment
            double positiveScore = 0.0;
            
            for (int t = 0; t < sequenceLength; t++) {
                int wordIdx = random.nextInt(vocabSize);
                
                // Create sentiment patterns
                if (wordIdx < 15) { // Positive words
                    positiveScore += 1.0;
                } else if (wordIdx >= 35) { // Negative words
                    positiveScore -= 1.0;
                }
                
                // One-hot encoding
                sequence[t * vocabSize + wordIdx] = 1.0;
            }
            
            trainX[i] = sequence;
            trainY[i] = positiveScore > 0 ? 1.0 : 0.0; // Binary sentiment
        }
        
        // Create test data
        int testSize = 40;
        testX = new double[testSize][sequenceLength * vocabSize];
        testY = new double[testSize];
        
        for (int i = 0; i < testSize; i++) {
            System.arraycopy(trainX[i], 0, testX[i], 0, trainX[i].length);
            testY[i] = trainY[i];
        }
    }
    
    @Test
    @DisplayName("Test RNN constructor and default parameters")
    void testConstructorAndDefaults() {
        RNNClassifier rnn = new RNNClassifier();
        assertNotNull(rnn);
        
        RNNClassifier customRnn = new RNNClassifier(64, 2, "LSTM");
        assertNotNull(customRnn);
        assertEquals(64, customRnn.getHiddenSize());
        assertEquals(2, customRnn.getNumLayers());
        assertEquals("LSTM", customRnn.getCellType());
    }
    
    @Test
    @DisplayName("Test RNN fit method")
    void testFit() {
        assertDoesNotThrow(() -> {
            classifier.setHiddenSize(32)
                     .setNumLayers(1)
                     .setMaxEpochs(10)
                     .setLearningRate(0.01)
                     .setBatchSize(16)
                     .fit(trainX, trainY);
        });
    }
    
    @Test
    @DisplayName("Test RNN predict method")
    void testPredict() {
        classifier.setHiddenSize(32)
                 .setNumLayers(1)
                 .setMaxEpochs(15)
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
    @DisplayName("Test RNN predict_proba method")
    void testPredictProba() {
        classifier.setHiddenSize(32)
                 .setNumLayers(1)
                 .setMaxEpochs(15)
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
    @DisplayName("Test different RNN cell types")
    void testCellTypes() {
        String[] cellTypes = {"LSTM", "GRU", "SimpleRNN"};
        
        for (String cellType : cellTypes) {
            RNNClassifier rnn = new RNNClassifier()
                .setCellType(cellType)
                .setHiddenSize(16)
                .setNumLayers(1)
                .setMaxEpochs(5)
                .setLearningRate(0.01)
                .setBatchSize(16);
            
            assertDoesNotThrow(() -> rnn.fit(trainX, trainY), 
                "Failed for cell type: " + cellType);
            
            double[] predictions = rnn.predict(testX);
            assertNotNull(predictions);
            assertEquals(testX.length, predictions.length);
        }
    }
    
    @Test
    @DisplayName("Test bidirectional RNN")
    void testBidirectionalRNN() {
        classifier.setCellType("LSTM")
                 .setHiddenSize(16)
                 .setNumLayers(1)
                 .setBidirectional(true)
                 .setMaxEpochs(10)
                 .setLearningRate(0.01)
                 .setBatchSize(16)
                 .fit(trainX, trainY);
        
        assertTrue(classifier.isBidirectional());
        
        double[] predictions = classifier.predict(testX);
        assertNotNull(predictions);
        assertEquals(testX.length, predictions.length);
    }
    
    @Test
    @DisplayName("Test multi-layer RNN")
    void testMultiLayerRNN() {
        classifier.setCellType("LSTM")
                 .setHiddenSize(24)
                 .setNumLayers(3)
                 .setMaxEpochs(10)
                 .setLearningRate(0.01)
                 .setBatchSize(16)
                 .fit(trainX, trainY);
        
        assertEquals(3, classifier.getNumLayers());
        
        double[] predictions = classifier.predict(testX);
        assertNotNull(predictions);
        assertEquals(testX.length, predictions.length);
    }
    
    @Test
    @DisplayName("Test RNN with attention mechanism")
    void testAttentionMechanism() {
        classifier.setCellType("LSTM")
                 .setHiddenSize(32)
                 .setNumLayers(2)
                 .setUseAttention(true)
                 .setReturnSequences(true)
                 .setMaxEpochs(10)
                 .setLearningRate(0.01)
                 .setBatchSize(16)
                 .fit(trainX, trainY);
        
        assertTrue(classifier.isUseAttention());
        
        double[] predictions = classifier.predict(testX);
        assertNotNull(predictions);
        assertEquals(testX.length, predictions.length);
    }
    
    @Test
    @DisplayName("Test RNN with dropout")
    void testDropoutRegularization() {
        classifier.setCellType("LSTM")
                 .setHiddenSize(32)
                 .setNumLayers(2)
                 .setDropoutRate(0.3)
                 .setRecurrentDropout(0.2)
                 .setMaxEpochs(10)
                 .setLearningRate(0.01)
                 .setBatchSize(16)
                 .fit(trainX, trainY);
        
        double[] predictions = classifier.predict(testX);
        assertNotNull(predictions);
        assertEquals(testX.length, predictions.length);
    }
    
    @Test
    @DisplayName("Test pre-defined RNN architectures")
    void testPreDefinedArchitectures() {
        // Test LSTM architecture
        RNNClassifier lstmRnn = new RNNClassifier()
            .useLSTMArchitecture()
            .setMaxEpochs(5)
            .setLearningRate(0.01)
            .setBatchSize(16);
        
        assertDoesNotThrow(() -> lstmRnn.fit(trainX, trainY));
        
        // Test GRU architecture
        RNNClassifier gruRnn = new RNNClassifier()
            .useGRUArchitecture()
            .setMaxEpochs(5)
            .setLearningRate(0.01)
            .setBatchSize(16);
        
        assertDoesNotThrow(() -> gruRnn.fit(trainX, trainY));
        
        // Test Attention LSTM architecture
        RNNClassifier attentionRnn = new RNNClassifier()
            .useAttentionLSTM()
            .setMaxEpochs(5)
            .setLearningRate(0.01)
            .setBatchSize(16);
        
        assertDoesNotThrow(() -> attentionRnn.fit(trainX, trainY));
    }
    
    @Test
    @DisplayName("Test RNN configuration builder pattern")
    void testBuilderPattern() {
        RNNClassifier rnn = new RNNClassifier()
            .setHiddenSize(64)
            .setNumLayers(2)
            .setCellType("LSTM")
            .setBidirectional(true)
            .setReturnSequences(true)
            .setUseAttention(true)
            .setLearningRate(0.001)
            .setMaxEpochs(20)
            .setBatchSize(32)
            .setDropoutRate(0.2)
            .setRecurrentDropout(0.1);
        
        assertNotNull(rnn);
        assertEquals(64, rnn.getHiddenSize());
        assertEquals(2, rnn.getNumLayers());
        assertEquals("LSTM", rnn.getCellType());
        assertTrue(rnn.isBidirectional());
        assertTrue(rnn.isUseAttention());
        
        assertDoesNotThrow(() -> rnn.fit(trainX, trainY));
    }
    
    @Test
    @DisplayName("Test RNN loss history tracking")
    void testLossHistoryTracking() {
        classifier.setHiddenSize(24)
                 .setNumLayers(1)
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
            assertTrue(lastLoss <= firstLoss * 1.1, "Loss should not increase significantly");
        }
    }
    
    @Test
    @DisplayName("Test multi-class RNN classification")
    void testMultiClassClassification() {
        // Create 3-class synthetic sequence data
        int numClasses = 3;
        int samplesPerClass = 40;
        int totalSamples = numClasses * samplesPerClass;
        int sequenceLength = 8;
        int vocabSize = 30;
        
        double[][] multiX = new double[totalSamples][sequenceLength * vocabSize];
        double[] multiY = new double[totalSamples];
        
        Random random = new Random(42);
        
        for (int cls = 0; cls < numClasses; cls++) {
            for (int i = 0; i < samplesPerClass; i++) {
                int idx = cls * samplesPerClass + i;
                
                double[] sequence = new double[sequenceLength * vocabSize];
                
                for (int t = 0; t < sequenceLength; t++) {
                    // Class-specific word distribution
                    int wordIdx;
                    switch (cls) {
                        case 0:
                            wordIdx = random.nextInt(10); // First 10 words
                            break;
                        case 1:
                            wordIdx = 10 + random.nextInt(10); // Middle 10 words
                            break;
                        case 2:
                            wordIdx = 20 + random.nextInt(10); // Last 10 words
                            break;
                        default:
                            wordIdx = random.nextInt(vocabSize);
                    }
                    
                    sequence[t * vocabSize + wordIdx] = 1.0;
                }
                
                multiX[idx] = sequence;
                multiY[idx] = cls;
            }
        }
        
        RNNClassifier multiClassifier = new RNNClassifier()
            .setCellType("LSTM")
            .setHiddenSize(32)
            .setNumLayers(2)
            .setBidirectional(true)
            .setMaxEpochs(15)
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
    @DisplayName("Test RNN with time series data")
    void testTimeSeriesClassification() {
        // Create synthetic time series data
        int sequenceLength = 20;
        int numFeatures = 5;
        int numSamples = 150;
        
        double[][] timeSeriesX = new double[numSamples][sequenceLength * numFeatures];
        double[] timeSeriesY = new double[numSamples];
        
        Random random = new Random(42);
        
        for (int i = 0; i < numSamples; i++) {
            double trend = random.nextGaussian();
            
            for (int t = 0; t < sequenceLength; t++) {
                for (int f = 0; f < numFeatures; f++) {
                    // Create trending time series
                    double value = trend * t / sequenceLength + random.nextGaussian() * 0.1;
                    timeSeriesX[i][t * numFeatures + f] = value;
                }
            }
            
            // Classification based on trend
            timeSeriesY[i] = trend > 0 ? 1.0 : 0.0;
        }
        
        RNNClassifier timeSeriesClassifier = new RNNClassifier()
            .setCellType("LSTM")
            .setHiddenSize(32)
            .setNumLayers(1)
            .setBidirectional(true)
            .setMaxEpochs(20)
            .setLearningRate(0.01)
            .setBatchSize(16);
        
        assertDoesNotThrow(() -> timeSeriesClassifier.fit(timeSeriesX, timeSeriesY));
        
        double[] predictions = timeSeriesClassifier.predict(timeSeriesX);
        assertNotNull(predictions);
        assertEquals(timeSeriesX.length, predictions.length);
        
        // Calculate accuracy
        double accuracy = calculateAccuracy(predictions, timeSeriesY);
        System.out.printf("Time series classification accuracy: %.3f%n", accuracy);
        assertTrue(accuracy > 0.6, "Time series accuracy should be reasonable: " + accuracy);
    }
    
    @Test
    @DisplayName("Test RNN invalid input handling")
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
            new RNNClassifier().predict(testX);
        });
        
        // Test invalid cell type
        assertThrows(IllegalArgumentException.class, () -> {
            new RNNClassifier().setCellType("INVALID").fit(trainX, trainY);
        });
    }
    
    @Test
    @DisplayName("Test RNN performance on longer sequences")
    void testLongerSequences() {
        // Create longer sequence data
        int longSeqLength = 30;
        int numFeatures = 10;
        int numSamples = 100;
        
        double[][] longSeqX = new double[numSamples][longSeqLength * numFeatures];
        double[] longSeqY = new double[numSamples];
        
        Random random = new Random(42);
        
        for (int i = 0; i < numSamples; i++) {
            double pattern = random.nextGaussian();
            
            for (int t = 0; t < longSeqLength; t++) {
                for (int f = 0; f < numFeatures; f++) {
                    // Create pattern in later part of sequence
                    double value = (t > longSeqLength / 2) ? pattern + random.nextGaussian() * 0.1 
                                                          : random.nextGaussian() * 0.1;
                    longSeqX[i][t * numFeatures + f] = value;
                }
            }
            
            longSeqY[i] = pattern > 0 ? 1.0 : 0.0;
        }
        
        RNNClassifier longSeqClassifier = new RNNClassifier()
            .setCellType("LSTM")
            .setHiddenSize(48)
            .setNumLayers(2)
            .setBidirectional(true)
            .setUseAttention(true) // Attention helps with longer sequences
            .setReturnSequences(true)
            .setMaxEpochs(15)
            .setLearningRate(0.001)
            .setBatchSize(8);
        
        long startTime = System.currentTimeMillis();
        assertDoesNotThrow(() -> longSeqClassifier.fit(longSeqX, longSeqY));
        long endTime = System.currentTimeMillis();
        
        System.out.printf("Long sequence RNN training time: %d ms%n", endTime - startTime);
        
        double[] predictions = longSeqClassifier.predict(longSeqX);
        double accuracy = calculateAccuracy(predictions, longSeqY);
        
        System.out.printf("Long sequence RNN accuracy: %.3f%n", accuracy);
        assertTrue(accuracy > 0.5, "Long sequence accuracy should be better than random: " + accuracy);
    }
    
    @Test
    @DisplayName("Test RNN memory efficiency with batching")
    void testMemoryEfficiency() {
        // Test with very small batches to ensure memory efficiency
        RNNClassifier memoryEfficientRnn = new RNNClassifier()
            .setCellType("LSTM")
            .setHiddenSize(16)
            .setNumLayers(1)
            .setBatchSize(2) // Very small batch size
            .setMaxEpochs(8)
            .setLearningRate(0.01);
        
        assertDoesNotThrow(() -> memoryEfficientRnn.fit(trainX, trainY));
        
        double[] predictions = memoryEfficientRnn.predict(testX);
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
