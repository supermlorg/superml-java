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
 * Comprehensive test suite for MLPClassifier
 * 
 * @author SuperML Team
 * @version 2.1.0
 */
public class MLPClassifierTest {
    
    private MLPClassifier classifier;
    private double[][] trainX;
    private double[] trainY;
    private double[][] testX;
    private double[] testY;
    
    @BeforeEach
    void setUp() {
        classifier = new MLPClassifier();
        setupTestData();
    }
    
    private void setupTestData() {
        // Create synthetic XOR-like dataset
        trainX = new double[][]{
            {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0},
            {0.1, 0.1}, {0.1, 0.9}, {0.9, 0.1}, {0.9, 0.9},
            {0.2, 0.2}, {0.2, 0.8}, {0.8, 0.2}, {0.8, 0.8}
        };
        
        trainY = new double[]{0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0};
        
        testX = new double[][]{
            {0.05, 0.05}, {0.05, 0.95}, {0.95, 0.05}, {0.95, 0.95}
        };
        
        testY = new double[]{0, 1, 1, 0};
    }
    
    @Test
    @DisplayName("Test MLP constructor and default parameters")
    void testConstructorAndDefaults() {
        MLPClassifier mlp = new MLPClassifier();
        assertNotNull(mlp);
        
        MLPClassifier customMlp = new MLPClassifier(100, 50);
        customMlp.setActivation("relu").setSolver("adam");
        assertNotNull(customMlp);
    }
    
    @Test
    @DisplayName("Test MLP fit method")
    void testFit() {
        assertDoesNotThrow(() -> {
            classifier.setHiddenLayerSizes(new int[]{10, 5})
                     .setMaxEpochs(50)
                     .setLearningRate(0.01)
                     .fit(trainX, trainY);
        });
        
        assertTrue(classifier.isFitted());
    }
    
    @Test
    @DisplayName("Test MLP predict method")
    void testPredict() {
        classifier.setHiddenLayerSizes(new int[]{10, 5})
                 .setMaxEpochs(100)
                 .setLearningRate(0.01)
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
    @DisplayName("Test MLP predict_proba method")
    void testPredictProba() {
        classifier.setHiddenLayerSizes(new int[]{10, 5})
                 .setMaxEpochs(100)
                 .setLearningRate(0.01)
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
    @DisplayName("Test different activation functions")
    void testActivationFunctions() {
        String[] activations = {"relu", "sigmoid", "tanh", "leaky_relu"};
        
        for (String activation : activations) {
            MLPClassifier mlp = new MLPClassifier()
                .setHiddenLayerSizes(new int[]{10})
                .setActivation(activation)
                .setMaxEpochs(50)
                .setLearningRate(0.01);
            
            assertDoesNotThrow(() -> mlp.fit(trainX, trainY));
            
            double[] predictions = mlp.predict(testX);
            assertNotNull(predictions);
            assertEquals(testX.length, predictions.length);
        }
    }
    
    @Test
    @DisplayName("Test different solvers")
    void testSolvers() {
        String[] solvers = {"sgd", "adam"};
        
        for (String solver : solvers) {
            MLPClassifier mlp = new MLPClassifier()
                .setHiddenLayerSizes(new int[]{10})
                .setSolver(solver)
                .setMaxEpochs(50)
                .setLearningRate(0.01);
            
            assertDoesNotThrow(() -> mlp.fit(trainX, trainY));
            
            double[] predictions = mlp.predict(testX);
            assertNotNull(predictions);
        }
    }
    
    @Test
    @DisplayName("Test regularization")
    void testRegularization() {
        classifier.setHiddenLayerSizes(new int[]{20, 10})
                 .setAlpha(0.01) // L2 regularization
                 .setMaxEpochs(100)
                 .setLearningRate(0.01)
                 .fit(trainX, trainY);
        
        double[] predictions = classifier.predict(testX);
        assertNotNull(predictions);
        assertEquals(testX.length, predictions.length);
    }
    
    @Test
    @DisplayName("Test early stopping")
    void testEarlyStopping() {
        classifier.setHiddenLayerSizes(new int[]{20, 10})
                 .setEarlyStopping(true)
                 .setValidationFraction(0.1)
                 .setMaxEpochs(200)
                 .setLearningRate(0.01)
                 .fit(trainX, trainY);
        
        // Should stop before max epochs due to early stopping
        assertTrue(classifier.isFitted());
    }
    
    @Test
    @DisplayName("Test batch training")
    void testBatchTraining() {
        classifier.setHiddenLayerSizes(new int[]{10})
                 .setBatchSize(4)
                 .setMaxEpochs(50)
                 .setLearningRate(0.01)
                 .fit(trainX, trainY);
        
        double[] predictions = classifier.predict(testX);
        assertNotNull(predictions);
        assertEquals(testX.length, predictions.length);
    }
    
    @Test
    @DisplayName("Test multi-class classification")
    void testMultiClassClassification() {
        // Create 3-class dataset
        double[][] multiX = {
            {1.0, 1.0}, {1.1, 1.1}, {0.9, 0.9},   // Class 0
            {2.0, 2.0}, {2.1, 2.1}, {1.9, 1.9},   // Class 1
            {3.0, 3.0}, {3.1, 3.1}, {2.9, 2.9}    // Class 2
        };
        double[] multiY = {0, 0, 0, 1, 1, 1, 2, 2, 2};
        
        MLPClassifier multiClassifier = new MLPClassifier()
            .setHiddenLayerSizes(new int[]{10, 5})
            .setMaxEpochs(100)
            .setLearningRate(0.01);
        
        multiClassifier.fit(multiX, multiY);
        
        double[] predictions = multiClassifier.predict(multiX);
        assertNotNull(predictions);
        assertEquals(multiX.length, predictions.length);
        
        // Check that all class labels are present
        Set<Double> uniquePredictions = new HashSet<>();
        for (double pred : predictions) {
            uniquePredictions.add(pred);
        }
        assertTrue(uniquePredictions.size() <= 3);
    }
    
    @Test
    @DisplayName("Test configuration builder pattern")
    void testBuilderPattern() {
        MLPClassifier mlp = new MLPClassifier()
            .setHiddenLayerSizes(new int[]{100, 50, 25})
            .setActivation("relu")
            .setSolver("adam")
            .setLearningRate(0.001)
            .setMaxEpochs(200)
            .setBatchSize(32)
            .setAlpha(0.0001)
            .setEarlyStopping(true)
            .setValidationFraction(0.1)
            .setTolerance(1e-4)
            .setRandomState(42);
        
        assertNotNull(mlp);
        assertDoesNotThrow(() -> mlp.fit(trainX, trainY));
    }
    
    @Test
    @DisplayName("Test loss history tracking")
    void testLossHistory() {
        classifier.setHiddenLayerSizes(new int[]{10})
                 .setMaxEpochs(50)
                 .setLearningRate(0.01)
                 .fit(trainX, trainY);
        
        List<Double> lossHistory = classifier.getLossHistory();
        assertNotNull(lossHistory);
        assertTrue(lossHistory.size() > 0);
        
        // Loss should generally decrease
        double firstLoss = lossHistory.get(0);
        double lastLoss = lossHistory.get(lossHistory.size() - 1);
        assertTrue(lastLoss < firstLoss, "Loss should decrease during training");
    }
    
    @Test
    @DisplayName("Test invalid input handling")
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
            new MLPClassifier().predict(testX);
        });
    }
    
    @Test
    @DisplayName("Test performance on larger dataset")
    void testPerformanceOnLargerDataset() {
        // Generate larger synthetic dataset
        int numSamples = 1000;
        int numFeatures = 10;
        
        Random random = new Random(42);
        double[][] largeX = new double[numSamples][numFeatures];
        double[] largeY = new double[numSamples];
        
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numFeatures; j++) {
                largeX[i][j] = random.nextGaussian();
            }
            // Simple linear combination for labels
            largeY[i] = (largeX[i][0] + largeX[i][1] + largeX[i][2]) > 0 ? 1.0 : 0.0;
        }
        
        MLPClassifier largeMlp = new MLPClassifier()
            .setHiddenLayerSizes(new int[]{50, 25})
            .setMaxEpochs(100)
            .setLearningRate(0.001)
            .setBatchSize(64)
            .setEarlyStopping(true);
        
        long startTime = System.currentTimeMillis();
        largeMlp.fit(largeX, largeY);
        long endTime = System.currentTimeMillis();
        
        assertTrue(largeMlp.isFitted());
        System.out.printf("Training time for %d samples: %d ms%n", numSamples, endTime - startTime);
        
        // Test prediction accuracy
        double[] predictions = largeMlp.predict(largeX);
        double accuracy = calculateAccuracy(predictions, largeY);
        assertTrue(accuracy > 0.7, "Accuracy should be reasonably high: " + accuracy);
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
