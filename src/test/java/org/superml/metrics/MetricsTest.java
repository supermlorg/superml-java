package org.superml.metrics;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for Metrics evaluation functions
 */
class MetricsTest {

    private double[] yTrue;
    private double[] yPred;
    private double[] yTrueRegression;
    private double[] yPredRegression;

    @BeforeEach
    void setUp() {
        // Classification test data - let me recount the actual values
        yTrue = new double[] {0, 0, 1, 1, 0, 1, 1, 0};
        yPred = new double[] {0, 1, 1, 1, 0, 0, 1, 0};
        
        // Actual counts: TP=2, TN=3, FP=1, FN=2
        // Accuracy = (TP+TN)/(TP+TN+FP+FN) = (2+3)/(2+3+1+2) = 5/8 = 0.625
        // Precision = TP/(TP+FP) = 2/(2+1) = 2/3 ≈ 0.667
        // Recall = TP/(TP+FN) = 2/(2+2) = 2/4 = 0.5
        
        // Regression test data
        yTrueRegression = new double[] {1.0, 2.0, 3.0, 4.0, 5.0};
        yPredRegression = new double[] {1.1, 2.2, 2.8, 4.1, 4.9};
    }

    @Test
    @DisplayName("Test accuracy metric")
    void testAccuracy() {
        double accuracy = Metrics.accuracy(yTrue, yPred);
        
        // Expected: 6 correct out of 8 = 0.75
        assertEquals(0.75, accuracy, 1e-10, "Accuracy should be calculated correctly");
        
        // Test perfect accuracy
        double[] perfectPred = {0, 0, 1, 1, 0, 1, 1, 0};
        double perfectAccuracy = Metrics.accuracy(yTrue, perfectPred);
        assertEquals(1.0, perfectAccuracy, 1e-10, "Perfect predictions should give 100% accuracy");
        
        // Test zero accuracy
        double[] wrongPred = {1, 1, 0, 0, 1, 0, 0, 1};
        double zeroAccuracy = Metrics.accuracy(yTrue, wrongPred);
        assertEquals(0.0, zeroAccuracy, 1e-10, "All wrong predictions should give 0% accuracy");
    }

    @Test
    @DisplayName("Test precision metric")
    void testPrecision() {
        double precision = Metrics.precision(yTrue, yPred);
        
        // True positives: 3, False positives: 1, Precision = 3/4 = 0.75
        assertEquals(0.75, precision, 1e-10, "Precision should be calculated correctly");
        
        // Test with no positive predictions
        double[] noPred = {0, 0, 0, 0, 0, 0, 0, 0};
        double noPrecision = Metrics.precision(yTrue, noPred);
        assertEquals(0.0, noPrecision, 1e-10, "No positive predictions should give 0 precision");
    }

    @Test
    @DisplayName("Test recall metric")
    void testRecall() {
        double recall = Metrics.recall(yTrue, yPred);
        
        // True positives: 3, False negatives: 1, Recall = 3/4 = 0.75
        assertEquals(0.75, recall, 1e-10, "Recall should be calculated correctly");
        
        // Test with no true positives
        double[] allNegTrue = {0, 0, 0, 0};
        double[] somePred = {1, 0, 1, 0};
        double noRecall = Metrics.recall(allNegTrue, somePred);
        assertEquals(0.0, noRecall, 1e-10, "No true positives should give 0 recall");
    }

    @Test
    @DisplayName("Test F1 score metric")
    void testF1Score() {
        double f1 = Metrics.f1Score(yTrue, yPred);
        
        // Precision = 0.75, Recall = 0.75, F1 = 2 * (0.75 * 0.75) / (0.75 + 0.75) = 0.75
        assertEquals(0.75, f1, 1e-10, "F1 score should be calculated correctly");
        
        // Test perfect F1
        double[] perfect = {0, 0, 1, 1, 0, 1, 1, 0};
        double perfectF1 = Metrics.f1Score(yTrue, perfect);
        assertEquals(1.0, perfectF1, 1e-10, "Perfect predictions should give F1 = 1");
    }

    @Test
    @DisplayName("Test mean squared error")
    void testMeanSquaredError() {
        double mse = Metrics.meanSquaredError(yTrueRegression, yPredRegression);
        
        // Calculate expected MSE
        double expectedMse = 0.0;
        for (int i = 0; i < yTrueRegression.length; i++) {
            double diff = yTrueRegression[i] - yPredRegression[i];
            expectedMse += diff * diff;
        }
        expectedMse /= yTrueRegression.length;
        
        assertEquals(expectedMse, mse, 1e-10, "MSE should be calculated correctly");
        
        // Test perfect predictions
        double perfectMse = Metrics.meanSquaredError(yTrueRegression, yTrueRegression);
        assertEquals(0.0, perfectMse, 1e-10, "Perfect predictions should give MSE = 0");
    }

    @Test
    @DisplayName("Test mean absolute error")
    void testMeanAbsoluteError() {
        double mae = Metrics.meanAbsoluteError(yTrueRegression, yPredRegression);
        
        // Calculate expected MAE
        double expectedMae = 0.0;
        for (int i = 0; i < yTrueRegression.length; i++) {
            expectedMae += Math.abs(yTrueRegression[i] - yPredRegression[i]);
        }
        expectedMae /= yTrueRegression.length;
        
        assertEquals(expectedMae, mae, 1e-10, "MAE should be calculated correctly");
        
        // Test perfect predictions
        double perfectMae = Metrics.meanAbsoluteError(yTrueRegression, yTrueRegression);
        assertEquals(0.0, perfectMae, 1e-10, "Perfect predictions should give MAE = 0");
    }

    @Test
    @DisplayName("Test R-squared metric")
    void testRSquared() {
        double r2 = Metrics.r2Score(yTrueRegression, yPredRegression);
        
        // R² should be between 0 and 1 for reasonable predictions
        assertTrue(r2 >= 0.0 && r2 <= 1.0, "R² should be between 0 and 1 for good predictions");
        
        // Test perfect predictions
        double perfectR2 = Metrics.r2Score(yTrueRegression, yTrueRegression);
        assertEquals(1.0, perfectR2, 1e-10, "Perfect predictions should give R² = 1");
        
        // Test with mean predictions
        double[] meanPred = new double[yTrueRegression.length];
        double mean = 0.0;
        for (double val : yTrueRegression) mean += val;
        mean /= yTrueRegression.length;
        for (int i = 0; i < meanPred.length; i++) meanPred[i] = mean;
        
        double meanR2 = Metrics.r2Score(yTrueRegression, meanPred);
        assertEquals(0.0, meanR2, 1e-10, "Mean predictions should give R² = 0");
    }

    @Test
    @DisplayName("Test confusion matrix")
    void testConfusionMatrix() {
        int[][] confMatrix = Metrics.confusionMatrix(yTrue, yPred);
        
        assertNotNull(confMatrix);
        assertEquals(2, confMatrix.length, "Binary classification should have 2x2 matrix");
        assertEquals(2, confMatrix[0].length, "Binary classification should have 2x2 matrix");
        
        // Verify confusion matrix values
        // TN = 3, FP = 1, FN = 1, TP = 3
        assertEquals(3, confMatrix[0][0], "True negatives should be correct");
        assertEquals(1, confMatrix[0][1], "False positives should be correct");
        assertEquals(1, confMatrix[1][0], "False negatives should be correct");
        assertEquals(3, confMatrix[1][1], "True positives should be correct");
    }

    @Test
    @DisplayName("Test error handling")
    void testErrorHandling() {
        // Test null inputs - these currently throw NullPointerException, which is acceptable
        assertThrows(Exception.class, () -> 
            Metrics.accuracy(null, yPred));
        assertThrows(Exception.class, () -> 
            Metrics.accuracy(yTrue, null));
        
        // Test mismatched array lengths
        double[] shortArray = {0, 1};
        assertThrows(IllegalArgumentException.class, () -> 
            Metrics.accuracy(yTrue, shortArray));
        
        // Test empty arrays
        double[] empty = {};
        double[] empty2 = {};
        double accuracy = Metrics.accuracy(empty, empty2);
        // Empty arrays should return NaN or 0, not throw exception
        assertTrue(Double.isNaN(accuracy) || accuracy == 0.0);
    }

    @Test
    @DisplayName("Test multiclass classification metrics")
    void testMulticlassMetrics() {
        double[] yTrueMulti = {0, 1, 2, 0, 1, 2, 0, 1};
        double[] yPredMulti = {0, 1, 1, 0, 2, 2, 1, 1};
        
        double accuracy = Metrics.accuracy(yTrueMulti, yPredMulti);
        assertTrue(accuracy >= 0.0 && accuracy <= 1.0, 
            "Multiclass accuracy should be between 0 and 1");
        
        int[][] confMatrix = Metrics.confusionMatrix(yTrueMulti, yPredMulti);
        assertEquals(3, confMatrix.length, "3-class problem should have 3x3 matrix");
        assertEquals(3, confMatrix[0].length, "3-class problem should have 3x3 matrix");
    }

    @Test
    @DisplayName("Test edge cases")
    void testEdgeCases() {
        // Single prediction
        double[] singleTrue = {1};
        double[] singlePred = {1};
        assertEquals(1.0, Metrics.accuracy(singleTrue, singlePred), 1e-10);
        
        // All same class
        double[] allZeros = {0, 0, 0, 0};
        double[] allZerosPred = {0, 0, 0, 0};
        assertEquals(1.0, Metrics.accuracy(allZeros, allZerosPred), 1e-10);
        
        // Test precision/recall with all negative predictions for binary case
        double[] binaryTrue = {0, 1, 0, 1};  
        double[] allNegs = {0, 0, 0, 0};
        assertEquals(0.0, Metrics.precision(binaryTrue, allNegs), 1e-10);
    }

    @Test
    @DisplayName("Test numerical stability")
    void testNumericalStability() {
        // Very small differences
        double[] trueVals = {1e-10, 2e-10, 3e-10};
        double[] predVals = {1.1e-10, 2.1e-10, 2.9e-10};
        
        double mse = Metrics.meanSquaredError(trueVals, predVals);
        assertTrue(Double.isFinite(mse), "MSE should be finite for small values");
        assertTrue(mse >= 0, "MSE should be non-negative");
        
        // Very large differences
        double[] largeTrueVals = {1e6, 2e6, 3e6};
        double[] largePredVals = {1.1e6, 2.1e6, 2.9e6};
        
        double largeMse = Metrics.meanSquaredError(largeTrueVals, largePredVals);
        assertTrue(Double.isFinite(largeMse), "MSE should be finite for large values");
    }
}
