package org.superml.linear_model;

import org.junit.jupiter.api.Test;
import org.superml.datasets.Datasets;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test cases for multiclass classification implementations.
 */
public class MulticlassTest {
    
    @Test
    public void testLogisticRegressionMulticlassOvR() {
        // Generate a simple multiclass dataset
        Datasets.ClassificationData data = Datasets.makeClassification(100, 4, 3);
        double[][] X = data.X;
        double[] y = convertToDouble(data.y);
        
        // Test LogisticRegression with One-vs-Rest
        LogisticRegression lr = new LogisticRegression()
                .setMultiClass("ovr")
                .setMaxIter(100);
        
        lr.fit(X, y);
        
        // Verify it's fitted
        assertTrue(lr.getClasses().length == 3);
        
        // Test predictions
        double[] predictions = lr.predict(X);
        assertEquals(X.length, predictions.length);
        
        // Test probabilities
        double[][] probabilities = lr.predictProba(X);
        assertEquals(X.length, probabilities.length);
        assertEquals(3, probabilities[0].length);
        
        // Verify probabilities sum to approximately 1
        for (int i = 0; i < Math.min(10, X.length); i++) {
            double sum = 0;
            for (int j = 0; j < probabilities[i].length; j++) {
                sum += probabilities[i][j];
            }
            assertEquals(1.0, sum, 0.001, "Probabilities should sum to 1");
        }
    }
    
    @Test
    public void testLogisticRegressionMulticlassSoftmax() {
        // Generate a simple multiclass dataset
        Datasets.ClassificationData data = Datasets.makeClassification(100, 4, 3);
        double[][] X = data.X;
        double[] y = convertToDouble(data.y);
        
        // Test LogisticRegression with Softmax
        LogisticRegression lr = new LogisticRegression()
                .setMultiClass("multinomial")
                .setMaxIter(100);
        
        lr.fit(X, y);
        
        // Verify it's fitted
        assertTrue(lr.getClasses().length == 3);
        
        // Test predictions
        double[] predictions = lr.predict(X);
        assertEquals(X.length, predictions.length);
        
        // Test probabilities
        double[][] probabilities = lr.predictProba(X);
        assertEquals(X.length, probabilities.length);
        assertEquals(3, probabilities[0].length);
        
        // Verify probabilities sum to approximately 1
        for (int i = 0; i < Math.min(10, X.length); i++) {
            double sum = 0;
            for (int j = 0; j < probabilities[i].length; j++) {
                sum += probabilities[i][j];
            }
            assertEquals(1.0, sum, 0.001, "Probabilities should sum to 1");
        }
    }
    
    @Test
    public void testOneVsRestClassifier() {
        // Generate a simple multiclass dataset
        Datasets.ClassificationData data = Datasets.makeClassification(100, 4, 3);
        double[][] X = data.X;
        double[] y = convertToDouble(data.y);
        
        // Test OneVsRestClassifier
        LogisticRegression baseClassifier = new LogisticRegression().setMaxIter(100);
        OneVsRestClassifier ovr = new OneVsRestClassifier(baseClassifier);
        
        ovr.fit(X, y);
        
        // Verify it's fitted
        assertEquals(3, ovr.getClasses().length);
        assertEquals(3, ovr.getClassifiers().size());
        
        // Test predictions
        double[] predictions = ovr.predict(X);
        assertEquals(X.length, predictions.length);
        
        // Test probabilities
        double[][] probabilities = ovr.predictProba(X);
        assertEquals(X.length, probabilities.length);
        assertEquals(3, probabilities[0].length);
    }
    
    @Test
    public void testSoftmaxRegression() {
        // Generate a simple multiclass dataset
        Datasets.ClassificationData data = Datasets.makeClassification(100, 4, 3);
        double[][] X = data.X;
        double[] y = convertToDouble(data.y);
        
        // Test SoftmaxRegression
        SoftmaxRegression softmax = new SoftmaxRegression().setMaxIter(100);
        
        softmax.fit(X, y);
        
        // Verify it's fitted
        assertEquals(3, softmax.getClasses().length);
        
        // Test predictions
        double[] predictions = softmax.predict(X);
        assertEquals(X.length, predictions.length);
        
        // Test probabilities
        double[][] probabilities = softmax.predictProba(X);
        assertEquals(X.length, probabilities.length);
        assertEquals(3, probabilities[0].length);
        
        // Verify probabilities sum to exactly 1 (softmax property)
        for (int i = 0; i < Math.min(10, X.length); i++) {
            double sum = 0;
            for (int j = 0; j < probabilities[i].length; j++) {
                sum += probabilities[i][j];
            }
            assertEquals(1.0, sum, 0.0001, "Softmax probabilities should sum to 1");
        }
        
        // Test weight matrix dimensions
        double[][] weights = softmax.getWeights();
        assertEquals(3, weights.length); // Number of classes
        assertEquals(5, weights[0].length); // Number of features + 1 (bias)
    }
    
    @Test
    public void testBinaryClassificationStillWorks() {
        // Generate binary dataset
        Datasets.ClassificationData data = Datasets.makeClassification(100, 4, 2);
        double[][] X = data.X;
        double[] y = convertToDouble(data.y);
        
        // Test that binary classification still works with the updated LogisticRegression
        LogisticRegression lr = new LogisticRegression().setMaxIter(100);
        
        lr.fit(X, y);
        
        // Verify it's fitted
        assertEquals(2, lr.getClasses().length);
        
        // Test predictions
        double[] predictions = lr.predict(X);
        assertEquals(X.length, predictions.length);
        
        // Test probabilities
        double[][] probabilities = lr.predictProba(X);
        assertEquals(X.length, probabilities.length);
        assertEquals(2, probabilities[0].length);
    }
    
    @Test
    public void testSoftmaxBinaryFallback() {
        // Generate binary dataset
        Datasets.ClassificationData data = Datasets.makeClassification(100, 4, 2);
        double[][] X = data.X;
        double[] y = convertToDouble(data.y);
        
        // Test that SoftmaxRegression handles binary case correctly
        SoftmaxRegression softmax = new SoftmaxRegression().setMaxIter(100);
        
        softmax.fit(X, y);
        
        // Verify it's fitted
        assertEquals(2, softmax.getClasses().length);
        
        // Test predictions
        double[] predictions = softmax.predict(X);
        assertEquals(X.length, predictions.length);
        
        // Test probabilities
        double[][] probabilities = softmax.predictProba(X);
        assertEquals(X.length, probabilities.length);
        assertEquals(2, probabilities[0].length);
        
        // Verify probabilities sum to 1
        for (int i = 0; i < Math.min(10, X.length); i++) {
            double sum = probabilities[i][0] + probabilities[i][1];
            assertEquals(1.0, sum, 0.0001, "Binary probabilities should sum to 1");
        }
    }
    
    private double[] convertToDouble(int[] intArray) {
        double[] doubleArray = new double[intArray.length];
        for (int i = 0; i < intArray.length; i++) {
            doubleArray[i] = intArray[i];
        }
        return doubleArray;
    }
}
