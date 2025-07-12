package org.superml.linear_model;

import org.superml.datasets.Datasets;
import org.superml.model_selection.ModelSelection;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for LogisticRegression.
 */
public class LogisticRegressionTest {
    
    @Test
    public void testFitAndPredict() {
        // Generate a simple classification dataset
        Datasets.Dataset dataset = Datasets.makeClassification(100, 2, 2, 42);
        
        // Split the data
        ModelSelection.TrainTestSplit split = ModelSelection.trainTestSplit(
            dataset.data, dataset.target, 0.2, 42);
        
        // Create and train the model
        LogisticRegression lr = new LogisticRegression(0.1, 1000);
        lr.fit(split.XTrain, split.yTrain);
        
        // Make predictions
        double[] predictions = lr.predict(split.XTest);
        
        // Check that predictions are not null and have correct length
        assertNotNull(predictions);
        assertEquals(split.XTest.length, predictions.length);
        
        // Check that predictions are binary (0 or 1)
        for (double prediction : predictions) {
            assertTrue(prediction == 0.0 || prediction == 1.0);
        }
        
        // Check that score method works
        double accuracy = lr.score(split.XTest, split.yTest);
        assertTrue(accuracy >= 0.0 && accuracy <= 1.0);
    }
    
    @Test
    public void testPredictProba() {
        // Generate a simple dataset
        double[][] X = {{1.0, 2.0}, {2.0, 1.0}, {-1.0, -2.0}, {-2.0, -1.0}};
        double[] y = {1.0, 1.0, 0.0, 0.0};
        
        LogisticRegression lr = new LogisticRegression(0.1, 100);
        lr.fit(X, y);
        
        double[][] probabilities = lr.predictProba(X);
        
        // Check dimensions
        assertEquals(X.length, probabilities.length);
        assertEquals(2, probabilities[0].length);
        
        // Check that probabilities sum to 1
        for (int i = 0; i < probabilities.length; i++) {
            double sum = probabilities[i][0] + probabilities[i][1];
            assertEquals(1.0, sum, 0.001);
        }
        
        // Check that probabilities are between 0 and 1
        for (int i = 0; i < probabilities.length; i++) {
            for (int j = 0; j < probabilities[i].length; j++) {
                assertTrue(probabilities[i][j] >= 0.0 && probabilities[i][j] <= 1.0);
            }
        }
    }
    
    @Test
    public void testParameterManagement() {
        LogisticRegression lr = new LogisticRegression();
        
        // Test setting learning rate
        lr.setLearningRate(0.05);
        assertEquals(0.05, lr.getLearningRate());
        assertEquals(0.05, lr.getParams().get("learning_rate"));
        
        // Test setting max iterations
        lr.setMaxIter(500);
        assertEquals(500, lr.getMaxIter());
        assertEquals(500, lr.getParams().get("max_iter"));
    }
    
    @Test
    public void testGetClasses() {
        double[][] X = {{1.0, 2.0}, {2.0, 1.0}, {-1.0, -2.0}, {-2.0, -1.0}};
        double[] y = {1.0, 1.0, 0.0, 0.0};
        
        LogisticRegression lr = new LogisticRegression();
        lr.fit(X, y);
        
        double[] classes = lr.getClasses();
        assertEquals(2, classes.length);
        assertEquals(0.0, classes[0]);
        assertEquals(1.0, classes[1]);
    }
    
    @Test
    public void testNotFittedError() {
        LogisticRegression lr = new LogisticRegression();
        double[][] X = {{1.0, 2.0}, {2.0, 1.0}};
        
        // Should throw exception when calling predict before fit
        assertThrows(IllegalStateException.class, () -> lr.predict(X));
        assertThrows(IllegalStateException.class, () -> lr.predictProba(X));
        assertThrows(IllegalStateException.class, () -> lr.getClasses());
    }
}
