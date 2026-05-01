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

package org.superml.linear_model;

import org.superml.datasets.Datasets;
import org.junit.jupiter.api.Test;
import java.util.Collections;
import java.util.List;
import java.util.ArrayList;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for LogisticRegression.
 */
public class LogisticRegressionTest {
    
    @Test
    public void testFitAndPredict() {
        // Generate a simple classification dataset
        Datasets.ClassificationData dataset = Datasets.makeClassification(100, 2, 2);
        
        // Convert int array to double array
        double[] target = new double[dataset.y.length];
        for (int i = 0; i < dataset.y.length; i++) {
            target[i] = dataset.y[i];
        }
        
        // Split the data (simple train-test split)
        TrainTestSplit split = trainTestSplit(dataset.X, target, 0.2, 42);
        
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
    
    @Test
    void testFitAndPredictBinaryClassification() {
        LogisticRegression model = new LogisticRegression(0.01, 1000);

        // Training data for binary classification
        double[][] X = {
            {1.0, 2.0},
            {2.0, 3.0},
            {3.0, 4.0},
            {4.0, 5.0}
        };
        double[] y = {0, 0, 1, 1};

        // Fit the model
        model.fit(X, y);

        // Test predictions
        double[] testFeatures = {2.5, 3.5};
        double prediction = model.predict(new double[][]{testFeatures})[0];

        assertEquals(1.0, prediction, "Prediction should match the expected class");
    }

    @Test
    void testPredictProbabilities() {
        LogisticRegression model = new LogisticRegression(0.01, 1000);

        // Training data for binary classification
        double[][] X = {
            {1.0, 2.0},
            {2.0, 3.0},
            {3.0, 4.0},
            {4.0, 5.0}
        };
        double[] y = {0, 0, 1, 1};

        // Fit the model
        model.fit(X, y);

        // Test probabilities
        double[] testFeatures = {2.5, 3.5};
        double[] probabilities = model.predictProba(new double[][]{testFeatures})[0];

        assertEquals(2, probabilities.length, "There should be two probabilities for binary classification");
        assertTrue(probabilities[0] >= 0 && probabilities[0] <= 1, "Probability of class 0 should be valid");
        assertTrue(probabilities[1] >= 0 && probabilities[1] <= 1, "Probability of class 1 should be valid");
        assertEquals(1.0, probabilities[0] + probabilities[1], 1e-6, "Probabilities should sum to 1");
    }

    @Test
    void testScore() {
        LogisticRegression model = new LogisticRegression(0.1, 1000); // Increased learning rate for better convergence

        // Training data for binary classification
        double[][] X = {
            {1.0, 2.0},
            {2.0, 3.0},
            {3.0, 4.0},
            {4.0, 5.0}
        };
        double[] y = {0, 0, 1, 1};

        // Fit the model
        model.fit(X, y);

        // Test score
        double score = model.score(X, y);
        assertEquals(1.0, score, "Score should be 1.0 for perfectly classified training data");
    }
    
    // Simple train-test split utility to avoid circular dependency
    private static class TrainTestSplit {
        public final double[][] XTrain;
        public final double[][] XTest;
        public final double[] yTrain;
        public final double[] yTest;
        
        public TrainTestSplit(double[][] XTrain, double[][] XTest, 
                             double[] yTrain, double[] yTest) {
            this.XTrain = XTrain;
            this.XTest = XTest;
            this.yTrain = yTrain;
            this.yTest = yTest;
        }
    }
    
    private static TrainTestSplit trainTestSplit(double[][] X, double[] y, 
                                               double testSize, int randomState) {
        if (X.length != y.length) {
            throw new IllegalArgumentException("X and y must have the same number of samples");
        }
        
        int nSamples = X.length;
        int nTest = (int) Math.round(nSamples * testSize);
        int nTrain = nSamples - nTest;
        
        // Create indices and shuffle them
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < nSamples; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, new Random(randomState));
        
        // Split data
        double[][] XTrain = new double[nTrain][];
        double[][] XTest = new double[nTest][];
        double[] yTrain = new double[nTrain];
        double[] yTest = new double[nTest];
        
        for (int i = 0; i < nTrain; i++) {
            int idx = indices.get(i);
            XTrain[i] = X[idx].clone();
            yTrain[i] = y[idx];
        }
        
        for (int i = 0; i < nTest; i++) {
            int idx = indices.get(i + nTrain);
            XTest[i] = X[idx].clone();
            yTest[i] = y[idx];
        }
        
        return new TrainTestSplit(XTrain, XTest, yTrain, yTest);
    }
}
