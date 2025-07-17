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

package org.superml.tree;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

import java.util.*;

/**
 * Test cases for XGBoost implementation
 * 
 * @author SuperML Team
 * @version 2.1.0
 */
public class XGBoostTest {
    
    private double[][] XTrain;
    private double[] yTrain;
    private double[][] XTest;
    private double[] yTest;
    
    @BeforeEach
    void setUp() {
        // Generate simple linearly separable dataset
        Random random = new Random(42);
        int trainSamples = 100;
        int testSamples = 30;
        int features = 4;
        
        // Training data
        XTrain = new double[trainSamples][features];
        yTrain = new double[trainSamples];
        
        for (int i = 0; i < trainSamples; i++) {
            for (int j = 0; j < features; j++) {
                XTrain[i][j] = random.nextGaussian();
            }
            // Simple decision rule: sum of first two features
            double score = XTrain[i][0] + XTrain[i][1];
            yTrain[i] = score > 0 ? 1.0 : 0.0;
        }
        
        // Test data
        XTest = new double[testSamples][features];
        yTest = new double[testSamples];
        
        for (int i = 0; i < testSamples; i++) {
            for (int j = 0; j < features; j++) {
                XTest[i][j] = random.nextGaussian();
            }
            double score = XTest[i][0] + XTest[i][1];
            yTest[i] = score > 0 ? 1.0 : 0.0;
        }
    }
    
    @Test
    void testBasicTraining() {
        XGBoost xgb = new XGBoost()
            .setNEstimators(10)
            .setLearningRate(0.3)
            .setMaxDepth(3)
            .setRandomState(42);
        
        // Training should not throw exceptions
        assertDoesNotThrow(() -> {
            xgb.fit(XTrain, yTrain);
        });
        
        // Model should be fitted
        assertTrue(xgb.isFitted());
        
        // Should produce predictions
        double[] predictions = xgb.predict(XTest);
        assertNotNull(predictions);
        assertEquals(XTest.length, predictions.length);
        
        // Predictions should be 0 or 1
        for (double pred : predictions) {
            assertTrue(pred == 0.0 || pred == 1.0);
        }
    }
    
    @Test
    void testHyperparameters() {
        XGBoost xgb = new XGBoost()
            .setNEstimators(50)
            .setLearningRate(0.1)
            .setMaxDepth(6)
            .setGamma(0.1)
            .setLambda(1.0)
            .setAlpha(0.1)
            .setSubsample(0.8)
            .setColsampleBytree(0.8)
            .setMinChildWeight(1)
            .setRandomState(42);
        
        // Check getter methods
        assertEquals(50, xgb.getConfiguredNEstimators());
        assertEquals(0.1, xgb.getLearningRate(), 1e-6);
        assertEquals(6, xgb.getMaxDepth());
        assertEquals(0.1, xgb.getGamma(), 1e-6);
        assertEquals(1.0, xgb.getLambda(), 1e-6);
        assertEquals(0.1, xgb.getAlpha(), 1e-6);
        assertEquals(0.8, xgb.getSubsample(), 1e-6);
        assertEquals(0.8, xgb.getColsampleBytree(), 1e-6);
        assertEquals(1, xgb.getMinChildWeight());
        assertEquals(42, xgb.getRandomState());
    }
    
    @Test
    void testPredictProba() {
        XGBoost xgb = new XGBoost()
            .setNEstimators(20)
            .setRandomState(42);
        
        xgb.fit(XTrain, yTrain);
        
        double[][] probas = xgb.predictProba(XTest);
        assertNotNull(probas);
        assertEquals(XTest.length, probas.length);
        
        // Each sample should have probabilities for 2 classes
        for (double[] proba : probas) {
            assertEquals(2, proba.length);
            // Probabilities should sum to 1 (approximately)
            double sum = proba[0] + proba[1];
            assertEquals(1.0, sum, 1e-6);
            // Probabilities should be non-negative
            assertTrue(proba[0] >= 0.0);
            assertTrue(proba[1] >= 0.0);
        }
    }
    
    @Test
    void testFeatureImportance() {
        XGBoost xgb = new XGBoost()
            .setNEstimators(30)
            .setRandomState(42);
        
        xgb.fit(XTrain, yTrain);
        
        Map<String, double[]> importanceStats = xgb.getFeatureImportanceStats();
        assertNotNull(importanceStats);
        
        // Should have different types of importance
        assertTrue(importanceStats.containsKey("weight"));
        assertTrue(importanceStats.containsKey("gain"));
        assertTrue(importanceStats.containsKey("cover"));
        
        double[] weightImportance = importanceStats.get("weight");
        assertEquals(XTrain[0].length, weightImportance.length);
        
        // First two features should have higher importance
        // (since they determine the target)
        double feature0Importance = weightImportance[0];
        double feature2Importance = weightImportance[2];
        assertTrue(feature0Importance >= feature2Importance);
    }
    
    @Test
    void testEarlyStopping() {
        XGBoost xgb = new XGBoost()
            .setNEstimators(100)
            .setLearningRate(0.1)
            .setEarlyStoppingRounds(10)
            .setValidationFraction(0.2)
            .setRandomState(42);
        
        xgb.fit(XTrain, yTrain);
        
        // Should have stopped early (less than 100 estimators)
        assertTrue(xgb.getNEstimators() < 100);
        
        // Should have evaluation results
        Map<String, List<Double>> evalResults = xgb.getEvalResults();
        assertNotNull(evalResults);
        assertFalse(evalResults.isEmpty());
    }
    
    @Test
    void testConsistentPredictions() {
        XGBoost xgb = new XGBoost()
            .setNEstimators(20)
            .setRandomState(42);
        
        xgb.fit(XTrain, yTrain);
        
        // Multiple calls should produce same results
        double[] preds1 = xgb.predict(XTest);
        double[] preds2 = xgb.predict(XTest);
        
        assertArrayEquals(preds1, preds2, 1e-10);
    }
    
    @Test
    void testInvalidInputs() {
        XGBoost xgb = new XGBoost();
        
        // Should throw exception for invalid parameters
        assertThrows(IllegalArgumentException.class, () -> {
            xgb.setNEstimators(-1);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            xgb.setLearningRate(-0.1);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            xgb.setMaxDepth(0);
        });
        
        // Should throw exception when predicting before fitting
        assertThrows(IllegalStateException.class, () -> {
            xgb.predict(XTest);
        });
    }
    
    @Test
    void testAccuracyOnSimpleDataset() {
        XGBoost xgb = new XGBoost()
            .setNEstimators(50)
            .setLearningRate(0.2)
            .setMaxDepth(4)
            .setRandomState(42);
        
        xgb.fit(XTrain, yTrain);
        double[] predictions = xgb.predict(XTest);
        
        // Calculate accuracy
        int correct = 0;
        for (int i = 0; i < yTest.length; i++) {
            if (predictions[i] == yTest[i]) {
                correct++;
            }
        }
        double accuracy = (double) correct / yTest.length;
        
        // Should achieve reasonable accuracy on this simple dataset
        assertTrue(accuracy >= 0.7, "Accuracy should be at least 70%, got: " + accuracy);
    }
}
