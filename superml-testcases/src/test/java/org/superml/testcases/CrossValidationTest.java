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

package org.superml.model_selection;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.superml.linear_model.LogisticRegression;
import org.superml.linear_model.LinearRegression;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for Cross-Validation functionality
 */
class CrossValidationTest {
    
    private double[][] X;
    private double[] yClassification;
    private double[] yRegression;
    private LogisticRegression classifier;
    private LinearRegression regressor;
    
    @BeforeEach
    void setUp() {
        // Create test data for classification (binary)
        X = new double[][] {
            {1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0},
            {5.0, 6.0}, {6.0, 7.0}, {7.0, 8.0}, {8.0, 9.0},
            {1.5, 2.5}, {2.5, 3.5}, {3.5, 4.5}, {4.5, 5.5},
            {5.5, 6.5}, {6.5, 7.5}, {7.5, 8.5}, {8.5, 9.5}
        };
        
        yClassification = new double[] {
            0, 0, 1, 1, 1, 1, 1, 1,
            0, 0, 1, 1, 1, 1, 1, 1
        };
        
        yRegression = new double[] {
            3.0, 7.0, 11.0, 15.0, 19.0, 23.0, 27.0, 31.0,
            5.0, 9.0, 13.0, 17.0, 21.0, 25.0, 29.0, 33.0
        };
        
        classifier = new LogisticRegression();
        regressor = new LinearRegression();
    }
    
    @Test
    @DisplayName("Test basic cross-validation with default config")
    void testBasicCrossValidation() {
        CrossValidation.CrossValidationResults results = 
            CrossValidation.crossValidate(classifier, X, yClassification);
        
        assertNotNull(results);
        assertEquals(5, results.getNumFolds());
        assertTrue(results.getMeanScores().containsKey("accuracy"));
        assertTrue(results.getMeanScore("accuracy") >= 0.0);
        assertTrue(results.getMeanScore("accuracy") <= 1.0);
        assertTrue(results.getStdScore("accuracy") >= 0.0);
    }
    
    @Test
    @DisplayName("Test cross-validation with custom number of folds")
    void testCustomFolds() {
        CrossValidation.CrossValidationResults results = 
            CrossValidation.crossValidate(classifier, X, yClassification, 3);
        
        assertNotNull(results);
        assertEquals(3, results.getNumFolds());
        
        // Check that we have scores for all folds
        for (double[] scores : results.getScores().values()) {
            assertEquals(3, scores.length);
        }
    }
    
    @Test
    @DisplayName("Test cross-validation with custom configuration")
    void testCustomConfiguration() {
        CrossValidation.CrossValidationConfig config = 
            new CrossValidation.CrossValidationConfig()
                .setFolds(4)
                .setShuffle(false)
                .setRandomSeed(42L)
                .setMetrics("accuracy", "precision", "recall");
        
        CrossValidation.CrossValidationResults results = 
            CrossValidation.crossValidate(classifier, X, yClassification, config);
        
        assertNotNull(results);
        assertEquals(4, results.getNumFolds());
        assertTrue(results.getMeanScores().containsKey("accuracy"));
        assertTrue(results.getMeanScores().containsKey("precision"));
        assertTrue(results.getMeanScores().containsKey("recall"));
    }
    
    @Test
    @DisplayName("Test cross-validation metrics validation")
    void testMetricsValidation() {
        CrossValidation.CrossValidationConfig config = 
            new CrossValidation.CrossValidationConfig()
                .setMetrics("accuracy", "f1");
        
        CrossValidation.CrossValidationResults results = 
            CrossValidation.crossValidate(classifier, X, yClassification, config);
        
        assertNotNull(results);
        
        // All metrics should be between 0 and 1
        for (String metric : results.getMeanScores().keySet()) {
            double score = results.getMeanScore(metric);
            assertTrue(score >= 0.0 && score <= 1.0, 
                "Metric " + metric + " should be between 0 and 1, got: " + score);
        }
    }
    
    @Test
    @DisplayName("Test cross-validation results structure")
    void testResultsStructure() {
        CrossValidation.CrossValidationResults results = 
            CrossValidation.crossValidate(classifier, X, yClassification);
        
        assertNotNull(results.getScores());
        assertNotNull(results.getMeanScores());
        assertNotNull(results.getStdScores());
        
        // Check that mean and std scores have same keys as raw scores
        assertEquals(results.getScores().keySet(), results.getMeanScores().keySet());
        assertEquals(results.getScores().keySet(), results.getStdScores().keySet());
        
        // Check toString method
        String resultString = results.toString();
        assertNotNull(resultString);
        assertTrue(resultString.contains("Cross-Validation Results"));
        assertTrue(resultString.contains("folds"));
    }
    
    @Test
    @DisplayName("Test regression cross-validation")
    void testRegressionCrossValidation() {
        CrossValidation.CrossValidationConfig config = 
            new CrossValidation.CrossValidationConfig();
        
        CrossValidation.CrossValidationResults results = 
            CrossValidation.crossValidateRegression(regressor, X, yRegression, config);
        
        assertNotNull(results);
        assertTrue(results.getMeanScores().containsKey("mse"));
        assertTrue(results.getMeanScores().containsKey("mae"));
        assertTrue(results.getMeanScores().containsKey("r2"));
        
        // MSE and MAE should be non-negative
        assertTrue(results.getMeanScore("mse") >= 0.0);
        assertTrue(results.getMeanScore("mae") >= 0.0);
        
        // R² should be between -∞ and 1, but typically positive for decent models
        assertTrue(results.getMeanScore("r2") <= 1.0);
    }
    
    @Test
    @DisplayName("Test cross-validation with parallel execution")
    void testParallelExecution() {
        CrossValidation.CrossValidationConfig config = 
            new CrossValidation.CrossValidationConfig()
                .setParallel(true)
                .setFolds(3);
        
        CrossValidation.CrossValidationResults results = 
            CrossValidation.crossValidate(classifier, X, yClassification, config);
        
        assertNotNull(results);
        assertEquals(3, results.getNumFolds());
        assertTrue(results.getMeanScore("accuracy") >= 0.0);
    }
    
    @Test
    @DisplayName("Test cross-validation reproducibility with random seed")
    void testReproducibility() {
        CrossValidation.CrossValidationConfig config1 = 
            new CrossValidation.CrossValidationConfig()
                .setRandomSeed(42L)
                .setShuffle(true);
        
        CrossValidation.CrossValidationConfig config2 = 
            new CrossValidation.CrossValidationConfig()
                .setRandomSeed(42L)
                .setShuffle(true);
        
        CrossValidation.CrossValidationResults results1 = 
            CrossValidation.crossValidate(classifier, X, yClassification, config1);
        CrossValidation.CrossValidationResults results2 = 
            CrossValidation.crossValidate(classifier, X, yClassification, config2);
        
        // Results should be identical with same seed
        assertEquals(results1.getMeanScore("accuracy"), results2.getMeanScore("accuracy"), 1e-10);
    }
    
    @Test
    @DisplayName("Test error handling for invalid inputs")
    void testErrorHandling() {
        // Test mismatched X and y lengths
        double[] shortY = {0, 1};
        assertThrows(IllegalArgumentException.class, () -> {
            CrossValidation.crossValidate(classifier, X, shortY);
        });
        
        // Test invalid number of folds
        assertThrows(IllegalArgumentException.class, () -> {
            new CrossValidation.CrossValidationConfig().setFolds(1);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            new CrossValidation.CrossValidationConfig().setFolds(0);
        });
    }
    
    @Test
    @DisplayName("Test cross-validation with minimum viable data")
    void testMinimumData() {
        // Test with exactly enough data for 2-fold CV
        double[][] minX = {{1.0, 2.0}, {3.0, 4.0}};
        double[] minY = {0, 1};
        
        CrossValidation.CrossValidationResults results = 
            CrossValidation.crossValidate(classifier, minX, minY, 2);
        
        assertNotNull(results);
        assertEquals(2, results.getNumFolds());
    }
    
    @Test
    @DisplayName("Test configuration builder pattern")
    void testConfigurationBuilder() {
        CrossValidation.CrossValidationConfig config = 
            new CrossValidation.CrossValidationConfig()
                .setFolds(3)
                .setShuffle(true)
                .setRandomSeed(123L)
                .setParallel(false)
                .addMetric("accuracy")
                .addMetric("f1");
        
        assertEquals(3, config.getFolds());
        assertTrue(config.isShuffle());
        assertEquals(123L, config.getRandomSeed());
        assertFalse(config.isParallel());
        assertTrue(config.getMetrics().contains("accuracy"));
        assertTrue(config.getMetrics().contains("f1"));
    }
    
    @Test
    @DisplayName("Test cross-validation with different metrics")
    void testDifferentMetrics() {
        CrossValidation.CrossValidationConfig config = 
            new CrossValidation.CrossValidationConfig()
                .setMetrics("accuracy", "precision", "recall", "f1");
        
        CrossValidation.CrossValidationResults results = 
            CrossValidation.crossValidate(classifier, X, yClassification, config);
        
        // All requested metrics should be present
        assertEquals(4, results.getMeanScores().size());
        assertTrue(results.getMeanScores().containsKey("accuracy"));
        assertTrue(results.getMeanScores().containsKey("precision"));
        assertTrue(results.getMeanScores().containsKey("recall"));
        assertTrue(results.getMeanScores().containsKey("f1"));
    }
}
