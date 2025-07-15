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
import org.superml.linear_model.Ridge;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for Hyperparameter Tuning functionality
 */
class HyperparameterTuningTest {
    
    private double[][] X;
    private double[] yClassification;
    private double[] yRegression;
    private LogisticRegression classifier;
    private LinearRegression regressor;
    private Ridge ridgeRegressor;
    
    @BeforeEach
    void setUp() {
        // Create test data for classification (binary)
        X = new double[][] {
            {1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0},
            {5.0, 6.0}, {6.0, 7.0}, {7.0, 8.0}, {8.0, 9.0},
            {1.5, 2.5}, {2.5, 3.5}, {3.5, 4.5}, {4.5, 5.5},
            {5.5, 6.5}, {6.5, 7.5}, {7.5, 8.5}, {8.5, 9.5},
            {0.5, 1.5}, {1.5, 2.5}, {2.5, 3.5}, {3.5, 4.5}
        };
        
        yClassification = new double[] {
            0, 0, 1, 1, 1, 1, 1, 1,
            0, 0, 1, 1, 1, 1, 1, 1,
            0, 0, 1, 1
        };
        
        yRegression = new double[] {
            3.0, 7.0, 11.0, 15.0, 19.0, 23.0, 27.0, 31.0,
            5.0, 9.0, 13.0, 17.0, 21.0, 25.0, 29.0, 33.0,
            4.0, 8.0, 12.0, 16.0
        };
        
        classifier = new LogisticRegression();
        regressor = new LinearRegression();
        ridgeRegressor = new Ridge();
    }
    
    @Test
    @DisplayName("Test Grid Search CV with basic configuration")
    void testGridSearchBasic() {
        List<HyperparameterTuning.ParameterSpec> specs = Arrays.asList(
            HyperparameterTuning.ParameterSpec.discrete("learningRate", 0.01, 0.1, 0.5),
            HyperparameterTuning.ParameterSpec.discrete("maxIterations", 100, 500)
        );
        
        HyperparameterTuning.TuningConfig config = new HyperparameterTuning.TuningConfig();
        
        HyperparameterTuning.TuningResults results = 
            HyperparameterTuning.GridSearch.search(classifier, X, yClassification, specs, config);
        
        assertNotNull(results);
        assertNotNull(results.getBestParameters());
        assertTrue(results.getBestScore() >= 0.0);
        assertTrue(results.getBestScore() <= 1.0);
        assertEquals(6, results.getAllCombinations().size()); // 3 * 2 = 6 combinations
    }
    
    @Test
    @DisplayName("Test Grid Search with Ridge regression parameters")
    void testGridSearchRidge() {
        List<HyperparameterTuning.ParameterSpec> specs = Arrays.asList(
            HyperparameterTuning.ParameterSpec.discrete("alpha", 0.1, 1.0, 10.0)
        );
        
        HyperparameterTuning.TuningConfig config = new HyperparameterTuning.TuningConfig()
            .setScoringMetric("r2");
        
        HyperparameterTuning.TuningResults results = 
            HyperparameterTuning.GridSearch.searchRegressor(ridgeRegressor, X, yRegression, specs, config);
        
        assertNotNull(results);
        assertTrue(results.getBestParameters().getParameters().containsKey("alpha"));
        assertEquals(3, results.getAllCombinations().size());
    }
    
    @Test
    @DisplayName("Test Random Search with basic configuration")
    void testRandomSearchBasic() {
        List<HyperparameterTuning.ParameterSpec> specs = Arrays.asList(
            HyperparameterTuning.ParameterSpec.discrete("learningRate", 0.01, 0.05, 0.1, 0.2),
            HyperparameterTuning.ParameterSpec.discrete("maxIterations", 100, 200, 500, 1000)
        );
        
        HyperparameterTuning.TuningConfig config = new HyperparameterTuning.TuningConfig()
            .setMaxIterations(5);
        
        HyperparameterTuning.TuningResults results = 
            HyperparameterTuning.RandomSearch.search(classifier, X, yClassification, specs, config);
        
        assertNotNull(results);
        assertNotNull(results.getBestParameters());
        assertTrue(results.getBestScore() >= 0.0);
        assertTrue(results.getBestScore() <= 1.0);
        assertEquals(5, results.getAllCombinations().size());
    }
    
    @Test
    @DisplayName("Test parameter specifications")
    void testParameterSpecifications() {
        // Test discrete parameters
        HyperparameterTuning.ParameterSpec discrete = 
            HyperparameterTuning.ParameterSpec.discrete("param1", 1, 2, 3);
        
        assertEquals("param1", discrete.getName());
        assertEquals(3, discrete.getValues().length);
        assertEquals(HyperparameterTuning.ParameterSpec.ParameterType.DISCRETE, discrete.getType());
        
        // Test continuous parameters
        HyperparameterTuning.ParameterSpec continuous = 
            HyperparameterTuning.ParameterSpec.continuous("param2", 0.0, 1.0, 5);
        
        assertEquals("param2", continuous.getName());
        assertEquals(5, continuous.getValues().length);
        assertEquals(HyperparameterTuning.ParameterSpec.ParameterType.CONTINUOUS, continuous.getType());
        
        // Test integer parameters
        HyperparameterTuning.ParameterSpec integer = 
            HyperparameterTuning.ParameterSpec.integer("param3", 10, 15);
        
        assertEquals("param3", integer.getName());
        assertEquals(6, integer.getValues().length); // 10, 11, 12, 13, 14, 15
        assertEquals(HyperparameterTuning.ParameterSpec.ParameterType.INTEGER, integer.getType());
    }
    
    @Test
    @DisplayName("Test tuning configuration builder")
    void testTuningConfigBuilder() {
        HyperparameterTuning.TuningConfig config = 
            new HyperparameterTuning.TuningConfig()
                .setScoringMetric("f1")
                .setCvFolds(3)
                .setParallel(true)
                .setRandomSeed(42L)
                .setVerbose(false)
                .setMaxIterations(10);
        
        assertEquals("f1", config.getScoringMetric());
        assertEquals(3, config.getCvFolds());
        assertTrue(config.isParallel());
        assertEquals(42L, (long) config.getRandomSeed());
        assertFalse(config.isVerbose());
        assertEquals(10, config.getMaxIterations());
    }
    
    @Test
    @DisplayName("Test tuning results structure")
    void testTuningResults() {
        List<HyperparameterTuning.ParameterSpec> specs = Arrays.asList(
            HyperparameterTuning.ParameterSpec.discrete("learningRate", 0.01, 0.1)
        );
        
        HyperparameterTuning.TuningConfig config = new HyperparameterTuning.TuningConfig();
        
        HyperparameterTuning.TuningResults results = 
            HyperparameterTuning.GridSearch.search(classifier, X, yClassification, specs, config);
        
        assertNotNull(results.getBestParameters());
        assertNotNull(results.getAllCombinations());
        assertNotNull(results.getAllScores());
        assertTrue(results.getBestScore() >= 0.0);
        assertTrue(results.getSearchTimeMs() >= 0);
        assertEquals("accuracy", results.getScoringMetric());
        
        // Test toString method
        String resultString = results.toString();
        assertNotNull(resultString);
        assertTrue(resultString.contains("Best Score"));
        assertTrue(resultString.contains("Best Parameters"));
    }
    
    @Test
    @DisplayName("Test parameter combination")
    void testParameterCombination() {
        Map<String, Object> params = Map.of("param1", 0.1, "param2", 100);
        HyperparameterTuning.ParameterCombination combination = 
            new HyperparameterTuning.ParameterCombination(params);
        
        assertEquals(0.1, combination.get("param1"));
        assertEquals(100, combination.get("param2"));
        assertEquals(2, combination.getParameters().size());
        
        // Test equality
        HyperparameterTuning.ParameterCombination combination2 = 
            new HyperparameterTuning.ParameterCombination(params);
        assertEquals(combination, combination2);
        assertEquals(combination.hashCode(), combination2.hashCode());
        
        // Test toString
        assertNotNull(combination.toString());
    }
    
    @Test
    @DisplayName("Test parallel execution")
    void testParallelExecution() {
        List<HyperparameterTuning.ParameterSpec> specs = Arrays.asList(
            HyperparameterTuning.ParameterSpec.discrete("learningRate", 0.01, 0.05, 0.1),
            HyperparameterTuning.ParameterSpec.discrete("maxIterations", 100, 200)
        );
        
        HyperparameterTuning.TuningConfig config = new HyperparameterTuning.TuningConfig()
            .setParallel(true);
        
        long startTime = System.currentTimeMillis();
        HyperparameterTuning.TuningResults results = 
            HyperparameterTuning.GridSearch.search(classifier, X, yClassification, specs, config);
        long endTime = System.currentTimeMillis();
        
        assertNotNull(results);
        assertEquals(6, results.getAllCombinations().size()); // 3 * 2 = 6 combinations
        
        // Parallel execution should complete (timing test is environment-dependent)
        assertTrue(endTime > startTime);
    }
    
    @Test
    @DisplayName("Test reproducibility with random seed")
    void testReproducibility() {
        List<HyperparameterTuning.ParameterSpec> specs = Arrays.asList(
            HyperparameterTuning.ParameterSpec.discrete("learningRate", 0.01, 0.05, 0.1, 0.2)
        );
        
        HyperparameterTuning.TuningConfig config1 = new HyperparameterTuning.TuningConfig()
            .setMaxIterations(3)
            .setRandomSeed(42L);
        
        HyperparameterTuning.TuningConfig config2 = new HyperparameterTuning.TuningConfig()
            .setMaxIterations(3)
            .setRandomSeed(42L);
        
        HyperparameterTuning.TuningResults results1 = 
            HyperparameterTuning.RandomSearch.search(classifier, X, yClassification, specs, config1);
        HyperparameterTuning.TuningResults results2 = 
            HyperparameterTuning.RandomSearch.search(classifier, X, yClassification, specs, config2);
        
        // Results should be identical with same seed
        assertEquals(results1.getBestScore(), results2.getBestScore(), 1e-10);
    }
    
    @Test
    @DisplayName("Test convenience methods")
    void testConvenienceMethods() {
        // Test grid search convenience method
        HyperparameterTuning.TuningResults gridResults = HyperparameterTuning.gridSearch(
            classifier, X, yClassification,
            HyperparameterTuning.ParameterSpec.discrete("learningRate", 0.01, 0.1)
        );
        
        assertNotNull(gridResults);
        assertEquals(2, gridResults.getAllCombinations().size());
        
        // Test random search convenience method
        HyperparameterTuning.TuningResults randomResults = HyperparameterTuning.randomSearch(
            classifier, X, yClassification, 3,
            HyperparameterTuning.ParameterSpec.discrete("learningRate", 0.01, 0.05, 0.1, 0.2)
        );
        
        assertNotNull(randomResults);
        assertEquals(3, randomResults.getAllCombinations().size());
    }
    
    @Test
    @DisplayName("Test error handling for invalid parameters")
    void testErrorHandling() {
        // Test mismatched X and y lengths - this should trigger exception in cross-validation
        double[] shortY = {0, 1};
        List<HyperparameterTuning.ParameterSpec> specs = Arrays.asList(
            HyperparameterTuning.ParameterSpec.discrete("learningRate", 0.1)
        );
        
        assertThrows(Exception.class, () -> {
            HyperparameterTuning.GridSearch.search(classifier, X, shortY, specs, 
                new HyperparameterTuning.TuningConfig());
        });
        
        // Test invalid parameter values that would cause reflection errors
        // This is more likely to cause actual exceptions
        HyperparameterTuning.ParameterSpec invalidSpec = 
            HyperparameterTuning.ParameterSpec.discrete("nonExistentParameter", "invalidValue");
        
        // This should complete without error but with warnings in the log
        HyperparameterTuning.TuningResults results = HyperparameterTuning.GridSearch.search(
            classifier, X, yClassification, Arrays.asList(invalidSpec), 
            new HyperparameterTuning.TuningConfig());
        
        // The search should still complete, just with parameter application warnings
        assertNotNull(results);
    }
    
    @Test
    @DisplayName("Test different scoring metrics")
    void testDifferentScoringMetrics() {
        List<HyperparameterTuning.ParameterSpec> specs = Arrays.asList(
            HyperparameterTuning.ParameterSpec.discrete("learningRate", 0.01, 0.1)
        );
        
        HyperparameterTuning.TuningConfig config = new HyperparameterTuning.TuningConfig()
            .setScoringMetric("f1");
        
        HyperparameterTuning.TuningResults results = 
            HyperparameterTuning.GridSearch.search(classifier, X, yClassification, specs, config);
        
        assertNotNull(results);
        assertEquals("f1", results.getScoringMetric());
        // F1 score should be between 0 and 1
        assertTrue(results.getBestScore() >= 0.0);
        assertTrue(results.getBestScore() <= 1.0);
    }
    
    @Test
    @DisplayName("Test parameter random value generation")
    void testParameterRandomValues() {
        HyperparameterTuning.ParameterSpec spec = 
            HyperparameterTuning.ParameterSpec.discrete("test", 1, 2, 3, 4, 5);
        
        java.util.Random random = new java.util.Random(42);
        
        // Generate multiple random values and ensure they're from the valid set
        for (int i = 0; i < 20; i++) {
            Object value = spec.getRandomValue(random);
            assertTrue(value instanceof Integer);
            int intValue = (Integer) value;
            assertTrue(intValue >= 1 && intValue <= 5);
        }
    }
    
    @Test
    @DisplayName("Test cross-validation integration")
    void testCrossValidationIntegration() {
        List<HyperparameterTuning.ParameterSpec> specs = Arrays.asList(
            HyperparameterTuning.ParameterSpec.discrete("learningRate", 0.01, 0.1)
        );
        
        HyperparameterTuning.TuningConfig config = new HyperparameterTuning.TuningConfig()
            .setCvFolds(3);
        
        HyperparameterTuning.TuningResults results = 
            HyperparameterTuning.GridSearch.search(classifier, X, yClassification, specs, config);
        
        assertNotNull(results);
        // Each parameter combination should be evaluated with 3-fold CV
        assertEquals(2, results.getAllCombinations().size());
    }
}
