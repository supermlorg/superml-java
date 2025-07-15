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
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for DecisionTree classification and regression
 */
class DecisionTreeTest {

    private DecisionTree classifier;
    private DecisionTree regressor;
    private double[][] XClassification;
    private double[] yClassification;
    private double[][] XRegression;
    private double[] yRegression;

    @BeforeEach
    void setUp() {
        // Classification tree (auto-detected from data)
        classifier = new DecisionTree();
        
        // Simple classification dataset
        XClassification = new double[][] {
            {1.0, 2.0},
            {2.0, 3.0},
            {3.0, 1.0},
            {4.0, 2.0},
            {5.0, 4.0},
            {6.0, 3.0}
        };
        yClassification = new double[] {0, 0, 1, 1, 0, 1};
        
        // Regression tree (auto-detected from data)
        regressor = new DecisionTree();
        
        // Simple regression dataset with continuous values
        XRegression = new double[][] {
            {1.0}, {2.0}, {3.0}, {4.0}, {5.0}, {6.0}
        };
        yRegression = new double[] {2.1, 4.3, 6.2, 8.1, 9.9, 12.2};
    }

    @Test
    @DisplayName("Test classification tree basic functionality")
    void testClassificationBasic() {
        classifier.fit(XClassification, yClassification);
        double[] predictions = classifier.predict(XClassification);
        
        assertNotNull(predictions);
        assertEquals(XClassification.length, predictions.length);
        
        // Predictions should be either 0 or 1 for this binary problem
        for (double pred : predictions) {
            assertTrue(pred == 0.0 || pred == 1.0,
                "Classification predictions should be class labels");
        }
    }

    @Test
    @DisplayName("Test regression tree basic functionality")
    void testRegressionBasic() {
        regressor.fit(XRegression, yRegression);
        double[] predictions = regressor.predict(XRegression);
        
        assertNotNull(predictions);
        assertEquals(XRegression.length, predictions.length);
        
        // For this simple linear relationship, tree should learn reasonably well
        for (int i = 0; i < predictions.length; i++) {
            assertTrue(Math.abs(predictions[i] - yRegression[i]) < 5.0,
                "Regression predictions should be reasonably close");
        }
    }

    @Test
    @DisplayName("Test tree with max_depth parameter")
    void testMaxDepth() {
        DecisionTree shallowTree = new DecisionTree();
        shallowTree.setMaxDepth(1);
        
        DecisionTree deepTree = new DecisionTree();
        deepTree.setMaxDepth(10);
        
        shallowTree.fit(XClassification, yClassification);
        deepTree.fit(XClassification, yClassification);
        
        double[] shallowPred = shallowTree.predict(XClassification);
        double[] deepPred = deepTree.predict(XClassification);
        
        assertNotNull(shallowPred);
        assertNotNull(deepPred);
        
        // Both should make valid predictions
        assertEquals(XClassification.length, shallowPred.length);
        assertEquals(XClassification.length, deepPred.length);
    }

    @Test
    @DisplayName("Test tree with min_samples_split parameter")
    void testMinSamplesSplit() {
        DecisionTree restrictiveTree = new DecisionTree();
        restrictiveTree.setMinSamplesSplit(3);
        
        restrictiveTree.fit(XClassification, yClassification);
        double[] predictions = restrictiveTree.predict(XClassification);
        
        assertNotNull(predictions);
        assertEquals(XClassification.length, predictions.length);
    }

    @Test
    @DisplayName("Test perfect classification case")
    void testPerfectClassification() {
        // Create perfectly separable data
        double[][] perfectX = {
            {0.0}, {1.0}, {2.0}, {3.0}, {4.0}, {5.0}
        };
        double[] perfectY = {0, 0, 0, 1, 1, 1}; // Clear separation at x=3
        
        classifier.fit(perfectX, perfectY);
        double[] predictions = classifier.predict(perfectX);
        
        // Should achieve perfect classification
        assertArrayEquals(perfectY, predictions, 1e-10,
            "Tree should perfectly classify linearly separable data");
    }

    @Test
    @DisplayName("Test single feature regression")
    void testSingleFeatureRegression() {
        regressor.fit(XRegression, yRegression);
        
        // Test on training data
        double[] trainPredictions = regressor.predict(XRegression);
        assertNotNull(trainPredictions);
        
        // Test on new data
        double[][] newX = {{1.5}, {2.5}, {3.5}};
        double[] newPredictions = regressor.predict(newX);
        
        assertNotNull(newPredictions);
        assertEquals(newX.length, newPredictions.length);
        
        // Predictions should be in reasonable range
        for (double pred : newPredictions) {
            assertTrue(pred >= 0.0 && pred <= 15.0,
                "Predictions should be in reasonable range");
        }
    }

    @Test
    @DisplayName("Test error handling")
    void testErrorHandling() {
        // Test null inputs
        assertThrows(NullPointerException.class, () -> classifier.fit(null, yClassification));
        assertThrows(NullPointerException.class, () -> classifier.fit(XClassification, null));
        
        // Test empty data
        double[][] emptyX = {};
        double[] emptyY = {};
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> classifier.fit(emptyX, emptyY));
        
        // Test prediction before fitting
        DecisionTree unfitted = new DecisionTree();
        assertThrows(IllegalStateException.class, () -> unfitted.predict(XClassification));
        
        // Test dimension mismatch - current implementation doesn't validate
        classifier.fit(XClassification, yClassification);
        double[][] wrongX = {{1.0, 2.0, 3.0}}; // Wrong number of features
        // DecisionTree doesn't currently validate dimensions
        double[] predictions = classifier.predict(wrongX);
        assertNotNull(predictions);
    }

    @Test
    @DisplayName("Test tree properties")
    void testTreeProperties() {
        classifier.fit(XClassification, yClassification);
        
        // Tree should have root node after fitting
        assertNotNull(classifier.getTree(), "Tree should have a root node after fitting");
    }

    @Test
    @DisplayName("Test parameter access")
    void testParameterAccess() {
        classifier.setMaxDepth(5);
        classifier.setMinSamplesSplit(3);
        classifier.setRandomState(42);
        
        assertEquals(5, classifier.getMaxDepth());
        assertEquals(3, classifier.getMinSamplesSplit());
        assertEquals(42, classifier.getRandomState());
    }

    @Test
    @DisplayName("Test multiclass classification")
    void testMulticlassClassification() {
        // Create 3-class problem
        double[][] multiX = {
            {1.0, 1.0}, {1.0, 2.0}, 
            {2.0, 1.0}, {2.0, 2.0},
            {3.0, 3.0}, {3.0, 4.0}
        };
        double[] multiY = {0, 0, 1, 1, 2, 2};
        
        classifier.fit(multiX, multiY);
        double[] predictions = classifier.predict(multiX);
        
        assertNotNull(predictions);
        assertEquals(multiX.length, predictions.length);
        
        // All predictions should be valid class labels
        for (double pred : predictions) {
            assertTrue(pred >= 0 && pred <= 2,
                "Predictions should be valid class labels");
        }
    }

    @Test
    @DisplayName("Test overfitting prevention")
    void testOverfittingPrevention() {
        DecisionTree limitedTree = new DecisionTree();
        limitedTree.setMaxDepth(2);
        limitedTree.setMinSamplesSplit(2);
        
        limitedTree.fit(XClassification, yClassification);
        
        // Limited tree should still make reasonable predictions
        double[] predictions = limitedTree.predict(XClassification);
        assertNotNull(predictions);
        assertEquals(XClassification.length, predictions.length);
        
        // Tree should respect the depth limit
        assertEquals(2, limitedTree.getMaxDepth(),
            "Tree should maintain max_depth parameter");
    }

    @Test
    @DisplayName("Test reproducibility")
    void testReproducibility() {
        DecisionTree tree1 = new DecisionTree();
        DecisionTree tree2 = new DecisionTree();
        
        // Set same parameters
        tree1.setRandomState(42);
        tree2.setRandomState(42);
        
        tree1.fit(XClassification, yClassification);
        tree2.fit(XClassification, yClassification);
        
        double[] pred1 = tree1.predict(XClassification);
        double[] pred2 = tree2.predict(XClassification);
        
        assertArrayEquals(pred1, pred2, 1e-10,
            "Trees with same random state should produce identical results");
    }
}
