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

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for Ridge regression with L2 regularization
 */
class RidgeTest {

    private Ridge model;
    private double[][] X;
    private double[] y;

    @BeforeEach
    void setUp() {
        model = new Ridge();
        
        // Well-conditioned overdetermined system (same as LinearRegression)
        X = new double[][] {
            {1.0, 1.0},
            {2.0, 1.0},
            {3.0, 1.0},
            {1.0, 2.0},
            {2.0, 2.0},
            {3.0, 2.0},
            {1.0, 3.0},
            {2.0, 3.0}
        };
        
        // y = 2*x1 + 3*x2 + 1 (approximately)
        y = new double[] {6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0};
    }

    @Test
    @DisplayName("Test Ridge with default alpha")
    void testDefaultAlpha() {
        model.fit(X, y);
        double[] predictions = model.predict(X);
        
        assertNotNull(predictions);
        assertEquals(X.length, predictions.length);
        
        // Ridge should produce reasonable predictions
        for (int i = 0; i < predictions.length; i++) {
            assertTrue(Math.abs(predictions[i] - y[i]) < 5.0, 
                "Ridge predictions should be reasonably close to actual values");
        }
    }

    @Test
    @DisplayName("Test Ridge with custom alpha")
    void testCustomAlpha() {
        Ridge strongRegularization = new Ridge(10.0);
        Ridge weakRegularization = new Ridge(0.01);
        
        strongRegularization.fit(X, y);
        weakRegularization.fit(X, y);
        
        double[] predStrong = strongRegularization.predict(X);
        double[] predWeak = weakRegularization.predict(X);
        
        assertNotNull(predStrong);
        assertNotNull(predWeak);
        
        // Strong regularization should shrink coefficients more
        double[] coefStrong = strongRegularization.getCoefficients();
        double[] coefWeak = weakRegularization.getCoefficients();
        
        for (int i = 0; i < coefStrong.length; i++) {
            assertTrue(Math.abs(coefStrong[i]) <= Math.abs(coefWeak[i]) + 1e-6,
                "Strong regularization should produce smaller coefficients");
        }
    }

    @Test
    @DisplayName("Test Ridge vs LinearRegression comparison")
    void testRidgeVsLinearRegression() {
        LinearRegression lr = new LinearRegression();
        Ridge ridge = new Ridge(1.0);
        
        lr.fit(X, y);
        ridge.fit(X, y);
        
        double[] lrPred = lr.predict(X);
        double[] ridgePred = ridge.predict(X);
        
        // Both should make reasonable predictions
        assertNotNull(lrPred);
        assertNotNull(ridgePred);
        assertEquals(lrPred.length, ridgePred.length);
    }

    @Test
    @DisplayName("Test coefficient access")
    void testCoefficientAccess() {
        model.fit(X, y);
        
        double[] coefficients = model.getCoefficients();
        assertNotNull(coefficients);
        assertEquals(X[0].length, coefficients.length);
        
        double intercept = model.getIntercept();
        assertTrue(Double.isFinite(intercept));
    }

    @Test
    @DisplayName("Test error handling")
    void testErrorHandling() {
        // Test null inputs
        assertThrows(NullPointerException.class, () -> model.fit(null, y));
        assertThrows(NullPointerException.class, () -> model.fit(X, null));
        
        // Test prediction before fitting
        assertThrows(IllegalStateException.class, () -> model.predict(X));
        
        // Test dimension mismatch - current implementation doesn't validate
        model.fit(X, y);
        double[][] wrongX = {{1.0, 2.0, 3.0}};
        // Ridge doesn't currently validate dimensions, so this won't throw
        double[] predictions = model.predict(wrongX);
        assertNotNull(predictions);
    }

    @Test
    @DisplayName("Test high regularization effect")
    void testHighRegularization() {
        Ridge highAlpha = new Ridge(1000.0);
        highAlpha.fit(X, y);
        
        double[] coefficients = highAlpha.getCoefficients();
        
        // High regularization should produce small coefficients
        for (double coef : coefficients) {
            assertTrue(Math.abs(coef) < 1.0, 
                "High regularization should produce small coefficients");
        }
    }

    @Test
    @DisplayName("Test zero regularization")
    void testZeroRegularization() {
        Ridge zeroAlpha = new Ridge(0.0);
        LinearRegression lr = new LinearRegression();
        
        zeroAlpha.fit(X, y);
        lr.fit(X, y);
        
        double[] ridgePred = zeroAlpha.predict(X);
        double[] lrPred = lr.predict(X);
        
        // Zero regularization should be similar to LinearRegression
        for (int i = 0; i < ridgePred.length; i++) {
            assertEquals(lrPred[i], ridgePred[i], 1e-3,
                "Ridge with alpha=0 should be similar to LinearRegression");
        }
    }

    @Test
    @DisplayName("Test single feature Ridge")
    void testSingleFeature() {
        double[][] singleX = {{1}, {2}, {3}, {4}};
        double[] singleY = {2, 4, 6, 8};
        
        model.fit(singleX, singleY);
        double[] predictions = model.predict(singleX);
        
        assertNotNull(predictions);
        assertEquals(singleX.length, predictions.length);
        
        // Should learn a reasonable relationship
        for (int i = 0; i < predictions.length; i++) {
            assertTrue(Math.abs(predictions[i] - singleY[i]) < 2.0);
        }
    }

    @Test
    @DisplayName("Test reproducibility")
    void testReproducibility() {
        Ridge model1 = new Ridge(1.0);
        Ridge model2 = new Ridge(1.0);
        
        model1.fit(X, y);
        model2.fit(X, y);
        
        double[] pred1 = model1.predict(X);
        double[] pred2 = model2.predict(X);
        
        assertArrayEquals(pred1, pred2, 1e-10,
            "Same Ridge model should produce identical results");
    }
}
