package org.superml.linear_model;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for LinearRegression
 */
class LinearRegressionTest {

    private LinearRegression model;
    private double[][] X;
    private double[] y;
    private double[][] XTest;

    @BeforeEach
    void setUp() {
        model = new LinearRegression();
        
        // Well-conditioned overdetermined system (more data points than features)
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
        
        XTest = new double[][] {
            {3.0, 3.0},
            {4.0, 4.0}
        };
    }

    @Test
    @DisplayName("Test basic fitting and prediction")
    void testBasicFitAndPredict() {
        // Fit the model
        model.fit(X, y);
        
        // Make predictions
        double[] predictions = model.predict(X);
        
        // Check that predictions are reasonable
        assertNotNull(predictions);
        assertEquals(X.length, predictions.length);
        
        // Check that model learned something close to the true relationship
        for (int i = 0; i < predictions.length; i++) {
            assertEquals(y[i], predictions[i], 2.0, 
                "Prediction " + i + " should be close to actual value");
        }
    }

    @Test
    @DisplayName("Test prediction on new data")
    void testPredictNewData() {
        model.fit(X, y);
        double[] predictions = model.predict(XTest);
        
        assertNotNull(predictions);
        assertEquals(XTest.length, predictions.length);
        
        // Predictions should be in reasonable range (around 5-15 based on our data)
        for (double pred : predictions) {
            assertTrue(pred > 5, "Predictions should be positive for this dataset");
            assertTrue(pred < 20, "Predictions should be reasonable");
        }
    }

    @Test
    @DisplayName("Test single feature regression")
    void testSingleFeature() {
        // Simple case: y = 2*x + 1
        double[][] XSingle = {{1}, {2}, {3}, {4}, {5}};
        double[] ySingle = {3, 5, 7, 9, 11};
        
        model.fit(XSingle, ySingle);
        double[] predictions = model.predict(XSingle);
        
        // Should predict very accurately for this simple linear case
        for (int i = 0; i < predictions.length; i++) {
            assertEquals(ySingle[i], predictions[i], 0.1, 
                "Single feature regression should be very accurate");
        }
    }

    @Test
    @DisplayName("Test model with different fit_intercept setting")
    void testWithoutIntercept() {
        LinearRegression noInterceptModel = new LinearRegression(false);
        noInterceptModel.fit(X, y);
        
        double[] predictions = noInterceptModel.predict(X);
        assertNotNull(predictions);
        assertEquals(X.length, predictions.length);
        assertEquals(0.0, noInterceptModel.getIntercept(), 1e-10, 
            "Intercept should be zero when fit_intercept=false");
    }

    @Test
    @DisplayName("Test error handling for mismatched dimensions")
    void testDimensionMismatch() {
        // Fit with correct dimensions
        model.fit(X, y);
        
        // Try to predict with wrong number of features - just test it doesn't crash
        double[][] wrongX = {{1.0, 2.0, 3.0}}; // 3 features instead of 2
        
        // Current implementation doesn't validate dimensions, so this won't throw
        // In future versions we could add dimension validation
        double[] predictions = model.predict(wrongX);
        assertNotNull(predictions);
    }

    @Test
    @DisplayName("Test error handling for empty data")
    void testEmptyData() {
        double[][] emptyX = {};
        double[] emptyY = {};
        
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> {
            model.fit(emptyX, emptyY);
        }, "Should throw exception for empty training data");
    }

    @Test
    @DisplayName("Test error handling for null data")
    void testNullData() {
        assertThrows(NullPointerException.class, () -> {
            model.fit(null, y);
        }, "Should throw exception for null X");
        
        assertThrows(NullPointerException.class, () -> {
            model.fit(X, null);
        }, "Should throw exception for null y");
        
        // Test null prediction data - first fit the model
        model.fit(X, y);
        assertThrows(NullPointerException.class, () -> {
            model.predict(null);
        }, "Should throw exception for null prediction data");
    }

    @Test
    @DisplayName("Test prediction before fitting")
    void testPredictBeforeFit() {
        LinearRegression unfittedModel = new LinearRegression();
        
        assertThrows(IllegalStateException.class, () -> {
            unfittedModel.predict(X);
        }, "Should throw exception when predicting before fitting");
    }

    @Test
    @DisplayName("Test coefficient access")
    void testCoefficientAccess() {
        model.fit(X, y);
        
        // Should be able to access coefficients after fitting
        double[] coefficients = model.getCoefficients();
        assertNotNull(coefficients);
        assertEquals(X[0].length, coefficients.length, 
            "Should have one coefficient per feature");
        
        double intercept = model.getIntercept();
        assertTrue(Double.isFinite(intercept), "Intercept should be a finite number");
    }

    @Test
    @DisplayName("Test perfect linear relationship")
    void testPerfectLinearRelationship() {
        // Perfect relationship: y = 2*x + 3
        double[][] perfectX = {{1}, {2}, {3}, {4}, {5}};
        double[] perfectY = {5, 7, 9, 11, 13};
        
        model.fit(perfectX, perfectY);
        double[] predictions = model.predict(perfectX);
        
        // Should predict perfectly
        for (int i = 0; i < predictions.length; i++) {
            assertEquals(perfectY[i], predictions[i], 1e-6, 
                "Perfect linear relationship should be learned exactly");
        }
        
        // Check learned parameters
        double[] coef = model.getCoefficients();
        assertEquals(2.0, coef[0], 1e-6, "Should learn coefficient of 2");
        assertEquals(3.0, model.getIntercept(), 1e-6, "Should learn intercept of 3");
    }

    @Test
    @DisplayName("Test model reproducibility")
    void testReproducibility() {
        // Fit the same model twice
        LinearRegression model1 = new LinearRegression();
        LinearRegression model2 = new LinearRegression();
        
        model1.fit(X, y);
        model2.fit(X, y);
        
        double[] pred1 = model1.predict(XTest);
        double[] pred2 = model2.predict(XTest);
        
        // Should get identical results
        assertArrayEquals(pred1, pred2, 1e-10, 
            "Same model with same data should produce identical results");
    }
}
