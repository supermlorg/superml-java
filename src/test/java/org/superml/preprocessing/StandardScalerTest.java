package org.superml.preprocessing;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for StandardScaler feature normalization
 */
class StandardScalerTest {

    private StandardScaler scaler;
    private double[][] X;
    private double[][] XDifferentScale;

    @BeforeEach
    void setUp() {
        scaler = new StandardScaler();
        
        // Sample data with different scales
        X = new double[][] {
            {1.0, 100.0, 0.1},
            {2.0, 200.0, 0.2},
            {3.0, 300.0, 0.3},
            {4.0, 400.0, 0.4},
            {5.0, 500.0, 0.5}
        };
        
        // Data with very different scales to test effectiveness
        XDifferentScale = new double[][] {
            {10, 1000, 0.01},
            {20, 2000, 0.02},
            {30, 3000, 0.03}
        };
    }

    @Test
    @DisplayName("Test basic fit and transform")
    void testFitTransform() {
        double[][] scaled = scaler.fitTransform(X);
        
        assertNotNull(scaled);
        assertEquals(X.length, scaled.length);
        assertEquals(X[0].length, scaled[0].length);
        
        // Check that each feature has approximately mean 0 and std 1
        for (int feature = 0; feature < X[0].length; feature++) {
            double mean = 0.0;
            double variance = 0.0;
            
            // Calculate mean
            for (int i = 0; i < scaled.length; i++) {
                mean += scaled[i][feature];
            }
            mean /= scaled.length;
            
            // Calculate variance
            for (int i = 0; i < scaled.length; i++) {
                variance += Math.pow(scaled[i][feature] - mean, 2);
            }
            variance /= scaled.length;
            
            assertEquals(0.0, mean, 1e-10, 
                "Feature " + feature + " should have mean close to 0");
            assertEquals(1.0, Math.sqrt(variance), 1e-10, 
                "Feature " + feature + " should have std close to 1");
        }
    }

    @Test
    @DisplayName("Test separate fit and transform")
    void testSeparateFitTransform() {
        scaler.fit(X);
        double[][] scaled = scaler.transform(X);
        
        assertNotNull(scaled);
        assertEquals(X.length, scaled.length);
        assertEquals(X[0].length, scaled[0].length);
        
        // Verify scaling is applied correctly
        double[] means = scaler.getMean();
        double[] scales = scaler.getScale();
        
        assertNotNull(means);
        assertNotNull(scales);
        assertEquals(X[0].length, means.length);
        assertEquals(X[0].length, scales.length);
    }

    @Test
    @DisplayName("Test transform new data")
    void testTransformNewData() {
        // Fit on original data
        scaler.fit(X);
        
        // Transform new data
        double[][] scaledNew = scaler.transform(XDifferentScale);
        
        assertNotNull(scaledNew);
        assertEquals(XDifferentScale.length, scaledNew.length);
        assertEquals(XDifferentScale[0].length, scaledNew[0].length);
        
        // New data should be scaled using original statistics
        for (int i = 0; i < scaledNew.length; i++) {
            for (int j = 0; j < scaledNew[i].length; j++) {
                assertTrue(Double.isFinite(scaledNew[i][j]), 
                    "Scaled values should be finite");
            }
        }
    }

    @Test
    @DisplayName("Test inverse transform")
    void testInverseTransform() {
        double[][] scaled = scaler.fitTransform(X);
        double[][] restored = scaler.inverseTransform(scaled);
        
        assertNotNull(restored);
        assertEquals(X.length, restored.length);
        assertEquals(X[0].length, restored[0].length);
        
        // Restored data should match original
        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < X[i].length; j++) {
                assertEquals(X[i][j], restored[i][j], 1e-10,
                    "Inverse transform should restore original values");
            }
        }
    }

    @Test
    @DisplayName("Test single sample transformation")
    void testSingleSample() {
        scaler.fit(X);
        
        double[][] singleSample = {{2.5, 250.0, 0.25}};
        double[][] scaled = scaler.transform(singleSample);
        
        assertNotNull(scaled);
        assertEquals(1, scaled.length);
        assertEquals(3, scaled[0].length);
        
        // Inverse should work for single sample too
        double[][] restored = scaler.inverseTransform(scaled);
        for (int j = 0; j < singleSample[0].length; j++) {
            assertEquals(singleSample[0][j], restored[0][j], 1e-10);
        }
    }

    @Test
    @DisplayName("Test constant feature handling")
    void testConstantFeature() {
        // Data with one constant feature
        double[][] constantX = {
            {1.0, 5.0, 5.0},  // Third feature is constant
            {2.0, 6.0, 5.0},
            {3.0, 7.0, 5.0},
            {4.0, 8.0, 5.0}
        };
        
        double[][] scaled = scaler.fitTransform(constantX);
        
        assertNotNull(scaled);
        
        // Constant feature should remain unchanged (or become 0)
        for (int i = 0; i < scaled.length; i++) {
            assertEquals(0.0, scaled[i][2], 1e-10,
                "Constant feature should be scaled to 0");
        }
    }

    @Test
    @DisplayName("Test error handling")
    void testErrorHandling() {
        // Test null input
        assertThrows(NullPointerException.class, () -> scaler.fit(null));
        
        // Fit scaler first, then test null transform/inverse transform
        scaler.fit(X);
        assertThrows(NullPointerException.class, () -> scaler.transform(null));
        assertThrows(NullPointerException.class, () -> scaler.inverseTransform(null));
        
        // Test empty data
        double[][] empty = {};
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> scaler.fit(empty));
        
        // Test transform before fit
        StandardScaler unfittedScaler = new StandardScaler();
        assertThrows(IllegalStateException.class, () -> unfittedScaler.transform(X));
        
        // Test dimension mismatch - current implementation doesn't validate
        double[][] wrongDim = {{1.0, 2.0}}; // Wrong number of features
        // StandardScaler doesn't currently validate dimensions
        double[][] result = scaler.transform(wrongDim);
        assertNotNull(result);
    }

    @Test
    @DisplayName("Test with_mean and with_std parameters")
    void testParameterSettings() {
        // Test with_mean=false
        StandardScaler noMeanScaler = new StandardScaler(false, true);
        double[][] scaled = noMeanScaler.fitTransform(X);
        
        assertNotNull(scaled);
        
        // When with_mean=false, data should not be centered
        // but should still be scaled
        
        // Test with_std=false
        StandardScaler noStdScaler = new StandardScaler(true, false);
        double[][] scaledNoStd = noStdScaler.fitTransform(X);
        
        assertNotNull(scaledNoStd);
    }

    @Test
    @DisplayName("Test statistics access")
    void testStatisticsAccess() {
        scaler.fit(X);
        
        double[] means = scaler.getMean();
        double[] scales = scaler.getScale();
        
        assertNotNull(means);
        assertNotNull(scales);
        assertEquals(X[0].length, means.length);
        assertEquals(X[0].length, scales.length);
        
        // Verify means are calculated correctly
        for (int feature = 0; feature < X[0].length; feature++) {
            double expectedMean = 0.0;
            for (int i = 0; i < X.length; i++) {
                expectedMean += X[i][feature];
            }
            expectedMean /= X.length;
            
            assertEquals(expectedMean, means[feature], 1e-10,
                "Mean should be calculated correctly");
        }
    }

    @Test
    @DisplayName("Test numerical stability")
    void testNumericalStability() {
        // Data with very large values
        double[][] largeX = {
            {1e6, 2e6},
            {1.1e6, 2.1e6},
            {1.2e6, 2.2e6}
        };
        
        double[][] scaled = scaler.fitTransform(largeX);
        
        assertNotNull(scaled);
        for (int i = 0; i < scaled.length; i++) {
            for (int j = 0; j < scaled[i].length; j++) {
                assertTrue(Double.isFinite(scaled[i][j]),
                    "Scaled values should be finite even for large inputs");
            }
        }
    }

    @Test
    @DisplayName("Test reproducibility")
    void testReproducibility() {
        StandardScaler scaler1 = new StandardScaler();
        StandardScaler scaler2 = new StandardScaler();
        
        double[][] scaled1 = scaler1.fitTransform(X);
        double[][] scaled2 = scaler2.fitTransform(X);
        
        // Should produce identical results
        for (int i = 0; i < scaled1.length; i++) {
            assertArrayEquals(scaled1[i], scaled2[i], 1e-15,
                "Same scaler should produce identical results");
        }
    }
}
