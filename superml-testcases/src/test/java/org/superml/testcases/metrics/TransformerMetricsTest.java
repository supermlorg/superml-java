package org.superml.transformers.metrics;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import java.util.Random;

/**
 * Test class for Transformer Metrics validation.
 * Focuses on core attention matrix validation and metrics computation.
 */
public class TransformerMetricsTest {

    private RealMatrix testAttentionMatrix;
    private Random random;
    
    @BeforeEach
    public void setUp() {
        random = new Random(42); // Fixed seed for reproducible tests
        testAttentionMatrix = generateTestAttentionMatrix(4, 4);
    }

    @Test
    public void testAttentionMatrixValidation() {
        // Test that attention matrix rows sum to approximately 1.0
        int numRows = testAttentionMatrix.getRowDimension();
        
        for (int i = 0; i < numRows; i++) {
            double[] row = testAttentionMatrix.getRow(i);
            double rowSum = 0.0;
            for (double value : row) {
                rowSum += value;
                // Each attention value should be non-negative
                assertTrue(value >= 0.0, "Attention values should be non-negative");
            }
            // Row should sum to approximately 1.0 (allowing for floating point precision)
            assertEquals(1.0, rowSum, 1e-6, "Attention row should sum to 1.0");
        }
    }

    @Test
    public void testAttentionMatrixDimensions() {
        // Test matrix dimensions
        assertEquals(testAttentionMatrix.getRowDimension(), 
            testAttentionMatrix.getColumnDimension(), "Matrix should be square");
        
        assertTrue(testAttentionMatrix.getRowDimension() > 0, "Matrix should have positive dimensions");
    }

    @Test
    public void testAttentionEntropyCalculation() {
        // Test entropy calculation for attention distribution
        double[] uniformDistribution = {0.25, 0.25, 0.25, 0.25};
        double entropy = calculateEntropy(uniformDistribution);
        
        // Uniform distribution should have maximum entropy for 4 elements
        double expectedMaxEntropy = Math.log(4.0) / Math.log(2.0); // log2(4) = 2.0
        assertEquals(expectedMaxEntropy, entropy, 1e-6, "Uniform distribution should have maximum entropy");
    }

    @Test
    public void testAttentionSparsity() {
        // Test sparsity calculation
        double[][] sparseData = {
            {1.0, 0.0, 0.0, 0.0},
            {0.0, 1.0, 0.0, 0.0},
            {0.0, 0.0, 1.0, 0.0},
            {0.0, 0.0, 0.0, 1.0}
        };
        RealMatrix sparseMatrix = new Array2DRowRealMatrix(sparseData);
        
        double sparsity = calculateSparsity(sparseMatrix);
        assertEquals(0.75, sparsity, 1e-6, "Diagonal matrix should have 75% sparsity");
    }

    @Test
    public void testBasicMetricsComputation() {
        // Test basic statistical metrics
        double[] values = {1.0, 2.0, 3.0, 4.0, 5.0};
        
        double mean = calculateMean(values);
        assertEquals(3.0, mean, 1e-6, "Mean should be 3.0");
        
        double variance = calculateVariance(values, mean);
        assertEquals(2.5, variance, 1e-6, "Variance should be 2.5");
    }

    // Helper methods

    private RealMatrix generateTestAttentionMatrix(int rows, int cols) {
        double[][] data = new double[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            // Generate random weights
            double[] weights = new double[cols];
            double sum = 0.0;
            
            for (int j = 0; j < cols; j++) {
                weights[j] = random.nextDouble();
                sum += weights[j];
            }
            
            // Normalize to sum to 1.0 (proper attention distribution)
            for (int j = 0; j < cols; j++) {
                data[i][j] = weights[j] / sum;
            }
        }
        
        return new Array2DRowRealMatrix(data);
    }

    private double calculateEntropy(double[] distribution) {
        double entropy = 0.0;
        for (double p : distribution) {
            if (p > 0) {
                entropy -= p * (Math.log(p) / Math.log(2.0));
            }
        }
        return entropy;
    }

    private double calculateSparsity(RealMatrix matrix) {
        int totalElements = matrix.getRowDimension() * matrix.getColumnDimension();
        int zeroElements = 0;
        
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                if (Math.abs(matrix.getEntry(i, j)) < 1e-10) {
                    zeroElements++;
                }
            }
        }
        
        return (double) zeroElements / totalElements;
    }

    private double calculateMean(double[] values) {
        double sum = 0.0;
        for (double value : values) {
            sum += value;
        }
        return sum / values.length;
    }

    private double calculateVariance(double[] values, double mean) {
        double sum = 0.0;
        for (double value : values) {
            double diff = value - mean;
            sum += diff * diff;
        }
        return sum / (values.length - 1);
    }
}
