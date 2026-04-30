package org.superml.examples.transformers;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.superml.metrics.transformers.TransformerAnalysisMetrics;
import org.superml.metrics.transformers.TransformerAnalysisMetrics.*;
import org.superml.visualization.transformers.TransformerAnalysisVisualization;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Integration tests for transformer analysis functionality across
 * metrics and visualization modules.
 */
public class TransformerAnalysisIntegrationTest {

    @Test
    public void testEndToEndAnalysis() {
        // Test different attention matrix sizes
        int[] sizes = {16, 32, 64};
        for (int size : sizes) {
            System.out.printf("%nTesting %dx%d attention matrix%n", size, size);

            // Generate synthetic attention data
            RealMatrix attentionMatrix = generateSyntheticAttention(size);

            // Run attention pattern analysis
            AttentionAnalysis analysis =
                TransformerAnalysisMetrics.analyzeAttentionPatterns(attentionMatrix);

            // Validate metrics
            validateAttentionMetrics(analysis);

            // Visualize results
            TransformerAnalysisVisualization.visualizeAttentionAnalysis(analysis);
        }
    }

    @Test
    public void testGradientFlowAnalysis() {
        // Test gradient flow through multiple layers
        int numLayers = 4;
        int size = 32;

        // Generate synthetic layer data
        List<RealMatrix> gradients = new ArrayList<>();
        List<RealMatrix> activations = new ArrayList<>();

        for (int i = 0; i < numLayers; i++) {
            gradients.add(generateSyntheticGradients(size));
            activations.add(generateSyntheticActivations(size));
        }

        // Run gradient flow analysis
        GradientFlowAnalysis analysis =
            TransformerAnalysisMetrics.analyzeGradientFlow(gradients, activations);

        // Validate metrics
        validateGradientMetrics(analysis);

        // Visualize results
        TransformerAnalysisVisualization.visualizeGradientFlow(analysis);
    }

    private void validateAttentionMetrics(AttentionAnalysis analysis) {
        // Pattern metrics validation
        assertTrue(analysis.patternMetrics.sparsity >= 0 &&
                  analysis.patternMetrics.sparsity <= 1,
                  "Sparsity should be between 0 and 1");

        assertTrue(analysis.patternMetrics.headDiversity >= 0 &&
                  analysis.patternMetrics.headDiversity <= 1,
                  "Head diversity should be between 0 and 1");

        // Distribution metrics validation
        assertTrue(analysis.distributionMetrics.stdDev >= 0,
                  "Standard deviation should be positive");

        assertTrue(analysis.distributionMetrics.percentile5 <
                  analysis.distributionMetrics.percentile95,
                  "5th percentile should be less than 95th");
    }

    private void validateGradientMetrics(GradientFlowAnalysis analysis) {
        // Gradient metrics validation
        assertTrue(analysis.gradientMetrics.maxGradientNorm > 0,
                  "Max gradient norm should be positive");

        assertTrue(analysis.gradientMetrics.gradientCorrelation >= -1 &&
                  analysis.gradientMetrics.gradientCorrelation <= 1,
                  "Gradient correlation should be between -1 and 1");

        // Activation metrics validation
        for (Double mean : analysis.activationMetrics.meanActivations) {
            assertTrue(!Double.isInfinite(mean) && !Double.isNaN(mean),
                      "Mean activation should be finite");
        }

        for (Double deadUnits : analysis.activationMetrics.deadUnits) {
            assertTrue(deadUnits >= 0 && deadUnits <= 1,
                      "Dead units ratio should be between 0 and 1");
        }
    }

    private RealMatrix generateSyntheticAttention(int size) {
        Random rand = new Random(42); // Fixed seed for reproducibility
        double[][] attention = new double[size][size];

        for (int i = 0; i < size; i++) {
            double sum = 0;
            for (int j = 0; j < size; j++) {
                // Generate attention weights with local patterns
                double distance = Math.abs(i - j) / (double) size;
                attention[i][j] = rand.nextDouble() * Math.exp(-distance * 2);
                sum += attention[i][j];
            }
            // Normalize to sum to 1
            for (int j = 0; j < size; j++) {
                attention[i][j] /= sum;
            }
        }

        return new Array2DRowRealMatrix(attention);
    }

    private RealMatrix generateSyntheticGradients(int size) {
        Random rand = new Random(42);
        double[][] gradients = new double[size][size];

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                // Generate gradients following normal distribution
                gradients[i][j] = rand.nextGaussian() * 0.01;
            }
        }

        return new Array2DRowRealMatrix(gradients);
    }

    private RealMatrix generateSyntheticActivations(int size) {
        Random rand = new Random(42);
        double[][] activations = new double[size][size];

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                // Generate activations with ReLU-like pattern
                double value = rand.nextGaussian();
                activations[i][j] = Math.max(0, value);
            }
        }

        return new Array2DRowRealMatrix(activations);
    }
}
