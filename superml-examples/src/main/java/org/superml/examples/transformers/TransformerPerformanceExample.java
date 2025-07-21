package org.superml.examples.transformers;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.superml.metrics.transformers.TransformerPerformanceMetrics;
import org.superml.metrics.transformers.TransformerPerformanceMetrics.AttentionEvaluation;
import org.superml.visualization.transformers.TransformerPerformanceVisualization;

/**
 * Example demonstrating transformer performance analysis using SuperML's
 * cross-cutting metrics and visualization modules.
 */
public class TransformerPerformanceExample {

    public static void main(String[] args) {
        System.out.println("🚀 SuperML Transformer Performance Analysis Example");
        System.out.println("================================================");

        // Test different matrix sizes to analyze scaling behavior
        int[] dimensions = {64, 128, 256, 512, 1024};
        int detectedCacheLineSize = detectSystemCacheLineSize();

        for (int dim : dimensions) {
            System.out.printf("%nAnalyzing %dx%d attention matrix...%n", dim, dim);

            // Generate sample attention matrix
            RealMatrix attentionMatrix = generateAttentionMatrix(dim);

            // Evaluate performance using metrics module
            AttentionEvaluation evaluation = TransformerPerformanceMetrics
                .evaluateAttentionPerformance(attentionMatrix, detectedCacheLineSize);

            // Visualize results using visualization module
            TransformerPerformanceVisualization.visualizePerformanceResults(evaluation);
        }
    }

    private static int detectSystemCacheLineSize() {
        // Simple cache line size detection
        int[] testSizes = {32, 64, 128};
        long[] accessTimes = new long[testSizes.length];

        for (int i = 0; i < testSizes.length; i++) {
            int size = testSizes[i];
            double[] array = new double[4096];

            long startTime = System.nanoTime();
            for (int iter = 0; iter < 1000; iter++) {
                for (int j = 0; j < array.length; j += size/8) {
                    array[j] += 1.0;
                }
            }
            accessTimes[i] = System.nanoTime() - startTime;
        }

        // Find optimal size based on access times
        int bestSize = 64; // Default cache line size
        double maxRatio = 0;
        for (int i = 1; i < accessTimes.length; i++) {
            double ratio = (double) accessTimes[i] / accessTimes[i-1];
            if (ratio > maxRatio) {
                maxRatio = ratio;
                bestSize = testSizes[i-1];
            }
        }

        return bestSize;
    }

    private static RealMatrix generateAttentionMatrix(int size) {
        double[][] matrix = new double[size][size];

        for (int i = 0; i < size; i++) {
            double sum = 0;
            // Generate random attention weights
            for (int j = 0; j < size; j++) {
                matrix[i][j] = Math.random();
                sum += matrix[i][j];
            }
            // Normalize to valid attention weights (sum to 1)
            for (int j = 0; j < size; j++) {
                matrix[i][j] /= sum;
            }
        }

        return new Array2DRowRealMatrix(matrix);
    }
}
