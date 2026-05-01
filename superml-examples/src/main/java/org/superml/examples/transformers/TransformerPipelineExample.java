package org.superml.examples.transformers;

import org.superml.pipeline.Pipeline;
import org.superml.preprocessing.StandardScaler;
import org.superml.transformers.attention.MultiHeadAttention;
// Note: Using custom metrics implementation since TransformerMetrics is not available
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

/**
 * SuperML Transformer Pipeline Integration Example
 *
 * Demonstrates integration of transformers with SuperML's pipeline system
 * following the framework's cross-cutting functionality patterns.
 *
 * This example shows:
 * - Pipeline integration with transformers
 * - Cross-cutting metrics analysis
 * - Realistic transformer data patterns
 *
 * @author SuperML Team
 * @version 2.1.0
 */
public class TransformerPipelineExample {

    public static void main(String[] args) {
        System.out.println("🚀 SuperML Transformer Pipeline Integration Example");
        System.out.println("=================================================");

        try {
            // Example 1: Basic Pipeline with Transformers
            demonstrateBasicPipeline();

            // Example 2: Advanced Analysis with Metrics
            demonstrateAdvancedAnalysis();

            // Example 3: Performance Validation
            demonstratePerformanceValidation();

            System.out.println("\n✅ All transformer pipeline examples completed successfully!");

        } catch (Exception e) {
            System.err.println("❌ Error in transformer pipeline: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Demonstrates basic transformer integration with SuperML Pipeline system.
     */
    private static void demonstrateBasicPipeline() {
        System.out.println("\n1. 🔧 Basic Transformer Pipeline");
        System.out.println("=================================");

        // Create a pipeline with preprocessing and attention
        Pipeline pipeline = new Pipeline()
            .addStep("scaler", new StandardScaler())
            .addStep("attention", new MultiHeadAttention(512, 8)
                .setDropout(0.1)
                .setUseBias(true));

        // Generate sample transformer data
        double[][] inputSequence = generateTransformerData(10, 512);
        System.out.printf("Generated input: %d tokens × %d dimensions%n",
            inputSequence.length, inputSequence[0].length);

        // Fit and transform through pipeline
        // Note: Using dummy labels since this is for demonstration
        double[] dummyLabels = new double[inputSequence.length];
        double[][] transformed = pipeline.fit(inputSequence, dummyLabels).transform(inputSequence);
        System.out.printf("Transformed output: %d tokens × %d dimensions%n",
            transformed.length, transformed[0].length);

        // Basic validation
        validateTransformation(inputSequence, transformed);
    }

    /**
     * Demonstrates advanced transformer analysis using cross-cutting metrics.
     */
    private static void demonstrateAdvancedAnalysis() {
        System.out.println("\n2. 📊 Advanced Transformer Analysis");
        System.out.println("===================================");

        // Generate larger sequence for analysis
        double[][] sequence = generateTransformerData(32, 256);
        RealMatrix attentionMatrix = new Array2DRowRealMatrix(sequence);

        // Apply custom attention metrics analysis
        AttentionMetrics metrics = evaluateAttention(attentionMatrix);

        System.out.printf("Attention Metrics Analysis:%n");
        System.out.printf("  Average Attention: %.4f%n", metrics.getAverageAttention());
        System.out.printf("  Attention Entropy: %.4f%n", metrics.getEntropy());

        // Additional transformer-specific analysis
        analyzeTransformerCharacteristics(attentionMatrix);
    }

    /**
     * Demonstrates performance validation following SuperML patterns.
     */
    private static void demonstratePerformanceValidation() {
        System.out.println("\n3. ⚡ Performance Validation");
        System.out.println("============================");

        // Test different sequence lengths for performance characteristics
        int[] sequenceLengths = {16, 32, 64, 128};

        for (int seqLen : sequenceLengths) {
            long startTime = System.nanoTime();

            // Generate and process sequence
            double[][] data = generateTransformerData(seqLen, 256);
            RealMatrix matrix = new Array2DRowRealMatrix(data);
            AttentionMetrics metrics = evaluateAttention(matrix);

            long endTime = System.nanoTime();
            double timeMs = (endTime - startTime) / 1_000_000.0;

            System.out.printf("Sequence %3d: %.2f ms (%.4f avg attention)%n",
                seqLen, timeMs, metrics.getAverageAttention());
        }
    }

    /**
     * Analyzes transformer-specific characteristics following SuperML metrics patterns.
     */
    private static void analyzeTransformerCharacteristics(RealMatrix matrix) {
        System.out.printf("%nTransformer Characteristics:%n");

        // Matrix properties
        int size = matrix.getRowDimension();
        double frobeniusNorm = calculateFrobeniusNorm(matrix);
        double sparsity = calculateSparsity(matrix);

        System.out.printf("  Matrix Size: %dx%d%n", size, size);
        System.out.printf("  Frobenius Norm: %.4f%n", frobeniusNorm);
        System.out.printf("  Sparsity: %.2f%%%n", sparsity * 100);

        // Attention distribution analysis
        double[] rowSums = calculateRowSums(matrix);
        double avgRowSum = calculateMean(rowSums);
        double stdRowSum = calculateStandardDeviation(rowSums, avgRowSum);

        System.out.printf("  Row Sum Statistics:%n");
        System.out.printf("    Mean: %.4f%n", avgRowSum);
        System.out.printf("    Std Dev: %.4f%n", stdRowSum);

        // Validate attention properties
        validateAttentionProperties(rowSums, avgRowSum);
    }

    /**
     * Generates realistic transformer-style data with positional encoding patterns.
     */
    private static double[][] generateTransformerData(int sequenceLength, int dimensions) {
        double[][] data = new double[sequenceLength][dimensions];

        for (int i = 0; i < sequenceLength; i++) {
            for (int j = 0; j < dimensions; j++) {
                // Base random embeddings
                data[i][j] = (Math.random() - 0.5) * 2.0; // Range [-1, 1]

                // Add positional encoding-like patterns
                double pos = i / 10000.0;
                double dimScale = Math.pow(dimensions, j / (double) dimensions);

                if (j % 2 == 0) {
                    data[i][j] *= Math.sin(pos * dimScale);
                } else {
                    data[i][j] *= Math.cos(pos * dimScale);
                }

                // Add some transformer-like attention patterns
                if (i > 0 && j < dimensions / 4) {
                    data[i][j] += 0.1 * data[i-1][j]; // Sequential dependency
                }
            }
        }
        return data;
    }

    /**
     * Validates transformation results following SuperML validation patterns.
     */
    private static void validateTransformation(double[][] input, double[][] output) {
        System.out.printf("Transformation Validation:%n");

        // Dimension preservation
        boolean dimensionsMatch = (input.length == output.length &&
                                 input[0].length == output[0].length);
        System.out.printf("  Dimensions preserved: %s%n", dimensionsMatch ? "✅" : "❌");

        // Check for valid numerical output
        boolean hasValidOutput = hasValidNumericalValues(output);
        System.out.printf("  Valid numerical output: %s%n", hasValidOutput ? "✅" : "❌");

        // Calculate transformation magnitude
        double inputMagnitude = calculateMatrixMagnitude(input);
        double outputMagnitude = calculateMatrixMagnitude(output);
        double magnitudeRatio = outputMagnitude / inputMagnitude;

        System.out.printf("  Magnitude ratio: %.4f%n", magnitudeRatio);
    }

    // Utility methods following SuperML framework patterns

    private static double calculateFrobeniusNorm(RealMatrix matrix) {
        double sum = 0;
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                double val = matrix.getEntry(i, j);
                sum += val * val;
            }
        }
        return Math.sqrt(sum);
    }

    private static double calculateSparsity(RealMatrix matrix) {
        int zeros = 0;
        int total = matrix.getRowDimension() * matrix.getColumnDimension();

        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                if (Math.abs(matrix.getEntry(i, j)) < 1e-6) {
                    zeros++;
                }
            }
        }

        return (double) zeros / total;
    }

    private static double[] calculateRowSums(RealMatrix matrix) {
        int rows = matrix.getRowDimension();
        double[] rowSums = new double[rows];

        for (int i = 0; i < rows; i++) {
            double sum = 0;
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                sum += matrix.getEntry(i, j);
            }
            rowSums[i] = sum;
        }

        return rowSums;
    }

    private static double calculateMean(double[] values) {
        double sum = 0;
        for (double value : values) {
            sum += value;
        }
        return sum / values.length;
    }

    private static double calculateStandardDeviation(double[] values, double mean) {
        double sumSquaredDiffs = 0;
        for (double value : values) {
            double diff = value - mean;
            sumSquaredDiffs += diff * diff;
        }
        return Math.sqrt(sumSquaredDiffs / values.length);
    }

    private static boolean hasValidNumericalValues(double[][] matrix) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                if (Double.isNaN(matrix[i][j]) || Double.isInfinite(matrix[i][j])) {
                    return false;
                }
            }
        }
        return true;
    }

    private static double calculateMatrixMagnitude(double[][] matrix) {
        double sum = 0;
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                sum += matrix[i][j] * matrix[i][j];
            }
        }
        return Math.sqrt(sum);
    }

    private static void validateAttentionProperties(double[] rowSums, double avgRowSum) {
        // Check if attention weights are reasonable
        boolean reasonable = avgRowSum > 0.1 && avgRowSum < 10.0;
        System.out.printf("    Reasonable magnitudes: %s%n", reasonable ? "✅" : "❌");

        // Check for consistent attention distribution
        int validRows = 0;
        for (double sum : rowSums) {
            if (sum > 0 && !Double.isNaN(sum)) {
                validRows++;
            }
        }
        double validRatio = (double) validRows / rowSums.length;
        System.out.printf("    Valid attention rows: %.1f%%%n", validRatio * 100);
    }

    /**
     * Custom AttentionMetrics class to replace missing TransformerMetrics
     */
    private static class AttentionMetrics {
        private final double averageAttention;
        private final double entropy;

        public AttentionMetrics(double averageAttention, double entropy) {
            this.averageAttention = averageAttention;
            this.entropy = entropy;
        }

        public double getAverageAttention() {
            return averageAttention;
        }

        public double getEntropy() {
            return entropy;
        }
    }

    /**
     * Custom attention evaluation method to replace missing TransformerMetrics.evaluateAttention
     */
    private static AttentionMetrics evaluateAttention(RealMatrix matrix) {
        // Calculate average attention
        double sum = 0;
        int count = 0;
        
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                sum += Math.abs(matrix.getEntry(i, j));
                count++;
            }
        }
        double avgAttention = sum / count;

        // Calculate entropy (simplified version)
        double entropy = 0;
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            double[] row = matrix.getRow(i);
            double rowSum = 0;
            
            // Normalize row to create probability distribution
            for (double val : row) {
                rowSum += Math.abs(val);
            }
            
            if (rowSum > 0) {
                for (double val : row) {
                    double p = Math.abs(val) / rowSum;
                    if (p > 0) {
                        entropy -= p * Math.log(p) / Math.log(2.0);
                    }
                }
            }
        }
        entropy /= matrix.getRowDimension(); // Average entropy per row

        return new AttentionMetrics(avgAttention, entropy);
    }
}
