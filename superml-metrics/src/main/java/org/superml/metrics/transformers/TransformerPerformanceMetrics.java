package org.superml.metrics.transformers;

import org.apache.commons.math3.linear.RealMatrix;
import java.util.Map;
import java.util.HashMap;

/**
 * Performance metrics for transformer model evaluation.
 * Follows SuperML metrics specialist pattern for transformer-specific analysis.
 */
public class TransformerPerformanceMetrics {

    /**
     * Evaluates attention mechanism performance including memory and cache metrics.
     */
    public static AttentionEvaluation evaluateAttentionPerformance(RealMatrix attentionMatrix, int cacheLineSize) {
        long startTime = System.nanoTime();
        MemoryMetrics memoryMetrics = measureMemoryUsage(() ->
            calculateMatrixStatistics(attentionMatrix));

        CacheMetrics cacheMetrics = analyzeCachePerformance(attentionMatrix, cacheLineSize);
        long endTime = System.nanoTime();

        return new AttentionEvaluation(
            calculateAttentionStatistics(attentionMatrix),
            memoryMetrics,
            cacheMetrics,
            (endTime - startTime) / 1_000_000.0 // Convert to ms
        );
    }

    private static AttentionStatistics calculateAttentionStatistics(RealMatrix attentionMatrix) {
        double avgAttention = calculateAverageAttention(attentionMatrix);
        double entropy = calculateAttentionEntropy(attentionMatrix);
        double sparsity = calculateSparsity(attentionMatrix);

        return new AttentionStatistics(avgAttention, entropy, sparsity);
    }

    private static double calculateAverageAttention(RealMatrix matrix) {
        double sum = 0;
        int rows = matrix.getRowDimension();
        int cols = matrix.getColumnDimension();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                sum += matrix.getEntry(i, j);
            }
        }

        return sum / (rows * cols);
    }

    private static double calculateAttentionEntropy(RealMatrix matrix) {
        double entropy = 0;
        int rows = matrix.getRowDimension();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                double p = matrix.getEntry(i, j);
                if (p > 0) {
                    entropy -= p * Math.log(p);
                }
            }
        }

        return entropy;
    }

    private static double calculateSparsity(RealMatrix matrix) {
        int rows = matrix.getRowDimension();
        int cols = matrix.getColumnDimension();
        int zeros = 0;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (Math.abs(matrix.getEntry(i, j)) < 1e-10) {
                    zeros++;
                }
            }
        }

        return (double) zeros / (rows * cols);
    }

    private static CacheMetrics analyzeCachePerformance(RealMatrix matrix, int cacheLineSize) {
        int matrixSize = matrix.getRowDimension();
        double cacheLinesAccessed = (matrixSize * matrixSize * 8 + cacheLineSize - 1) / cacheLineSize;

        // Measure row-major vs column-major access times
        long rowMajorTime = measureRowMajorAccess(matrix);
        long colMajorTime = measureColumnMajorAccess(matrix);

        double cacheEfficiency = 1.0 - ((double) colMajorTime - rowMajorTime) / colMajorTime;

        return new CacheMetrics(
            cacheLinesAccessed,
            rowMajorTime / 1_000_000.0, // Convert to ms
            colMajorTime / 1_000_000.0,
            cacheEfficiency
        );
    }

    private static long measureRowMajorAccess(RealMatrix matrix) {
        long startTime = System.nanoTime();
        double sum = 0;
        int rows = matrix.getRowDimension();
        int cols = matrix.getColumnDimension();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                sum += matrix.getEntry(i, j);
            }
        }

        return System.nanoTime() - startTime;
    }

    private static long measureColumnMajorAccess(RealMatrix matrix) {
        long startTime = System.nanoTime();
        double sum = 0;
        int rows = matrix.getRowDimension();
        int cols = matrix.getColumnDimension();

        for (int j = 0; j < cols; j++) {
            for (int i = 0; i < rows; i++) {
                sum += matrix.getEntry(i, j);
            }
        }

        return System.nanoTime() - startTime;
    }

    private static MemoryMetrics measureMemoryUsage(Runnable operation) {
        Runtime runtime = Runtime.getRuntime();
        runtime.gc();

        long memoryBefore = runtime.totalMemory() - runtime.freeMemory();
        long startTime = System.nanoTime();

        operation.run();

        long endTime = System.nanoTime();
        long memoryAfter = runtime.totalMemory() - runtime.freeMemory();

        return new MemoryMetrics(
            Math.max(0, memoryAfter - memoryBefore),
            (endTime - startTime) / 1_000_000.0 // Convert to ms
        );
    }

    private static Map<String, Double> calculateMatrixStatistics(RealMatrix matrix) {
        Map<String, Double> stats = new HashMap<>();
        double sum = 0;
        double sumSquared = 0;
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;

        int rows = matrix.getRowDimension();
        int cols = matrix.getColumnDimension();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double value = matrix.getEntry(i, j);
                sum += value;
                sumSquared += value * value;
                min = Math.min(min, value);
                max = Math.max(max, value);
            }
        }

        double mean = sum / (rows * cols);
        double variance = (sumSquared / (rows * cols)) - (mean * mean);

        stats.put("mean", mean);
        stats.put("std", Math.sqrt(variance));
        stats.put("min", min);
        stats.put("max", max);

        return stats;
    }

    public static class AttentionEvaluation {
        public final AttentionStatistics statistics;
        public final MemoryMetrics memoryMetrics;
        public final CacheMetrics cacheMetrics;
        public final double totalTimeMs;

        public AttentionEvaluation(AttentionStatistics statistics,
                                 MemoryMetrics memoryMetrics,
                                 CacheMetrics cacheMetrics,
                                 double totalTimeMs) {
            this.statistics = statistics;
            this.memoryMetrics = memoryMetrics;
            this.cacheMetrics = cacheMetrics;
            this.totalTimeMs = totalTimeMs;
        }
    }

    public static class AttentionStatistics {
        public final double averageAttention;
        public final double entropy;
        public final double sparsity;

        public AttentionStatistics(double averageAttention, double entropy, double sparsity) {
            this.averageAttention = averageAttention;
            this.entropy = entropy;
            this.sparsity = sparsity;
        }
    }

    public static class MemoryMetrics {
        public final long bytesAllocated;
        public final double processingTimeMs;

        public MemoryMetrics(long bytesAllocated, double processingTimeMs) {
            this.bytesAllocated = bytesAllocated;
            this.processingTimeMs = processingTimeMs;
        }
    }

    public static class CacheMetrics {
        public final double cacheLinesAccessed;
        public final double rowMajorTimeMs;
        public final double colMajorTimeMs;
        public final double cacheEfficiency;

        public CacheMetrics(double cacheLinesAccessed,
                          double rowMajorTimeMs,
                          double colMajorTimeMs,
                          double cacheEfficiency) {
            this.cacheLinesAccessed = cacheLinesAccessed;
            this.rowMajorTimeMs = rowMajorTimeMs;
            this.colMajorTimeMs = colMajorTimeMs;
            this.cacheEfficiency = cacheEfficiency;
        }
    }
}
