package org.superml.visualization.transformers;

import org.superml.metrics.transformers.TransformerPerformanceMetrics.AttentionEvaluation;
import org.superml.metrics.transformers.TransformerPerformanceMetrics.CacheMetrics;

/**
 * Visualization utilities for transformer performance metrics.
 * Follows SuperML visualization specialist pattern for transformer analysis.
 */
public class TransformerPerformanceVisualization {

    public static void visualizePerformanceResults(AttentionEvaluation evaluation) {
        printHeader("Transformer Performance Analysis");

        // Attention Statistics
        printSection("Attention Statistics");
        System.out.printf("Average Attention: %.4f%n", evaluation.statistics.averageAttention);
        System.out.printf("Attention Entropy: %.4f%n", evaluation.statistics.entropy);
        System.out.printf("Attention Sparsity: %.2f%%%n", evaluation.statistics.sparsity * 100);

        // Memory Metrics
        printSection("Memory Usage");
        System.out.printf("Memory Allocated: %.2f MB%n", evaluation.memoryMetrics.bytesAllocated / (1024.0 * 1024.0));
        System.out.printf("Processing Time: %.3f ms%n", evaluation.memoryMetrics.processingTimeMs);

        // Cache Performance
        printSection("Cache Performance");
        visualizeCacheMetrics(evaluation.cacheMetrics);

        // Overall Performance
        printSection("Overall Performance");
        System.out.printf("Total Execution Time: %.3f ms%n", evaluation.totalTimeMs);
    }

    private static void visualizeCacheMetrics(CacheMetrics metrics) {
        System.out.printf("Cache Lines Accessed: %.0f%n", metrics.cacheLinesAccessed);
        System.out.printf("Cache Efficiency: %.1f%%%n", metrics.cacheEfficiency * 100);

        // Performance comparison bar chart
        System.out.println("\nAccess Pattern Comparison:");
        visualizeBar("Row-major ", metrics.rowMajorTimeMs, metrics.colMajorTimeMs);
        visualizeBar("Col-major ", metrics.colMajorTimeMs, metrics.colMajorTimeMs);
    }

    private static void visualizeBar(String label, double value, double maxValue) {
        int width = 40;
        int filled = (int) (value / maxValue * width);
        System.out.printf("%s |%s%s| %.2fms%n",
            label,
            "=".repeat(filled),
            " ".repeat(width - filled),
            value);
    }

    private static void printHeader(String text) {
        String border = "=".repeat(text.length());
        System.out.println("\n" + border);
        System.out.println(text);
        System.out.println(border);
    }

    private static void printSection(String section) {
        System.out.printf("%n%s:%n", section);
        System.out.println("-".repeat(section.length() + 1));
    }
}
