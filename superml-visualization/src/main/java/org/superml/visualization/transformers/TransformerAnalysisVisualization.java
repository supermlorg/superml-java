package org.superml.visualization.transformers;

import org.superml.metrics.transformers.TransformerAnalysisMetrics.*;
import java.util.List;

/**
 * Specialized visualization utilities for transformer analysis metrics.
 * Follows SuperML visualization specialist pattern.
 */
public class TransformerAnalysisVisualization {

    public static void visualizeAttentionAnalysis(AttentionAnalysis analysis) {
        printHeader("Attention Pattern Analysis");

        // Pattern metrics visualization
        printSection("Pattern Metrics");
        System.out.printf("Matrix Size: %d x %d%n",
            analysis.patternMetrics.size,
            analysis.patternMetrics.size);
        System.out.printf("Sparsity: %.2f%%%n",
            analysis.patternMetrics.sparsity * 100);
        System.out.printf("Entropy: %.4f%n",
            analysis.patternMetrics.entropy);
        System.out.printf("Head Diversity: %.4f%n",
            analysis.patternMetrics.headDiversity);

        // Head importance visualization
        printSection("Head Importance Distribution");
        visualizeHeadImportance(analysis.patternMetrics.headImportance);

        // Distribution metrics
        printSection("Attention Distribution");
        visualizeDistributionMetrics(analysis.distributionMetrics);
    }

    public static void visualizeGradientFlow(GradientFlowAnalysis analysis) {
        printHeader("Gradient Flow Analysis");

        // Layer-wise gradient metrics
        printSection("Layer-wise Gradient Analysis");
        visualizeLayerGradients(analysis.gradientMetrics);

        // Activation patterns
        printSection("Layer Activation Patterns");
        visualizeLayerActivations(analysis.activationMetrics);

        // Overall gradient health
        printSection("Gradient Health Summary");
        visualizeGradientHealth(analysis.gradientMetrics);
    }

    private static void visualizeHeadImportance(double[] importance) {
        int maxBarWidth = 40;
        System.out.println("Head Importance (normalized):");

        for (int i = 0; i < importance.length; i++) {
            int barWidth = (int)(importance[i] * maxBarWidth);
            System.out.printf("Head %2d: |%s| %.3f%n",
                i,
                "=".repeat(barWidth) + " ".repeat(maxBarWidth - barWidth),
                importance[i]);
        }
    }

    private static void visualizeDistributionMetrics(DistributionMetrics metrics) {
        System.out.printf("Mean: %.4f%n", metrics.mean);
        System.out.printf("Std Dev: %.4f%n", metrics.stdDev);
        System.out.printf("Skewness: %.4f%n", metrics.skewness);
        System.out.printf("Kurtosis: %.4f%n", metrics.kurtosis);
        System.out.println("\nValue Distribution:");

        // ASCII histogram of 5-95 percentile range
        int bins = 20;
        double range = metrics.percentile95 - metrics.percentile5;
        double binWidth = range / bins;

        for (int i = 0; i < bins; i++) {
            double binStart = metrics.percentile5 + i * binWidth;
            System.out.printf("%.3f: ", binStart);

            // Calculate approximate height based on normal distribution
            double z = (binStart - metrics.mean) / metrics.stdDev;
            int height = (int)(20 * Math.exp(-0.5 * z * z));
            System.out.println("*".repeat(height));
        }
    }

    private static void visualizeLayerGradients(LayerGradientMetrics metrics) {
        System.out.println("Gradient Norm Distribution:");
        for (int i = 0; i < metrics.layerNorms.size(); i++) {
            double norm = metrics.layerNorms.get(i);
            double variance = metrics.gradientVariance.get(i);

            System.out.printf("Layer %2d: ", i);
            visualizeBar(norm / metrics.maxGradientNorm);
            System.out.printf(" Norm: %.4f (Var: %.4f)%n", norm, variance);
        }

        System.out.printf("%nGradient Correlation: %.4f%n",
            metrics.gradientCorrelation);
    }

    private static void visualizeLayerActivations(LayerActivationMetrics metrics) {
        System.out.println("Layer Activation Patterns:");

        for (int i = 0; i < metrics.meanActivations.size(); i++) {
            double mean = metrics.meanActivations.get(i);
            double range = metrics.activationRanges.get(i);
            double dead = metrics.deadUnits.get(i);

            System.out.printf("Layer %2d:%n", i);
            System.out.printf("  Mean: %.4f | Range: %.4f | Dead: %.2f%%%n",
                mean, range, dead * 100);

            // Visualize activation distribution
            System.out.print("  Distribution: ");
            visualizeBar(mean / range);
            System.out.println();
        }
    }

    private static void visualizeGradientHealth(LayerGradientMetrics metrics) {
        double avgNorm = metrics.layerNorms.stream()
            .mapToDouble(Double::doubleValue)
            .average()
            .orElse(0.0);

        double avgVariance = metrics.gradientVariance.stream()
            .mapToDouble(Double::doubleValue)
            .average()
            .orElse(0.0);

        System.out.println("Gradient Health Indicators:");
        System.out.printf("Average Gradient Norm: %.4f%n", avgNorm);
        System.out.printf("Average Gradient Variance: %.4f%n", avgVariance);
        System.out.printf("Max Gradient Norm: %.4f%n", metrics.maxGradientNorm);

        // Gradient stability assessment
        double stabilityScore = calculateStabilityScore(
            metrics.layerNorms,
            metrics.gradientVariance,
            metrics.gradientCorrelation
        );

        System.out.print("\nGradient Stability: ");
        visualizeHealthIndicator(stabilityScore);
        System.out.printf(" (%.2f)%n", stabilityScore);
    }

    private static double calculateStabilityScore(
            List<Double> norms,
            List<Double> variances,
            double correlation) {
        double normStability = 1.0 - calculateCoeffOfVariation(norms);
        double varStability = 1.0 - calculateMeanNormalizedVariance(variances);
        double corrStability = (correlation + 1.0) / 2.0; // Map [-1,1] to [0,1]

        return (normStability + varStability + corrStability) / 3.0;
    }

    private static double calculateCoeffOfVariation(List<Double> values) {
        double mean = values.stream()
            .mapToDouble(Double::doubleValue)
            .average()
            .orElse(0.0);

        double variance = values.stream()
            .mapToDouble(v -> (v - mean) * (v - mean))
            .average()
            .orElse(0.0);

        return Math.sqrt(variance) / mean;
    }

    private static double calculateMeanNormalizedVariance(List<Double> variances) {
        double maxVar = variances.stream()
            .mapToDouble(Double::doubleValue)
            .max()
            .orElse(1.0);

        return variances.stream()
            .mapToDouble(v -> v / maxVar)
            .average()
            .orElse(0.0);
    }

    private static void visualizeBar(double value) {
        int width = 40;
        int filled = (int)(value * width);
        System.out.print("|" + "=".repeat(filled) +
                        " ".repeat(width - filled) + "|");
    }

    private static void visualizeHealthIndicator(double score) {
        String indicator;
        if (score > 0.8) {
            indicator = "✓✓✓ Excellent";
        } else if (score > 0.6) {
            indicator = "✓✓  Good";
        } else if (score > 0.4) {
            indicator = "✓   Fair";
        } else {
            indicator = "⚠   Poor";
        }
        System.out.print(indicator);
    }

    private static void printHeader(String text) {
        System.out.println("\n" + "=".repeat(text.length()));
        System.out.println(text);
        System.out.println("=".repeat(text.length()));
    }

    private static void printSection(String text) {
        System.out.println("\n" + text);
        System.out.println("-".repeat(text.length()));
    }
}
