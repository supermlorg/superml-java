package org.superml.visualization.transformers.advanced;

import org.superml.metrics.transformers.advanced.AdvancedGradientMetrics.*;
import java.util.List;
import java.util.function.Function;
import java.util.stream.IntStream;

/**
 * Interactive visualization for advanced transformer gradient analysis.
 * Follows SuperML visualization specialist pattern.
 */
public class AdvancedGradientVisualization {

    private static final int DEFAULT_PLOT_WIDTH = 80;
    private static final String HEADER_STYLE = "=";
    private static final String SUBHEADER_STYLE = "-";

    /**
     * Visualizes comprehensive gradient analysis with interaction options.
     */
    public static void visualizeGradientAnalysis(
            GradientAdvancedAnalysis analysis,
            VisualizationOptions options) {

        if (options.showHessianMetrics) {
            visualizeHessianMetrics(analysis.hessianMetrics);
        }

        if (options.showLandscapeMetrics) {
            visualizeLandscapeMetrics(analysis.landscapeMetrics);
        }

        if (options.showOptimizationMetrics) {
            visualizeOptimizationMetrics(analysis.optimizationMetrics);
        }

        if (options.showSummary) {
            visualizeAnalysisSummary(analysis);
        }
    }

    private static void visualizeHessianMetrics(HessianApproximation metrics) {
        printHeader("Hessian Analysis");

        // Eigenvalue spectrum visualization
        System.out.println("\nEigenvalue Spectrum:");
        visualizeTimeSeries(metrics.eigenvalueEstimates,
            "Layer", "Eigenvalue", true);

        // Condition number trend
        System.out.println("\nCondition Number Trend:");
        visualizeTimeSeries(metrics.conditionNumbers,
            "Layer", "Condition Number", true);

        // Stability indicators
        double avgCondition = metrics.conditionNumbers.stream()
            .mapToDouble(Double::doubleValue)
            .average()
            .orElse(0.0);

        System.out.printf("%nStability Indicators:%n");
        System.out.printf("Average Condition Number: %.2f%n", avgCondition);
        visualizeStabilityGauge(avgCondition);
    }

    private static void visualizeLandscapeMetrics(GradientLandscapeMetrics metrics) {
        printHeader("Loss Landscape Analysis");

        // Flatness/sharpness comparison
        System.out.println("\nLoss Surface Geometry:");
        visualizeParallelSeries(
            metrics.flatness,
            metrics.sharpness,
            "Layer",
            "Flatness vs Sharpness"
        );

        // Optimization path visualization
        System.out.println("\nOptimization Path Length:");
        visualizeTimeSeries(metrics.pathLengths,
            "Step", "Path Length", false);

        // Surface characteristics summary
        double avgFlatness = average(metrics.flatness);
        double avgSharpness = average(metrics.sharpness);
        System.out.printf("%nSurface Characteristics:%n");
        System.out.printf("Average Flatness: %.4f%n", avgFlatness);
        System.out.printf("Average Sharpness: %.4f%n", avgSharpness);
        System.out.printf("Flatness/Sharpness Ratio: %.4f%n",
            avgFlatness / (avgSharpness + 1e-12));
    }

    private static void visualizeOptimizationMetrics(OptimizationMetrics metrics) {
        printHeader("Optimization Dynamics");

        // Gradient norm progression
        System.out.println("\nGradient Norm Progression:");
        visualizeTimeSeries(metrics.gradientNorms,
            "Step", "Gradient Norm", true);

        // Update ratio analysis
        System.out.println("\nParameter Update Ratios:");
        visualizeTimeSeries(metrics.updateRatios,
            "Step", "Update Ratio", false);

        // Effective learning rate
        System.out.println("\nEffective Learning Rate:");
        visualizeTimeSeries(metrics.effectiveLearningRates,
            "Step", "Effective LR", false);

        // Optimization health indicators
        visualizeOptimizationHealth(metrics);
    }

    private static void visualizeAnalysisSummary(GradientAdvancedAnalysis analysis) {
        printHeader("Analysis Summary");

        // Optimization quality score
        double optimScore = calculateOptimizationScore(analysis);
        System.out.printf("%nOptimization Quality: %.2f/10.0%n", optimScore);
        visualizeQualityBar(optimScore / 10.0);

        // Key metrics summary
        System.out.println("\nKey Metrics:");
        summarizeKeyMetrics(analysis);

        // Recommendations based on analysis
        System.out.println("\nRecommendations:");
        generateRecommendations(analysis);
    }

    private static void visualizeTimeSeries(
            List<Double> values,
            String xLabel,
            String yLabel,
            boolean logScale) {

        int width = DEFAULT_PLOT_WIDTH;
        int height = 20;

        double minVal = values.stream().mapToDouble(Double::doubleValue).min().orElse(0);
        double maxVal = values.stream().mapToDouble(Double::doubleValue).max().orElse(1);

        if (logScale) {
            minVal = Math.log10(Math.max(minVal, 1e-12));
            maxVal = Math.log10(Math.max(maxVal, 1e-12));
        }

        System.out.printf("%s vs %s:%n", yLabel, xLabel);

        // Plot frame
        String horizontalBorder = "+" + "-".repeat(width + 2) + "+";
        System.out.println(horizontalBorder);

        // Plot data points
        for (int i = height - 1; i >= 0; i--) {
            System.out.print("|");
            for (int j = 0; j < width; j++) {
                int dataIndex = j * values.size() / width;
                if (dataIndex < values.size()) {
                    double value = values.get(dataIndex);
                    if (logScale) {
                        value = Math.log10(Math.max(value, 1e-12));
                    }
                    double normalizedValue = (value - minVal) / (maxVal - minVal);
                    System.out.print(getPlotChar(normalizedValue, i / (double) height));
                } else {
                    System.out.print(" ");
                }
            }
            System.out.println("|");
        }

        System.out.println(horizontalBorder);
    }

    private static void visualizeParallelSeries(
            List<Double> series1,
            List<Double> series2,
            String xLabel,
            String title) {

        System.out.printf("%s:%n", title);
        int width = DEFAULT_PLOT_WIDTH;

        // Normalize both series
        double max1 = series1.stream().mapToDouble(Double::doubleValue).max().orElse(1.0);
        double max2 = series2.stream().mapToDouble(Double::doubleValue).max().orElse(1.0);

        for (int i = 0; i < Math.min(series1.size(), series2.size()); i++) {
            double norm1 = series1.get(i) / max1;
            double norm2 = series2.get(i) / max2;

            int pos1 = (int)(norm1 * width);
            int pos2 = (int)(norm2 * width);

            StringBuilder line = new StringBuilder(" ".repeat(width));
            line.setCharAt(Math.min(pos1, width - 1), '*');
            line.setCharAt(Math.min(pos2, width - 1), '+');

            System.out.printf("%3d |%s|%n", i, line);
        }
    }

    private static void visualizeStabilityGauge(double condition) {
        int width = 40;
        double normalizedValue = Math.min(Math.log10(condition) / 6.0, 1.0);
        int position = (int)(normalizedValue * width);

        System.out.print("Stability: [");
        for (int i = 0; i < width; i++) {
            if (i < position) {
                System.out.print("=");
            } else if (i == position) {
                System.out.print(">");
            } else {
                System.out.print(" ");
            }
        }
        System.out.println("]");
    }

    private static void visualizeOptimizationHealth(OptimizationMetrics metrics) {
        System.out.println("\nOptimization Health Indicators:");

        // Gradient norm stability
        double normStability = calculateNormStability(metrics.gradientNorms);
        System.out.printf("Gradient Stability: %.2f%n", normStability);
        visualizeHealthBar(normStability);

        // Learning rate efficiency
        double lrEfficiency = calculateLREfficiency(metrics.effectiveLearningRates);
        System.out.printf("Learning Rate Efficiency: %.2f%n", lrEfficiency);
        visualizeHealthBar(lrEfficiency);

        // Update ratio consistency
        double updateConsistency = calculateUpdateConsistency(metrics.updateRatios);
        System.out.printf("Update Consistency: %.2f%n", updateConsistency);
        visualizeHealthBar(updateConsistency);
    }

    private static void visualizeHealthBar(double value) {
        int width = 40;
        int filled = (int)(value * width);
        System.out.print("[");
        System.out.print("=".repeat(filled));
        System.out.print(" ".repeat(width - filled));
        System.out.println("]");
    }

    private static void visualizeQualityBar(double quality) {
        int width = 50;
        int filled = (int)(quality * width);
        System.out.print("[");
        for (int i = 0; i < width; i++) {
            if (i < filled) {
                System.out.print(i < width / 3 ? "!" :
                               i < 2 * width / 3 ? "=" : "#");
            } else {
                System.out.print(" ");
            }
        }
        System.out.println("]");
    }

    private static String getPlotChar(double normalizedValue, double threshold) {
        return normalizedValue >= threshold ? "█" : " ";
    }

    private static double average(List<Double> values) {
        return values.stream()
            .mapToDouble(Double::doubleValue)
            .average()
            .orElse(0.0);
    }

    private static double calculateNormStability(List<Double> norms) {
        if (norms.isEmpty()) return 0.0;
        double max = norms.stream().mapToDouble(Double::doubleValue).max().orElse(1.0);
        double variation = norms.stream()
            .mapToDouble(n -> Math.abs(n - max))
            .average()
            .orElse(0.0);
        return Math.max(0.0, 1.0 - variation / max);
    }

    private static double calculateLREfficiency(List<Double> learningRates) {
        if (learningRates.isEmpty()) return 0.0;
        double optimal = learningRates.stream()
            .mapToDouble(Double::doubleValue)
            .max()
            .orElse(1.0);
        return learningRates.stream()
            .mapToDouble(lr -> lr / optimal)
            .average()
            .orElse(0.0);
    }

    private static double calculateUpdateConsistency(List<Double> updateRatios) {
        if (updateRatios.isEmpty()) return 0.0;
        double mean = average(updateRatios);
        double variance = updateRatios.stream()
            .mapToDouble(r -> Math.pow(r - mean, 2))
            .average()
            .orElse(0.0);
        return Math.exp(-variance);
    }

    private static double calculateOptimizationScore(GradientAdvancedAnalysis analysis) {
        double conditionScore = analysis.hessianMetrics.conditionNumbers.stream()
            .mapToDouble(c -> Math.exp(-Math.log10(c) / 6.0))
            .average()
            .orElse(0.0);

        double landscapeScore = analysis.landscapeMetrics.flatness.stream()
            .mapToDouble(f -> Math.exp(-f))
            .average()
            .orElse(0.0);

        double optimizationScore = calculateNormStability(
            analysis.optimizationMetrics.gradientNorms);

        return (conditionScore + landscapeScore + optimizationScore) * 10.0 / 3.0;
    }

    private static void summarizeKeyMetrics(GradientAdvancedAnalysis analysis) {
        // Condition number trend
        System.out.println("Condition Numbers:");
        printMetricSummary(analysis.hessianMetrics.conditionNumbers);

        // Loss landscape
        System.out.println("\nLoss Surface:");
        System.out.printf("Avg Flatness: %.4f%n",
            average(analysis.landscapeMetrics.flatness));
        System.out.printf("Avg Sharpness: %.4f%n",
            average(analysis.landscapeMetrics.sharpness));

        // Optimization dynamics
        System.out.println("\nOptimization:");
        System.out.printf("Final Gradient Norm: %.4e%n",
            analysis.optimizationMetrics.gradientNorms.get(
                analysis.optimizationMetrics.gradientNorms.size() - 1));
    }

    private static void generateRecommendations(GradientAdvancedAnalysis analysis) {
        // Based on condition numbers
        double avgCondition = average(analysis.hessianMetrics.conditionNumbers);
        if (avgCondition > 1000) {
            System.out.println("⚠ High condition numbers detected:");
            System.out.println("  → Consider gradient clipping");
            System.out.println("  → Try reducing learning rate");
        }

        // Based on landscape
        double avgSharpness = average(analysis.landscapeMetrics.sharpness);
        if (avgSharpness > 0.1) {
            System.out.println("⚠ Sharp loss landscape detected:");
            System.out.println("  → Consider adding regularization");
            System.out.println("  → Try learning rate warmup");
        }

        // Based on optimization
        List<Double> norms = analysis.optimizationMetrics.gradientNorms;
        if (norms.get(norms.size() - 1) > 0.1) {
            System.out.println("⚠ Large final gradients:");
            System.out.println("  → Training may need more iterations");
            System.out.println("  → Consider adjusting batch size");
        }
    }

    private static void printMetricSummary(List<Double> values) {
        double min = values.stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
        double max = values.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
        double avg = average(values);

        System.out.printf("Min: %.2e | Avg: %.2e | Max: %.2e%n", min, avg, max);
    }

    private static void printHeader(String text) {
        System.out.println("\n" + HEADER_STYLE.repeat(text.length()));
        System.out.println(text);
        System.out.println(HEADER_STYLE.repeat(text.length()));
    }

    public static class VisualizationOptions {
        public final boolean showHessianMetrics;
        public final boolean showLandscapeMetrics;
        public final boolean showOptimizationMetrics;
        public final boolean showSummary;

        public VisualizationOptions(
                boolean showHessianMetrics,
                boolean showLandscapeMetrics,
                boolean showOptimizationMetrics,
                boolean showSummary) {
            this.showHessianMetrics = showHessianMetrics;
            this.showLandscapeMetrics = showLandscapeMetrics;
            this.showOptimizationMetrics = showOptimizationMetrics;
            this.showSummary = showSummary;
        }

        public static VisualizationOptions all() {
            return new VisualizationOptions(true, true, true, true);
        }

        public static VisualizationOptions summary() {
            return new VisualizationOptions(false, false, false, true);
        }
    }
}
