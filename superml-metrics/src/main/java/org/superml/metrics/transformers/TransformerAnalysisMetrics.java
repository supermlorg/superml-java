package org.superml.metrics.transformers;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import java.util.*;

/**
 * Specialized metrics for analyzing transformer model characteristics
 * including gradient flow, attention patterns, and memory efficiency.
 */
public class TransformerAnalysisMetrics {

    /**
     * Analyzes attention pattern characteristics and health metrics
     */
    public static AttentionAnalysis analyzeAttentionPatterns(RealMatrix attentionMatrix) {
        AttentionPatternMetrics patternMetrics = calculatePatternMetrics(attentionMatrix);
        DistributionMetrics distributionMetrics = analyzeDistribution(attentionMatrix);

        return new AttentionAnalysis(patternMetrics, distributionMetrics);
    }

    /**
     * Analyzes gradient flow characteristics through attention layers
     */
    public static GradientFlowAnalysis analyzeGradientFlow(List<RealMatrix> gradients,
                                                          List<RealMatrix> layerOutputs) {
        LayerGradientMetrics gradientMetrics = calculateGradientMetrics(gradients);
        LayerActivationMetrics activationMetrics = calculateActivationMetrics(layerOutputs);

        return new GradientFlowAnalysis(gradientMetrics, activationMetrics);
    }

    private static AttentionPatternMetrics calculatePatternMetrics(RealMatrix attention) {
        int size = attention.getRowDimension();
        double sparsity = calculateSparsity(attention);
        double entropy = calculateEntropy(attention);
        double headDiversity = calculateHeadDiversity(attention);
        double[] headImportance = calculateHeadImportance(attention);

        return new AttentionPatternMetrics(
            size,
            sparsity,
            entropy,
            headDiversity,
            headImportance
        );
    }

    private static DistributionMetrics analyzeDistribution(RealMatrix attention) {
        DescriptiveStatistics stats = new DescriptiveStatistics();
        int rows = attention.getRowDimension();
        int cols = attention.getColumnDimension();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                stats.addValue(attention.getEntry(i, j));
            }
        }

        return new DistributionMetrics(
            stats.getMean(),
            stats.getStandardDeviation(),
            stats.getSkewness(),
            stats.getKurtosis(),
            stats.getPercentile(5),
            stats.getPercentile(95)
        );
    }

    private static LayerGradientMetrics calculateGradientMetrics(List<RealMatrix> gradients) {
        List<Double> layerNorms = new ArrayList<>();
        List<Double> gradientVariance = new ArrayList<>();
        double maxGradientNorm = 0.0;

        for (RealMatrix gradient : gradients) {
            double norm = calculateFrobeniusNorm(gradient);
            double variance = calculateVariance(gradient);

            layerNorms.add(norm);
            gradientVariance.add(variance);
            maxGradientNorm = Math.max(maxGradientNorm, norm);
        }

        return new LayerGradientMetrics(
            layerNorms,
            gradientVariance,
            maxGradientNorm,
            calculateGradientCorrelation(gradients)
        );
    }

    private static LayerActivationMetrics calculateActivationMetrics(List<RealMatrix> activations) {
        List<Double> meanActivations = new ArrayList<>();
        List<Double> activationRanges = new ArrayList<>();
        List<Double> deadUnits = new ArrayList<>();

        for (RealMatrix activation : activations) {
            meanActivations.add(calculateMean(activation));
            activationRanges.add(calculateRange(activation));
            deadUnits.add(calculateDeadUnits(activation));
        }

        return new LayerActivationMetrics(
            meanActivations,
            activationRanges,
            deadUnits
        );
    }

    // Utility methods for mathematical calculations
    private static double calculateSparsity(RealMatrix matrix) {
        int rows = matrix.getRowDimension();
        int cols = matrix.getColumnDimension();
        int zeros = 0;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (Math.abs(matrix.getEntry(i, j)) < 1e-6) {
                    zeros++;
                }
            }
        }

        return (double) zeros / (rows * cols);
    }

    private static double calculateEntropy(RealMatrix matrix) {
        double entropy = 0.0;
        int rows = matrix.getRowDimension();
        int cols = matrix.getColumnDimension();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double p = matrix.getEntry(i, j);
                if (p > 0) {
                    entropy -= p * Math.log(p);
                }
            }
        }

        return entropy / rows; // Normalize by number of rows
    }

    private static double calculateHeadDiversity(RealMatrix attention) {
        // Calculate diversity between attention heads using cosine similarity
        double diversity = 0.0;
        int heads = attention.getRowDimension();

        for (int i = 0; i < heads; i++) {
            for (int j = i + 1; j < heads; j++) {
                diversity += 1.0 - cosineSimilarity(
                    attention.getRow(i),
                    attention.getRow(j)
                );
            }
        }

        return diversity / (heads * (heads - 1) / 2.0);
    }

    private static double[] calculateHeadImportance(RealMatrix attention) {
        int heads = attention.getRowDimension();
        double[] importance = new double[heads];

        for (int i = 0; i < heads; i++) {
            importance[i] = calculateFrobeniusNorm(attention.getRowMatrix(i));
        }

        // Normalize importance scores
        double sum = Arrays.stream(importance).sum();
        for (int i = 0; i < heads; i++) {
            importance[i] /= sum;
        }

        return importance;
    }

    // Analysis result container classes
    public static class AttentionAnalysis {
        public final AttentionPatternMetrics patternMetrics;
        public final DistributionMetrics distributionMetrics;

        public AttentionAnalysis(AttentionPatternMetrics patternMetrics,
                               DistributionMetrics distributionMetrics) {
            this.patternMetrics = patternMetrics;
            this.distributionMetrics = distributionMetrics;
        }
    }

    public static class GradientFlowAnalysis {
        public final LayerGradientMetrics gradientMetrics;
        public final LayerActivationMetrics activationMetrics;

        public GradientFlowAnalysis(LayerGradientMetrics gradientMetrics,
                                  LayerActivationMetrics activationMetrics) {
            this.gradientMetrics = gradientMetrics;
            this.activationMetrics = activationMetrics;
        }
    }

    public static class AttentionPatternMetrics {
        public final int size;
        public final double sparsity;
        public final double entropy;
        public final double headDiversity;
        public final double[] headImportance;

        public AttentionPatternMetrics(int size, double sparsity, double entropy,
                                     double headDiversity, double[] headImportance) {
            this.size = size;
            this.sparsity = sparsity;
            this.entropy = entropy;
            this.headDiversity = headDiversity;
            this.headImportance = headImportance;
        }
    }

    public static class DistributionMetrics {
        public final double mean;
        public final double stdDev;
        public final double skewness;
        public final double kurtosis;
        public final double percentile5;
        public final double percentile95;

        public DistributionMetrics(double mean, double stdDev, double skewness,
                                 double kurtosis, double percentile5, double percentile95) {
            this.mean = mean;
            this.stdDev = stdDev;
            this.skewness = skewness;
            this.kurtosis = kurtosis;
            this.percentile5 = percentile5;
            this.percentile95 = percentile95;
        }
    }

    public static class LayerGradientMetrics {
        public final List<Double> layerNorms;
        public final List<Double> gradientVariance;
        public final double maxGradientNorm;
        public final double gradientCorrelation;

        public LayerGradientMetrics(List<Double> layerNorms, List<Double> gradientVariance,
                                  double maxGradientNorm, double gradientCorrelation) {
            this.layerNorms = layerNorms;
            this.gradientVariance = gradientVariance;
            this.maxGradientNorm = maxGradientNorm;
            this.gradientCorrelation = gradientCorrelation;
        }
    }

    public static class LayerActivationMetrics {
        public final List<Double> meanActivations;
        public final List<Double> activationRanges;
        public final List<Double> deadUnits;

        public LayerActivationMetrics(List<Double> meanActivations,
                                    List<Double> activationRanges,
                                    List<Double> deadUnits) {
            this.meanActivations = meanActivations;
            this.activationRanges = activationRanges;
            this.deadUnits = deadUnits;
        }
    }

    // Additional mathematical utility methods
    private static double calculateFrobeniusNorm(RealMatrix matrix) {
        double sum = 0.0;
        int rows = matrix.getRowDimension();
        int cols = matrix.getColumnDimension();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double val = matrix.getEntry(i, j);
                sum += val * val;
            }
        }

        return Math.sqrt(sum);
    }

    private static double calculateVariance(RealMatrix matrix) {
        DescriptiveStatistics stats = new DescriptiveStatistics();
        int rows = matrix.getRowDimension();
        int cols = matrix.getColumnDimension();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                stats.addValue(matrix.getEntry(i, j));
            }
        }

        return stats.getVariance();
    }

    private static double calculateGradientCorrelation(List<RealMatrix> gradients) {
        if (gradients.size() < 2) return 0.0;

        double correlation = 0.0;
        int count = 0;

        for (int i = 0; i < gradients.size() - 1; i++) {
            RealMatrix current = gradients.get(i);
            RealMatrix next = gradients.get(i + 1);

            if (current.getRowDimension() == next.getRowDimension() &&
                current.getColumnDimension() == next.getColumnDimension()) {
                correlation += cosineSimilarity(
                    flattenMatrix(current),
                    flattenMatrix(next)
                );
                count++;
            }
        }

        return count > 0 ? correlation / count : 0.0;
    }

    private static double cosineSimilarity(double[] v1, double[] v2) {
        double dotProduct = 0.0;
        double norm1 = 0.0;
        double norm2 = 0.0;

        for (int i = 0; i < v1.length; i++) {
            dotProduct += v1[i] * v2[i];
            norm1 += v1[i] * v1[i];
            norm2 += v2[i] * v2[i];
        }

        return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }

    private static double[] flattenMatrix(RealMatrix matrix) {
        int rows = matrix.getRowDimension();
        int cols = matrix.getColumnDimension();
        double[] flattened = new double[rows * cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                flattened[i * cols + j] = matrix.getEntry(i, j);
            }
        }

        return flattened;
    }

    private static double calculateMean(RealMatrix matrix) {
        double sum = 0.0;
        int rows = matrix.getRowDimension();
        int cols = matrix.getColumnDimension();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                sum += matrix.getEntry(i, j);
            }
        }

        return sum / (rows * cols);
    }

    private static double calculateRange(RealMatrix matrix) {
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        int rows = matrix.getRowDimension();
        int cols = matrix.getColumnDimension();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double val = matrix.getEntry(i, j);
                min = Math.min(min, val);
                max = Math.max(max, val);
            }
        }

        return max - min;
    }

    private static double calculateDeadUnits(RealMatrix matrix) {
        int rows = matrix.getRowDimension();
        int cols = matrix.getColumnDimension();
        int dead = 0;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (Math.abs(matrix.getEntry(i, j)) < 1e-6) {
                    dead++;
                }
            }
        }

        return (double) dead / (rows * cols);
    }
}
