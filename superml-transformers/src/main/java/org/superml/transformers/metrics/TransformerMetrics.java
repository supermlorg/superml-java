package org.superml.transformers.metrics;

import org.apache.commons.math3.linear.RealMatrix;

/**
 * Metrics for evaluating transformer model performance with focus on attention mechanism.
 */
public class TransformerMetrics {

    public static AttentionMetrics evaluateAttention(RealMatrix attentionScores) {
        double averageAttention = calculateAverageAttention(attentionScores);
        double entropy = calculateAttentionEntropy(attentionScores);
        return new AttentionMetrics(averageAttention, entropy);
    }

    private static double calculateAverageAttention(RealMatrix attentionScores) {
        double sum = 0;
        int rows = attentionScores.getRowDimension();
        int cols = attentionScores.getColumnDimension();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                sum += attentionScores.getEntry(i, j);
            }
        }

        return sum / (rows * cols);
    }

    private static double calculateAttentionEntropy(RealMatrix attentionScores) {
        double entropy = 0;
        int rows = attentionScores.getRowDimension();

        for (int i = 0; i < rows; i++) {
            double rowSum = 0;
            for (int j = 0; j < attentionScores.getColumnDimension(); j++) {
                double p = attentionScores.getEntry(i, j);
                if (p > 0) {
                    entropy -= p * Math.log(p);
                }
            }
        }

        return entropy;
    }

    public static class AttentionMetrics {
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

        @Override
        public String toString() {
            return String.format("AttentionMetrics{avgAttention=%.4f, entropy=%.4f}",
                               averageAttention, entropy);
        }
    }
}
