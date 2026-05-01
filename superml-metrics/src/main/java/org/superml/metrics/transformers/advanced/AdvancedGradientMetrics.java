package org.superml.metrics.transformers.advanced;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import java.util.List;
import java.util.ArrayList;

/**
 * Advanced gradient analysis metrics for transformer models.
 * Follows SuperML metrics specialist pattern for deep analysis.
 */
public class AdvancedGradientMetrics {

    /**
     * Analyzes gradient characteristics with Hessian approximation.
     */
    public static GradientAdvancedAnalysis analyzeGradients(
            List<RealMatrix> gradients,
            List<RealMatrix> parameters,
            double learningRate) {

        HessianApproximation hessianMetrics = approximateHessian(gradients, parameters);
        GradientLandscapeMetrics landscapeMetrics = analyzeLandscape(gradients, parameters);
        OptimizationMetrics optimizationMetrics = analyzeOptimization(gradients, learningRate);

        return new GradientAdvancedAnalysis(
            hessianMetrics,
            landscapeMetrics,
            optimizationMetrics
        );
    }

    private static HessianApproximation approximateHessian(
            List<RealMatrix> gradients,
            List<RealMatrix> parameters) {

        List<RealMatrix> hessianDiagonal = new ArrayList<>();
        List<Double> eigenvalueEstimates = new ArrayList<>();
        List<Double> conditionNumbers = new ArrayList<>();

        for (int i = 0; i < gradients.size() - 1; i++) {
            RealMatrix gradDiff = subtract(gradients.get(i + 1), gradients.get(i));
            RealMatrix paramDiff = subtract(parameters.get(i + 1), parameters.get(i));

            // Compute diagonal Hessian approximation using finite differences
            RealMatrix hessianEst = approximateLayerHessian(gradDiff, paramDiff);
            hessianDiagonal.add(hessianEst);

            // Estimate dominant eigenvalues using power iteration
            double maxEigenvalue = estimateMaxEigenvalue(hessianEst);
            double minEigenvalue = estimateMinEigenvalue(hessianEst);
            eigenvalueEstimates.add(maxEigenvalue);
            conditionNumbers.add(maxEigenvalue / Math.max(Math.abs(minEigenvalue), 1e-12));
        }

        return new HessianApproximation(
            hessianDiagonal,
            eigenvalueEstimates,
            conditionNumbers
        );
    }

    private static GradientLandscapeMetrics analyzeLandscape(
            List<RealMatrix> gradients,
            List<RealMatrix> parameters) {

        List<Double> flatness = new ArrayList<>();
        List<Double> sharpness = new ArrayList<>();
        List<Double> pathLengths = new ArrayList<>();

        for (int i = 0; i < gradients.size() - 1; i++) {
            // Measure local landscape geometry
            double localFlatness = measureLocalFlatness(gradients.get(i), parameters.get(i));
            flatness.add(localFlatness);

            // Measure sharpness through second-order variations
            double localSharpness = measureLocalSharpness(gradients.get(i), gradients.get(i + 1));
            sharpness.add(localSharpness);

            // Calculate optimization path length
            double pathLength = calculatePathLength(parameters.get(i), parameters.get(i + 1));
            pathLengths.add(pathLength);
        }

        return new GradientLandscapeMetrics(flatness, sharpness, pathLengths);
    }

    private static OptimizationMetrics analyzeOptimization(
            List<RealMatrix> gradients,
            double learningRate) {

        List<Double> gradientNorms = new ArrayList<>();
        List<Double> updateRatios = new ArrayList<>();
        List<Double> effectiveLRs = new ArrayList<>();

        for (int i = 0; i < gradients.size(); i++) {
            double norm = calculateFrobeniusNorm(gradients.get(i));
            gradientNorms.add(norm);

            if (i > 0) {
                // Measure relative update size
                double updateRatio = norm / gradientNorms.get(i - 1);
                updateRatios.add(updateRatio);

                // Estimate effective learning rate
                double effectiveLR = learningRate * updateRatio;
                effectiveLRs.add(effectiveLR);
            }
        }

        return new OptimizationMetrics(gradientNorms, updateRatios, effectiveLRs);
    }

    // Utility methods for matrix operations and metrics calculation
    private static RealMatrix approximateLayerHessian(RealMatrix gradDiff, RealMatrix paramDiff) {
        int rows = gradDiff.getRowDimension();
        int cols = gradDiff.getColumnDimension();
        RealMatrix hessian = new Array2DRowRealMatrix(rows, cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (Math.abs(paramDiff.getEntry(i, j)) > 1e-12) {
                    hessian.setEntry(i, j,
                        gradDiff.getEntry(i, j) / paramDiff.getEntry(i, j));
                }
            }
        }

        return hessian;
    }

    private static double estimateMaxEigenvalue(RealMatrix matrix) {
        int maxIters = 20;
        double[] vector = randomVector(matrix.getColumnDimension());

        for (int i = 0; i < maxIters; i++) {
            vector = powerIteration(matrix, vector);
        }

        return rayleighQuotient(matrix, vector);
    }

    private static double estimateMinEigenvalue(RealMatrix matrix) {
        // Use inverse iteration for smallest eigenvalue
        int maxIters = 20;
        double[] vector = randomVector(matrix.getColumnDimension());
        double shift = 1e-6;

        for (int i = 0; i < maxIters; i++) {
            vector = inverseIteration(matrix, vector, shift);
        }

        return rayleighQuotient(matrix, vector);
    }

    // Container classes for analysis results
    public static class GradientAdvancedAnalysis {
        public final HessianApproximation hessianMetrics;
        public final GradientLandscapeMetrics landscapeMetrics;
        public final OptimizationMetrics optimizationMetrics;

        public GradientAdvancedAnalysis(
                HessianApproximation hessianMetrics,
                GradientLandscapeMetrics landscapeMetrics,
                OptimizationMetrics optimizationMetrics) {
            this.hessianMetrics = hessianMetrics;
            this.landscapeMetrics = landscapeMetrics;
            this.optimizationMetrics = optimizationMetrics;
        }
    }

    public static class HessianApproximation {
        public final List<RealMatrix> hessianDiagonal;
        public final List<Double> eigenvalueEstimates;
        public final List<Double> conditionNumbers;

        public HessianApproximation(
                List<RealMatrix> hessianDiagonal,
                List<Double> eigenvalueEstimates,
                List<Double> conditionNumbers) {
            this.hessianDiagonal = hessianDiagonal;
            this.eigenvalueEstimates = eigenvalueEstimates;
            this.conditionNumbers = conditionNumbers;
        }
    }

    public static class GradientLandscapeMetrics {
        public final List<Double> flatness;
        public final List<Double> sharpness;
        public final List<Double> pathLengths;

        public GradientLandscapeMetrics(
                List<Double> flatness,
                List<Double> sharpness,
                List<Double> pathLengths) {
            this.flatness = flatness;
            this.sharpness = sharpness;
            this.pathLengths = pathLengths;
        }
    }

    public static class OptimizationMetrics {
        public final List<Double> gradientNorms;
        public final List<Double> updateRatios;
        public final List<Double> effectiveLearningRates;

        public OptimizationMetrics(
                List<Double> gradientNorms,
                List<Double> updateRatios,
                List<Double> effectiveLearningRates) {
            this.gradientNorms = gradientNorms;
            this.updateRatios = updateRatios;
            this.effectiveLearningRates = effectiveLearningRates;
        }
    }

    // Additional mathematical utility methods
    private static RealMatrix subtract(RealMatrix a, RealMatrix b) {
        return a.subtract(b);
    }

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

    private static double[] powerIteration(RealMatrix matrix, double[] vector) {
        double[] result = matrixVectorMultiply(matrix, vector);
        return normalize(result);
    }

    private static double[] inverseIteration(RealMatrix matrix, double[] vector, double shift) {
        // Simple implementation - in practice would use more stable methods
        double[] result = new double[vector.length];
        System.arraycopy(vector, 0, result, 0, vector.length);

        for (int i = 0; i < vector.length; i++) {
            double sum = 0;
            for (int j = 0; j < vector.length; j++) {
                if (i == j) {
                    sum += (matrix.getEntry(i, j) - shift) * vector[j];
                } else {
                    sum += matrix.getEntry(i, j) * vector[j];
                }
            }
            result[i] = sum;
        }

        return normalize(result);
    }

    private static double rayleighQuotient(RealMatrix matrix, double[] vector) {
        double[] mv = matrixVectorMultiply(matrix, vector);
        double numerator = dotProduct(vector, mv);
        double denominator = dotProduct(vector, vector);
        return numerator / denominator;
    }

    private static double[] matrixVectorMultiply(RealMatrix matrix, double[] vector) {
        double[] result = new double[vector.length];
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            double sum = 0;
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                sum += matrix.getEntry(i, j) * vector[j];
            }
            result[i] = sum;
        }
        return result;
    }

    private static double[] normalize(double[] vector) {
        double norm = Math.sqrt(dotProduct(vector, vector));
        double[] normalized = new double[vector.length];
        for (int i = 0; i < vector.length; i++) {
            normalized[i] = vector[i] / norm;
        }
        return normalized;
    }

    private static double dotProduct(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    private static double[] randomVector(int size) {
        double[] vector = new double[size];
        for (int i = 0; i < size; i++) {
            vector[i] = Math.random() - 0.5;
        }
        return normalize(vector);
    }

    private static double measureLocalFlatness(RealMatrix gradients, RealMatrix parameters) {
        double gradientNorm = calculateFrobeniusNorm(gradients);
        double parameterNorm = calculateFrobeniusNorm(parameters);
        return gradientNorm / (parameterNorm + 1e-12);
    }

    private static double measureLocalSharpness(RealMatrix grad1, RealMatrix grad2) {
        return calculateFrobeniusNorm(subtract(grad2, grad1));
    }

    private static double calculatePathLength(RealMatrix param1, RealMatrix param2) {
        return calculateFrobeniusNorm(subtract(param2, param1));
    }
}
