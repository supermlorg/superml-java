package org.superml.examples.transformers;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Integration test for transformer gradient analysis functionality.
 * Self-contained test following SuperML framework patterns.
 */
public class TransformerGradientTest {

    private static final int[] TEST_LAYER_CONFIGS = {
        4,  // Small model (testing)
        8,  // Medium model (like T5-small)
        12  // Large model (like BERT-base)
    };

    private static final int[] TEST_HIDDEN_SIZES = {
        64,   // Small for quick tests
        128   // Medium size for validation
    };

    private Random random;

    @BeforeEach
    public void setUp() {
        random = new Random(42); // Fixed seed for reproducibility
    }

    @Test
    public void testGradientAnalysisWithDifferentModelSizes() {
        System.out.println("Running Transformer Gradient Analysis Tests");
        System.out.println("=========================================");

        for (int numLayers : TEST_LAYER_CONFIGS) {
            for (int hiddenSize : TEST_HIDDEN_SIZES) {
                System.out.printf("%nTesting %d-layer model with hidden size %d%n",
                    numLayers, hiddenSize);

                // Generate synthetic model data
                ModelData modelData = generateModelData(numLayers, hiddenSize);

                // Run gradient analysis
                testGradientAnalysis(modelData);

                // Print separator for readability
                System.out.println("-".repeat(50));
            }
        }
    }

    private ModelData generateModelData(int numLayers, int hiddenSize) {
        List<RealMatrix> gradients = new ArrayList<>();
        List<RealMatrix> parameters = new ArrayList<>();

        // Generate synthetic data with realistic transformer patterns
        for (int layer = 0; layer < numLayers; layer++) {
            // Parameters follow Xavier/Glorot initialization
            double paramScale = Math.sqrt(2.0 / (hiddenSize + hiddenSize));
            RealMatrix params = generateParameters(hiddenSize, paramScale);
            parameters.add(params);

            // Gradients show typical transformer characteristics
            // - Vanishing gradient pattern in deeper layers
            // - Sparse updates in attention mechanisms
            double gradScale = Math.exp(-layer * 0.5) * 0.01;
            RealMatrix grads = generateGradients(hiddenSize, gradScale, layer);
            gradients.add(grads);
        }

        return new ModelData(gradients, parameters);
    }

    private RealMatrix generateParameters(int size, double scale) {
        double[][] data = new double[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                // Xavier/Glorot initialization pattern
                data[i][j] = (random.nextGaussian() * scale);
            }
        }
        return new Array2DRowRealMatrix(data);
    }

    private RealMatrix generateGradients(int size, double scale, int layer) {
        double[][] data = new double[size][size];

        // Add transformer-specific gradient patterns
        double attentionSparsity = 0.3 + (layer * 0.05); // Increasing sparsity with depth

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (random.nextDouble() < attentionSparsity) {
                    data[i][j] = 0.0; // Sparse attention updates
                } else {
                    // Gradient magnitude follows layer depth pattern
                    data[i][j] = random.nextGaussian() * scale;
                }
            }
        }
        return new Array2DRowRealMatrix(data);
    }

    private void testGradientAnalysis(ModelData modelData) {
        // Basic validation
        validateModelStructure(modelData);

        // Gradient analysis
        GradientAnalysisResult analysis = analyzeGradients(modelData);

        // Print analysis results
        printAnalysisResults(analysis);

        // Validate analysis results
        validateAnalysisResults(analysis);
    }

    private void validateModelStructure(ModelData modelData) {
        assertNotNull(modelData.gradients, "Gradients should not be null");
        assertNotNull(modelData.parameters, "Parameters should not be null");
        assertEquals(modelData.gradients.size(), modelData.parameters.size(), 
            "Should have same number of gradient and parameter matrices");

        int numLayers = modelData.gradients.size();
        for (int i = 0; i < numLayers; i++) {
            RealMatrix grads = modelData.gradients.get(i);
            RealMatrix params = modelData.parameters.get(i);

            assertEquals(grads.getRowDimension(), grads.getColumnDimension(),
                "Layer " + i + " matrices should be square");
            assertEquals(grads.getRowDimension(), params.getRowDimension(),
                "Layer " + i + " matrices should match sizes");
        }
    }

    private GradientAnalysisResult analyzeGradients(ModelData modelData) {
        int numLayers = modelData.gradients.size();
        double[] gradientNorms = new double[numLayers];
        double[] parameterNorms = new double[numLayers];
        double[] sparsityRatios = new double[numLayers];
        double[] gradientToParamRatios = new double[numLayers];

        for (int i = 0; i < numLayers; i++) {
            RealMatrix grads = modelData.gradients.get(i);
            RealMatrix params = modelData.parameters.get(i);

            gradientNorms[i] = calculateFrobeniusNorm(grads);
            parameterNorms[i] = calculateFrobeniusNorm(params);
            sparsityRatios[i] = calculateSparsity(grads);
            gradientToParamRatios[i] = gradientNorms[i] / parameterNorms[i];
        }

        return new GradientAnalysisResult(
            gradientNorms,
            parameterNorms,
            sparsityRatios,
            gradientToParamRatios
        );
    }

    private void validateAnalysisResults(GradientAnalysisResult analysis) {
        // Validate gradient vanishing pattern
        for (int i = 1; i < analysis.gradientNorms.length; i++) {
            assertTrue(analysis.gradientNorms[i] <= analysis.gradientNorms[i-1] * 1.5,
                "Gradient norm should typically decrease with depth");
        }

        // Validate parameter-to-gradient ratios
        for (int i = 0; i < analysis.gradientToParamRatios.length; i++) {
            double ratio = analysis.gradientToParamRatios[i];
            assertTrue(ratio > 1e-8 && ratio < 1e2,
                "Gradient/parameter ratio should be reasonable");
        }

        // Validate sparsity progression
        for (int i = 1; i < analysis.sparsityRatios.length; i++) {
            assertTrue(analysis.sparsityRatios[i] >= analysis.sparsityRatios[i-1] * 0.8,
                "Sparsity should generally increase with depth");
        }
    }

    private void printAnalysisResults(GradientAnalysisResult analysis) {
        System.out.println("\nGradient Analysis Results:");
        System.out.println("Layer | Grad Norm | Param Norm | Sparsity | G/P Ratio");
        System.out.println("-".repeat(60));

        for (int i = 0; i < analysis.gradientNorms.length; i++) {
            System.out.printf("%5d | %9.2e | %9.2e | %8.2f%% | %8.2e%n",
                i,
                analysis.gradientNorms[i],
                analysis.parameterNorms[i],
                analysis.sparsityRatios[i] * 100,
                analysis.gradientToParamRatios[i]);
        }
    }

    private double calculateFrobeniusNorm(RealMatrix matrix) {
        double sum = 0;
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                double val = matrix.getEntry(i, j);
                sum += val * val;
            }
        }
        return Math.sqrt(sum);
    }

    private double calculateSparsity(RealMatrix matrix) {
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

    private static class ModelData {
        final List<RealMatrix> gradients;
        final List<RealMatrix> parameters;

        ModelData(List<RealMatrix> gradients, List<RealMatrix> parameters) {
            this.gradients = gradients;
            this.parameters = parameters;
        }
    }

    private static class GradientAnalysisResult {
        final double[] gradientNorms;
        final double[] parameterNorms;
        final double[] sparsityRatios;
        final double[] gradientToParamRatios;

        GradientAnalysisResult(
                double[] gradientNorms,
                double[] parameterNorms,
                double[] sparsityRatios,
                double[] gradientToParamRatios) {
            this.gradientNorms = gradientNorms;
            this.parameterNorms = parameterNorms;
            this.sparsityRatios = sparsityRatios;
            this.gradientToParamRatios = gradientToParamRatios;
        }
    }
}
