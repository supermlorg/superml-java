package org.superml.examples.transformers.data;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import java.util.ArrayList;
import java.util.List;
import java.io.*;
import java.util.zip.GZIPOutputStream;
import java.nio.charset.StandardCharsets;

/**
 * Test data generator for transformer gradient analysis.
 * Creates synthetic gradient data following realistic patterns.
 */
public class TransformerGradientTestData {

    public static void main(String[] args) {
        generateTestDatasets();
    }

    public static void generateTestDatasets() {
        // BERT-style model (12 layers, 768 hidden)
        generateModelData("bert-base-uncased", 12, 768, 2e-4);

        // GPT2-style model (12 layers, 768 hidden)
        generateModelData("gpt2-small", 12, 768, 1e-4);

        // T5-style model (8 layers, 512 hidden)
        generateModelData("t5-small", 8, 512, 3e-4);
    }

    private static void generateModelData(
            String modelName,
            int numLayers,
            int hiddenSize,
            double learningRate) {

        List<RealMatrix> gradients = new ArrayList<>();
        List<RealMatrix> parameters = new ArrayList<>();

        // Generate realistic gradients with typical patterns
        for (int layer = 0; layer < numLayers; layer++) {
            // Parameters have xavier-like initialization
            RealMatrix params = generateParameters(hiddenSize, 1.0 / Math.sqrt(hiddenSize));
            parameters.add(params);

            // Gradients show vanishing pattern in deeper layers
            double gradScale = Math.exp(-layer * 0.5) * 0.1;
            RealMatrix grads = generateGradients(hiddenSize, gradScale);
            gradients.add(grads);
        }

        // Save to compressed file
        String filename = String.format("%s-gradient-data.gz", modelName);
        saveModelData(filename, gradients, parameters, learningRate);
    }

    private static RealMatrix generateParameters(int size, double scale) {
        double[][] data = new double[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                data[i][j] = (Math.random() * 2 - 1) * scale;
            }
        }
        return new Array2DRowRealMatrix(data);
    }

    private static RealMatrix generateGradients(int size, double scale) {
        double[][] data = new double[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                // Add some sparsity pattern
                if (Math.random() < 0.3) {
                    data[i][j] = 0;
                } else {
                    data[i][j] = (Math.random() * 2 - 1) * scale;
                }
            }
        }
        return new Array2DRowRealMatrix(data);
    }

    private static void saveModelData(
            String filename,
            List<RealMatrix> gradients,
            List<RealMatrix> parameters,
            double learningRate) {

        String testResourcePath = "superml-examples/src/test/resources/data/" + filename;

        try (PrintWriter writer = new PrintWriter(new OutputStreamWriter(
                new GZIPOutputStream(new FileOutputStream(testResourcePath)),
                StandardCharsets.UTF_8))) {

            // Write metadata
            writer.printf("%d,%d,%.6f%n",
                gradients.size(),
                gradients.get(0).getRowDimension(),
                learningRate);

            // Write gradients and parameters
            for (int i = 0; i < gradients.size(); i++) {
                writeMatrix(writer, gradients.get(i));
                writeMatrix(writer, parameters.get(i));
            }

        } catch (IOException e) {
            System.err.println("Error saving test data: " + e.getMessage());
        }
    }

    private static void writeMatrix(PrintWriter writer, RealMatrix matrix) {
        int rows = matrix.getRowDimension();
        int cols = matrix.getColumnDimension();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                writer.print(matrix.getEntry(i, j));
                if (j < cols - 1) writer.print(",");
            }
            writer.println();
        }
    }
}
