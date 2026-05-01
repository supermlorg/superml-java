package org.superml.examples.transformers;

import org.superml.transformers.models.TransformerModel;

/**
 * Working Transformer Implementation Example.
 * 
 * This example demonstrates the currently working transformer functionality:
 * 1. Encoder-only classification (BERT-style) - FULLY WORKING
 * 2. Performance metrics and analysis
 * 
 * @author SuperML Team
 * @version 2.1.0
 */
public class WorkingTransformerExample {
    
    public static void main(String[] args) {
        System.out.println("🎯 SuperML Working Transformer Example");
        System.out.println("=====================================");
        
        try {
            // Example 1: Encoder-only classification (FULLY WORKING)
            demonstrateEncoderOnlyClassification();
            
            // Example 2: Performance analysis
            performanceAnalysis();
            
            System.out.println("\n✅ All working transformer examples completed successfully!");
            
        } catch (Exception e) {
            System.err.println("❌ Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Demonstrates encoder-only architecture for sequence classification (BERT-style).
     * This is FULLY IMPLEMENTED and WORKING.
     */
    private static void demonstrateEncoderOnlyClassification() {
        System.out.println("\n1. 🎯 Encoder-Only Classification (WORKING)");
        System.out.println("==========================================");
        
        // Configuration
        int numLayers = 4;
        int dModel = 256;
        int numHeads = 8;
        int numClasses = 3;
        
        System.out.printf("📋 Configuration: %d layers, %d-dim, %d heads, %d classes\n", 
            numLayers, dModel, numHeads, numClasses);
        
        // Create encoder-only model
        TransformerModel model = TransformerModel.createEncoderOnly(numLayers, dModel, numHeads, numClasses);
        
        // Generate classification data (features must match model dimension)
        int batchSize = 8;
        double[][] documents = generateClassificationData(batchSize, dModel);
        double[] labels = {0, 1, 2, 0, 1, 2, 1, 0}; // 3-class classification
        
        System.out.printf("📊 Data: %d samples × %d features\n", batchSize, dModel);
        
        // Training
        System.out.println("\n🔥 Training:");
        long startTime = System.currentTimeMillis();
        model.fit(documents, labels);
        long trainingTime = System.currentTimeMillis() - startTime;
        System.out.printf("   ⏱️ Training time: %d ms\n", trainingTime);
        
        // Prediction
        System.out.println("\n🔮 Prediction:");
        startTime = System.currentTimeMillis();
        double[] predictions = model.predict(documents);
        long predictionTime = System.currentTimeMillis() - startTime;
        System.out.printf("   ⏱️ Prediction time: %d ms\n", predictionTime);
        
        // Probability prediction
        double[][] probabilities = model.predictProba(documents);
        
        // Results
        System.out.println("\n📊 Results:");
        for (int i = 0; i < Math.min(5, batchSize); i++) {
            System.out.printf("   Sample %d: True=%s, Pred=%s, Probs=[%.3f, %.3f, %.3f]\n",
                i, getLabel((int)labels[i]), getLabel((int)predictions[i]),
                probabilities[i][0], probabilities[i][1], probabilities[i][2]);
        }
        
        // Accuracy
        double accuracy = model.score(documents, labels);
        System.out.printf("\n🎯 Accuracy: %.1f%%\n", accuracy * 100);
        
        // Model info
        double[] classes = model.getClasses();
        System.out.printf("🏷️ Classes: ");
        for (int i = 0; i < classes.length; i++) {
            System.out.printf("%.0f", classes[i]);
            if (i < classes.length - 1) System.out.print(", ");
        }
        System.out.println();
        
        // Log probabilities
        System.out.println("\n📊 Log Probabilities (first 3 samples):");
        double[][] logProbs = model.predictLogProba(documents);
        for (int i = 0; i < Math.min(3, batchSize); i++) {
            System.out.printf("   Sample %d log probs: [%.3f, %.3f, %.3f]\n",
                i, logProbs[i][0], logProbs[i][1], logProbs[i][2]);
        }
    }
    
    /**
     * Analyzes performance characteristics of the transformer implementation.
     */
    private static void performanceAnalysis() {
        System.out.println("\n2. ⚡ Performance Analysis");
        System.out.println("========================");
        
        // Test different model sizes
        int[] modelSizes = {64, 128, 256};
        int[] layerCounts = {2, 4, 6};
        int batchSize = 10;
        
        System.out.println("📊 Performance Matrix (Training Time in ms):");
        System.out.println("Model Dim \\ Layers   2      4      6");
        System.out.println("--------------------------------");
        
        for (int dModel : modelSizes) {
            System.out.printf("%-8d     ", dModel);
            
            for (int numLayers : layerCounts) {
                try {
                    // Create model
                    TransformerModel model = TransformerModel.createEncoderOnly(numLayers, dModel, 4, 2);
                    
                    // Generate data
                    double[][] data = generateClassificationData(batchSize, dModel);
                    double[] labels = new double[batchSize];
                    for (int i = 0; i < batchSize; i++) {
                        labels[i] = i % 2; // Binary classification
                    }
                    
                    // Measure training time
                    long startTime = System.currentTimeMillis();
                    model.fit(data, labels);
                    long trainingTime = System.currentTimeMillis() - startTime;
                    
                    System.out.printf("%-6d ", trainingTime);
                    
                } catch (Exception e) {
                    System.out.printf("%-6s ", "ERROR");
                }
            }
            System.out.println();
        }
        
        // Memory usage analysis
        System.out.println("\n💾 Memory Analysis:");
        Runtime runtime = Runtime.getRuntime();
        runtime.gc(); // Suggest garbage collection
        
        long totalMemory = runtime.totalMemory() / (1024 * 1024);
        long freeMemory = runtime.freeMemory() / (1024 * 1024);
        long usedMemory = totalMemory - freeMemory;
        long maxMemory = runtime.maxMemory() / (1024 * 1024);
        
        System.out.printf("   📈 Used Memory: %d MB\n", usedMemory);
        System.out.printf("   📊 Total Memory: %d MB\n", totalMemory);
        System.out.printf("   🎯 Max Memory: %d MB\n", maxMemory);
        System.out.printf("   📉 Free Memory: %d MB\n", freeMemory);
        
        // Model complexity analysis
        System.out.println("\n🧮 Model Complexity Estimates:");
        for (int dModel : new int[]{128, 256, 512}) {
            for (int layers : new int[]{4, 8, 12}) {
                long params = estimateParameters(dModel, 8, layers);
                System.out.printf("   %d-dim, %d layers: ~%s parameters\n", 
                    dModel, layers, formatNumber(params));
            }
        }
    }
    
    // Utility methods
    
    private static double[][] generateClassificationData(int samples, int features) {
        double[][] data = new double[samples][features];
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < features; j++) {
                // Generate synthetic data with some patterns
                data[i][j] = Math.sin(i * 0.1 + j * 0.01) + 
                           (Math.random() - 0.5) * 0.2;
            }
        }
        return data;
    }
    
    private static String getLabel(int classId) {
        String[] labels = {"Class A", "Class B", "Class C"};
        return classId >= 0 && classId < labels.length ? labels[classId] : "Unknown";
    }
    
    private static long estimateParameters(int dModel, int numHeads, int numLayers) {
        // Rough parameter count estimation for transformer encoder
        long embeddingParams = 1000 * dModel; // Assuming vocab size of 1000
        long attentionParams = numLayers * dModel * dModel * 4; // Q, K, V, O projections
        long feedForwardParams = numLayers * dModel * 4 * dModel * 2; // Two FF layers
        long normParams = numLayers * dModel * 2; // LayerNorm parameters
        
        return embeddingParams + attentionParams + feedForwardParams + normParams;
    }
    
    private static String formatNumber(long number) {
        if (number >= 1_000_000) {
            return String.format("%.1fM", number / 1_000_000.0);
        } else if (number >= 1_000) {
            return String.format("%.1fK", number / 1_000.0);
        } else {
            return String.valueOf(number);
        }
    }
}
