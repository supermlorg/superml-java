package org.superml.examples.transformers;

import org.superml.transformers.models.TransformerEncoder;
import org.superml.transformers.models.TransformerDecoder;
import org.superml.transformers.models.TransformerModel;
import org.superml.transformers.models.TransformerModel.Architecture;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

/**
 * Complete Transformer Models Integration Example.
 * 
 * This example demonstrates:
 * 1. All three transformer architectures (Encoder-only, Decoder-only, Encoder-Decoder)
 * 2. SuperML Pipeline integration
 * 3. Real-world usage patterns
 * 4. Performance metrics and evaluation
 * 5. Text classification and generation workflows
 * 
 * @author SuperML Team
 * @version 2.1.0
 */
public class TransformerModelsExample {
    
    public static void main(String[] args) {
        System.out.println("🤖 SuperML Complete Transformer Models Demo");
        System.out.println("==========================================");
        
        try {
            // Example 1: Encoder-only for classification (BERT-style) - FULLY WORKING
            demonstrateEncoderOnlyClassification();
            
            // Example 2: Decoder-only info (architecture explanation)
            explainDecoderOnlyGeneration();
            
            // Example 3: Full encoder-decoder info (architecture explanation)
            explainEncoderDecoderSeq2Seq();
            
            // Example 4: Pipeline integration
            demonstratePipelineIntegration();
            
            // Example 5: Performance comparison (encoder-only focus)
            performanceComparison();
            
            System.out.println("\n✅ All transformer model examples completed successfully!");
            
        } catch (Exception e) {
            System.err.println("❌ Error running transformer examples: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Demonstrates encoder-only architecture for sequence classification (BERT-style).
     */
    private static void demonstrateEncoderOnlyClassification() {
        System.out.println("\n1. 📚 Encoder-Only Classification (BERT-style)");
        System.out.println("==============================================");
        
        // Configuration for text classification
        int vocabSize = 2000;      // Vocabulary size
        int dModel = 256;          // Model dimension
        int numHeads = 8;          // Number of attention heads
        int numLayers = 6;         // Number of transformer layers
        int feedForwardDim = 1024; // Feed-forward network dimension
        int maxSeqLength = 128;    // Maximum sequence length
        int numClasses = 4;        // Number of classification classes
        
        System.out.printf("📋 Model Configuration:\n");
        System.out.printf("   - Vocabulary Size: %d\n", vocabSize);
        System.out.printf("   - Model Dimension: %d\n", dModel);
        System.out.printf("   - Attention Heads: %d\n", numHeads);
        System.out.printf("   - Transformer Layers: %d\n", numLayers);
        System.out.printf("   - Classification Classes: %d\n", numClasses);
        
        // Create encoder-only model
        TransformerModel model = TransformerModel.createEncoderOnly(
            numLayers, dModel, numHeads, numClasses);
        
        // Generate synthetic text classification data
        int batchSize = 6;
        int seqLength = dModel; // Input features must match model dimension
        double[][] documents = generateTextClassificationData(batchSize, seqLength, vocabSize);
        double[] sentimentLabels = {0, 1, 2, 3, 1, 0}; // Multi-class sentiment
        
        System.out.printf("\n📊 Training Data:\n");
        System.out.printf("   - Batch Size: %d documents\n", batchSize);
        System.out.printf("   - Feature Dimension: %d (matches model dim)\n", seqLength);
        System.out.printf("   - Label Distribution: ");
        printLabelDistribution(sentimentLabels);
        
        // Training phase
        System.out.println("\n🎯 Training Phase:");
        long startTime = System.currentTimeMillis();
        model.fit(documents, sentimentLabels);
        long trainingTime = System.currentTimeMillis() - startTime;
        System.out.printf("   ✅ Training completed in %d ms\n", trainingTime);
        
        // Inference phase
        System.out.println("\n🔮 Inference Phase:");
        startTime = System.currentTimeMillis();
        double[] predictions = model.predict(documents);
        double[][] probabilities = model.predictProba(documents);
        long inferenceTime = System.currentTimeMillis() - startTime;
        System.out.printf("   ✅ Inference completed in %d ms\n", inferenceTime);
        
        // Results analysis
        System.out.println("\n📈 Classification Results:");
        for (int i = 0; i < batchSize; i++) {
            System.out.printf("   Document %d: True=%s, Pred=%s, Confidence=%.3f\n",
                i, getClassLabel((int)sentimentLabels[i]), getClassLabel((int)predictions[i]),
                probabilities[i][(int)predictions[i]]);
        }
        
        double accuracy = model.score(documents, sentimentLabels);
        System.out.printf("\n🎯 Model Accuracy: %.3f (%.1f%%)\n", accuracy, accuracy * 100);
    }
    
    /**
     * Explains decoder-only architecture for text generation (GPT-style).
     * Note: Full implementation in progress.
     */
    private static void explainDecoderOnlyGeneration() {
        System.out.println("\n2. ✍️ Decoder-Only Text Generation (GPT-style)");
        System.out.println("==============================================");
        
        // Configuration for text generation
        int vocabSize = 1500;
        int dModel = 256;
        int numHeads = 8;
        int numLayers = 4;
        int feedForwardDim = 1024;
        int maxSeqLength = 64;
        
        System.out.printf("📋 Generation Model Configuration:\n");
        System.out.printf("   - Autoregressive Decoder with %d layers\n", numLayers);
        System.out.printf("   - Causal Attention Masking\n");
        System.out.printf("   - Maximum Generation Length: %d tokens\n", maxSeqLength);
        
        System.out.println("\n🏗️ Architecture Overview:");
        System.out.println("   1. Causal Self-Attention (can only attend to previous tokens)");
        System.out.println("   2. Feed-Forward Networks");
        System.out.println("   3. Layer Normalization");
        System.out.println("   4. Autoregressive Generation (one token at a time)");
        
        System.out.println("\n� Use Cases:");
        System.out.println("   • Text Generation (stories, code, etc.)");
        System.out.println("   • Conversational AI");
        System.out.println("   • Code Completion");
        System.out.println("   • Creative Writing");
        
        System.out.println("   🚧 Status: Interface available, full implementation in progress");
        
        // Create decoder-only model (interface demonstration)
        try {
            TransformerModel decoderModel = TransformerModel.createDecoderOnly(
                numLayers, dModel, numHeads, vocabSize);
            System.out.println("   ✅ Model architecture created successfully");
            System.out.printf("   📊 Parameters: %s\n", estimateParameters(dModel, numHeads, numLayers, vocabSize, false, true));
            
            // Show interface capabilities
            System.out.println("   🔧 Available Methods:");
            System.out.println("      • Architecture: " + decoderModel.getArchitecture());
            System.out.println("      • Model Dimension: " + dModel);
            System.out.println("      • Vocabulary Size: " + vocabSize);
        } catch (Exception e) {
            System.out.println("   ⚠️ Note: " + e.getMessage());
        }
    }
    
    /**
     * Explains full encoder-decoder architecture for sequence-to-sequence tasks.
     * Note: Full implementation in progress.
     */
    private static void explainEncoderDecoderSeq2Seq() {
        System.out.println("\n3. 🔄 Encoder-Decoder Seq2Seq (Full Transformer)");
        System.out.println("================================================");
        
        // Configuration for sequence-to-sequence
        int vocabSize = 2500;
        int dModel = 256;
        int numHeads = 8;
        int numLayers = 6;
        int feedForwardDim = 1024;
        int maxSeqLength = 96;
        int numClasses = 8;
        
        System.out.printf("📋 Seq2Seq Model Configuration:\n");
        System.out.printf("   - Encoder Layers: %d\n", numLayers);
        System.out.printf("   - Decoder Layers: %d\n", numLayers);
        System.out.printf("   - Cross-Attention: Enabled\n");
        System.out.printf("   - Output Classes: %d\n", numClasses);
        
        System.out.println("\n🏗️ Architecture Overview:");
        System.out.println("   📥 ENCODER:");
        System.out.println("      1. Self-Attention (bidirectional)");
        System.out.println("      2. Feed-Forward Networks");
        System.out.println("      3. Layer Normalization");
        System.out.println("      4. Encodes input sequence into context");
        System.out.println();
        System.out.println("   � DECODER:");
        System.out.println("      1. Causal Self-Attention (autoregressive)");
        System.out.println("      2. Cross-Attention (attends to encoder output)");
        System.out.println("      3. Feed-Forward Networks");
        System.out.println("      4. Layer Normalization");
        
        System.out.println("\n📝 Use Cases:");
        System.out.println("   • Machine Translation (EN → DE, FR → EN, etc.)");
        System.out.println("   • Text Summarization");
        System.out.println("   • Question Answering");
        System.out.println("   • Code Translation");
        System.out.println("   • Speech Recognition");
        
        System.out.println("   � Status: Interface available, full implementation in progress");
        
        // Create encoder-decoder model (interface demonstration)
        try {
            TransformerModel encoderDecoderModel = TransformerModel.createEncoderDecoder(
                numLayers, numLayers, dModel, numHeads, vocabSize);
            System.out.println("   ✅ Model architecture created successfully");
            System.out.printf("   📊 Parameters: %s\n", estimateParameters(dModel, numHeads, numLayers, vocabSize, true, true));
            
            // Show interface capabilities
            System.out.println("   🔧 Available Methods:");
            System.out.println("      • Architecture: " + encoderDecoderModel.getArchitecture());
            System.out.println("      • Model Dimension: " + dModel);
            System.out.println("      • Vocabulary Size: " + vocabSize);
        } catch (Exception e) {
            System.out.println("   ⚠️ Note: " + e.getMessage());
        }
    }
    
    /**
     * Demonstrates transformer integration with SuperML Pipeline.
     */
    private static void demonstratePipelineIntegration() {
        System.out.println("\n4. 🔧 Pipeline Integration with SuperML");
        System.out.println("======================================");
        
        try {
            // Note: Pipeline integration would require additional preprocessing steps
            // This demonstrates the concept with a simplified transformer model
            
            int modelDim = 128;
            TransformerModel transformer = TransformerModel.createEncoderOnly(
                3, modelDim, 4, 3);
                
            // Generate pipeline data
            double[][] data = generateTextClassificationData(4, modelDim, 1000);
            double[] labels = {0, 1, 2, 0};
            
            System.out.println("📋 Pipeline Components:");
            System.out.println("   1. Input Preprocessing (Token IDs)");
            System.out.println("   2. Transformer Encoder");
            System.out.println("   3. Classification Head");
            
            // Direct model usage (Pipeline integration would require custom preprocessing)
            transformer.fit(data, labels);
            double[] pipelineResults = transformer.predict(data);
            
            System.out.println("\n✅ Pipeline execution completed");
            System.out.printf("   Results: ");
            for (double result : pipelineResults) {
                System.out.printf("%.0f ", result);
            }
            System.out.println();
            
        } catch (Exception e) {
            System.err.println("Pipeline integration note: " + e.getMessage());
        }
    }
    
    /**
     * Compares performance across different transformer architectures.
     */
    private static void performanceComparison() {
        System.out.println("\n5. ⚡ Performance Comparison");
        System.out.println("===========================");
        
        // Test configurations
        int vocabSize = 1000;
        int dModel = 128;
        int numHeads = 4;
        int numLayers = 3;
        int batchSize = 3;
        int seqLength = 16;
        
        // Generate test data
        double[][] testData = generateTextClassificationData(batchSize, dModel, vocabSize);
        double[] testLabels = {0, 1, 0};
        
        System.out.println("📊 Performance Metrics (Training Time):");
        
        // Test Encoder-only (WORKING)
        long startTime = System.currentTimeMillis();
        TransformerModel encoderOnly = TransformerModel.createEncoderOnly(
            numLayers, dModel, numHeads, 2);
        encoderOnly.fit(testData, testLabels);
        long encoderTime = System.currentTimeMillis() - startTime;
        
        // Test Decoder-only (ARCHITECTURE DEMO)
        startTime = System.currentTimeMillis();
        TransformerModel decoderOnly = TransformerModel.createDecoderOnly(
            numLayers, dModel, numHeads, vocabSize);
        // Note: fit() not yet implemented for decoder-only
        long decoderTime = System.currentTimeMillis() - startTime;
        
        // Test Encoder-Decoder (ARCHITECTURE DEMO)
        startTime = System.currentTimeMillis();
        TransformerModel encoderDecoder = TransformerModel.createEncoderDecoder(
            numLayers, numLayers, dModel, numHeads, vocabSize);
        // Note: fit() not yet implemented for encoder-decoder
        long encoderDecoderTime = System.currentTimeMillis() - startTime;
        
        // Report performance
        System.out.printf("   🔹 Encoder-Only:     %d ms (WORKING - includes training)\n", encoderTime);
        System.out.printf("   🔹 Decoder-Only:     %d ms (model creation only)\n", decoderTime);
        System.out.printf("   🔹 Encoder-Decoder:  %d ms (model creation only)\n", encoderDecoderTime);
        
        System.out.println("\n📈 Model Complexity (Parameters):");
        System.out.println("   🔹 Encoder-Only:     ~" + estimateParameters(dModel, numHeads, numLayers, vocabSize, true, false));
        System.out.println("   🔹 Decoder-Only:     ~" + estimateParameters(dModel, numHeads, numLayers, vocabSize, false, true));
        System.out.println("   🔹 Encoder-Decoder:  ~" + estimateParameters(dModel, numHeads, numLayers, vocabSize, true, true));
    }
    
    // Utility methods for data generation
    
    private static double[][] generateTextClassificationData(int batchSize, int seqLength, int vocabSize) {
        double[][] data = new double[batchSize][seqLength];
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < seqLength; j++) {
                // Generate token IDs with some semantic patterns
                data[i][j] = (i * 50 + j * 7 + (int)(Math.random() * vocabSize * 0.2)) % vocabSize;
            }
        }
        return data;
    }
    
    private static double[][] generateLanguageModelingData(int batchSize, int seqLength, int vocabSize) {
        double[][] data = new double[batchSize][seqLength];
        for (int i = 0; i < batchSize; i++) {
            int seed = i * 123;
            for (int j = 0; j < seqLength; j++) {
                // Generate coherent sequences
                data[i][j] = (seed + j * 17) % vocabSize;
            }
        }
        return data;
    }
    
    private static double[][] generateSeq2SeqSourceData(int batchSize, int seqLength, int vocabSize) {
        double[][] data = new double[batchSize][seqLength];
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < seqLength; j++) {
                // Generate structured source sequences
                data[i][j] = (i * 200 + j * 13 + (int)(Math.random() * vocabSize * 0.15)) % vocabSize;
            }
        }
        return data;
    }
    
    private static void printLabelDistribution(double[] labels) {
        int[] counts = new int[10]; // Assume max 10 classes
        for (double label : labels) {
            if (label >= 0 && label < counts.length) {
                counts[(int)label]++;
            }
        }
        for (int i = 0; i < counts.length; i++) {
            if (counts[i] > 0) {
                System.out.printf("Class%d:%d ", i, counts[i]);
            }
        }
        System.out.println();
    }
    
    private static String getClassLabel(int classId) {
        String[] labels = {"Negative", "Neutral", "Positive", "Very Positive"};
        return classId >= 0 && classId < labels.length ? labels[classId] : "Unknown";
    }
    
    private static String estimateParameters(int dModel, int numHeads, int numLayers, 
                                           int vocabSize, boolean hasEncoder, boolean hasDecoder) {
        // Rough parameter estimation
        long params = vocabSize * dModel; // Embedding
        if (hasEncoder) params += numLayers * (dModel * dModel * 4 + dModel * 2048 * 2); // Encoder
        if (hasDecoder) params += numLayers * (dModel * dModel * 6 + dModel * 2048 * 2); // Decoder
        params += dModel * vocabSize; // Output projection
        
        if (params > 1_000_000) {
            return String.format("%.1fM", params / 1_000_000.0);
        } else if (params > 1_000) {
            return String.format("%.1fK", params / 1_000.0);
        } else {
            return String.valueOf(params);
        }
    }
}
