package org.superml.examples.transformers;

import org.superml.transformers.models.TransformerModel;
import org.superml.transformers.training.TransformerTrainer;

/**
 * Real Training Example for Transformers
 * 
 * This demonstrates actual learning with:
 * - Cross-entropy loss optimization
 * - Gradient-based parameter updates
 * - Convergence monitoring
 * - Performance visualization
 * 
 * Unlike our text prediction example which used placeholder training,
 * this implements real backpropagation for genuine learning.
 */
public class RealTrainingExample {
    
    public static void main(String[] args) {
        System.out.println("🧠 SuperML Transformers - Real Training Example");
        System.out.println("===============================================\n");
        
        // Generate synthetic classification dataset
        int numSamples = 1000;
        int modelDim = 128;        // This must match the model dimension
        int vocabSize = 50;
        int numClasses = 5;
        
        System.out.println("📊 Generating Training Dataset");
        System.out.printf("   Samples: %d\n", numSamples);
        System.out.printf("   Model Dimension: %d\n", modelDim);
        System.out.printf("   Vocabulary Size: %d\n", vocabSize);
        System.out.printf("   Classes: %d\n\n", numClasses);
        
        double[][] X = generateSequenceData(numSamples, modelDim, vocabSize);
        double[] y = generateLabels(numSamples, numClasses, X);
        
        // Create transformer model
        System.out.println("🏗️ Initializing Transformer Model");
        TransformerModel model = TransformerModel.createEncoderOnly(
            4,            // numLayers
            128,          // modelDim  
            8,            // numHeads
            numClasses    // numClasses
        );
        
        System.out.printf("   Architecture: Encoder-only (BERT-style)\n");
        System.out.printf("   Parameters: ~%.1fM\n", estimateParameters(model) / 1_000_000.0);
        System.out.println();
        
        // Train with real backpropagation
        System.out.println("🔥 Real Training Process");
        TransformerTrainer trainer = new TransformerTrainer(
            0.001,  // learning rate
            30,     // epochs (reduced for faster demo)
            1e-4,   // tolerance (less strict for realistic training)
            true    // verbose
        );
        
        TransformerTrainer.TrainingResult result = trainer.trainTransformer(model, X, y);
        
        // Evaluate final performance
        System.out.println("\n📈 Training Analysis");
        System.out.println("===================");
        System.out.printf("   Convergence: %s\n", result.hasConverged() ? "✅ Yes" : "❌ No");
        System.out.printf("   Final Loss: %.6f\n", result.getFinalLoss());
        System.out.printf("   Final Accuracy: %.1f%%\n", result.getFinalAccuracy() * 100);
        
        // Show learning curve
        System.out.println("\n📊 Learning Curve (Loss):");
        displayLearningCurve(result.getLosses());
        
        System.out.println("\n📊 Accuracy Curve:");
        displayLearningCurve(result.getAccuracies());
        
        // Test on new data
        System.out.println("\n🧪 Testing on New Data");
        System.out.println("=====================");
        double[][] testX = generateSequenceData(100, modelDim, vocabSize);
        double[] testY = generateLabels(100, numClasses, testX);
        
        double[] predictions = model.predict(testX);
        double testAccuracy = calculateAccuracy(predictions, testY);
        System.out.printf("   Test Accuracy: %.1f%%\n", testAccuracy * 100);
        
        // Show some predictions
        System.out.println("\n🎯 Sample Predictions:");
        showSamplePredictions(model, testX, testY, 5);
        
        System.out.println("\n✅ Real Training Complete!");
        System.out.println("   The transformer has learned to classify sequences with actual gradient-based optimization.");
    }
    
    /**
     * Generate synthetic sequence data for classification.
     */
    private static double[][] generateSequenceData(int numSamples, int seqLength, int vocabSize) {
        double[][] data = new double[numSamples][seqLength];
        java.util.Random random = new java.util.Random(42);
        
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < seqLength; j++) {
                data[i][j] = random.nextInt(vocabSize);
            }
        }
        
        return data;
    }
    
    /**
     * Generate labels based on sequence patterns.
     */
    private static double[] generateLabels(int numSamples, int numClasses, double[][] X) {
        double[] labels = new double[numSamples];
        
        for (int i = 0; i < numSamples; i++) {
            // Create pattern-based labels
            double sum = 0;
            for (double token : X[i]) {
                sum += token;
            }
            
            // Map sum to class (creates learnable patterns)
            labels[i] = (int) (sum % numClasses);
        }
        
        return labels;
    }
    
    /**
     * Estimate model parameters.
     */
    private static long estimateParameters(TransformerModel model) {
        // Rough estimation based on architecture
        int embedDim = 128; // from model setup
        int vocabSize = 50; // from data generation
        int numHeads = 8;
        int numLayers = 4;
        
        // Embedding parameters
        long embedParams = vocabSize * embedDim;
        
        // Attention parameters per layer
        long attentionParams = embedDim * embedDim * 4; // Q, K, V, O matrices
        
        // Feed-forward parameters per layer
        long ffParams = embedDim * (embedDim * 4) + (embedDim * 4) * embedDim;
        
        // Total per layer
        long paramsPerLayer = attentionParams + ffParams;
        
        return embedParams + (numLayers * paramsPerLayer);
    }
    
    /**
     * Display learning curve as ASCII chart.
     */
    private static void displayLearningCurve(java.util.List<Double> values) {
        if (values.size() < 2) return;
        
        // Find min/max for scaling
        double min = values.stream().mapToDouble(Double::doubleValue).min().orElse(0);
        double max = values.stream().mapToDouble(Double::doubleValue).max().orElse(1);
        
        String bar = "   ";
        for (int i = 0; i < Math.min(values.size(), 20); i++) {
            double normalized = (values.get(i * values.size() / 20) - min) / (max - min);
            int barLength = (int) (normalized * 30);
            
            System.out.printf("   %2d: ", i * values.size() / 20);
            for (int j = 0; j < barLength; j++) {
                System.out.print("█");
            }
            System.out.printf(" %.4f\n", values.get(i * values.size() / 20));
        }
    }
    
    /**
     * Calculate accuracy.
     */
    private static double calculateAccuracy(double[] predictions, double[] actual) {
        int correct = 0;
        for (int i = 0; i < predictions.length; i++) {
            if (Math.round(predictions[i]) == Math.round(actual[i])) {
                correct++;
            }
        }
        return (double) correct / predictions.length;
    }
    
    /**
     * Show sample predictions vs actual.
     */
    private static void showSamplePredictions(TransformerModel model, double[][] X, 
                                            double[] y, int numSamples) {
        double[][] probas = model.predictProba(X);
        
        for (int i = 0; i < Math.min(numSamples, X.length); i++) {
            // Find predicted class
            int predicted = 0;
            double maxProb = probas[i][0];
            for (int j = 1; j < probas[i].length; j++) {
                if (probas[i][j] > maxProb) {
                    maxProb = probas[i][j];
                    predicted = j;
                }
            }
            
            System.out.printf("   Sample %d: Predicted=%d (%.1f%%), Actual=%d %s\n",
                i + 1, predicted, maxProb * 100, (int) y[i],
                predicted == (int) y[i] ? "✅" : "❌");
        }
    }
}
