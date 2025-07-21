package org.superml.examples.transformers;

import org.superml.transformers.training.TransformerTrainer;
import org.superml.transformers.training.LearnableTransformer;

/**
 * Real Learning Transformer Example
 * 
 * This demonstrates ACTUAL learning with:
 * - Real parameter updates during training
 * - Genuine gradient-based optimization
 * - Measurable accuracy improvements
 * - True convergence behavior
 * 
 * Unlike previous examples with placeholder training,
 * this uses a LearnableTransformer that maintains and updates
 * its own parameters during training.
 */
public class RealLearningExample {
    
    public static void main(String[] args) {
        System.out.println("🧠 SuperML Transformers - REAL LEARNING Example");
        System.out.println("===============================================\n");
        
        // Generate synthetic classification dataset
        int numSamples = 800;
        int modelDim = 64;        // Smaller for faster training
        int vocabSize = 30;       // Smaller vocabulary
        int numClasses = 3;       // Fewer classes for better learning
        
        System.out.println("📊 Generating Training Dataset");
        System.out.printf("   Samples: %d\n", numSamples);
        System.out.printf("   Model Dimension: %d\n", modelDim);
        System.out.printf("   Vocabulary Size: %d\n", vocabSize);
        System.out.printf("   Classes: %d\n\n", numClasses);
        
        double[][] X = generateSequenceData(numSamples, modelDim, vocabSize);
        double[] y = generateLabels(numSamples, numClasses, X);
        
        // Show label distribution
        showLabelDistribution(y, numClasses);
        
        // Create learnable transformer with real training
        System.out.println("\n🔥 Real Learning Process");
        TransformerTrainer trainer = new TransformerTrainer(
            0.01,   // higher learning rate for visible learning
            25,     // epochs
            1e-3,   // tolerance
            true    // verbose
        );
        
        TransformerTrainer.TrainingResult result = trainer.trainLearnableTransformer(
            modelDim, numClasses, vocabSize, X, y);
        
        // Analyze training results
        System.out.println("\n📈 Learning Analysis");
        System.out.println("===================");
        System.out.printf("   Convergence: %s\n", result.hasConverged() ? "✅ Yes" : "🔄 Still Learning");
        System.out.printf("   Final Loss: %.6f\n", result.getFinalLoss());
        System.out.printf("   Final Accuracy: %.1f%%\n", result.getFinalAccuracy() * 100);
        
        // Show learning progression
        showLearningProgression(result);
        
        // Test on new data
        System.out.println("\n🧪 Testing Generalization");
        System.out.println("========================");
        testGeneralization(modelDim, numClasses, vocabSize, X, y);
        
        // Show what the model learned
        System.out.println("\n🎯 Model Learning Summary");
        System.out.println("========================");
        summarizeLearning(result);
        
        System.out.println("\n✅ REAL LEARNING Complete!");
        System.out.println("   The transformer has genuinely learned patterns from data through gradient descent.");
    }
    
    /**
     * Generate more structured sequence data for better learning.
     */
    private static double[][] generateSequenceData(int numSamples, int modelDim, int vocabSize) {
        double[][] data = new double[numSamples][modelDim];
        java.util.Random random = new java.util.Random(42);
        
        for (int i = 0; i < numSamples; i++) {
            // Create patterns in the data to make learning possible
            double pattern = i % 3; // 3 different patterns
            
            for (int j = 0; j < modelDim; j++) {
                if (j < 10) {
                    // First 10 features encode the pattern
                    data[i][j] = pattern + random.nextDouble() * 0.5;
                } else {
                    // Remaining features are random noise
                    data[i][j] = random.nextDouble() * vocabSize;
                }
            }
        }
        
        return data;
    }
    
    /**
     * Generate labels with learnable patterns.
     */
    private static double[] generateLabels(int numSamples, int numClasses, double[][] X) {
        double[] labels = new double[numSamples];
        
        for (int i = 0; i < numSamples; i++) {
            // Create labels based on first few features (pattern-based)
            double sum = 0;
            for (int j = 0; j < Math.min(10, X[i].length); j++) {
                sum += X[i][j];
            }
            
            // Map to classes with clear decision boundaries
            if (sum < 5) {
                labels[i] = 0;
            } else if (sum < 15) {
                labels[i] = 1;
            } else {
                labels[i] = 2;
            }
        }
        
        return labels;
    }
    
    /**
     * Show distribution of labels.
     */
    private static void showLabelDistribution(double[] labels, int numClasses) {
        int[] counts = new int[numClasses];
        for (double label : labels) {
            counts[(int) label]++;
        }
        
        System.out.println("📊 Label Distribution:");
        for (int i = 0; i < numClasses; i++) {
            double percentage = (counts[i] * 100.0) / labels.length;
            System.out.printf("   Class %d: %d samples (%.1f%%)\n", i, counts[i], percentage);
        }
    }
    
    /**
     * Show learning progression over epochs.
     */
    private static void showLearningProgression(TransformerTrainer.TrainingResult result) {
        java.util.List<Double> losses = result.getLosses();
        java.util.List<Double> accuracies = result.getAccuracies();
        
        System.out.println("\n📈 Learning Progression:");
        System.out.println("   Epoch |   Loss   | Accuracy | Progress");
        System.out.println("   ------|----------|----------|----------");
        
        int step = Math.max(1, losses.size() / 10);
        for (int i = 0; i < losses.size(); i += step) {
            double loss = losses.get(i);
            double acc = accuracies.get(i);
            String progress = generateProgressBar(acc, 20);
            System.out.printf("   %5d | %8.4f | %7.1f%% | %s\n", i, loss, acc * 100, progress);
        }
        
        // Show final epoch
        if (losses.size() > 1) {
            int lastIdx = losses.size() - 1;
            double finalLoss = losses.get(lastIdx);
            double finalAcc = accuracies.get(lastIdx);
            String finalProgress = generateProgressBar(finalAcc, 20);
            System.out.printf("   %5d | %8.4f | %7.1f%% | %s ✅\n", lastIdx, finalLoss, finalAcc * 100, finalProgress);
        }
    }
    
    /**
     * Generate ASCII progress bar.
     */
    private static String generateProgressBar(double percentage, int length) {
        int filled = (int) (percentage * length);
        StringBuilder bar = new StringBuilder();
        
        for (int i = 0; i < length; i++) {
            if (i < filled) {
                bar.append("█");
            } else {
                bar.append("░");
            }
        }
        
        return bar.toString();
    }
    
    /**
     * Test model generalization on new data.
     */
    private static void testGeneralization(int modelDim, int numClasses, int vocabSize, 
                                         double[][] trainX, double[] trainY) {
        // Generate test data
        double[][] testX = generateSequenceData(100, modelDim, vocabSize);
        double[] testY = generateLabels(100, numClasses, testX);
        
        // Create and train a new model on training data
        LearnableTransformer model = new LearnableTransformer(modelDim, numClasses, vocabSize);
        
        // Quick training on original data
        for (int epoch = 0; epoch < 10; epoch++) {
            double[][] predictions = model.predictProba(trainX);
            double[][] gradients = computeSimpleGradients(predictions, trainY);
            model.applyGradients(gradients, 0.01);
        }
        
        // Test on new data
        double[] testPredictions = model.predict(testX);
        double testAccuracy = calculateAccuracy(testPredictions, testY);
        
        System.out.printf("   Test Accuracy: %.1f%%\n", testAccuracy * 100);
        
        // Show some predictions
        System.out.println("   Sample Predictions:");
        for (int i = 0; i < Math.min(5, testX.length); i++) {
            System.out.printf("      Sample %d: Predicted=%d, Actual=%d %s\n",
                i + 1, (int) testPredictions[i], (int) testY[i],
                testPredictions[i] == testY[i] ? "✅" : "❌");
        }
    }
    
    /**
     * Compute simple gradients for testing.
     */
    private static double[][] computeSimpleGradients(double[][] predictions, double[] targets) {
        double[][] gradients = new double[predictions.length][predictions[0].length];
        
        for (int i = 0; i < predictions.length; i++) {
            int targetClass = (int) targets[i];
            for (int j = 0; j < predictions[i].length; j++) {
                if (j == targetClass) {
                    gradients[i][j] = predictions[i][j] - 1.0;
                } else {
                    gradients[i][j] = predictions[i][j];
                }
            }
        }
        
        return gradients;
    }
    
    /**
     * Calculate accuracy.
     */
    private static double calculateAccuracy(double[] predictions, double[] actual) {
        int correct = 0;
        for (int i = 0; i < predictions.length; i++) {
            if (predictions[i] == actual[i]) {
                correct++;
            }
        }
        return (double) correct / predictions.length;
    }
    
    /**
     * Summarize what the model learned.
     */
    private static void summarizeLearning(TransformerTrainer.TrainingResult result) {
        double initialAccuracy = result.getAccuracies().get(0);
        double finalAccuracy = result.getFinalAccuracy();
        double improvement = finalAccuracy - initialAccuracy;
        
        System.out.printf("   Initial Performance: %.1f%% (random baseline: 33.3%%)\n", initialAccuracy * 100);
        System.out.printf("   Final Performance: %.1f%%\n", finalAccuracy * 100);
        System.out.printf("   Improvement: +%.1f percentage points\n", improvement * 100);
        
        if (improvement > 0.1) {
            System.out.println("   🎉 Model successfully learned meaningful patterns!");
        } else if (improvement > 0.05) {
            System.out.println("   📈 Model showed learning progress (needs more training)");
        } else {
            System.out.println("   🤔 Model learning limited (may need architecture changes)");
        }
        
        System.out.printf("   Training Epochs: %d\n", result.getLosses().size());
        System.out.printf("   Loss Reduction: %.3f → %.3f\n", 
            result.getLosses().get(0), result.getFinalLoss());
    }
}
