package examples;

import org.superml.inference.InferenceEngine;
import org.superml.linear_model.LogisticRegression;
import org.superml.datasets.Datasets;
import org.superml.model_selection.ModelSelection;
import org.superml.persistence.ModelPersistence;
import java.io.File;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

/**
 * Production inference example demonstrating real-time and batch prediction.
 * Shows how to deploy trained models for production use.
 */
public class InferenceExample {
    
    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("      SuperML Java - Production Inference Example");
        System.out.println("=".repeat(60));
        
        try {
            // 1. Train and save a model
            System.out.println("Training and saving model...");
            var dataset = Datasets.makeClassification(150, 4, 3, 42);
            var split = ModelSelection.trainTestSplit(dataset.data, dataset.target, 0.3, 42);
            
            var model = new LogisticRegression().setMaxIter(1000);
            model.fit(split.XTrain, split.yTrain);
            
            String modelPath = "temp_classifier_model.superml";
            ModelPersistence.save(model, modelPath);
            System.out.printf("✓ Model saved to: %s\n", modelPath);
            
            // 2. Create inference engine and load model
            System.out.println("\nInitializing inference engine...");
            var inferenceEngine = new InferenceEngine();
            
            // Load model into inference engine
            inferenceEngine.loadModel("demo_classifier", modelPath);
            System.out.println("✓ Model loaded into inference engine");
            
            // 3. Warm up the model
            System.out.println("Warming up model for optimal performance...");
            inferenceEngine.warmUp("demo_classifier", 50);
            System.out.println("✓ Model warmed up with 50 samples");
            
            // 4. Single prediction (real-time inference)
            System.out.println("\n" + "=".repeat(40));
            System.out.println("Real-time Inference:");
            System.out.println("=".repeat(40));
            
            double[] sampleInput = split.XTest[0];
            long startTime = System.nanoTime();
            double prediction = inferenceEngine.predict("demo_classifier", sampleInput);
            long inferenceTime = System.nanoTime() - startTime;
            
            System.out.printf("Sample input: [%.2f, %.2f, %.2f, %.2f]\n", 
                            sampleInput[0], sampleInput[1], sampleInput[2], sampleInput[3]);
            System.out.printf("Prediction: %d (Class %d)\n", (int)prediction, (int)prediction);
            System.out.printf("Inference time: %.2f μs\n", inferenceTime / 1000.0);
            
            // 5. Batch inference
            System.out.println("\n" + "=".repeat(40));
            System.out.println("Batch Inference:");
            System.out.println("=".repeat(40));
            
            // Process batch of samples
            double[][] batchInputs = new double[20][];
            for (int i = 0; i < 20; i++) {
                batchInputs[i] = split.XTest[i % split.XTest.length];
            }
            
            startTime = System.nanoTime();
            double[] batchPredictions = inferenceEngine.predict("demo_classifier", batchInputs);
            long batchTime = System.nanoTime() - startTime;
            
            System.out.printf("Batch size: %d samples\n", batchInputs.length);
            System.out.printf("Batch processing time: %.2f ms\n", batchTime / 1_000_000.0);
            System.out.printf("Average per sample: %.2f μs\n", batchTime / 1000.0 / batchInputs.length);
            
            // Show some batch results
            System.out.println("\nBatch results (first 10):");
            for (int i = 0; i < Math.min(10, batchPredictions.length); i++) {
                int predClass = (int) batchPredictions[i];
                System.out.printf("  Sample %2d: %d (Class %d)\n", i + 1, predClass, predClass);
            }
            
            // 6. Async predictions
            System.out.println("\nAsynchronous Inference:");
            System.out.println("-".repeat(25));
            
            @SuppressWarnings("unchecked")
            CompletableFuture<Double>[] futures = new CompletableFuture[5];
            for (int i = 0; i < 5; i++) {
                final int index = i;
                futures[i] = inferenceEngine.predictAsync("demo_classifier", split.XTest[index]);
            }
            
            // Wait for all async predictions
            for (int i = 0; i < futures.length; i++) {
                try {
                    double asyncPrediction = futures[i].get();
                    System.out.printf("Async prediction %d: %d (Class %d)\n", 
                                    i + 1, (int)asyncPrediction, (int)asyncPrediction);
                } catch (ExecutionException | InterruptedException e) {
                    System.err.printf("Async prediction %d failed: %s\n", i + 1, e.getMessage());
                }
            }
            
            // 7. Probability predictions
            System.out.println("\n" + "=".repeat(40));
            System.out.println("Probability Predictions:");
            System.out.println("=".repeat(40));
            
            double[] probabilities = inferenceEngine.predictProba("demo_classifier", sampleInput);
            System.out.println("Class probabilities for sample:");
            for (int i = 0; i < probabilities.length; i++) {
                System.out.printf("  Class %d: %.4f (%.1f%%)\n", i, probabilities[i], probabilities[i] * 100);
            }
            
            // 8. Performance monitoring
            System.out.println("\n" + "=".repeat(40));
            System.out.println("Performance Summary:");
            System.out.println("=".repeat(40));
            
            var metrics = inferenceEngine.getMetrics("demo_classifier");
            System.out.printf("Total inferences: %d\n", metrics.getTotalInferences());
            System.out.printf("Total samples: %d\n", metrics.getTotalSamples());
            System.out.printf("Average time per sample: %.2f μs\n", metrics.getAverageTimePerSampleMicros());
            System.out.printf("Throughput: %.0f samples/sec\n", metrics.getThroughputSamplesPerSecond());
            System.out.printf("Error rate: %.2f%%\n", metrics.getErrorRate() * 100);
            
            // 9. Model unloading and cleanup
            System.out.println("\n" + "=".repeat(40));
            System.out.println("Cleanup:");
            System.out.println("=".repeat(40));
            
            inferenceEngine.unloadModel("demo_classifier");
            System.out.println("✓ Model unloaded from inference engine");
            
            // Clean up saved model file
            new File(modelPath).delete();
            System.out.println("✓ Temporary model file cleaned up");
            
            System.out.println("\n✓ Inference example completed successfully!");
            System.out.println("💡 Production inference supports caching, async, and batch processing");
            
        } catch (Exception e) {
            System.err.println("❌ Example failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
