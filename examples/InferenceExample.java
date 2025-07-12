package examples;

import com.superml.inference.InferenceEngine;
import com.superml.inference.BatchInferenceProcessor;
import com.superml.linear_model.LogisticRegression;
import com.superml.datasets.Datasets;
import com.superml.model_selection.ModelSelection;
import com.superml.persistence.ModelPersistence;
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
            var dataset = Datasets.loadIris();
            var split = ModelSelection.trainTestSplit(dataset.data, dataset.target, 0.3, 42);
            
            var model = new LogisticRegression().setMaxIter(1000);
            model.fit(split.XTrain, split.yTrain);
            
            String modelPath = "examples/iris_model.superml";
            ModelPersistence.saveModel(model, modelPath);
            System.out.printf("✓ Model saved to: %s\n", modelPath);
            
            // 2. Create inference engine
            System.out.println("\nInitializing inference engine...");
            var inferenceEngine = new InferenceEngine()
                .setCacheSize(1000)
                .setWarmupSamples(100);
            
            // Load model into inference engine
            inferenceEngine.loadModel("iris_classifier", modelPath);
            System.out.println("✓ Model loaded into inference engine");
            
            // 3. Warm up the model
            System.out.println("Warming up model for optimal performance...");
            inferenceEngine.warmUp("iris_classifier", split.XTest);
            System.out.println("✓ Model warmed up");
            
            // 4. Single prediction (real-time inference)
            System.out.println("\n" + "=".repeat(40));
            System.out.println("Real-time Inference:");
            System.out.println("=".repeat(40));
            
            double[] sampleInput = split.XTest[0];
            long startTime = System.nanoTime();
            double prediction = inferenceEngine.predict("iris_classifier", sampleInput);
            long inferenceTime = System.nanoTime() - startTime;
            
            System.out.printf("Sample input: [%.2f, %.2f, %.2f, %.2f]\n", 
                            sampleInput[0], sampleInput[1], sampleInput[2], sampleInput[3]);
            System.out.printf("Prediction: %d (%s)\n", (int)prediction, dataset.targetNames[(int)prediction]);
            System.out.printf("Inference time: %.2f μs\n", inferenceTime / 1000.0);
            
            // 5. Async predictions
            System.out.println("\nAsynchronous Inference:");
            System.out.println("-".repeat(25));
            
            CompletableFuture<Double>[] futures = new CompletableFuture[5];
            for (int i = 0; i < 5; i++) {
                final int index = i;
                futures[i] = inferenceEngine.predictAsync("iris_classifier", split.XTest[index]);
            }
            
            // Wait for all async predictions
            for (int i = 0; i < futures.length; i++) {
                try {
                    double asyncPrediction = futures[i].get();
                    System.out.printf("Async prediction %d: %d (%s)\n", 
                                    i + 1, (int)asyncPrediction, dataset.targetNames[(int)asyncPrediction]);
                } catch (ExecutionException | InterruptedException e) {
                    System.err.printf("Async prediction %d failed: %s\n", i + 1, e.getMessage());
                }
            }
            
            // 6. Batch inference
            System.out.println("\n" + "=".repeat(40));
            System.out.println("Batch Inference:");
            System.out.println("=".repeat(40));
            
            var batchProcessor = new BatchInferenceProcessor(inferenceEngine);
            
            // Process batch of 20 samples
            double[][] batchInputs = new double[20][];
            for (int i = 0; i < 20; i++) {
                batchInputs[i] = split.XTest[i % split.XTest.length];
            }
            
            startTime = System.nanoTime();
            double[] batchPredictions = batchProcessor.processSerial("iris_classifier", batchInputs);
            long batchTime = System.nanoTime() - startTime;
            
            System.out.printf("Batch size: %d samples\n", batchInputs.length);
            System.out.printf("Batch processing time: %.2f ms\n", batchTime / 1_000_000.0);
            System.out.printf("Average per sample: %.2f μs\n", batchTime / 1000.0 / batchInputs.length);
            
            // Show some batch results
            System.out.println("\nBatch results (first 10):");
            for (int i = 0; i < Math.min(10, batchPredictions.length); i++) {
                int predClass = (int) batchPredictions[i];
                System.out.printf("  Sample %2d: %d (%s)\n", i + 1, predClass, dataset.targetNames[predClass]);
            }
            
            // 7. Performance monitoring
            System.out.println("\n" + "=".repeat(40));
            System.out.println("Performance Metrics:");
            System.out.println("=".repeat(40));
            
            var metrics = inferenceEngine.getMetrics("iris_classifier");
            System.out.printf("Total predictions: %d\n", metrics.getTotalPredictions());
            System.out.printf("Average latency: %.2f μs\n", metrics.getAverageLatency() / 1000.0);
            System.out.printf("Cache hit rate: %.1f%%\n", metrics.getCacheHitRate() * 100);
            System.out.printf("Throughput: %.0f predictions/sec\n", metrics.getThroughputPerSecond());
            
            // 8. Parallel batch processing
            System.out.println("\nParallel Batch Processing:");
            System.out.println("-".repeat(30));
            
            // Create larger batch for parallel processing
            double[][] largeBatch = new double[100][];
            for (int i = 0; i < 100; i++) {
                largeBatch[i] = split.XTest[i % split.XTest.length];
            }
            
            startTime = System.nanoTime();
            double[] parallelPredictions = batchProcessor.processParallel("iris_classifier", largeBatch);
            long parallelTime = System.nanoTime() - startTime;
            
            System.out.printf("Parallel batch size: %d samples\n", largeBatch.length);
            System.out.printf("Parallel processing time: %.2f ms\n", parallelTime / 1_000_000.0);
            System.out.printf("Speedup vs serial: %.1fx\n", 
                            (double) batchTime / parallelTime * largeBatch.length / batchInputs.length);
            
            // 9. Model unloading
            System.out.println("\n" + "=".repeat(40));
            System.out.println("Cleanup:");
            System.out.println("=".repeat(40));
            
            inferenceEngine.unloadModel("iris_classifier");
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
