package com.superml;

import com.superml.inference.InferenceEngine;
import com.superml.inference.BatchInferenceProcessor;
import com.superml.inference.InferenceMetrics;
import com.superml.datasets.Datasets;
import com.superml.linear_model.LogisticRegression;
import com.superml.linear_model.LinearRegression;
import com.superml.pipeline.Pipeline;
import com.superml.preprocessing.StandardScaler;
import com.superml.persistence.ModelPersistence;
import com.superml.model_selection.ModelSelection;
import com.superml.metrics.Metrics;

import java.util.concurrent.CompletableFuture;
import java.util.Arrays;

/**
 * Comprehensive demonstration of SuperML Java Inference Layer capabilities.
 * Shows real-time and batch inference, model loading, caching, and performance monitoring.
 */
public class InferenceDemo {
    
    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("   SuperML Java - Inference Layer Demo");
        System.out.println("=".repeat(60));
        
        try {
            // Create and train some models first
            prepareModels();
            
            // Demonstrate inference capabilities
            demonstrateBasicInference();
            demonstrateModelCaching();
            demonstrateBatchInference();
            demonstrateAsyncInference();
            demonstratePerformanceMonitoring();
            demonstrateClassificationInference();
            
            System.out.println("\n" + "=".repeat(60));
            System.out.println("   Inference Demo Completed Successfully!");
            System.out.println("=".repeat(60));
            
        } catch (Exception e) {
            System.err.println("Demo failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void prepareModels() {
        System.out.println("\n1. PREPARING MODELS FOR INFERENCE");
        System.out.println("=".repeat(40));
        
        // Create classification dataset
        System.out.println("Creating classification dataset...");
        var classificationDataset = Datasets.makeClassification(1000, 10, 2, 42);
        var classificationSplit = ModelSelection.trainTestSplit(
            classificationDataset.data, classificationDataset.target, 0.2, 42);
        
        // Train classification model
        var classifier = new LogisticRegression().setMaxIter(1000);
        classifier.fit(classificationSplit.XTrain, classificationSplit.yTrain);
        
        // Save classification model with statistics
        ModelPersistence.saveWithStats(classifier, "demo_classifier", 
                                      "Demo classification model",
                                      classificationSplit.XTest, classificationSplit.yTest, null);
        
        // Create regression dataset
        System.out.println("Creating regression dataset...");
        var regressionDataset = Datasets.makeRegression(800, 8, 0.1, 42);
        var regressionSplit = ModelSelection.trainTestSplit(
            regressionDataset.data, regressionDataset.target, 0.2, 42);
        
        // Train regression model
        var regressor = new LinearRegression();
        regressor.fit(regressionSplit.XTrain, regressionSplit.yTrain);
        
        // Save regression model
        ModelPersistence.saveWithStats(regressor, "demo_regressor",
                                      "Demo regression model",
                                      regressionSplit.XTest, regressionSplit.yTest, null);
        
        // Create and save pipeline
        var pipeline = new Pipeline()
            .addStep("scaler", new StandardScaler())
            .addStep("classifier", new LogisticRegression().setMaxIter(1000));
        
        pipeline.fit(classificationSplit.XTrain, classificationSplit.yTrain);
        ModelPersistence.save(pipeline, "demo_pipeline", "Demo preprocessing pipeline", null);
        
        System.out.println("✓ Models prepared and saved successfully");
    }
    
    private static void demonstrateBasicInference() {
        System.out.println("\n2. BASIC INFERENCE OPERATIONS");
        System.out.println("=".repeat(40));
        
        // Create inference engine
        InferenceEngine engine = new InferenceEngine();
        
        try {
            // Load models
            System.out.println("Loading models...");
            var classifierInfo = engine.loadModel("classifier", "demo_classifier.superml");
            var regressorInfo = engine.loadModel("regressor", "demo_regressor.superml");
            
            System.out.println("✓ Loaded: " + classifierInfo);
            System.out.println("✓ Loaded: " + regressorInfo);
            
            // Single predictions
            System.out.println("\nMaking single predictions...");
            double[] classificationFeatures = {1.5, -0.5, 2.0, 1.0, -1.5, 0.5, 2.5, -1.0, 1.5, 0.0};
            double[] regressionFeatures = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
            
            double classificationPred = engine.predict("classifier", classificationFeatures);
            double regressionPred = engine.predict("regressor", regressionFeatures);
            
            System.out.printf("Classification prediction: %.3f\n", classificationPred);
            System.out.printf("Regression prediction: %.3f\n", regressionPred);
            
            // Batch predictions
            System.out.println("\nMaking batch predictions...");
            double[][] batchFeatures = {
                {1.5, -0.5, 2.0, 1.0, -1.5, 0.5, 2.5, -1.0, 1.5, 0.0},
                {-1.0, 1.5, -2.0, 0.5, 2.0, -1.5, -0.5, 2.5, -1.0, 1.0},
                {0.0, 0.0, 1.0, -1.0, 2.0, -2.0, 1.5, -1.5, 0.5, -0.5}
            };
            
            double[] batchPredictions = engine.predict("classifier", batchFeatures);
            System.out.println("Batch predictions: " + Arrays.toString(batchPredictions));
            
            // Probability predictions for classification
            System.out.println("\nPredicting class probabilities...");
            double[][] probabilities = engine.predictProba("classifier", batchFeatures);
            for (int i = 0; i < probabilities.length; i++) {
                System.out.printf("Sample %d probabilities: [%.3f, %.3f]\n", 
                                i, probabilities[i][0], probabilities[i][1]);
            }
            
        } finally {
            engine.shutdown();
        }
    }
    
    private static void demonstrateModelCaching() {
        System.out.println("\n3. MODEL CACHING AND MANAGEMENT");
        System.out.println("=".repeat(40));
        
        InferenceEngine engine = new InferenceEngine();
        
        try {
            // Load multiple models
            System.out.println("Loading models into cache...");
            engine.loadModel("model1", "demo_classifier.superml");
            engine.loadModel("model2", "demo_regressor.superml");
            engine.loadModel("pipeline", "demo_pipeline.superml");
            
            // Show loaded models
            var loadedModels = engine.getLoadedModels();
            System.out.println("Loaded models: " + loadedModels);
            
            // Model information
            for (String modelId : loadedModels) {
                var info = engine.getModelInfo(modelId);
                System.out.println("Model info: " + info);
            }
            
            // Warm up models
            System.out.println("\nWarming up models...");
            engine.warmUp("model1", 100);
            engine.warmUp("model2", 100);
            
            // Check model status
            System.out.println("Model1 loaded: " + engine.isModelLoaded("model1"));
            System.out.println("Model2 loaded: " + engine.isModelLoaded("model2"));
            System.out.println("NonExistent loaded: " + engine.isModelLoaded("nonexistent"));
            
            // Unload a model
            System.out.println("\nUnloading model2...");
            boolean unloaded = engine.unloadModel("model2");
            System.out.println("Model2 unloaded: " + unloaded);
            System.out.println("Remaining models: " + engine.getLoadedModels());
            
        } finally {
            engine.shutdown();
        }
    }
    
    private static void demonstrateBatchInference() {
        System.out.println("\n4. BATCH INFERENCE PROCESSING");
        System.out.println("=".repeat(40));
        
        InferenceEngine engine = new InferenceEngine();
        BatchInferenceProcessor processor = new BatchInferenceProcessor(engine);
        
        try {
            // Load model
            engine.loadModel("batch_model", "demo_classifier.superml");
            
            // Create test data
            System.out.println("Creating test data for batch processing...");
            var testData = Datasets.makeClassification(5000, 10, 2, 123);
            
            // Process large batch
            System.out.println("Processing batch with 5000 samples...");
            var config = new BatchInferenceProcessor.BatchConfig()
                .setBatchSize(500)
                .setShowProgress(true)
                .setProgressInterval(2);
            
            long startTime = System.currentTimeMillis();
            double[] predictions = processor.process(testData.data, "batch_model", config);
            long processingTime = System.currentTimeMillis() - startTime;
            
            System.out.printf("Processed %d samples in %dms (%.1f samples/sec)\n",
                            predictions.length, processingTime, 
                            predictions.length / (processingTime / 1000.0));
            
            // Show sample predictions
            System.out.println("Sample predictions: " + 
                             Arrays.toString(Arrays.copyOf(predictions, 10)));
            
        } finally {
            engine.shutdown();
        }
    }
    
    private static void demonstrateAsyncInference() {
        System.out.println("\n5. ASYNCHRONOUS INFERENCE");
        System.out.println("=".repeat(40));
        
        InferenceEngine engine = new InferenceEngine();
        
        try {
            // Load model
            engine.loadModel("async_model", "demo_classifier.superml");
            
            // Single async prediction
            System.out.println("Starting async single prediction...");
            double[] features = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
            
            CompletableFuture<Double> singleFuture = engine.predictAsync("async_model", features);
            
            // Batch async prediction
            System.out.println("Starting async batch prediction...");
            double[][] batchFeatures = new double[1000][10];
            for (int i = 0; i < 1000; i++) {
                for (int j = 0; j < 10; j++) {
                    batchFeatures[i][j] = Math.random() * 10 - 5;
                }
            }
            
            CompletableFuture<double[]> batchFuture = engine.predictAsync("async_model", batchFeatures);
            
            // Wait for results
            System.out.println("Waiting for async results...");
            Double singleResult = singleFuture.get();
            double[] batchResults = batchFuture.get();
            
            System.out.printf("Async single prediction: %.3f\n", singleResult);
            System.out.printf("Async batch prediction completed: %d results\n", batchResults.length);
            System.out.printf("First 5 batch results: %s\n", 
                            Arrays.toString(Arrays.copyOf(batchResults, 5)));
            
        } catch (Exception e) {
            System.err.println("Async inference failed: " + e.getMessage());
        } finally {
            engine.shutdown();
        }
    }
    
    private static void demonstratePerformanceMonitoring() {
        System.out.println("\n6. PERFORMANCE MONITORING");
        System.out.println("=".repeat(40));
        
        InferenceEngine engine = new InferenceEngine();
        
        try {
            // Load model
            engine.loadModel("perf_model", "demo_classifier.superml");
            
            // Make multiple predictions to generate metrics
            System.out.println("Running performance test...");
            double[][] testData = new double[100][10];
            for (int i = 0; i < 100; i++) {
                for (int j = 0; j < 10; j++) {
                    testData[i][j] = Math.random() * 10 - 5;
                }
            }
            
            // Run predictions with different batch sizes
            for (int batchSize : new int[]{1, 10, 50, 100}) {
                for (int i = 0; i < testData.length; i += batchSize) {
                    int endIndex = Math.min(i + batchSize, testData.length);
                    double[][] batch = Arrays.copyOfRange(testData, i, endIndex);
                    engine.predict("perf_model", batch);
                }
            }
            
            // Get and display metrics
            InferenceMetrics metrics = engine.getMetrics("perf_model");
            System.out.println("\nPerformance Metrics:");
            System.out.println("===================");
            System.out.printf("Total inferences: %d\n", metrics.getTotalInferences());
            System.out.printf("Total samples: %d\n", metrics.getTotalSamples());
            System.out.printf("Average time per inference: %.2f ms\n", metrics.getAverageInferenceTimeMs());
            System.out.printf("Average time per sample: %.2f μs\n", metrics.getAverageTimePerSampleMicros());
            System.out.printf("Min inference time: %.2f ms\n", metrics.getMinInferenceTimeMs());
            System.out.printf("Max inference time: %.2f ms\n", metrics.getMaxInferenceTimeMs());
            System.out.printf("Throughput: %.1f samples/sec\n", metrics.getThroughputSamplesPerSecond());
            System.out.printf("Error count: %d\n", metrics.getErrorCount());
            System.out.printf("Error rate: %.2f%%\n", metrics.getErrorRate());
            
            System.out.println("\nMetrics summary: " + metrics.getSummary());
            
            // Show all model metrics
            System.out.println("\nAll Model Metrics:");
            var allMetrics = engine.getAllMetrics();
            allMetrics.forEach((modelId, modelMetrics) -> 
                System.out.println(modelId + ": " + modelMetrics.getSummary()));
            
        } finally {
            engine.shutdown();
        }
    }
    
    private static void demonstrateClassificationInference() {
        System.out.println("\n7. CLASSIFICATION-SPECIFIC INFERENCE");
        System.out.println("=".repeat(40));
        
        InferenceEngine engine = new InferenceEngine();
        
        try {
            // Load classification model
            engine.loadModel("clf", "demo_classifier.superml");
            
            // Create test samples
            double[][] testSamples = {
                {2.0, -1.0, 3.0, 1.5, -2.0, 0.5, 3.5, -1.5, 2.5, 0.0},
                {-1.5, 2.0, -2.5, 0.5, 3.0, -1.0, -0.5, 2.5, -1.5, 1.0},
                {0.5, 0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 1.0, -1.0}
            };
            
            System.out.println("Making classification predictions...");
            
            // Class predictions
            double[] classPredictions = engine.predict("clf", testSamples);
            System.out.println("Class predictions: " + Arrays.toString(classPredictions));
            
            // Probability predictions
            double[][] probabilities = engine.predictProba("clf", testSamples);
            System.out.println("\nClass probabilities:");
            for (int i = 0; i < probabilities.length; i++) {
                System.out.printf("Sample %d: Class 0=%.3f, Class 1=%.3f (Predicted: %.0f)\n",
                                i, probabilities[i][0], probabilities[i][1], classPredictions[i]);
            }
            
            // Confidence analysis
            System.out.println("\nConfidence Analysis:");
            for (int i = 0; i < probabilities.length; i++) {
                double confidence = Math.max(probabilities[i][0], probabilities[i][1]);
                String confidenceLevel = confidence > 0.8 ? "High" : confidence > 0.6 ? "Medium" : "Low";
                System.out.printf("Sample %d: Confidence=%.3f (%s)\n", i, confidence, confidenceLevel);
            }
            
        } finally {
            engine.shutdown();
        }
    }
}
