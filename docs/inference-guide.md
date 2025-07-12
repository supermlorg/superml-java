---
title: "Inference Layer Guide"
description: "Production model deployment and high-performance inference with SuperML Java"
---

# Inference Layer Guide

Comprehensive guide to using the SuperML Java Inference Layer for production model deployment and high-performance inference.

## 🎯 Overview

The SuperML Java Inference Layer provides a complete solution for deploying trained models in production environments. It offers:

- **Model Loading and Caching** - Fast model loading with intelligent caching
- **High-Performance Inference** - Optimized for both single and batch predictions
- **Asynchronous Processing** - Non-blocking inference operations
- **Performance Monitoring** - Detailed metrics and performance tracking
- **Type Safety** - Compile-time type checking for models
- **Batch Processing** - Efficient processing of large datasets
- **Thread Safety** - Safe concurrent operations

## 🚀 Quick Start

### Basic Inference

```java
import com.superml.inference.InferenceEngine;

// Create inference engine
InferenceEngine engine = new InferenceEngine();

// Load a model
engine.loadModel("my_classifier", "models/classifier.superml");

// Single prediction
double[] features = {1.0, 2.0, 3.0, 4.0};
double prediction = engine.predict("my_classifier", features);

// Batch prediction
double[][] batchFeatures = { {1,2,3,4}, {5,6,7,8} };
double[] predictions = engine.predict("my_classifier", batchFeatures);

// Cleanup
engine.shutdown();
```

### Classification with Probabilities

```java
// Load classification model
engine.loadModel("classifier", "models/iris_classifier.superml");

// Predict classes
double[] features = {5.1, 3.5, 1.4, 0.2};
double prediction = engine.predict("classifier", features);

// Predict class probabilities
double[] probabilities = engine.predictProba("classifier", features);
System.out.printf("Predicted class: %.0f\n", prediction);
System.out.printf("Class probabilities: [%.3f, %.3f, %.3f]\n", 
                  probabilities[0], probabilities[1], probabilities[2]);
```

## 🔧 Core Components

### InferenceEngine

The main inference engine that handles model loading, caching, and predictions.

```java
// Default configuration
InferenceEngine engine = new InferenceEngine();

// Custom configuration
InferenceEngine.InferenceConfig config = new InferenceEngine.InferenceConfig(
    8,      // thread pool size
    true,   // validate input size
    true,   // validate finite values
    100     // max cache size
);
InferenceEngine engine = new InferenceEngine(config);
```

#### Key Methods

- `loadModel(modelId, filePath)` - Load and cache a model
- `predict(modelId, features)` - Make predictions
- `predictProba(modelId, features)` - Get class probabilities
- `predictAsync(modelId, features)` - Asynchronous prediction
- `getMetrics(modelId)` - Get performance metrics
- `warmUp(modelId, samples)` - Warm up model for optimal performance

### Model Management

```java
// Load models with type safety
LogisticRegression model = engine.loadModel("lr", "lr.superml", LogisticRegression.class);

// Check if model is loaded
boolean loaded = engine.isModelLoaded("lr");

// Get model information
InferenceEngine.ModelInfo info = engine.getModelInfo("lr");
System.out.println("Model: " + info.modelClass);
System.out.println("Description: " + info.description);

// List all loaded models
List<String> models = engine.getLoadedModels();

// Unload model
engine.unloadModel("lr");
```

### Batch Processing

For high-throughput processing of large datasets:

```java
import com.superml.inference.BatchInferenceProcessor;

// Create batch processor
BatchInferenceProcessor processor = new BatchInferenceProcessor(engine);

// Process CSV file
BatchInferenceProcessor.BatchResult result = processor.processCSV(
    "input.csv", "output.csv", "my_model");

// Custom batch configuration
BatchInferenceProcessor.BatchConfig config = new BatchInferenceProcessor.BatchConfig()
    .setBatchSize(1000)
    .setShowProgress(true)
    .setContinueOnError(true);

BatchInferenceProcessor.BatchResult result = processor.processCSV(
    "input.csv", "output.csv", "my_model", config);

System.out.println("Processed: " + result.getSummary());
```

## 📊 Performance Monitoring

### Inference Metrics

Track detailed performance metrics for each model:

```java
// Get metrics for a model
InferenceMetrics metrics = engine.getMetrics("my_model");

System.out.printf("Total inferences: %d\n", metrics.getTotalInferences());
System.out.printf("Total samples: %d\n", metrics.getTotalSamples());
System.out.printf("Average time: %.2f ms\n", metrics.getAverageInferenceTimeMs());
System.out.printf("Throughput: %.1f samples/sec\n", metrics.getThroughputSamplesPerSecond());
System.out.printf("Error rate: %.2f%%\n", metrics.getErrorRate());

// Get summary
System.out.println(metrics.getSummary());

// Clear metrics
engine.clearMetrics("my_model");
```

### Available Metrics

- **Timing**: Average, min, max inference times
- **Throughput**: Samples per second, inferences per second
- **Volume**: Total inferences, total samples processed
- **Reliability**: Error count, error rate
- **Efficiency**: Time per sample, batch efficiency

## ⚡ Asynchronous Inference

For non-blocking operations and improved throughput:

```java
import java.util.concurrent.CompletableFuture;

// Single async prediction
CompletableFuture<Double> future = engine.predictAsync("model", features);
future.thenAccept(prediction -> 
    System.out.println("Prediction: " + prediction));

// Batch async prediction
CompletableFuture<double[]> batchFuture = engine.predictAsync("model", batchFeatures);
double[] results = batchFuture.get(); // Wait for completion

// Multiple async operations
CompletableFuture<Double> future1 = engine.predictAsync("model1", features1);
CompletableFuture<Double> future2 = engine.predictAsync("model2", features2);

CompletableFuture.allOf(future1, future2).thenRun(() -> {
    System.out.println("All predictions completed");
});
```

## 🎛️ Advanced Configuration

### Input Validation

```java
InferenceEngine.InferenceConfig config = new InferenceEngine.InferenceConfig(
    Runtime.getRuntime().availableProcessors(), // Use all available cores
    true,  // Validate input size matches expected features
    true,  // Validate all values are finite (no NaN/Infinity)
    50     // Maximum models in cache
);

InferenceEngine engine = new InferenceEngine(config);
```

### Model Warm-up

Optimize performance by warming up models:

```java
// Load model
engine.loadModel("production_model", "models/prod.superml");

// Warm up with 1000 dummy samples
engine.warmUp("production_model", 1000);

// Now the model is optimized for production inference
```

### Batch Configuration

```java
BatchInferenceProcessor.BatchConfig batchConfig = 
    new BatchInferenceProcessor.BatchConfig()
        .setBatchSize(2000)           // Process 2000 samples per batch
        .setContinueOnError(true)     // Continue processing on errors
        .setShowProgress(true)        // Show progress updates
        .setProgressInterval(5)       // Update every 5 batches
        .setPredictionColumnName("score"); // Custom column name
```

## 🔍 Error Handling

### Exception Types

```java
try {
    engine.predict("nonexistent_model", features);
} catch (InferenceException e) {
    System.err.println("Inference failed: " + e.getMessage());
}
```

### Common Error Scenarios

1. **Model Not Loaded**: `InferenceException` when using unloaded model
2. **Input Validation**: Invalid feature dimensions or NaN values
3. **Type Mismatch**: Loading model with wrong expected type
4. **File Not Found**: Model file doesn't exist or is corrupted

## 📈 Best Practices

### 1. Model Loading Strategy

```java
// Load frequently-used models at startup
engine.loadModel("primary_classifier", "models/primary.superml");
engine.loadModel("fallback_model", "models/fallback.superml");

// Warm up critical models
engine.warmUp("primary_classifier", 1000);
```

### 2. Batch Size Optimization

```java
// For real-time inference: smaller batches
BatchConfig realtimeConfig = new BatchConfig().setBatchSize(100);

// For offline processing: larger batches
BatchConfig offlineConfig = new BatchConfig().setBatchSize(5000);
```

### 3. Resource Management

```java
// Always shutdown the engine when done
try (InferenceEngine engine = new InferenceEngine()) {
    // Use engine for inference
    engine.loadModel("model", "path/to/model");
    double prediction = engine.predict("model", features);
} // Automatically calls shutdown()
```

### 4. Monitoring and Alerting

```java
// Regular metrics monitoring
InferenceMetrics metrics = engine.getMetrics("production_model");
if (metrics.getErrorRate() > 5.0) {
    // Alert: High error rate detected
    alertingSystem.sendAlert("High inference error rate: " + metrics.getErrorRate() + "%");
}

if (metrics.getAverageInferenceTimeMs() > 100) {
    // Alert: Slow inference detected
    alertingSystem.sendAlert("Slow inference: " + metrics.getAverageInferenceTimeMs() + "ms");
}
```

## 🚀 Production Deployment

### Container Deployment

```dockerfile
FROM openjdk:11-jre-slim

COPY superml-app.jar /app/
COPY models/ /app/models/

WORKDIR /app
CMD ["java", "-jar", "superml-app.jar"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: superml-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: superml-inference
  template:
    metadata:
      labels:
        app: superml-inference
    spec:
      containers:
      - name: inference
        image: superml-inference:latest
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

### Load Balancing

```java
// Create multiple inference engines for load balancing
public class InferenceService {
    private final List<InferenceEngine> engines;
    private final AtomicInteger currentEngine = new AtomicInteger(0);
    
    public InferenceService(int engineCount) {
        engines = new ArrayList<>();
        for (int i = 0; i < engineCount; i++) {
            InferenceEngine engine = new InferenceEngine();
            engine.loadModel("model", "models/production.superml");
            engines.add(engine);
        }
    }
    
    public double predict(double[] features) {
        // Round-robin load balancing
        int index = currentEngine.getAndIncrement() % engines.size();
        return engines.get(index).predict("model", features);
    }
}
```

## 🔬 Performance Tuning

### JVM Optimization

```bash
java -Xmx4g -Xms2g \
     -XX:+UseG1GC \
     -XX:MaxGCPauseMillis=100 \
     -XX:+UseStringDeduplication \
     -jar superml-app.jar
```

### Threading Configuration

```java
// Configure thread pool based on workload
int cores = Runtime.getRuntime().availableProcessors();

// CPU-intensive workload
InferenceConfig cpuConfig = new InferenceConfig(cores, true, true, 100);

// I/O-intensive workload  
InferenceConfig ioConfig = new InferenceConfig(cores * 2, true, true, 100);
```

The Inference Layer provides a complete solution for deploying SuperML models in production environments with enterprise-grade performance, monitoring, and reliability features.
