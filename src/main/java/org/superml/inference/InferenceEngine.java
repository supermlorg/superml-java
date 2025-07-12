package org.superml.inference;

import org.superml.core.BaseEstimator;
import org.superml.core.SupervisedLearner;
import org.superml.core.Classifier;
import org.superml.core.Regressor;
import org.superml.persistence.ModelPersistence;
import org.superml.pipeline.Pipeline;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.Map;
import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * High-performance inference engine for trained SuperML models.
 * Provides model loading, caching, batch inference, and performance monitoring.
 * 
 * Features:
 * - Model caching for fast repeated inference
 * - Batch processing for efficiency
 * - Asynchronous inference capabilities
 * - Input validation and preprocessing
 * - Performance metrics and monitoring
 * - Thread-safe operations
 * 
 * Usage:
 * <pre>
 * InferenceEngine engine = new InferenceEngine();
 * 
 * // Load and cache model
 * engine.loadModel("my_classifier", "models/classifier.superml");
 * 
 * // Single prediction
 * double[] features = {1.0, 2.0, 3.0, 4.0};
 * double prediction = engine.predict("my_classifier", features);
 * 
 * // Batch prediction
 * double[][] batchFeatures = {{1,2,3,4}, {5,6,7,8}};
 * double[] predictions = engine.predict("my_classifier", batchFeatures);
 * 
 * // Async prediction
 * CompletableFuture&lt;Double&gt; future = engine.predictAsync("my_classifier", features);
 * </pre>
 */
public class InferenceEngine {
    private static final Logger logger = LoggerFactory.getLogger(InferenceEngine.class);
    
    // Model cache for fast repeated inference
    private final Map<String, LoadedModel> modelCache = new ConcurrentHashMap<>();
    
    // Thread pool for async operations
    private final ExecutorService executorService;
    
    // Performance monitoring
    private final Map<String, InferenceMetrics> metricsMap = new ConcurrentHashMap<>();
    
    // Configuration
    private final InferenceConfig config;
    
    /**
     * Create inference engine with default configuration.
     */
    public InferenceEngine() {
        this(new InferenceConfig());
    }
    
    /**
     * Create inference engine with custom configuration.
     * @param config inference configuration
     */
    public InferenceEngine(InferenceConfig config) {
        this.config = config;
        this.executorService = Executors.newFixedThreadPool(config.threadPoolSize);
        logger.info("InferenceEngine initialized with {} threads", config.threadPoolSize);
    }
    
    /**
     * Load a model from file and cache it for inference.
     * @param modelId unique identifier for the model
     * @param filePath path to the saved model file
     * @return model metadata
     */
    public ModelInfo loadModel(String modelId, String filePath) {
        return loadModel(modelId, filePath, BaseEstimator.class);
    }
    
    /**
     * Load a model with type checking and cache it for inference.
     * @param <T> expected model type
     * @param modelId unique identifier for the model
     * @param filePath path to the saved model file
     * @param expectedClass expected model class
     * @return model metadata
     */
    public <T extends BaseEstimator> ModelInfo loadModel(String modelId, String filePath, Class<T> expectedClass) {
        if (modelId == null || modelId.trim().isEmpty()) {
            throw new IllegalArgumentException("Model ID cannot be null or empty");
        }
        
        long startTime = System.currentTimeMillis();
        
        try {
            // Load model from persistence
            T model = ModelPersistence.load(filePath, expectedClass);
            ModelPersistence.ModelMetadata metadata = ModelPersistence.getMetadata(filePath);
            
            // Create loaded model wrapper
            LoadedModel loadedModel = new LoadedModel(model, metadata, filePath);
            
            // Cache the model
            modelCache.put(modelId, loadedModel);
            
            // Initialize metrics
            metricsMap.put(modelId, new InferenceMetrics(modelId));
            
            long loadTime = System.currentTimeMillis() - startTime;
            logger.info("Model '{}' loaded successfully in {}ms", modelId, loadTime);
            
            return new ModelInfo(modelId, metadata.modelClass, metadata.description, 
                               metadata.savedAt.toString(), loadTime, filePath);
            
        } catch (Exception e) {
            logger.error("Failed to load model '{}' from {}", modelId, filePath, e);
            throw new InferenceException("Failed to load model: " + e.getMessage(), e);
        }
    }
    
    /**
     * Check if a model is loaded and cached.
     * @param modelId model identifier
     * @return true if model is cached
     */
    public boolean isModelLoaded(String modelId) {
        return modelCache.containsKey(modelId);
    }
    
    /**
     * Unload a model from cache.
     * @param modelId model identifier
     * @return true if model was unloaded
     */
    public boolean unloadModel(String modelId) {
        LoadedModel removed = modelCache.remove(modelId);
        if (removed != null) {
            metricsMap.remove(modelId);
            logger.info("Model '{}' unloaded from cache", modelId);
            return true;
        }
        return false;
    }
    
    /**
     * Get list of loaded model IDs.
     * @return list of model identifiers
     */
    public List<String> getLoadedModels() {
        return new ArrayList<>(modelCache.keySet());
    }
    
    /**
     * Make a single prediction.
     * @param modelId model identifier
     * @param features input features (single sample)
     * @return prediction
     */
    public double predict(String modelId, double[] features) {
        double[][] batchFeatures = {features};
        double[] predictions = predict(modelId, batchFeatures);
        return predictions[0];
    }
    
    /**
     * Make batch predictions.
     * @param modelId model identifier
     * @param features input features (multiple samples)
     * @return predictions
     */
    public double[] predict(String modelId, double[][] features) {
        LoadedModel loadedModel = getLoadedModel(modelId);
        InferenceMetrics metrics = metricsMap.get(modelId);
        
        long startTime = System.nanoTime();
        
        try {
            // Validate input
            validateInput(features, loadedModel);
            
            // Make predictions
            double[] predictions;
            BaseEstimator model = loadedModel.model;
            
            if (model instanceof SupervisedLearner) {
                predictions = ((SupervisedLearner) model).predict(features);
            } else {
                throw new InferenceException("Model does not support prediction");
            }
            
            // Update metrics
            long inferenceTime = System.nanoTime() - startTime;
            metrics.recordInference(features.length, inferenceTime);
            
            logger.debug("Predicted {} samples with model '{}' in {}μs", 
                        features.length, modelId, inferenceTime / 1000);
            
            return predictions;
            
        } catch (Exception e) {
            metrics.recordError();
            logger.error("Prediction failed for model '{}'", modelId, e);
            throw new InferenceException("Prediction failed: " + e.getMessage(), e);
        }
    }
    
    /**
     * Make asynchronous single prediction.
     * @param modelId model identifier
     * @param features input features
     * @return future containing prediction
     */
    public CompletableFuture<Double> predictAsync(String modelId, double[] features) {
        return CompletableFuture.supplyAsync(() -> predict(modelId, features), executorService);
    }
    
    /**
     * Make asynchronous batch predictions.
     * @param modelId model identifier
     * @param features input features
     * @return future containing predictions
     */
    public CompletableFuture<double[]> predictAsync(String modelId, double[][] features) {
        return CompletableFuture.supplyAsync(() -> predict(modelId, features), executorService);
    }
    
    /**
     * Predict class probabilities (for classification models).
     * @param modelId model identifier
     * @param features input features
     * @return class probabilities
     */
    public double[][] predictProba(String modelId, double[][] features) {
        LoadedModel loadedModel = getLoadedModel(modelId);
        BaseEstimator model = loadedModel.model;
        
        if (!(model instanceof Classifier)) {
            throw new InferenceException("Model is not a classifier");
        }
        
        InferenceMetrics metrics = metricsMap.get(modelId);
        long startTime = System.nanoTime();
        
        try {
            validateInput(features, loadedModel);
            
            double[][] probabilities = ((Classifier) model).predictProba(features);
            
            long inferenceTime = System.nanoTime() - startTime;
            metrics.recordInference(features.length, inferenceTime);
            
            return probabilities;
            
        } catch (Exception e) {
            metrics.recordError();
            throw new InferenceException("Probability prediction failed: " + e.getMessage(), e);
        }
    }
    
    /**
     * Predict class probabilities for single sample.
     * @param modelId model identifier
     * @param features input features
     * @return class probabilities
     */
    public double[] predictProba(String modelId, double[] features) {
        double[][] batchFeatures = {features};
        double[][] probabilities = predictProba(modelId, batchFeatures);
        return probabilities[0];
    }
    
    /**
     * Get model information.
     * @param modelId model identifier
     * @return model information
     */
    public ModelInfo getModelInfo(String modelId) {
        LoadedModel loadedModel = getLoadedModel(modelId);
        ModelPersistence.ModelMetadata metadata = loadedModel.metadata;
        
        return new ModelInfo(modelId, metadata.modelClass, metadata.description,
                           metadata.savedAt.toString(), 0, loadedModel.filePath);
    }
    
    /**
     * Get inference metrics for a model.
     * @param modelId model identifier
     * @return inference metrics
     */
    public InferenceMetrics getMetrics(String modelId) {
        InferenceMetrics metrics = metricsMap.get(modelId);
        if (metrics == null) {
            throw new InferenceException("No metrics found for model: " + modelId);
        }
        return metrics;
    }
    
    /**
     * Get aggregated metrics for all models.
     * @return aggregated metrics
     */
    public Map<String, InferenceMetrics> getAllMetrics() {
        return new ConcurrentHashMap<>(metricsMap);
    }
    
    /**
     * Clear metrics for a model.
     * @param modelId model identifier
     */
    public void clearMetrics(String modelId) {
        InferenceMetrics metrics = metricsMap.get(modelId);
        if (metrics != null) {
            metrics.reset();
        }
    }
    
    /**
     * Clear all metrics.
     */
    public void clearAllMetrics() {
        metricsMap.values().forEach(InferenceMetrics::reset);
    }
    
    /**
     * Warm up a model by running dummy predictions.
     * @param modelId model identifier
     * @param sampleSize number of samples for warmup
     */
    public void warmUp(String modelId, int sampleSize) {
        LoadedModel loadedModel = getLoadedModel(modelId);
        
        try {
            // Get expected input size from metadata or model
            int inputSize = loadedModel.getExpectedInputSize();
            
            // Create dummy data
            double[][] dummyData = new double[sampleSize][inputSize];
            for (int i = 0; i < sampleSize; i++) {
                for (int j = 0; j < inputSize; j++) {
                    dummyData[i][j] = Math.random();
                }
            }
            
            // Run predictions to warm up JVM
            long startTime = System.currentTimeMillis();
            predict(modelId, dummyData);
            long warmupTime = System.currentTimeMillis() - startTime;
            
            logger.info("Model '{}' warmed up with {} samples in {}ms", 
                       modelId, sampleSize, warmupTime);
            
        } catch (Exception e) {
            logger.warn("Failed to warm up model '{}'", modelId, e);
        }
    }
    
    /**
     * Shutdown the inference engine and cleanup resources.
     */
    public void shutdown() {
        executorService.shutdown();
        modelCache.clear();
        metricsMap.clear();
        logger.info("InferenceEngine shutdown completed");
    }
    
    // Private helper methods
    
    private LoadedModel getLoadedModel(String modelId) {
        LoadedModel loadedModel = modelCache.get(modelId);
        if (loadedModel == null) {
            throw new InferenceException("Model not loaded: " + modelId);
        }
        return loadedModel;
    }
    
    private void validateInput(double[][] features, LoadedModel loadedModel) {
        if (features == null || features.length == 0) {
            throw new InferenceException("Input features cannot be null or empty");
        }
        
        if (config.validateInputSize) {
            int expectedSize = loadedModel.getExpectedInputSize();
            for (int i = 0; i < features.length; i++) {
                if (features[i].length != expectedSize) {
                    throw new InferenceException(
                        String.format("Expected %d features but got %d at sample %d", 
                                    expectedSize, features[i].length, i));
                }
            }
        }
        
        if (config.validateFiniteValues) {
            for (int i = 0; i < features.length; i++) {
                for (int j = 0; j < features[i].length; j++) {
                    if (!Double.isFinite(features[i][j])) {
                        throw new InferenceException(
                            String.format("Invalid value at sample %d, feature %d: %f", 
                                        i, j, features[i][j]));
                    }
                }
            }
        }
    }
    
    // Inner classes
    
    /**
     * Wrapper for a loaded model with metadata.
     */
    private static class LoadedModel {
        final BaseEstimator model;
        final ModelPersistence.ModelMetadata metadata;
        final String filePath;
        final LocalDateTime loadTime;
        
        LoadedModel(BaseEstimator model, ModelPersistence.ModelMetadata metadata, String filePath) {
            this.model = model;
            this.metadata = metadata;
            this.filePath = filePath;
            this.loadTime = LocalDateTime.now();
        }
        
        int getExpectedInputSize() {
            // Try to infer input size from model or metadata
            // This is a simplified implementation
            Map<String, Object> customMetadata = metadata.customMetadata;
            if (customMetadata != null && customMetadata.containsKey("test_features")) {
                return (Integer) customMetadata.get("test_features");
            }
            
            // Default fallback
            return -1; // Unknown size
        }
    }
    
    /**
     * Configuration for inference engine.
     */
    public static class InferenceConfig {
        public final int threadPoolSize;
        public final boolean validateInputSize;
        public final boolean validateFiniteValues;
        public final long maxCacheSize;
        
        public InferenceConfig() {
            this(Runtime.getRuntime().availableProcessors(), true, true, 100);
        }
        
        public InferenceConfig(int threadPoolSize, boolean validateInputSize, 
                             boolean validateFiniteValues, long maxCacheSize) {
            this.threadPoolSize = threadPoolSize;
            this.validateInputSize = validateInputSize;
            this.validateFiniteValues = validateFiniteValues;
            this.maxCacheSize = maxCacheSize;
        }
    }
    
    /**
     * Model information container.
     */
    public static class ModelInfo {
        public final String modelId;
        public final String modelClass;
        public final String description;
        public final String savedAt;
        public final long loadTimeMs;
        public final String filePath;
        
        ModelInfo(String modelId, String modelClass, String description, 
                 String savedAt, long loadTimeMs, String filePath) {
            this.modelId = modelId;
            this.modelClass = modelClass;
            this.description = description;
            this.savedAt = savedAt;
            this.loadTimeMs = loadTimeMs;
            this.filePath = filePath;
        }
        
        @Override
        public String toString() {
            return String.format("ModelInfo{id='%s', class='%s', description='%s'}", 
                               modelId, modelClass, description);
        }
    }
}
