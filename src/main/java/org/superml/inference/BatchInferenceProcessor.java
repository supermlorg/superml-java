package org.superml.inference;

import org.superml.core.BaseEstimator;
import org.superml.datasets.DataLoaders;
import org.superml.persistence.ModelPersistence;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.CompletableFuture;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * Batch inference processor for high-throughput model inference.
 * Optimized for processing large datasets with efficient memory usage.
 * 
 * Features:
 * - Streaming data processing
 * - Configurable batch sizes
 * - Memory-efficient processing
 * - Progress monitoring
 * - CSV input/output support
 * - Async processing
 * 
 * Usage:
 * <pre>
 * BatchInferenceProcessor processor = new BatchInferenceProcessor(inferenceEngine);
 * 
 * // Process CSV file
 * BatchResult result = processor.processCSV("input.csv", "output.csv", "my_model");
 * 
 * // Process with custom batch size
 * BatchConfig config = new BatchConfig().setBatchSize(1000);
 * BatchResult result = processor.processCSV("input.csv", "output.csv", "my_model", config);
 * </pre>
 */
public class BatchInferenceProcessor {
    private static final Logger logger = LoggerFactory.getLogger(BatchInferenceProcessor.class);
    
    private final InferenceEngine inferenceEngine;
    
    /**
     * Create batch processor with inference engine.
     * @param inferenceEngine inference engine to use
     */
    public BatchInferenceProcessor(InferenceEngine inferenceEngine) {
        if (inferenceEngine == null) {
            throw new IllegalArgumentException("InferenceEngine cannot be null");
        }
        this.inferenceEngine = inferenceEngine;
    }
    
    /**
     * Process CSV file with batch inference.
     * @param inputFile input CSV file path
     * @param outputFile output CSV file path
     * @param modelId model identifier
     * @return batch processing result
     */
    public BatchResult processCSV(String inputFile, String outputFile, String modelId) {
        return processCSV(inputFile, outputFile, modelId, new BatchConfig());
    }
    
    /**
     * Process CSV file with custom configuration.
     * @param inputFile input CSV file path
     * @param outputFile output CSV file path
     * @param modelId model identifier
     * @param config batch processing configuration
     * @return batch processing result
     */
    public BatchResult processCSV(String inputFile, String outputFile, String modelId, BatchConfig config) {
        if (!Files.exists(Paths.get(inputFile))) {
            throw new InferenceException("Input file not found: " + inputFile);
        }
        
        if (!inferenceEngine.isModelLoaded(modelId)) {
            throw new InferenceException("Model not loaded: " + modelId);
        }
        
        long startTime = System.currentTimeMillis();
        int totalSamples = 0;
        int batchCount = 0;
        List<String> errors = new ArrayList<>();
        
        try {
            logger.info("Starting batch processing: {} -> {} using model '{}'", 
                       inputFile, outputFile, modelId);
            
            // Load input data
            DataLoaders.FeatureData featureData = DataLoaders.loadFeaturesFromCsv(inputFile);
            double[][] features = featureData.X;
            totalSamples = features.length;
            
            logger.info("Loaded {} samples with {} features", totalSamples, featureData.nFeatures);
            
            // Process in batches
            List<double[]> allPredictions = new ArrayList<>();
            int batchSize = config.batchSize;
            
            for (int i = 0; i < features.length; i += batchSize) {
                int endIndex = Math.min(i + batchSize, features.length);
                double[][] batchFeatures = new double[endIndex - i][];
                
                // Copy batch data
                System.arraycopy(features, i, batchFeatures, 0, endIndex - i);
                
                try {
                    // Make predictions
                    double[] batchPredictions = inferenceEngine.predict(modelId, batchFeatures);
                    
                    // Store predictions
                    for (double prediction : batchPredictions) {
                        allPredictions.add(new double[]{prediction});
                    }
                    
                    batchCount++;
                    
                    if (config.showProgress && batchCount % config.progressInterval == 0) {
                        double progress = (double) (i + batchFeatures.length) / totalSamples * 100;
                        logger.info("Progress: {:.1f}% ({} batches processed)", progress, batchCount);
                    }
                    
                } catch (Exception e) {
                    String error = String.format("Batch %d failed: %s", batchCount + 1, e.getMessage());
                    errors.add(error);
                    logger.error(error, e);
                    
                    if (!config.continueOnError) {
                        throw new InferenceException("Batch processing failed: " + error, e);
                    }
                    
                    // Add null predictions for failed batch
                    for (int j = 0; j < batchFeatures.length; j++) {
                        allPredictions.add(new double[]{Double.NaN});
                    }
                }
            }
            
            // Save results to CSV
            if (config.saveResults) {
                saveResults(outputFile, featureData, allPredictions, config);
            }
            
            long processingTime = System.currentTimeMillis() - startTime;
            
            BatchResult result = new BatchResult(
                totalSamples, batchCount, processingTime, errors,
                inputFile, outputFile, modelId
            );
            
            logger.info("Batch processing completed: {}", result.getSummary());
            return result;
            
        } catch (Exception e) {
            logger.error("Batch processing failed", e);
            throw new InferenceException("Batch processing failed: " + e.getMessage(), e);
        }
    }
    
    /**
     * Process feature matrix with batch inference.
     * @param features input features
     * @param modelId model identifier
     * @return predictions
     */
    public double[] process(double[][] features, String modelId) {
        return process(features, modelId, new BatchConfig());
    }
    
    /**
     * Process feature matrix with custom configuration.
     * @param features input features
     * @param modelId model identifier
     * @param config batch configuration
     * @return predictions
     */
    public double[] process(double[][] features, String modelId, BatchConfig config) {
        if (features == null || features.length == 0) {
            return new double[0];
        }
        
        List<Double> allPredictions = new ArrayList<>();
        int batchSize = config.batchSize;
        
        for (int i = 0; i < features.length; i += batchSize) {
            int endIndex = Math.min(i + batchSize, features.length);
            double[][] batchFeatures = new double[endIndex - i][];
            
            System.arraycopy(features, i, batchFeatures, 0, endIndex - i);
            
            double[] batchPredictions = inferenceEngine.predict(modelId, batchFeatures);
            for (double prediction : batchPredictions) {
                allPredictions.add(prediction);
            }
        }
        
        return allPredictions.stream().mapToDouble(Double::doubleValue).toArray();
    }
    
    /**
     * Process CSV file asynchronously.
     * @param inputFile input CSV file
     * @param outputFile output CSV file
     * @param modelId model identifier
     * @return future containing batch result
     */
    public CompletableFuture<BatchResult> processCSVAsync(String inputFile, String outputFile, String modelId) {
        return CompletableFuture.supplyAsync(() -> processCSV(inputFile, outputFile, modelId));
    }
    
    /**
     * Process CSV file asynchronously with configuration.
     * @param inputFile input CSV file
     * @param outputFile output CSV file
     * @param modelId model identifier
     * @param config batch configuration
     * @return future containing batch result
     */
    public CompletableFuture<BatchResult> processCSVAsync(String inputFile, String outputFile, 
                                                         String modelId, BatchConfig config) {
        return CompletableFuture.supplyAsync(() -> processCSV(inputFile, outputFile, modelId, config));
    }
    
    // Private helper methods
    
    private void saveResults(String outputFile, DataLoaders.FeatureData featureData, 
                           List<double[]> predictions, BatchConfig config) throws IOException {
        // Create output data combining features and predictions
        List<String[]> outputLines = new ArrayList<>();
        
        // Create header
        String[] header = new String[featureData.featureNames.length + 1];
        System.arraycopy(featureData.featureNames, 0, header, 0, featureData.featureNames.length);
        header[header.length - 1] = config.predictionColumnName;
        outputLines.add(header);
        
        // Add data rows
        for (int i = 0; i < featureData.nSamples; i++) {
            String[] row = new String[header.length];
            
            // Copy features
            for (int j = 0; j < featureData.nFeatures; j++) {
                row[j] = String.valueOf(featureData.X[i][j]);
            }
            
            // Add prediction
            if (i < predictions.size()) {
                row[row.length - 1] = String.valueOf(predictions.get(i)[0]);
            } else {
                row[row.length - 1] = "NaN";
            }
            
            outputLines.add(row);
        }
        
        // Write to file
        try (java.io.PrintWriter writer = new java.io.PrintWriter(outputFile)) {
            for (String[] line : outputLines) {
                writer.println(String.join(",", line));
            }
        }
        
        logger.info("Results saved to: {}", outputFile);
    }
    
    /**
     * Configuration for batch processing.
     */
    public static class BatchConfig {
        public int batchSize = 1000;
        public boolean continueOnError = true;
        public boolean saveResults = true;
        public boolean showProgress = true;
        public int progressInterval = 10;
        public String predictionColumnName = "prediction";
        
        public BatchConfig setBatchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }
        
        public BatchConfig setContinueOnError(boolean continueOnError) {
            this.continueOnError = continueOnError;
            return this;
        }
        
        public BatchConfig setSaveResults(boolean saveResults) {
            this.saveResults = saveResults;
            return this;
        }
        
        public BatchConfig setShowProgress(boolean showProgress) {
            this.showProgress = showProgress;
            return this;
        }
        
        public BatchConfig setProgressInterval(int progressInterval) {
            this.progressInterval = progressInterval;
            return this;
        }
        
        public BatchConfig setPredictionColumnName(String predictionColumnName) {
            this.predictionColumnName = predictionColumnName;
            return this;
        }
    }
    
    /**
     * Result of batch processing operation.
     */
    public static class BatchResult {
        public final int totalSamples;
        public final int batchCount;
        public final long processingTimeMs;
        public final List<String> errors;
        public final String inputFile;
        public final String outputFile;
        public final String modelId;
        
        BatchResult(int totalSamples, int batchCount, long processingTimeMs, 
                   List<String> errors, String inputFile, String outputFile, String modelId) {
            this.totalSamples = totalSamples;
            this.batchCount = batchCount;
            this.processingTimeMs = processingTimeMs;
            this.errors = new ArrayList<>(errors);
            this.inputFile = inputFile;
            this.outputFile = outputFile;
            this.modelId = modelId;
        }
        
        public boolean hasErrors() {
            return !errors.isEmpty();
        }
        
        public double getSamplesPerSecond() {
            return totalSamples / (processingTimeMs / 1000.0);
        }
        
        public double getAverageTimePerSample() {
            return (double) processingTimeMs / totalSamples;
        }
        
        public String getSummary() {
            return String.format(
                "BatchResult: %d samples in %d batches, %.1f samples/s, %d errors, %.2fs total",
                totalSamples, batchCount, getSamplesPerSecond(), errors.size(), processingTimeMs / 1000.0
            );
        }
        
        @Override
        public String toString() {
            return getSummary();
        }
    }
}
