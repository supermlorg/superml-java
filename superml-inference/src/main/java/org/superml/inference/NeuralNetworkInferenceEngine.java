/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.superml.inference;

import org.superml.core.Estimator;
import org.superml.neural.MLPClassifier;
import org.superml.neural.CNNClassifier;
import org.superml.neural.RNNClassifier;
import org.superml.preprocessing.NeuralNetworkPreprocessor;

import java.util.*;
import java.util.concurrent.*;

/**
 * High-performance inference engine for neural networks.
 * Provides optimized batch processing and real-time inference capabilities.
 */
public class NeuralNetworkInferenceEngine {
    
    private final ExecutorService executorService;
    private final int batchSize;
    private final boolean enableParallelization;
    
    /**
     * Inference configuration
     */
    public static class InferenceConfig {
        public final int batchSize;
        public final boolean enablePreprocessing;
        public final boolean enableParallelization;
        public final int maxThreads;
        public final long timeoutMs;
        
        public InferenceConfig(int batchSize, boolean enablePreprocessing, 
                              boolean enableParallelization, int maxThreads, long timeoutMs) {
            this.batchSize = batchSize;
            this.enablePreprocessing = enablePreprocessing;
            this.enableParallelization = enableParallelization;
            this.maxThreads = maxThreads;
            this.timeoutMs = timeoutMs;
        }
        
        public static InferenceConfig defaults() {
            return new InferenceConfig(32, true, true, 
                Runtime.getRuntime().availableProcessors(), 5000);
        }
    }
    
    /**
     * Inference result with timing and metadata
     */
    public static class InferenceResult {
        public final double[] predictions;
        public final double[][] probabilities; // For classification
        public final long inferenceTimeMs;
        public final int batchSize;
        public final Map<String, Object> metadata;
        
        public InferenceResult(double[] predictions, double[][] probabilities,
                              long inferenceTimeMs, int batchSize, Map<String, Object> metadata) {
            this.predictions = predictions;
            this.probabilities = probabilities;
            this.inferenceTimeMs = inferenceTimeMs;
            this.batchSize = batchSize;
            this.metadata = metadata;
        }
    }
    
    public NeuralNetworkInferenceEngine(InferenceConfig config) {
        this.batchSize = config.batchSize;
        this.enableParallelization = config.enableParallelization;
        
        if (enableParallelization) {
            this.executorService = Executors.newFixedThreadPool(config.maxThreads);
        } else {
            this.executorService = null;
        }
    }
    
    /**
     * Perform batch inference on neural network models
     */
    public InferenceResult batchInference(Estimator model, double[][] X, InferenceConfig config) {
        long startTime = System.currentTimeMillis();
        
        try {
            // Preprocess if enabled
            double[][] processedX = X;
            if (config.enablePreprocessing && model instanceof MLPClassifier) {
                NeuralNetworkPreprocessor preprocessor = new NeuralNetworkPreprocessor(
                    NeuralNetworkPreprocessor.NetworkType.MLP).configureMLP();
                preprocessor.fit(X); // Fit with input data only
                processedX = preprocessor.transform(X);
            }
            
            // Perform inference
            double[] predictions;
            double[][] probabilities = null;
            
            if (config.enableParallelization && X.length > config.batchSize) {
                predictions = parallelInference(model, processedX, config);
            } else {
                predictions = sequentialInference(model, processedX);
            }
            
            // Get probabilities for classification models
            if (model instanceof MLPClassifier || model instanceof CNNClassifier) {
                probabilities = getProbabilities(model, processedX);
            }
            
            long inferenceTime = System.currentTimeMillis() - startTime;
            
            // Create metadata
            Map<String, Object> metadata = new HashMap<>();
            metadata.put("model_type", model.getClass().getSimpleName());
            metadata.put("input_shape", Arrays.asList(X.length, X[0].length));
            metadata.put("preprocessing_enabled", config.enablePreprocessing);
            metadata.put("parallelization_enabled", config.enableParallelization);
            
            return new InferenceResult(predictions, probabilities, inferenceTime, 
                                     X.length, metadata);
            
        } catch (Exception e) {
            throw new RuntimeException("Inference failed", e);
        }
    }
    
    /**
     * Sequential inference for smaller datasets
     */
    private double[] sequentialInference(Estimator model, double[][] X) {
        if (model instanceof MLPClassifier) {
            return ((MLPClassifier) model).predict(X);
        } else if (model instanceof CNNClassifier) {
            return ((CNNClassifier) model).predict(X);
        } else if (model instanceof RNNClassifier) {
            return ((RNNClassifier) model).predict(X);
        } else {
            throw new IllegalArgumentException("Unsupported model type for inference");
        }
    }
    
    /**
     * Parallel inference for large datasets
     */
    private double[] parallelInference(Estimator model, double[][] X, InferenceConfig config) 
            throws InterruptedException, ExecutionException {
        
        int numBatches = (int) Math.ceil((double) X.length / config.batchSize);
        List<Future<double[]>> futures = new ArrayList<>();
        
        // Submit batch inference tasks
        for (int i = 0; i < numBatches; i++) {
            final int start = i * config.batchSize;
            final int end = Math.min(start + config.batchSize, X.length);
            final double[][] batch = Arrays.copyOfRange(X, start, end);
            
            Future<double[]> future = executorService.submit(() -> {
                return sequentialInference(model, batch);
            });
            futures.add(future);
        }
        
        // Collect results
        List<Double> allPredictions = new ArrayList<>();
        for (Future<double[]> future : futures) {
            try {
                double[] batchPredictions = future.get(config.timeoutMs, TimeUnit.MILLISECONDS);
                for (double pred : batchPredictions) {
                    allPredictions.add(pred);
                }
            } catch (java.util.concurrent.TimeoutException e) {
                throw new RuntimeException("Inference timed out", e);
            }
        }
        
        return allPredictions.stream().mapToDouble(Double::doubleValue).toArray();
    }
    
    /**
     * Get prediction probabilities for classification models
     */
    private double[][] getProbabilities(Estimator model, double[][] X) {
        // This would require the neural network models to support probability prediction
        // For now, return null - would need to implement predictProba methods
        return null;
    }
    
    /**
     * Real-time single sample inference
     */
    public double predictSingle(Estimator model, double[] sample, InferenceConfig config) {
        double[][] X = new double[][]{sample};
        InferenceResult result = batchInference(model, X, config);
        return result.predictions[0];
    }
    
    /**
     * Streaming inference for continuous data
     */
    public static class StreamingInference {
        private final Estimator model;
        private final InferenceConfig config;
        private final Queue<double[]> inputBuffer;
        private final Queue<Double> outputBuffer;
        
        public StreamingInference(Estimator model, InferenceConfig config, int bufferSize) {
            this.model = model;
            this.config = config;
            this.inputBuffer = new LinkedList<>();
            this.outputBuffer = new LinkedList<>();
        }
        
        /**
         * Add sample to streaming buffer
         */
        public synchronized void addSample(double[] sample) {
            inputBuffer.offer(sample);
            
            // Process batch when buffer is full
            if (inputBuffer.size() >= config.batchSize) {
                processBatch();
            }
        }
        
        /**
         * Get next prediction from output buffer
         */
        public synchronized Double getNextPrediction() {
            return outputBuffer.poll();
        }
        
        /**
         * Process accumulated batch
         */
        private void processBatch() {
            List<double[]> batch = new ArrayList<>();
            while (!inputBuffer.isEmpty() && batch.size() < config.batchSize) {
                batch.add(inputBuffer.poll());
            }
            
            if (!batch.isEmpty()) {
                double[][] X = batch.toArray(new double[0][]);
                NeuralNetworkInferenceEngine engine = new NeuralNetworkInferenceEngine(config);
                InferenceResult result = engine.batchInference(model, X, config);
                
                // Add predictions to output buffer
                for (double prediction : result.predictions) {
                    outputBuffer.offer(prediction);
                }
            }
        }
        
        /**
         * Flush remaining samples in buffer
         */
        public synchronized void flush() {
            if (!inputBuffer.isEmpty()) {
                processBatch();
            }
        }
    }
    
    /**
     * Model performance profiler
     */
    public static class InferenceProfiler {
        
        /**
         * Profile model inference performance
         */
        public static Map<String, Object> profileModel(Estimator model, double[][] X, 
                                                      InferenceConfig config, int numRuns) {
            Map<String, Object> profile = new HashMap<>();
            List<Long> inferenceTimes = new ArrayList<>();
            
            NeuralNetworkInferenceEngine engine = new NeuralNetworkInferenceEngine(config);
            
            // Warm-up runs
            for (int i = 0; i < 5; i++) {
                engine.batchInference(model, X, config);
            }
            
            // Actual profiling runs
            for (int i = 0; i < numRuns; i++) {
                InferenceResult result = engine.batchInference(model, X, config);
                inferenceTimes.add(result.inferenceTimeMs);
            }
            
            // Calculate statistics
            double avgTime = inferenceTimes.stream().mapToLong(Long::longValue).average().orElse(0.0);
            double minTime = inferenceTimes.stream().mapToLong(Long::longValue).min().orElse(0);
            double maxTime = inferenceTimes.stream().mapToLong(Long::longValue).max().orElse(0);
            
            double throughput = (double) X.length / (avgTime / 1000.0); // samples per second
            
            profile.put("avg_inference_time_ms", avgTime);
            profile.put("min_inference_time_ms", minTime);
            profile.put("max_inference_time_ms", maxTime);
            profile.put("throughput_samples_per_sec", throughput);
            profile.put("num_samples", X.length);
            profile.put("num_runs", numRuns);
            profile.put("batch_size", config.batchSize);
            
            return profile;
        }
    }
    
    /**
     * Close inference engine and cleanup resources
     */
    public void close() {
        if (executorService != null) {
            executorService.shutdown();
            try {
                if (!executorService.awaitTermination(5, TimeUnit.SECONDS)) {
                    executorService.shutdownNow();
                }
            } catch (InterruptedException e) {
                executorService.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }
}
