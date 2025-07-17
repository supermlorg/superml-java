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

package org.superml.preprocessing;

import org.superml.core.BaseEstimator;
import org.superml.core.UnsupervisedLearner;
import java.util.Arrays;

/**
 * Specialized preprocessing for neural networks with different data types.
 * Provides optimized preprocessing for MLP, CNN, and RNN models.
 */
public class NeuralNetworkPreprocessor extends BaseEstimator implements UnsupervisedLearner {
    
    public enum NetworkType {
        MLP,    // Multi-Layer Perceptron (tabular data)
        CNN,    // Convolutional Neural Network (image data)
        RNN     // Recurrent Neural Network (sequence data)
    }
    
    private NetworkType networkType;
    private StandardScaler scaler;
    private boolean fitted = false;
    
    // CNN-specific parameters
    private int imageHeight = -1;
    private int imageWidth = -1;
    private int channels = 1;
    
    // RNN-specific parameters
    private int sequenceLength = -1;
    private int featuresPerTimestep = -1;
    private boolean normalizePerSequence = false;
    
    public NeuralNetworkPreprocessor(NetworkType networkType) {
        this.networkType = networkType;
        this.scaler = new StandardScaler();
    }
    
    // ==================== MLP Preprocessing ====================
    
    /**
     * Configure for MLP (tabular data) preprocessing
     */
    public NeuralNetworkPreprocessor configureMLP() {
        this.networkType = NetworkType.MLP;
        return this;
    }
    
    /**
     * Preprocess tabular data for MLP
     */
    public double[][] preprocessMLP(double[][] X) {
        if (networkType != NetworkType.MLP) {
            throw new IllegalStateException("Preprocessor not configured for MLP");
        }
        
        // 1. Standard scaling (essential for neural networks)
        if (!fitted) {
            scaler.fit(X);
            fitted = true;
        }
        double[][] scaled = scaler.transform(X);
        
        // 2. Clip outliers (beyond 3 standard deviations)
        return clipOutliers(scaled, 3.0);
    }
    
    // ==================== CNN Preprocessing ====================
    
    /**
     * Configure for CNN (image data) preprocessing
     */
    public NeuralNetworkPreprocessor configureCNN(int height, int width, int channels) {
        this.networkType = NetworkType.CNN;
        this.imageHeight = height;
        this.imageWidth = width;
        this.channels = channels;
        return this;
    }
    
    /**
     * Preprocess image data for CNN
     */
    public double[][] preprocessCNN(double[][] X) {
        if (networkType != NetworkType.CNN) {
            throw new IllegalStateException("Preprocessor not configured for CNN");
        }
        
        int nSamples = X.length;
        int totalFeatures = imageHeight * imageWidth * channels;
        
        if (X[0].length != totalFeatures) {
            throw new IllegalArgumentException(
                String.format("Expected %d features for %dx%dx%d images, got %d", 
                    totalFeatures, imageHeight, imageWidth, channels, X[0].length));
        }
        
        double[][] processed = new double[nSamples][totalFeatures];
        
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < totalFeatures; j++) {
                // Normalize pixels to [-1, 1] range (common for CNNs)
                processed[i][j] = 2.0 * X[i][j] - 1.0;
                
                // Clip to ensure bounds
                processed[i][j] = Math.max(-1.0, Math.min(1.0, processed[i][j]));
            }
        }
        
        fitted = true;
        return processed;
    }
    
    // ==================== RNN Preprocessing ====================
    
    /**
     * Configure for RNN (sequence data) preprocessing
     */
    public NeuralNetworkPreprocessor configureRNN(int sequenceLength, int featuresPerTimestep, 
                                                  boolean normalizePerSequence) {
        this.networkType = NetworkType.RNN;
        this.sequenceLength = sequenceLength;
        this.featuresPerTimestep = featuresPerTimestep;
        this.normalizePerSequence = normalizePerSequence;
        return this;
    }
    
    /**
     * Preprocess sequence data for RNN
     */
    public double[][] preprocessRNN(double[][] X) {
        if (networkType != NetworkType.RNN) {
            throw new IllegalStateException("Preprocessor not configured for RNN");
        }
        
        int nSamples = X.length;
        int totalFeatures = sequenceLength * featuresPerTimestep;
        
        if (X[0].length != totalFeatures) {
            throw new IllegalArgumentException(
                String.format("Expected %d features for sequences of length %d with %d features per timestep, got %d", 
                    totalFeatures, sequenceLength, featuresPerTimestep, X[0].length));
        }
        
        double[][] processed = new double[nSamples][totalFeatures];
        
        if (normalizePerSequence) {
            // Normalize each sequence independently
            for (int i = 0; i < nSamples; i++) {
                processed[i] = normalizeSequence(X[i]);
            }
        } else {
            // Global normalization across all sequences
            if (!fitted) {
                scaler.fit(X);
                fitted = true;
            }
            processed = scaler.transform(X);
            
            // Apply temporal smoothing
            for (int i = 0; i < nSamples; i++) {
                processed[i] = temporalSmoothing(processed[i]);
            }
        }
        
        fitted = true;
        return processed;
    }
    
    // ==================== UnsupervisedLearner Interface ====================
    
    @Override
    public NeuralNetworkPreprocessor fit(double[][] X) {
        switch (networkType) {
            case MLP:
                preprocessMLP(X);
                break;
            case CNN:
                preprocessCNN(X);
                break;
            case RNN:
                preprocessRNN(X);
                break;
            default:
                throw new IllegalStateException("Network type not configured");
        }
        return this;
    }
    
    @Override
    public double[][] transform(double[][] X) {
        switch (networkType) {
            case MLP:
                return preprocessMLP(X);
            case CNN:
                return preprocessCNN(X);
            case RNN:
                return preprocessRNN(X);
            default:
                throw new IllegalStateException("Network type not configured");
        }
    }
    
    // ==================== Utility Methods ====================
    
    /**
     * Clip outliers beyond specified number of standard deviations
     */
    private double[][] clipOutliers(double[][] X, double stdThreshold) {
        int nSamples = X.length;
        int nFeatures = X[0].length;
        double[][] clipped = new double[nSamples][nFeatures];
        
        for (int i = 0; i < nSamples; i++) {
            System.arraycopy(X[i], 0, clipped[i], 0, nFeatures);
            for (int j = 0; j < nFeatures; j++) {
                if (Math.abs(clipped[i][j]) > stdThreshold) {
                    clipped[i][j] = Math.signum(clipped[i][j]) * stdThreshold;
                }
            }
        }
        
        return clipped;
    }
    
    /**
     * Normalize a single sequence to zero mean and unit variance
     */
    private double[] normalizeSequence(double[] sequence) {
        double[] normalized = new double[sequence.length];
        
        // Calculate mean
        double mean = Arrays.stream(sequence).average().orElse(0.0);
        
        // Calculate standard deviation
        double variance = Arrays.stream(sequence)
            .map(x -> Math.pow(x - mean, 2))
            .average().orElse(1.0);
        double std = Math.sqrt(variance);
        
        // Avoid division by zero
        if (std == 0.0) std = 1.0;
        
        // Normalize
        for (int i = 0; i < sequence.length; i++) {
            normalized[i] = (sequence[i] - mean) / std;
        }
        
        return normalized;
    }
    
    /**
     * Apply temporal smoothing to reduce noise in sequences
     */
    private double[] temporalSmoothing(double[] sequence) {
        double[] smoothed = new double[sequence.length];
        double alpha = 0.3; // Smoothing factor
        
        // Reshape to timesteps x features
        int timesteps = sequenceLength;
        int features = featuresPerTimestep;
        
        for (int f = 0; f < features; f++) {
            smoothed[f] = sequence[f]; // First timestep unchanged
            
            for (int t = 1; t < timesteps; t++) {
                int idx = t * features + f;
                int prevIdx = (t - 1) * features + f;
                smoothed[idx] = alpha * sequence[idx] + (1 - alpha) * smoothed[prevIdx];
            }
        }
        
        return smoothed;
    }
    
    // ==================== Getters and Setters ====================
    
    public NetworkType getNetworkType() {
        return networkType;
    }
    
    public boolean isFitted() {
        return fitted;
    }
    
    public StandardScaler getScaler() {
        return scaler;
    }
    
    // CNN getters
    public int getImageHeight() { return imageHeight; }
    public int getImageWidth() { return imageWidth; }
    public int getChannels() { return channels; }
    
    // RNN getters
    public int getSequenceLength() { return sequenceLength; }
    public int getFeaturesPerTimestep() { return featuresPerTimestep; }
    public boolean isNormalizePerSequence() { return normalizePerSequence; }
    
    /**
     * Get preprocessing summary for the configured network type
     */
    public String getPreprocessingSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("Neural Network Preprocessor Configuration:\n");
        sb.append("Network Type: ").append(networkType).append("\n");
        
        switch (networkType) {
            case MLP:
                sb.append("- Standard scaling (mean=0, std=1)\n");
                sb.append("- Outlier clipping (±3σ)\n");
                break;
            case CNN:
                sb.append("- Image dimensions: ").append(imageHeight)
                  .append("x").append(imageWidth).append("x").append(channels).append("\n");
                sb.append("- Pixel normalization to [-1, 1]\n");
                sb.append("- Boundary clipping\n");
                break;
            case RNN:
                sb.append("- Sequence length: ").append(sequenceLength).append("\n");
                sb.append("- Features per timestep: ").append(featuresPerTimestep).append("\n");
                sb.append("- Per-sequence normalization: ").append(normalizePerSequence).append("\n");
                if (!normalizePerSequence) {
                    sb.append("- Global scaling + temporal smoothing\n");
                }
                break;
        }
        
        sb.append("Fitted: ").append(fitted);
        return sb.toString();
    }
}
