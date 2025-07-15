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

package org.superml.drift;

/**
 * Configuration class for drift detection parameters.
 * Centralizes all thresholds and settings for data and concept drift detection.
 */
public class DriftConfig {
    
    // Data drift thresholds
    private double ksignificanceLevel = 0.05;
    private double psiThreshold = 0.2;
    private double jsThreshold = 0.1;
    private double chiSquareSignificanceLevel = 0.05;
    private double momentsThreshold = 0.1;
    
    // Concept drift thresholds
    private double accuracyDropThreshold = 0.05;
    private double confidenceDropThreshold = 0.1;
    private double entropyChangeThreshold = 0.2;
    private double accuracyThreshold = 0.01; // For binary classification accuracy
    
    // Window sizes and limits
    private int slidingWindowSize = 1000;
    private int minSlidingWindowSize = 50;
    private int maxHistorySize = 10000;
    private int confidenceWindowSize = 500;
    private int distributionWindowSize = 500;
    
    // Minimum samples for drift detection
    private int minHistoryForConfidenceDrift = 100;
    private int minHistoryForDistributionDrift = 100;
    private int minSamplesForDDM = 30;
    private int minSamplesForEDDM = 30;
    private int minFeaturesForOverallDrift = 1;
    private int minTestsForDrift = 2;
    
    // ADWIN parameters
    private double adwinConfidence = 0.002; // 99.8% confidence level
    private int maxAdwinWindowSize = 2000;
    
    // PSI binning parameters
    private int psiBins = 10;
    private double psiMinBinSize = 0.01;
    
    // Statistical test parameters
    private boolean enableKsTest = true;
    private boolean enablePsiTest = true;
    private boolean enableJsTest = true;
    private boolean enableChiSquareTest = true;
    private boolean enableMomentsTest = true;
    
    // Alert and reporting settings
    private boolean enableAlerts = true;
    private boolean enableDetailedLogging = false;
    private String outputFormat = "JSON"; // JSON, CSV, or CONSOLE
    
    // Default constructor with sensible defaults
    public DriftConfig() {
        // All defaults are set above
    }
    
    // Builder pattern for easy configuration
    public static class Builder {
        private DriftConfig config = new DriftConfig();
        
        // Data drift threshold builders
        public Builder ksSignificanceLevel(double level) {
            config.ksignificanceLevel = level;
            return this;
        }
        
        public Builder psiThreshold(double threshold) {
            config.psiThreshold = threshold;
            return this;
        }
        
        public Builder jsThreshold(double threshold) {
            config.jsThreshold = threshold;
            return this;
        }
        
        public Builder chiSquareSignificanceLevel(double level) {
            config.chiSquareSignificanceLevel = level;
            return this;
        }
        
        public Builder momentsThreshold(double threshold) {
            config.momentsThreshold = threshold;
            return this;
        }
        
        // Concept drift threshold builders
        public Builder accuracyDropThreshold(double threshold) {
            config.accuracyDropThreshold = threshold;
            return this;
        }
        
        public Builder confidenceDropThreshold(double threshold) {
            config.confidenceDropThreshold = threshold;
            return this;
        }
        
        public Builder entropyChangeThreshold(double threshold) {
            config.entropyChangeThreshold = threshold;
            return this;
        }
        
        public Builder accuracyThreshold(double threshold) {
            config.accuracyThreshold = threshold;
            return this;
        }
        
        // Window size builders
        public Builder slidingWindowSize(int size) {
            config.slidingWindowSize = size;
            return this;
        }
        
        public Builder minSlidingWindowSize(int size) {
            config.minSlidingWindowSize = size;
            return this;
        }
        
        public Builder maxHistorySize(int size) {
            config.maxHistorySize = size;
            return this;
        }
        
        public Builder confidenceWindowSize(int size) {
            config.confidenceWindowSize = size;
            return this;
        }
        
        public Builder distributionWindowSize(int size) {
            config.distributionWindowSize = size;
            return this;
        }
        
        // Minimum samples builders
        public Builder minHistoryForConfidenceDrift(int samples) {
            config.minHistoryForConfidenceDrift = samples;
            return this;
        }
        
        public Builder minHistoryForDistributionDrift(int samples) {
            config.minHistoryForDistributionDrift = samples;
            return this;
        }
        
        public Builder minSamplesForDDM(int samples) {
            config.minSamplesForDDM = samples;
            return this;
        }
        
        public Builder minSamplesForEDDM(int samples) {
            config.minSamplesForEDDM = samples;
            return this;
        }
        
        // ADWIN builders
        public Builder adwinConfidence(double confidence) {
            config.adwinConfidence = confidence;
            return this;
        }
        
        public Builder maxAdwinWindowSize(int size) {
            config.maxAdwinWindowSize = size;
            return this;
        }
        
        // PSI builders
        public Builder psiBins(int bins) {
            config.psiBins = bins;
            return this;
        }
        
        public Builder psiMinBinSize(double minSize) {
            config.psiMinBinSize = minSize;
            return this;
        }
        
        // Test enablement builders
        public Builder enableKsTest(boolean enable) {
            config.enableKsTest = enable;
            return this;
        }
        
        public Builder enablePsiTest(boolean enable) {
            config.enablePsiTest = enable;
            return this;
        }
        
        public Builder enableJsTest(boolean enable) {
            config.enableJsTest = enable;
            return this;
        }
        
        public Builder enableChiSquareTest(boolean enable) {
            config.enableChiSquareTest = enable;
            return this;
        }
        
        public Builder enableMomentsTest(boolean enable) {
            config.enableMomentsTest = enable;
            return this;
        }
        
        // Alert and reporting builders
        public Builder enableAlerts(boolean enable) {
            config.enableAlerts = enable;
            return this;
        }
        
        public Builder enableDetailedLogging(boolean enable) {
            config.enableDetailedLogging = enable;
            return this;
        }
        
        public Builder outputFormat(String format) {
            config.outputFormat = format;
            return this;
        }
        
        public DriftConfig build() {
            validateConfig();
            return config;
        }
        
        private void validateConfig() {
            // Validate thresholds
            if (config.ksignificanceLevel <= 0 || config.ksignificanceLevel >= 1) {
                throw new IllegalArgumentException("KS significance level must be between 0 and 1");
            }
            if (config.psiThreshold < 0) {
                throw new IllegalArgumentException("PSI threshold must be non-negative");
            }
            if (config.jsThreshold < 0) {
                throw new IllegalArgumentException("JS threshold must be non-negative");
            }
            
            // Validate window sizes
            if (config.slidingWindowSize <= 0) {
                throw new IllegalArgumentException("Sliding window size must be positive");
            }
            if (config.minSlidingWindowSize <= 0) {
                throw new IllegalArgumentException("Minimum sliding window size must be positive");
            }
            if (config.minSlidingWindowSize > config.slidingWindowSize) {
                throw new IllegalArgumentException("Minimum sliding window size cannot exceed sliding window size");
            }
            
            // Validate ADWIN parameters
            if (config.adwinConfidence <= 0 || config.adwinConfidence >= 1) {
                throw new IllegalArgumentException("ADWIN confidence must be between 0 and 1");
            }
            
            // Validate PSI parameters
            if (config.psiBins <= 1) {
                throw new IllegalArgumentException("PSI bins must be greater than 1");
            }
            if (config.psiMinBinSize <= 0 || config.psiMinBinSize >= 1) {
                throw new IllegalArgumentException("PSI minimum bin size must be between 0 and 1");
            }
        }
    }
    
    // Preset configurations for common use cases
    public static DriftConfig sensitiveDetection() {
        return new Builder()
            .ksSignificanceLevel(0.1)
            .psiThreshold(0.1)
            .jsThreshold(0.05)
            .accuracyDropThreshold(0.02)
            .confidenceDropThreshold(0.05)
            .build();
    }
    
    public static DriftConfig conservativeDetection() {
        return new Builder()
            .ksSignificanceLevel(0.01)
            .psiThreshold(0.3)
            .jsThreshold(0.2)
            .accuracyDropThreshold(0.1)
            .confidenceDropThreshold(0.2)
            .build();
    }
    
    public static DriftConfig balancedDetection() {
        return new DriftConfig(); // Uses default values
    }
    
    public static DriftConfig fastDetection() {
        return new Builder()
            .slidingWindowSize(500)
            .minSlidingWindowSize(25)
            .confidenceWindowSize(250)
            .distributionWindowSize(250)
            .minSamplesForDDM(15)
            .minSamplesForEDDM(15)
            .build();
    }
    
    public static DriftConfig robustDetection() {
        return new Builder()
            .slidingWindowSize(2000)
            .minSlidingWindowSize(100)
            .confidenceWindowSize(1000)
            .distributionWindowSize(1000)
            .minSamplesForDDM(50)
            .minSamplesForEDDM(50)
            .build();
    }
    
    // Getters
    public double getKsSignificanceLevel() { return ksignificanceLevel; }
    public double getPsiThreshold() { return psiThreshold; }
    public double getJsThreshold() { return jsThreshold; }
    public double getChiSquareSignificanceLevel() { return chiSquareSignificanceLevel; }
    public double getMomentsThreshold() { return momentsThreshold; }
    
    public double getAccuracyDropThreshold() { return accuracyDropThreshold; }
    public double getConfidenceDropThreshold() { return confidenceDropThreshold; }
    public double getEntropyChangeThreshold() { return entropyChangeThreshold; }
    public double getAccuracyThreshold() { return accuracyThreshold; }
    
    public int getSlidingWindowSize() { return slidingWindowSize; }
    public int getMinSlidingWindowSize() { return minSlidingWindowSize; }
    public int getMaxHistorySize() { return maxHistorySize; }
    public int getConfidenceWindowSize() { return confidenceWindowSize; }
    public int getDistributionWindowSize() { return distributionWindowSize; }
    
    public int getMinHistoryForConfidenceDrift() { return minHistoryForConfidenceDrift; }
    public int getMinHistoryForDistributionDrift() { return minHistoryForDistributionDrift; }
    public int getMinSamplesForDDM() { return minSamplesForDDM; }
    public int getMinSamplesForEDDM() { return minSamplesForEDDM; }
    public int getMinFeaturesForOverallDrift() { return minFeaturesForOverallDrift; }
    public int getMinTestsForDrift() { return minTestsForDrift; }
    
    public double getAdwinConfidence() { return adwinConfidence; }
    public int getMaxAdwinWindowSize() { return maxAdwinWindowSize; }
    
    public int getPsiBins() { return psiBins; }
    public double getPsiMinBinSize() { return psiMinBinSize; }
    
    public boolean isKsTestEnabled() { return enableKsTest; }
    public boolean isPsiTestEnabled() { return enablePsiTest; }
    public boolean isJsTestEnabled() { return enableJsTest; }
    public boolean isChiSquareTestEnabled() { return enableChiSquareTest; }
    public boolean isMomentsTestEnabled() { return enableMomentsTest; }
    
    public boolean isAlertsEnabled() { return enableAlerts; }
    public boolean isDetailedLoggingEnabled() { return enableDetailedLogging; }
    public String getOutputFormat() { return outputFormat; }
    
    @Override
    public String toString() {
        return String.format("DriftConfig{" +
            "ksSignificanceLevel=%.3f, psiThreshold=%.3f, jsThreshold=%.3f, " +
            "accuracyDropThreshold=%.3f, slidingWindowSize=%d, " +
            "enabledTests=[KS=%b, PSI=%b, JS=%b, ChiSq=%b, Moments=%b]" +
            "}", ksignificanceLevel, psiThreshold, jsThreshold, 
            accuracyDropThreshold, slidingWindowSize,
            enableKsTest, enablePsiTest, enableJsTest, enableChiSquareTest, enableMomentsTest);
    }
}
