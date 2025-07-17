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

import java.util.ArrayList;
import java.util.List;

/**
 * ADWIN (Adaptive Windowing) implementation for detecting concept drift
 * using adaptive sliding windows with statistical change detection.
 * 
 * Reference: Bifet, A. &amp; Gavaldà, R. "Learning from Time-Changing Data with Adaptive Windowing" (2007)
 */
public class ADWINDriftDetector {
    
    private final DriftConfig config;
    private final List<Double> window;
    private double windowSum;
    private int windowSize;
    private double confidence;
    
    // ADWIN parameters
    private static final double DEFAULT_CONFIDENCE = 0.002; // 99.8% confidence level
    private static final int MIN_WINDOW_SIZE = 5;
    private static final int MAX_BUCKETS = 5;
    
    public ADWINDriftDetector(DriftConfig config) {
        this.config = config;
        this.window = new ArrayList<>();
        this.confidence = config.getAdwinConfidence() > 0 ? config.getAdwinConfidence() : DEFAULT_CONFIDENCE;
        reset();
    }
    
    /**
     * Process a new data point and detect change.
     * @param value New data point (e.g., accuracy, error rate, or confidence score)
     * @return ADWIN change detection result
     */
    public ADWINResult detectDrift(double value) {
        // Add new element to window
        window.add(value);
        windowSum += value;
        windowSize++;
        
        // Limit window size for computational efficiency
        while (window.size() > config.getMaxAdwinWindowSize()) {
            double removed = window.remove(0);
            windowSum -= removed;
            windowSize--;
        }
        
        // Need minimum window size for meaningful detection
        if (windowSize < MIN_WINDOW_SIZE) {
            return new ADWINResult(false, windowSum / windowSize, windowSize);
        }
        
        // Check for change using ADWIN algorithm
        boolean changeDetected = detectChange();
        
        double windowMean = windowSum / windowSize;
        
        if (changeDetected) {
            // Keep only recent data after change point
            int changePoint = findOptimalCutPoint();
            if (changePoint > 0 && changePoint < window.size()) {
                // Remove elements before change point
                for (int i = 0; i < changePoint; i++) {
                    double removed = window.remove(0);
                    windowSum -= removed;
                    windowSize--;
                }
            }
        }
        
        return new ADWINResult(changeDetected, windowMean, windowSize);
    }
    
    /**
     * Detect change using ADWIN's statistical test.
     */
    private boolean detectChange() {
        if (windowSize < 2 * MIN_WINDOW_SIZE) {
            return false;
        }
        
        // Try different cut points to find significant change
        for (int i = MIN_WINDOW_SIZE; i <= windowSize - MIN_WINDOW_SIZE; i++) {
            if (hasSignificantChange(i)) {
                return true;
            }
        }
        
        return false;
    }
    
    /**
     * Test if there's a significant change at the given cut point.
     */
    private boolean hasSignificantChange(int cutPoint) {
        // Calculate means for both sub-windows
        double sum1 = 0.0, sum2 = 0.0;
        int n1 = cutPoint;
        int n2 = windowSize - cutPoint;
        
        for (int i = 0; i < cutPoint; i++) {
            sum1 += window.get(i);
        }
        for (int i = cutPoint; i < windowSize; i++) {
            sum2 += window.get(i);
        }
        
        double mean1 = sum1 / n1;
        double mean2 = sum2 / n2;
        
        // Calculate variances
        double var1 = 0.0, var2 = 0.0;
        for (int i = 0; i < cutPoint; i++) {
            var1 += Math.pow(window.get(i) - mean1, 2);
        }
        for (int i = cutPoint; i < windowSize; i++) {
            var2 += Math.pow(window.get(i) - mean2, 2);
        }
        
        var1 /= (n1 - 1);
        var2 /= (n2 - 1);
        
        // Hoeffding bound for detecting change
        double harmonicMean = 2.0 / (1.0 / n1 + 1.0 / n2);
        double epsilon = Math.sqrt((1.0 / (2.0 * harmonicMean)) * Math.log(4.0 / confidence));
        
        // Check if difference in means exceeds threshold
        return Math.abs(mean1 - mean2) > epsilon;
    }
    
    /**
     * Find the optimal cut point for change detection.
     */
    private int findOptimalCutPoint() {
        double maxDifference = 0.0;
        int bestCutPoint = 0;
        
        for (int i = MIN_WINDOW_SIZE; i <= windowSize - MIN_WINDOW_SIZE; i++) {
            double sum1 = 0.0, sum2 = 0.0;
            
            for (int j = 0; j < i; j++) {
                sum1 += window.get(j);
            }
            for (int j = i; j < windowSize; j++) {
                sum2 += window.get(j);
            }
            
            double mean1 = sum1 / i;
            double mean2 = sum2 / (windowSize - i);
            double difference = Math.abs(mean1 - mean2);
            
            if (difference > maxDifference) {
                maxDifference = difference;
                bestCutPoint = i;
            }
        }
        
        return bestCutPoint;
    }
    
    /**
     * Get current window statistics.
     */
    public WindowStatistics getWindowStatistics() {
        if (windowSize == 0) {
            return new WindowStatistics(0.0, 0.0, 0, 0.0, 0.0);
        }
        
        double mean = windowSum / windowSize;
        
        // Calculate variance and standard deviation
        double variance = 0.0;
        for (double value : window) {
            variance += Math.pow(value - mean, 2);
        }
        variance /= windowSize;
        double stdDev = Math.sqrt(variance);
        
        // Calculate min and max
        double min = window.stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
        double max = window.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
        
        return new WindowStatistics(mean, stdDev, windowSize, min, max);
    }
    
    /**
     * Reset ADWIN detector state.
     */
    public void reset() {
        window.clear();
        windowSum = 0.0;
        windowSize = 0;
    }
    
    public static class ADWINResult {
        public final boolean changeDetected;
        public final double windowMean;
        public final int windowSize;
        
        public ADWINResult(boolean changeDetected, double windowMean, int windowSize) {
            this.changeDetected = changeDetected;
            this.windowMean = windowMean;
            this.windowSize = windowSize;
        }
    }
    
    public static class WindowStatistics {
        public final double mean;
        public final double standardDeviation;
        public final int size;
        public final double min;
        public final double max;
        
        public WindowStatistics(double mean, double standardDeviation, int size, double min, double max) {
            this.mean = mean;
            this.standardDeviation = standardDeviation;
            this.size = size;
            this.min = min;
            this.max = max;
        }
    }
}
