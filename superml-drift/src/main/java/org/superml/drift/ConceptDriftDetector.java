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

import org.superml.core.Estimator;
import java.util.*;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * Concept drift detection algorithms for monitoring changes in model performance
 * and prediction behavior over time. Implements various drift detectors suitable
 * for streaming data scenarios.
 */
public class ConceptDriftDetector {
    
    private final DriftConfig config;
    private final Queue<PredictionRecord> predictionHistory;
    private final SlidingWindowAccuracy slidingWindow;
    private final DDMDriftDetector ddmDetector;
    private final EDDMDriftDetector eddmDetector;
    private final ADWINDriftDetector adwinDetector;
    
    // Performance tracking
    private double baselineAccuracy;
    private Date lastDriftDetection;
    private int totalPredictions;
    private int correctPredictions;
    
    public ConceptDriftDetector(DriftConfig config) {
        this.config = config;
        this.predictionHistory = new ConcurrentLinkedQueue<>();
        this.slidingWindow = new SlidingWindowAccuracy(config.getSlidingWindowSize());
        this.ddmDetector = new DDMDriftDetector(config);
        this.eddmDetector = new EDDMDriftDetector(config);
        this.adwinDetector = new ADWINDriftDetector(config);
        this.totalPredictions = 0;
        this.correctPredictions = 0;
    }
    
    /**
     * Set baseline performance from training/validation data.
     * @param accuracy Baseline accuracy to compare against
     */
    public void setBaselineAccuracy(double accuracy) {
        this.baselineAccuracy = accuracy;
        System.out.printf("📊 Baseline accuracy set: %.4f\n", accuracy);
    }
    
    /**
     * Record a new prediction and check for concept drift.
     * @param prediction Model prediction
     * @param actualLabel True label (if available)
     * @param confidence Prediction confidence score
     * @param timestamp When the prediction was made
     * @return Concept drift detection result
     */
    public ConceptDriftResult recordPrediction(double prediction, Double actualLabel, 
                                             double confidence, Date timestamp) {
        PredictionRecord record = new PredictionRecord(prediction, actualLabel, confidence, timestamp);
        predictionHistory.offer(record);
        totalPredictions++;
        
        // Maintain sliding window size
        while (predictionHistory.size() > config.getMaxHistorySize()) {
            predictionHistory.poll();
        }
        
        // Update accuracy if ground truth is available
        boolean isCorrect = false;
        if (actualLabel != null) {
            isCorrect = Math.abs(prediction - actualLabel) < config.getAccuracyThreshold();
            if (isCorrect) correctPredictions++;
            
            slidingWindow.addPrediction(isCorrect);
        }
        
        // Check for drift using multiple detectors
        ConceptDriftResult result = detectConceptDrift(record, isCorrect);
        
        if (result.isDriftDetected()) {
            lastDriftDetection = timestamp;
            System.out.printf("🚨 Concept drift detected at prediction %d! Method: %s\n", 
                totalPredictions, result.getDetectionMethod());
        }
        
        return result;
    }
    
    /**
     * Batch process multiple predictions for drift detection.
     * @param predictions Array of predictions
     * @param actualLabels Array of true labels (can contain nulls)
     * @param confidences Array of confidence scores
     * @return Summary of drift detection results
     */
    public BatchDriftResult processBatch(double[] predictions, Double[] actualLabels, 
                                       double[] confidences) {
        List<ConceptDriftResult> results = new ArrayList<>();
        Date batchTimestamp = new Date();
        
        for (int i = 0; i < predictions.length; i++) {
            Double actualLabel = (actualLabels != null && i < actualLabels.length) ? actualLabels[i] : null;
            double confidence = (confidences != null && i < confidences.length) ? confidences[i] : 0.5;
            
            ConceptDriftResult result = recordPrediction(predictions[i], actualLabel, confidence, batchTimestamp);
            results.add(result);
        }
        
        // Analyze batch results
        long driftDetections = results.stream().mapToLong(r -> r.isDriftDetected() ? 1 : 0).sum();
        double avgConfidence = Arrays.stream(confidences).average().orElse(0.5);
        
        return new BatchDriftResult(results, driftDetections, avgConfidence, 
            getCurrentAccuracy(), batchTimestamp);
    }
    
    /**
     * Detect concept drift using multiple algorithms.
     */
    private ConceptDriftResult detectConceptDrift(PredictionRecord record, boolean isCorrect) {
        Map<String, DriftTestResult> testResults = new HashMap<>();
        boolean overallDrift = false;
        String primaryMethod = "None";
        
        // 1. Sliding window accuracy check
        if (slidingWindow.getWindowSize() >= config.getMinSlidingWindowSize()) {
            double windowAccuracy = slidingWindow.getAccuracy();
            double accuracyDrop = baselineAccuracy - windowAccuracy;
            boolean accuracyDrift = accuracyDrop > config.getAccuracyDropThreshold();
            
            testResults.put("SlidingWindow", new DriftTestResult(accuracyDrop, accuracyDrift, windowAccuracy));
            
            if (accuracyDrift) {
                overallDrift = true;
                primaryMethod = "SlidingWindow";
            }
        }
        
        // 2. DDM (Drift Detection Method)
        if (record.actualLabel != null) {
            DDMDriftDetector.DDMResult ddmResult = ddmDetector.detectDrift(isCorrect);
            testResults.put("DDM", new DriftTestResult(ddmResult.errorRate, ddmResult.isDrift, ddmResult.errorRate));
            
            if (ddmResult.isDrift && !overallDrift) {
                overallDrift = true;
                primaryMethod = "DDM";
            }
        }
        
        // 3. EDDM (Early Drift Detection Method)
        if (record.actualLabel != null) {
            EDDMDriftDetector.EDDMResult eddmResult = eddmDetector.detectDrift(isCorrect);
            testResults.put("EDDM", new DriftTestResult(eddmResult.averageDistance, eddmResult.isDrift, eddmResult.averageDistance));
            
            if (eddmResult.isDrift && !overallDrift) {
                overallDrift = true;
                primaryMethod = "EDDM";
            }
        }
        
        // 4. ADWIN (Adaptive Windowing)
        ADWINDriftDetector.ADWINResult adwinResult = adwinDetector.detectDrift(record.confidence);
        testResults.put("ADWIN", new DriftTestResult(adwinResult.changeDetected ? 1.0 : 0.0, 
            adwinResult.changeDetected, adwinResult.windowMean));
        
        if (adwinResult.changeDetected && !overallDrift) {
            overallDrift = true;
            primaryMethod = "ADWIN";
        }
        
        // 5. Confidence-based drift detection
        ConfidenceDriftResult confResult = detectConfidenceDrift();
        testResults.put("Confidence", new DriftTestResult(confResult.driftScore, confResult.isDrift, confResult.avgConfidence));
        
        if (confResult.isDrift && !overallDrift) {
            overallDrift = true;
            primaryMethod = "Confidence";
        }
        
        // 6. Prediction distribution drift
        PredictionDistributionResult distResult = detectPredictionDistributionDrift();
        testResults.put("Distribution", new DriftTestResult(distResult.driftScore, distResult.isDrift, distResult.entropy));
        
        if (distResult.isDrift && !overallDrift) {
            overallDrift = true;
            primaryMethod = "Distribution";
        }
        
        return new ConceptDriftResult(overallDrift, primaryMethod, testResults, 
            record.timestamp, getCurrentAccuracy());
    }
    
    /**
     * Detect drift based on confidence score changes.
     */
    private ConfidenceDriftResult detectConfidenceDrift() {
        if (predictionHistory.size() < config.getMinHistoryForConfidenceDrift()) {
            return new ConfidenceDriftResult(false, 0.0, 0.5);
        }
        
        List<PredictionRecord> recentRecords = new ArrayList<>(predictionHistory);
        int windowSize = Math.min(config.getConfidenceWindowSize(), recentRecords.size());
        
        // Calculate confidence statistics for recent window vs historical
        double recentAvgConfidence = recentRecords.subList(recentRecords.size() - windowSize, recentRecords.size())
            .stream().mapToDouble(r -> r.confidence).average().orElse(0.5);
            
        double historicalAvgConfidence = recentRecords.subList(0, Math.max(1, recentRecords.size() - windowSize))
            .stream().mapToDouble(r -> r.confidence).average().orElse(0.5);
        
        double confidenceDrop = historicalAvgConfidence - recentAvgConfidence;
        boolean isDrift = confidenceDrop > config.getConfidenceDropThreshold();
        
        return new ConfidenceDriftResult(isDrift, confidenceDrop, recentAvgConfidence);
    }
    
    /**
     * Detect drift in prediction distribution (prediction histogram changes).
     */
    private PredictionDistributionResult detectPredictionDistributionDrift() {
        if (predictionHistory.size() < config.getMinHistoryForDistributionDrift()) {
            return new PredictionDistributionResult(false, 0.0, 0.0);
        }
        
        List<PredictionRecord> recentRecords = new ArrayList<>(predictionHistory);
        int windowSize = Math.min(config.getDistributionWindowSize(), recentRecords.size() / 2);
        
        // Split into historical and recent windows
        List<Double> historicalPredictions = recentRecords.subList(0, recentRecords.size() - windowSize)
            .stream().map(r -> r.prediction).collect(java.util.stream.Collectors.toList());
            
        List<Double> recentPredictions = recentRecords.subList(recentRecords.size() - windowSize, recentRecords.size())
            .stream().map(r -> r.prediction).collect(java.util.stream.Collectors.toList());
        
        // Calculate entropy for both distributions
        double historicalEntropy = calculatePredictionEntropy(historicalPredictions);
        double recentEntropy = calculatePredictionEntropy(recentPredictions);
        
        // Simple distribution shift detection (can be enhanced with KS test)
        double entropyChange = Math.abs(historicalEntropy - recentEntropy);
        boolean isDrift = entropyChange > config.getEntropyChangeThreshold();
        
        return new PredictionDistributionResult(isDrift, entropyChange, recentEntropy);
    }
    
    /**
     * Calculate entropy of prediction distribution.
     */
    private double calculatePredictionEntropy(List<Double> predictions) {
        if (predictions.isEmpty()) return 0.0;
        
        // Create histogram bins
        double min = predictions.stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
        double max = predictions.stream().mapToDouble(Double::doubleValue).max().orElse(1.0);
        int numBins = Math.min(10, predictions.size() / 5);
        numBins = Math.max(numBins, 2);
        
        double binWidth = (max - min) / numBins;
        if (binWidth == 0) return 0.0;
        
        int[] binCounts = new int[numBins];
        for (double pred : predictions) {
            int binIndex = Math.min((int) ((pred - min) / binWidth), numBins - 1);
            binCounts[binIndex]++;
        }
        
        // Calculate entropy
        double entropy = 0.0;
        for (int count : binCounts) {
            if (count > 0) {
                double probability = (double) count / predictions.size();
                entropy -= probability * Math.log(probability) / Math.log(2);
            }
        }
        
        return entropy;
    }
    
    /**
     * Get current accuracy rate.
     */
    public double getCurrentAccuracy() {
        return totalPredictions > 0 ? (double) correctPredictions / totalPredictions : 0.0;
    }
    
    /**
     * Get drift detection statistics.
     */
    public DriftStatistics getStatistics() {
        return new DriftStatistics(
            totalPredictions,
            correctPredictions,
            getCurrentAccuracy(),
            baselineAccuracy,
            slidingWindow.getAccuracy(),
            lastDriftDetection,
            predictionHistory.size()
        );
    }
    
    /**
     * Reset drift detection state (e.g., after model retraining).
     */
    public void reset() {
        predictionHistory.clear();
        slidingWindow.reset();
        ddmDetector.reset();
        eddmDetector.reset();
        adwinDetector.reset();
        totalPredictions = 0;
        correctPredictions = 0;
        lastDriftDetection = null;
        System.out.println("🔄 Concept drift detector reset");
    }
    
    // Helper classes for sliding window accuracy
    private static class SlidingWindowAccuracy {
        private final Queue<Boolean> window;
        private final int maxSize;
        private int correctCount;
        
        public SlidingWindowAccuracy(int maxSize) {
            this.maxSize = maxSize;
            this.window = new LinkedList<>();
            this.correctCount = 0;
        }
        
        public void addPrediction(boolean isCorrect) {
            if (window.size() >= maxSize) {
                Boolean removed = window.poll();
                if (removed != null && removed) {
                    correctCount--;
                }
            }
            
            window.offer(isCorrect);
            if (isCorrect) {
                correctCount++;
            }
        }
        
        public double getAccuracy() {
            return window.isEmpty() ? 0.0 : (double) correctCount / window.size();
        }
        
        public int getWindowSize() {
            return window.size();
        }
        
        public void reset() {
            window.clear();
            correctCount = 0;
        }
    }
    
    // Data classes for results
    
    public static class PredictionRecord {
        public final double prediction;
        public final Double actualLabel; // Can be null if ground truth not available
        public final double confidence;
        public final Date timestamp;
        
        public PredictionRecord(double prediction, Double actualLabel, double confidence, Date timestamp) {
            this.prediction = prediction;
            this.actualLabel = actualLabel;
            this.confidence = confidence;
            this.timestamp = timestamp;
        }
    }
    
    public static class ConceptDriftResult {
        private final boolean driftDetected;
        private final String detectionMethod;
        private final Map<String, DriftTestResult> testResults;
        private final Date timestamp;
        private final double currentAccuracy;
        
        public ConceptDriftResult(boolean driftDetected, String detectionMethod,
                                Map<String, DriftTestResult> testResults, Date timestamp, double currentAccuracy) {
            this.driftDetected = driftDetected;
            this.detectionMethod = detectionMethod;
            this.testResults = testResults;
            this.timestamp = timestamp;
            this.currentAccuracy = currentAccuracy;
        }
        
        // Getters
        public boolean isDriftDetected() { return driftDetected; }
        public String getDetectionMethod() { return detectionMethod; }
        public Map<String, DriftTestResult> getTestResults() { return testResults; }
        public Date getTimestamp() { return timestamp; }
        public double getCurrentAccuracy() { return currentAccuracy; }
    }
    
    public static class BatchDriftResult {
        private final List<ConceptDriftResult> individualResults;
        private final long driftDetections;
        private final double averageConfidence;
        private final double batchAccuracy;
        private final Date timestamp;
        
        public BatchDriftResult(List<ConceptDriftResult> individualResults, long driftDetections,
                               double averageConfidence, double batchAccuracy, Date timestamp) {
            this.individualResults = individualResults;
            this.driftDetections = driftDetections;
            this.averageConfidence = averageConfidence;
            this.batchAccuracy = batchAccuracy;
            this.timestamp = timestamp;
        }
        
        // Getters
        public List<ConceptDriftResult> getIndividualResults() { return individualResults; }
        public long getDriftDetections() { return driftDetections; }
        public double getAverageConfidence() { return averageConfidence; }
        public double getBatchAccuracy() { return batchAccuracy; }
        public Date getTimestamp() { return timestamp; }
    }
    
    public static class DriftTestResult {
        public final double score;
        public final boolean isDrift;
        public final double additionalInfo;
        
        public DriftTestResult(double score, boolean isDrift, double additionalInfo) {
            this.score = score;
            this.isDrift = isDrift;
            this.additionalInfo = additionalInfo;
        }
    }
    
    public static class ConfidenceDriftResult {
        public final boolean isDrift;
        public final double driftScore;
        public final double avgConfidence;
        
        public ConfidenceDriftResult(boolean isDrift, double driftScore, double avgConfidence) {
            this.isDrift = isDrift;
            this.driftScore = driftScore;
            this.avgConfidence = avgConfidence;
        }
    }
    
    public static class PredictionDistributionResult {
        public final boolean isDrift;
        public final double driftScore;
        public final double entropy;
        
        public PredictionDistributionResult(boolean isDrift, double driftScore, double entropy) {
            this.isDrift = isDrift;
            this.driftScore = driftScore;
            this.entropy = entropy;
        }
    }
    
    public static class DriftStatistics {
        public final int totalPredictions;
        public final int correctPredictions;
        public final double overallAccuracy;
        public final double baselineAccuracy;
        public final double slidingWindowAccuracy;
        public final Date lastDriftDetection;
        public final int historySize;
        
        public DriftStatistics(int totalPredictions, int correctPredictions, double overallAccuracy,
                              double baselineAccuracy, double slidingWindowAccuracy, 
                              Date lastDriftDetection, int historySize) {
            this.totalPredictions = totalPredictions;
            this.correctPredictions = correctPredictions;
            this.overallAccuracy = overallAccuracy;
            this.baselineAccuracy = baselineAccuracy;
            this.slidingWindowAccuracy = slidingWindowAccuracy;
            this.lastDriftDetection = lastDriftDetection;
            this.historySize = historySize;
        }
    }
}
