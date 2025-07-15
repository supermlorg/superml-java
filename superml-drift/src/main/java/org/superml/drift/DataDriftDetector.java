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

import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest;
import org.apache.commons.math3.stat.inference.ChiSquareTest;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import java.util.*;
import java.util.stream.IntStream;

/**
 * Data drift detection algorithms for monitoring distribution changes in features.
 * Implements various statistical tests to detect when input data distribution
 * differs significantly from the training distribution.
 */
public class DataDriftDetector {
    
    private final DriftConfig config;
    private final Map<String, double[]> referenceDistributions;
    private final Map<String, String> featureTypes; // "continuous" or "categorical"
    
    public DataDriftDetector(DriftConfig config) {
        this.config = config;
        this.referenceDistributions = new HashMap<>();
        this.featureTypes = new HashMap<>();
    }
    
    /**
     * Set reference (training) data for drift detection.
     * @param features Feature matrix from training data
     * @param featureNames Names of the features
     * @param categoricalFeatures Indices of categorical features
     */
    public void setReferenceData(double[][] features, String[] featureNames, Set<Integer> categoricalFeatures) {
        System.out.println("📊 Setting reference data for drift detection...");
        
        for (int i = 0; i < featureNames.length; i++) {
            String featureName = featureNames[i];
            double[] featureValues = extractFeatureColumn(features, i);
            
            referenceDistributions.put(featureName, featureValues);
            featureTypes.put(featureName, categoricalFeatures.contains(i) ? "categorical" : "continuous");
        }
        
        System.out.printf("-> Reference data set: %d features, %d samples\n", 
            featureNames.length, features.length);
    }
    
    /**
     * Detect drift in current data compared to reference data.
     * @param currentFeatures Current feature matrix
     * @param featureNames Feature names
     * @return Drift detection results
     */
    /**
     * Detect drift between reference and current data for multiple features.
     * @param currentFeatures Current data matrix (samples x features)
     * @param featureNames Names of the features
     * @return Comprehensive drift detection result
     */
    public DataDriftResult detectDrift(double[][] currentFeatures, String[] featureNames) {
        System.out.println("🔍 Detecting data drift...");
        
        List<FeatureDriftResult> featureResults = new ArrayList<>();
        int significantDrifts = 0;
        
        for (int i = 0; i < featureNames.length; i++) {
            String featureName = featureNames[i];
            double[] currentValues = extractFeatureColumn(currentFeatures, i);
            double[] referenceValues = referenceDistributions.get(featureName);
            
            if (referenceValues == null) {
                System.err.println("⚠️  No reference data for feature: " + featureName);
                continue;
            }
            
            FeatureDriftResult result = detectFeatureDrift(
                featureName, referenceValues, currentValues);
            featureResults.add(result);
            
            if (result.isDriftDetected()) {
                significantDrifts++;
                System.out.printf("🚨 Drift detected in %s: score=%.4f (p=%.4f)\n",
                    featureName, result.getDriftScore(), result.getPValue());
            }
        }
        
        double overallDriftScore = calculateOverallDriftScore(featureResults);
        boolean overallDrift = significantDrifts >= config.getMinFeaturesForOverallDrift();
        
        System.out.printf("📈 Drift analysis complete: %d/%d features drifted, overall score: %.4f\n",
            significantDrifts, featureResults.size(), overallDriftScore);
        
        return new DataDriftResult(featureResults, overallDriftScore, overallDrift, 
            new Date(), significantDrifts);
    }
    
    /**
     * Detect drift between reference and current data for a single feature.
     * Convenience method for single feature comparison.
     * @param referenceData Reference data array
     * @param currentData Current data array  
     * @param featureName Name of the feature
     * @return Drift detection result
     */
    public DataDriftResult detectDrift(double[] referenceData, double[] currentData, String featureName) {
        System.out.printf("🔍 Detecting data drift for feature '%s'...\n", featureName);
        
        // Perform drift detection directly for single feature
        FeatureDriftResult featureResult = detectContinuousDrift(featureName, referenceData, currentData);
        
        // Calculate overall drift score and detection
        double driftScore = featureResult.getDriftScore();
        boolean isDrift = featureResult.isDriftDetected();
        
        if (isDrift) {
            System.out.printf("🚨 Drift detected in %s: score=%.4f\n", featureName, driftScore);
        } else {
            System.out.printf("-> No drift detected in %s: score=%.4f\n", featureName, driftScore);
        }
        
        // Create feature result list
        List<FeatureDriftResult> featureResults = new ArrayList<>();
        featureResults.add(featureResult);
        
        return new DataDriftResult(featureResults, driftScore, isDrift, new Date(), isDrift ? 1 : 0);
    }
    
    /**
     * Calculate drift score for a single feature's test results.
     */
    private double calculateFeatureDriftScore(Map<String, TestResult> testResults) {
        return testResults.values().stream()
                .mapToDouble(result -> result.score)
                .average()
                .orElse(0.0);
    }
    
    /**
     * Detect drift for a single feature using appropriate statistical test.
     */
    private FeatureDriftResult detectFeatureDrift(String featureName, double[] reference, double[] current) {
        String featureType = featureTypes.get(featureName);
        
        if ("categorical".equals(featureType)) {
            return detectCategoricalDrift(featureName, reference, current);
        } else {
            return detectContinuousDrift(featureName, reference, current);
        }
    }
    
    /**
     * Detect drift in continuous features using multiple tests.
     */
    private FeatureDriftResult detectContinuousDrift(String featureName, double[] reference, double[] current) {
        Map<String, TestResult> testResults = new HashMap<>();
        
        // 1. Kolmogorov-Smirnov Test
        KolmogorovSmirnovTest ksTest = new KolmogorovSmirnovTest();
        double ksPValue = ksTest.kolmogorovSmirnovTest(reference, current);
        double ksStatistic = ksTest.kolmogorovSmirnovStatistic(reference, current);
        testResults.put("KS", new TestResult(ksStatistic, ksPValue, ksPValue < config.getKsSignificanceLevel()));
        
        // 2. Population Stability Index (PSI)
        double psi = calculatePSI(reference, current);
        boolean psiDrift = psi > config.getPsiThreshold();
        testResults.put("PSI", new TestResult(psi, 0.0, psiDrift)); // PSI doesn't have p-value
        
        // 3. Jensen-Shannon Divergence
        double jsDivergence = calculateJensenShannonDivergence(reference, current);
        boolean jsDrift = jsDivergence > config.getJsThreshold();
        testResults.put("JS", new TestResult(jsDivergence, 0.0, jsDrift));
        
        // 4. Statistical moments comparison
        double[] momentsDrift = calculateMomentsDrift(reference, current);
        boolean momentsDriftDetected = Arrays.stream(momentsDrift).anyMatch(d -> Math.abs(d) > config.getMomentsThreshold());
        testResults.put("Moments", new TestResult(Arrays.stream(momentsDrift).max().orElse(0.0), 0.0, momentsDriftDetected));
        
        // Combine test results
        boolean overallDrift = testResults.values().stream().mapToInt(r -> r.isDriftDetected ? 1 : 0).sum() 
                              >= config.getMinTestsForDrift();
        double combinedScore = combineTestScores(testResults);
        double combinedPValue = Math.min(ksPValue, 1.0); // Use KS p-value as primary
        
        return new FeatureDriftResult(featureName, "continuous", combinedScore, combinedPValue, 
            overallDrift, testResults);
    }
    
    /**
     * Detect drift in categorical features using Chi-Square test.
     */
    private FeatureDriftResult detectCategoricalDrift(String featureName, double[] reference, double[] current) {
        Map<String, TestResult> testResults = new HashMap<>();
        
        // Get unique categories and their frequencies
        Map<Double, Integer> refCounts = getValueCounts(reference);
        Map<Double, Integer> currCounts = getValueCounts(current);
        
        // Align categories between reference and current
        Set<Double> allCategories = new HashSet<>(refCounts.keySet());
        allCategories.addAll(currCounts.keySet());
        
        if (allCategories.size() < 2) {
            // Can't perform chi-square with less than 2 categories
            return new FeatureDriftResult(featureName, "categorical", 0.0, 1.0, false, testResults);
        }
        
        // Prepare contingency table
        long[] refFreqs = new long[allCategories.size()];
        long[] currFreqs = new long[allCategories.size()];
        
        int idx = 0;
        for (Double category : allCategories) {
            refFreqs[idx] = refCounts.getOrDefault(category, 0);
            currFreqs[idx] = currCounts.getOrDefault(category, 0);
            idx++;
        }
        
        // Chi-Square test
        ChiSquareTest chiSquareTest = new ChiSquareTest();
        try {
            // Convert long arrays to double arrays for Commons Math
            double[] refFreqsDouble = new double[refFreqs.length];
            double[] currFreqsDouble = new double[currFreqs.length];
            for (int i = 0; i < refFreqs.length; i++) {
                refFreqsDouble[i] = (double) refFreqs[i];
                currFreqsDouble[i] = (double) currFreqs[i];
            }
            
            double chiSquarePValue = chiSquareTest.chiSquareTest(refFreqsDouble, currFreqs);
            double chiSquareStatistic = chiSquareTest.chiSquare(refFreqsDouble, currFreqs);
            boolean chiSquareDrift = chiSquarePValue < config.getChiSquareSignificanceLevel();
            
            testResults.put("ChiSquare", new TestResult(chiSquareStatistic, chiSquarePValue, chiSquareDrift));
            
            // PSI for categorical data
            double psi = calculateCategoricalPSI(refCounts, currCounts, reference.length, current.length);
            boolean psiDrift = psi > config.getPsiThreshold();
            testResults.put("PSI", new TestResult(psi, 0.0, psiDrift));
            
            boolean overallDrift = chiSquareDrift || psiDrift;
            double combinedScore = Math.max(chiSquareStatistic / 10.0, psi); // Normalize chi-square
            
            return new FeatureDriftResult(featureName, "categorical", combinedScore, chiSquarePValue, 
                overallDrift, testResults);
                
        } catch (Exception e) {
            System.err.println("Error in categorical drift detection for " + featureName + ": " + e.getMessage());
            return new FeatureDriftResult(featureName, "categorical", 0.0, 1.0, false, testResults);
        }
    }
    
    /**
     * Calculate Population Stability Index for continuous features.
     */
    private double calculatePSI(double[] reference, double[] current) {
        // Create bins based on reference data quantiles
        DescriptiveStatistics refStats = new DescriptiveStatistics(reference);
        int numBins = Math.min(10, reference.length / 20); // At least 20 samples per bin
        numBins = Math.max(numBins, 3); // At least 3 bins
        
        double[] binEdges = new double[numBins + 1];
        binEdges[0] = refStats.getMin() - 1e-10;
        binEdges[numBins] = refStats.getMax() + 1e-10;
        
        for (int i = 1; i < numBins; i++) {
            binEdges[i] = refStats.getPercentile((double) i * 100.0 / numBins);
        }
        
        // Calculate bin proportions
        double[] refProportions = calculateBinProportions(reference, binEdges);
        double[] currProportions = calculateBinProportions(current, binEdges);
        
        // Calculate PSI
        double psi = 0.0;
        for (int i = 0; i < numBins; i++) {
            double refProp = Math.max(refProportions[i], 1e-10); // Avoid log(0)
            double currProp = Math.max(currProportions[i], 1e-10);
            
            psi += (currProp - refProp) * Math.log(currProp / refProp);
        }
        
        return psi;
    }
    
    /**
     * Calculate PSI for categorical features.
     */
    private double calculateCategoricalPSI(Map<Double, Integer> refCounts, Map<Double, Integer> currCounts,
                                         int refTotal, int currTotal) {
        Set<Double> allCategories = new HashSet<>(refCounts.keySet());
        allCategories.addAll(currCounts.keySet());
        
        double psi = 0.0;
        for (Double category : allCategories) {
            double refProp = Math.max(refCounts.getOrDefault(category, 0).doubleValue() / refTotal, 1e-10);
            double currProp = Math.max(currCounts.getOrDefault(category, 0).doubleValue() / currTotal, 1e-10);
            
            psi += (currProp - refProp) * Math.log(currProp / refProp);
        }
        
        return psi;
    }
    
    /**
     * Calculate Jensen-Shannon Divergence between two distributions.
     */
    private double calculateJensenShannonDivergence(double[] reference, double[] current) {
        // Create histograms
        DescriptiveStatistics refStats = new DescriptiveStatistics(reference);
        int numBins = Math.min(20, reference.length / 10);
        numBins = Math.max(numBins, 5);
        
        double min = Math.min(refStats.getMin(), Arrays.stream(current).min().orElse(0.0));
        double max = Math.max(refStats.getMax(), Arrays.stream(current).max().orElse(0.0));
        
        double[] binEdges = new double[numBins + 1];
        double binWidth = (max - min) / numBins;
        for (int i = 0; i <= numBins; i++) {
            binEdges[i] = min + i * binWidth;
        }
        binEdges[numBins] += 1e-10; // Ensure last value is included
        
        double[] refHist = calculateBinProportions(reference, binEdges);
        double[] currHist = calculateBinProportions(current, binEdges);
        
        // Calculate JS divergence
        double jsDivergence = 0.0;
        for (int i = 0; i < numBins; i++) {
            double p = Math.max(refHist[i], 1e-10);
            double q = Math.max(currHist[i], 1e-10);
            double m = (p + q) / 2.0;
            
            double kl1 = p * Math.log(p / m);
            double kl2 = q * Math.log(q / m);
            
            jsDivergence += 0.5 * (kl1 + kl2);
        }
        
        return jsDivergence;
    }
    
    /**
     * Calculate drift in statistical moments (mean, std, skewness, kurtosis).
     */
    private double[] calculateMomentsDrift(double[] reference, double[] current) {
        DescriptiveStatistics refStats = new DescriptiveStatistics(reference);
        DescriptiveStatistics currStats = new DescriptiveStatistics(current);
        
        double meanDrift = Math.abs(currStats.getMean() - refStats.getMean()) / 
                          (refStats.getStandardDeviation() + 1e-10);
        double stdDrift = Math.abs(currStats.getStandardDeviation() - refStats.getStandardDeviation()) / 
                         (refStats.getStandardDeviation() + 1e-10);
        double skewDrift = Math.abs(currStats.getSkewness() - refStats.getSkewness());
        double kurtDrift = Math.abs(currStats.getKurtosis() - refStats.getKurtosis());
        
        return new double[]{meanDrift, stdDrift, skewDrift, kurtDrift};
    }
    
    // Helper methods
    
    private double[] extractFeatureColumn(double[][] matrix, int columnIndex) {
        return IntStream.range(0, matrix.length)
                .mapToDouble(i -> matrix[i][columnIndex])
                .toArray();
    }
    
    private double[] calculateBinProportions(double[] data, double[] binEdges) {
        int numBins = binEdges.length - 1;
        int[] counts = new int[numBins];
        
        for (double value : data) {
            for (int i = 0; i < numBins; i++) {
                if (value >= binEdges[i] && value < binEdges[i + 1]) {
                    counts[i]++;
                    break;
                }
            }
        }
        
        double[] proportions = new double[numBins];
        for (int i = 0; i < numBins; i++) {
            proportions[i] = (double) counts[i] / data.length;
        }
        
        return proportions;
    }
    
    private Map<Double, Integer> getValueCounts(double[] data) {
        Map<Double, Integer> counts = new HashMap<>();
        for (double value : data) {
            counts.merge(value, 1, Integer::sum);
        }
        return counts;
    }
    
    private double combineTestScores(Map<String, TestResult> testResults) {
        // Weight different tests
        double score = 0.0;
        score += testResults.getOrDefault("KS", new TestResult(0, 0, false)).score * 0.3;
        score += testResults.getOrDefault("PSI", new TestResult(0, 0, false)).score * 0.3;
        score += testResults.getOrDefault("JS", new TestResult(0, 0, false)).score * 0.2;
        score += testResults.getOrDefault("Moments", new TestResult(0, 0, false)).score * 0.2;
        return score;
    }
    
    private double calculateOverallDriftScore(List<FeatureDriftResult> featureResults) {
        return featureResults.stream()
                .mapToDouble(FeatureDriftResult::getDriftScore)
                .average()
                .orElse(0.0);
    }
    
    /**
     * Reset drift detector state.
     */
    public void reset() {
        // Clear any internal state if needed
        // For this stateless detector, no action needed
        System.out.println("🔄 Data drift detector reset");
    }
    
    // Data classes for results
    
    public static class TestResult {
        public final double score;
        public final double pValue;
        public final boolean isDriftDetected;
        
        public TestResult(double score, double pValue, boolean isDriftDetected) {
            this.score = score;
            this.pValue = pValue;
            this.isDriftDetected = isDriftDetected;
        }
    }
    
    public static class FeatureDriftResult {
        private final String featureName;
        private final String featureType;
        private final double driftScore;
        private final double pValue;
        private final boolean driftDetected;
        private final Map<String, TestResult> testResults;
        
        public FeatureDriftResult(String featureName, String featureType, double driftScore, 
                                 double pValue, boolean driftDetected, Map<String, TestResult> testResults) {
            this.featureName = featureName;
            this.featureType = featureType;
            this.driftScore = driftScore;
            this.pValue = pValue;
            this.driftDetected = driftDetected;
            this.testResults = testResults;
        }
        
        // Getters
        public String getFeatureName() { return featureName; }
        public String getFeatureType() { return featureType; }
        public double getDriftScore() { return driftScore; }
        public double getPValue() { return pValue; }
        public boolean isDriftDetected() { return driftDetected; }
        public Map<String, TestResult> getTestResults() { return testResults; }
    }
    
    public static class DataDriftResult {
        private final List<FeatureDriftResult> featureResults;
        private final double overallDriftScore;
        private final boolean overallDriftDetected;
        private final Date timestamp;
        private final int driftedFeaturesCount;
        
        public DataDriftResult(List<FeatureDriftResult> featureResults, double overallDriftScore,
                              boolean overallDriftDetected, Date timestamp, int driftedFeaturesCount) {
            this.featureResults = featureResults;
            this.overallDriftScore = overallDriftScore;
            this.overallDriftDetected = overallDriftDetected;
            this.timestamp = timestamp;
            this.driftedFeaturesCount = driftedFeaturesCount;
        }
        
        // Getters
        public List<FeatureDriftResult> getFeatureResults() { return featureResults; }
        public double getOverallDriftScore() { return overallDriftScore; }
        public boolean isOverallDriftDetected() { return overallDriftDetected; }
        public boolean isOverallDrift() { return overallDriftDetected; } // Alias for compatibility
        public Date getTimestamp() { return timestamp; }
        public int getDriftedFeaturesCount() { return driftedFeaturesCount; }
        
        public String getPrimaryMethod() {
            // Return the first method that detected drift
            for (FeatureDriftResult feature : featureResults) {
                for (Map.Entry<String, TestResult> entry : feature.getTestResults().entrySet()) {
                    if (entry.getValue().isDriftDetected) {
                        return entry.getKey();
                    }
                }
            }
            return "None";
        }
    }
}
