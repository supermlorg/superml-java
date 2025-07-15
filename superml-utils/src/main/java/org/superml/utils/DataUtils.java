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

package org.superml.utils;

import org.apache.commons.csv.*;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Utility class for common data processing and analysis tasks in SuperML.
 */
public class DataUtils {
    
    private static final ObjectMapper objectMapper = new ObjectMapper();
    
    /**
     * Load data from CSV file.
     * @param filePath Path to the CSV file
     * @param hasHeader Whether the CSV has a header row
     * @param targetColumn Index of the target column (-1 if no target)
     * @return DataSet containing features and labels
     */
    public static DataSet loadCSV(String filePath, boolean hasHeader, int targetColumn) throws IOException {
        System.out.println("📊 Loading CSV data from: " + filePath);
        
        List<double[]> features = new ArrayList<>();
        List<Double> targets = new ArrayList<>();
        List<String> headers = new ArrayList<>();
        
        try (Reader reader = Files.newBufferedReader(Paths.get(filePath));
             CSVParser csvParser = new CSVParser(reader, CSVFormat.DEFAULT)) {
            
            boolean isFirstRow = true;
            for (CSVRecord record : csvParser) {
                if (isFirstRow && hasHeader) {
                    // Store headers
                    for (String value : record) {
                        headers.add(value);
                    }
                    isFirstRow = false;
                    continue;
                }
                
                List<Double> row = new ArrayList<>();
                Double target = null;
                
                for (int i = 0; i < record.size(); i++) {
                    try {
                        double value = Double.parseDouble(record.get(i));
                        if (i == targetColumn) {
                            target = value;
                        } else {
                            row.add(value);
                        }
                    } catch (NumberFormatException e) {
                        // Handle non-numeric values
                        if (i == targetColumn) {
                            target = encodeStringTarget(record.get(i));
                        } else {
                            row.add(0.0); // Or use proper encoding
                        }
                    }
                }
                
                if (!row.isEmpty()) {
                    features.add(row.stream().mapToDouble(Double::doubleValue).toArray());
                    if (target != null) {
                        targets.add(target);
                    }
                }
            }
        }
        
        double[][] X = features.toArray(new double[0][]);
        double[] y = targets.isEmpty() ? null : targets.stream().mapToDouble(Double::doubleValue).toArray();
        
        System.out.printf("-> Loaded %d samples with %d features\n", X.length, X[0].length);
        
        return new DataSet(X, y, headers);
    }
    
    /**
     * Save data to CSV file.
     */
    public static void saveCSV(DataSet dataset, String filePath, boolean includeHeaders) throws IOException {
        try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(Paths.get(filePath)))) {
            
            // Write headers if requested
            if (includeHeaders && !dataset.headers.isEmpty()) {
                writer.println(String.join(",", dataset.headers));
            }
            
            // Write data
            for (int i = 0; i < dataset.X.length; i++) {
                List<String> row = new ArrayList<>();
                
                // Add features
                for (double feature : dataset.X[i]) {
                    row.add(String.valueOf(feature));
                }
                
                // Add target if available
                if (dataset.y != null && i < dataset.y.length) {
                    row.add(String.valueOf(dataset.y[i]));
                }
                
                writer.println(String.join(",", row));
            }
        }
        
        System.out.println("💾 Data saved to: " + filePath);
    }
    
    /**
     * Split dataset into training and testing sets.
     */
    public static TrainTestSplit trainTestSplit(double[][] X, double[] y, double testSize, int randomSeed) {
        System.out.printf("✂️  Splitting data: %.0f%% train, %.0f%% test\n", 
            (1 - testSize) * 100, testSize * 100);
        
        Random random = new Random(randomSeed);
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < X.length; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, random);
        
        int trainSize = (int) ((1 - testSize) * X.length);
        
        double[][] XTrain = new double[trainSize][];
        double[][] XTest = new double[X.length - trainSize][];
        double[] yTrain = new double[trainSize];
        double[] yTest = new double[X.length - trainSize];
        
        for (int i = 0; i < trainSize; i++) {
            int idx = indices.get(i);
            XTrain[i] = X[idx];
            yTrain[i] = y[idx];
        }
        
        for (int i = 0; i < X.length - trainSize; i++) {
            int idx = indices.get(i + trainSize);
            XTest[i] = X[idx];
            yTest[i] = y[idx];
        }
        
        return new TrainTestSplit(XTrain, XTest, yTrain, yTest);
    }
    
    /**
     * Normalize features using standard scaling (z-score).
     */
    public static NormalizationResult standardScale(double[][] X) {
        System.out.println("📐 Applying standard scaling...");
        
        int numFeatures = X[0].length;
        double[] means = new double[numFeatures];
        double[] stds = new double[numFeatures];
        
        // Calculate means
        for (int j = 0; j < numFeatures; j++) {
            double sum = 0;
            for (int i = 0; i < X.length; i++) {
                sum += X[i][j];
            }
            means[j] = sum / X.length;
        }
        
        // Calculate standard deviations
        for (int j = 0; j < numFeatures; j++) {
            double sumSquaredDiffs = 0;
            for (int i = 0; i < X.length; i++) {
                sumSquaredDiffs += Math.pow(X[i][j] - means[j], 2);
            }
            stds[j] = Math.sqrt(sumSquaredDiffs / X.length);
            if (stds[j] == 0) stds[j] = 1; // Avoid division by zero
        }
        
        // Apply scaling
        double[][] scaledX = new double[X.length][numFeatures];
        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < numFeatures; j++) {
                scaledX[i][j] = (X[i][j] - means[j]) / stds[j];
            }
        }
        
        return new NormalizationResult(scaledX, means, stds);
    }
    
    /**
     * Apply learned normalization parameters to new data.
     */
    public static double[][] applyNormalization(double[][] X, double[] means, double[] stds) {
        double[][] scaledX = new double[X.length][X[0].length];
        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < X[i].length; j++) {
                scaledX[i][j] = (X[i][j] - means[j]) / stds[j];
            }
        }
        return scaledX;
    }
    
    /**
     * Normalize features using min-max scaling.
     */
    public static NormalizationResult minMaxScale(double[][] X) {
        System.out.println("📐 Applying min-max scaling...");
        
        int numFeatures = X[0].length;
        double[] mins = new double[numFeatures];
        double[] maxs = new double[numFeatures];
        
        // Find min and max for each feature
        for (int j = 0; j < numFeatures; j++) {
            mins[j] = Double.MAX_VALUE;
            maxs[j] = Double.MIN_VALUE;
            
            for (int i = 0; i < X.length; i++) {
                mins[j] = Math.min(mins[j], X[i][j]);
                maxs[j] = Math.max(maxs[j], X[i][j]);
            }
        }
        
        // Apply scaling
        double[][] scaledX = new double[X.length][numFeatures];
        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < numFeatures; j++) {
                double range = maxs[j] - mins[j];
                if (range == 0) {
                    scaledX[i][j] = 0;
                } else {
                    scaledX[i][j] = (X[i][j] - mins[j]) / range;
                }
            }
        }
        
        return new NormalizationResult(scaledX, mins, maxs);
    }
    
    /**
     * Generate statistical summary of the dataset.
     */
    public static DataSummary summarizeData(double[][] X, double[] y) {
        System.out.println("📊 Generating data summary...");
        
        DataSummary summary = new DataSummary();
        summary.numSamples = X.length;
        summary.numFeatures = X[0].length;
        
        // Feature statistics
        summary.featureStats = new FeatureStatistics[X[0].length];
        for (int j = 0; j < X[0].length; j++) {
            double[] feature = new double[X.length];
            for (int i = 0; i < X.length; i++) {
                feature[i] = X[i][j];
            }
            
            DescriptiveStatistics stats = new DescriptiveStatistics(feature);
            summary.featureStats[j] = new FeatureStatistics(
                stats.getMean(),
                stats.getStandardDeviation(),
                stats.getMin(),
                stats.getMax(),
                stats.getPercentile(25),
                stats.getPercentile(75)
            );
        }
        
        // Target statistics (if available)
        if (y != null) {
            DescriptiveStatistics targetStats = new DescriptiveStatistics(y);
            summary.targetStats = new FeatureStatistics(
                targetStats.getMean(),
                targetStats.getStandardDeviation(),
                targetStats.getMin(),
                targetStats.getMax(),
                targetStats.getPercentile(25),
                targetStats.getPercentile(75)
            );
            
            // Check if classification or regression
            Set<Double> uniqueValues = Arrays.stream(y).boxed().collect(Collectors.toSet());
            summary.isClassification = uniqueValues.size() <= 20 && uniqueValues.stream().allMatch(v -> v == Math.floor(v));
            summary.numClasses = uniqueValues.size();
        }
        
        return summary;
    }
    
    /**
     * Calculate correlation matrix for features.
     */
    public static double[][] calculateCorrelationMatrix(double[][] X) {
        System.out.println("🔗 Calculating correlation matrix...");
        
        int numFeatures = X[0].length;
        double[][] correlationMatrix = new double[numFeatures][numFeatures];
        PearsonsCorrelation correlation = new PearsonsCorrelation();
        
        for (int i = 0; i < numFeatures; i++) {
            for (int j = 0; j < numFeatures; j++) {
                if (i == j) {
                    correlationMatrix[i][j] = 1.0;
                } else {
                    double[] featureI = new double[X.length];
                    double[] featureJ = new double[X.length];
                    
                    for (int k = 0; k < X.length; k++) {
                        featureI[k] = X[k][i];
                        featureJ[k] = X[k][j];
                    }
                    
                    correlationMatrix[i][j] = correlation.correlation(featureI, featureJ);
                }
            }
        }
        
        return correlationMatrix;
    }
    
    /**
     * Find highly correlated feature pairs.
     */
    public static List<CorrelationPair> findHighlyCorrelatedFeatures(double[][] X, double threshold) {
        double[][] correlationMatrix = calculateCorrelationMatrix(X);
        List<CorrelationPair> highCorrelations = new ArrayList<>();
        
        for (int i = 0; i < correlationMatrix.length; i++) {
            for (int j = i + 1; j < correlationMatrix[i].length; j++) {
                double correlation = Math.abs(correlationMatrix[i][j]);
                if (correlation > threshold) {
                    highCorrelations.add(new CorrelationPair(i, j, correlationMatrix[i][j]));
                }
            }
        }
        
        highCorrelations.sort((a, b) -> Double.compare(Math.abs(b.correlation), Math.abs(a.correlation)));
        
        return highCorrelations;
    }
    
    /**
     * Remove features with low variance.
     */
    public static FeatureSelectionResult removeConstantFeatures(double[][] X, double varianceThreshold) {
        System.out.println("🔍 Removing constant features...");
        
        List<Integer> keepIndices = new ArrayList<>();
        
        for (int j = 0; j < X[0].length; j++) {
            double[] feature = new double[X.length];
            for (int i = 0; i < X.length; i++) {
                feature[i] = X[i][j];
            }
            
            double variance = StatUtils.variance(feature);
            if (variance > varianceThreshold) {
                keepIndices.add(j);
            }
        }
        
        // Create new feature matrix
        double[][] filteredX = new double[X.length][keepIndices.size()];
        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < keepIndices.size(); j++) {
                filteredX[i][j] = X[i][keepIndices.get(j)];
            }
        }
        
        System.out.printf("-> Kept %d/%d features\n", keepIndices.size(), X[0].length);
        
        return new FeatureSelectionResult(filteredX, keepIndices);
    }
    
    /**
     * Generate synthetic classification dataset.
     */
    public static DataSet generateClassificationData(int numSamples, int numFeatures, int numClasses, int randomSeed) {
        System.out.printf("🎲 Generating synthetic classification data: %d samples, %d features, %d classes\n",
            numSamples, numFeatures, numClasses);
        
        Random random = new Random(randomSeed);
        double[][] X = new double[numSamples][numFeatures];
        double[] y = new double[numSamples];
        
        for (int i = 0; i < numSamples; i++) {
            // Generate features
            for (int j = 0; j < numFeatures; j++) {
                X[i][j] = random.nextGaussian();
            }
            
            // Generate class based on simple linear combination
            double sum = 0;
            for (int j = 0; j < numFeatures; j++) {
                sum += X[i][j] * (j + 1);
            }
            y[i] = Math.abs((int) sum) % numClasses;
        }
        
        List<String> headers = new ArrayList<>();
        for (int i = 0; i < numFeatures; i++) {
            headers.add("feature_" + i);
        }
        headers.add("target");
        
        return new DataSet(X, y, headers);
    }
    
    /**
     * Generate synthetic regression dataset.
     */
    public static DataSet generateRegressionData(int numSamples, int numFeatures, double noise, int randomSeed) {
        System.out.printf("🎲 Generating synthetic regression data: %d samples, %d features, noise=%.2f\n",
            numSamples, numFeatures, noise);
        
        Random random = new Random(randomSeed);
        double[][] X = new double[numSamples][numFeatures];
        double[] y = new double[numSamples];
        
        // Generate random coefficients
        double[] coefficients = new double[numFeatures];
        for (int j = 0; j < numFeatures; j++) {
            coefficients[j] = random.nextGaussian();
        }
        
        for (int i = 0; i < numSamples; i++) {
            // Generate features
            double target = 0;
            for (int j = 0; j < numFeatures; j++) {
                X[i][j] = random.nextGaussian();
                target += X[i][j] * coefficients[j];
            }
            
            // Add noise
            target += random.nextGaussian() * noise;
            y[i] = target;
        }
        
        List<String> headers = new ArrayList<>();
        for (int i = 0; i < numFeatures; i++) {
            headers.add("feature_" + i);
        }
        headers.add("target");
        
        return new DataSet(X, y, headers);
    }
    
    private static double encodeStringTarget(String value) {
        // Simple string to number encoding
        return value.hashCode() % 1000;
    }
    
    // Data classes
    
    public static class DataSet {
        public final double[][] X;
        public final double[] y;
        public final List<String> headers;
        
        public DataSet(double[][] X, double[] y, List<String> headers) {
            this.X = X;
            this.y = y;
            this.headers = headers != null ? headers : new ArrayList<>();
        }
    }
    
    public static class TrainTestSplit {
        public final double[][] XTrain;
        public final double[][] XTest;
        public final double[] yTrain;
        public final double[] yTest;
        
        public TrainTestSplit(double[][] XTrain, double[][] XTest, double[] yTrain, double[] yTest) {
            this.XTrain = XTrain;
            this.XTest = XTest;
            this.yTrain = yTrain;
            this.yTest = yTest;
        }
    }
    
    public static class NormalizationResult {
        public final double[][] scaledX;
        public final double[] parameters1; // means or mins
        public final double[] parameters2; // stds or maxs
        
        public NormalizationResult(double[][] scaledX, double[] parameters1, double[] parameters2) {
            this.scaledX = scaledX;
            this.parameters1 = parameters1;
            this.parameters2 = parameters2;
        }
    }
    
    public static class DataSummary {
        public int numSamples;
        public int numFeatures;
        public FeatureStatistics[] featureStats;
        public FeatureStatistics targetStats;
        public boolean isClassification;
        public int numClasses;
    }
    
    public static class FeatureStatistics {
        public final double mean;
        public final double std;
        public final double min;
        public final double max;
        public final double q25;
        public final double q75;
        
        public FeatureStatistics(double mean, double std, double min, double max, double q25, double q75) {
            this.mean = mean;
            this.std = std;
            this.min = min;
            this.max = max;
            this.q25 = q25;
            this.q75 = q75;
        }
    }
    
    public static class CorrelationPair {
        public final int feature1;
        public final int feature2;
        public final double correlation;
        
        public CorrelationPair(int feature1, int feature2, double correlation) {
            this.feature1 = feature1;
            this.feature2 = feature2;
            this.correlation = correlation;
        }
    }
    
    public static class FeatureSelectionResult {
        public final double[][] filteredX;
        public final List<Integer> selectedFeatures;
        
        public FeatureSelectionResult(double[][] filteredX, List<Integer> selectedFeatures) {
            this.filteredX = filteredX;
            this.selectedFeatures = selectedFeatures;
        }
    }
}
