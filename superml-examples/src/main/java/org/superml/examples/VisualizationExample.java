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

package org.superml.examples;

import org.superml.core.Estimator;
import org.superml.linear_model.LogisticRegression;
import org.superml.linear_model.LinearRegression;
import org.superml.datasets.Datasets;
import org.superml.utils.DataUtils;
import org.superml.metrics.Metrics;

// Import visualization components
import org.superml.visualization.Visualization;
import org.superml.visualization.VisualizationFactory;
import org.superml.visualization.classification.ConfusionMatrix;
import org.superml.visualization.regression.ScatterPlot;

/**
 * SuperML Java 2.0.0 - Comprehensive Visualization Example
 * 
 * This example demonstrates:
 * 1. Classification visualization with enhanced confusion matrix
 * 2. Regression visualization with scatter plots and metrics
 * 3. Clustering visualization with cluster plots
 * 4. Advanced visualization features and customization
 */
public class VisualizationExample {
    
    public static void main(String[] args) {
        System.out.println("=== SuperML 2.0.0 - Comprehensive Visualization Example ===\n");
        
        try {
            // Demo 1: Classification Visualization
            demonstrateClassificationVisualization();
            
            // Demo 2: Regression Visualization
            demonstrateRegressionVisualization();
            
            // Demo 3: Clustering Visualization
            demonstrateClusteringVisualization();
            
            System.out.println("✅ All visualization examples completed successfully!");
            
        } catch (Exception e) {
            System.err.println("❌ Error in visualization example: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void demonstrateClassificationVisualization() {
        System.out.println("🎯 DEMO 1: Classification Visualization");
        System.out.println("=" + "=".repeat(40));
        
        // Generate multi-class dataset
        double[][] X = generateSyntheticData(300, 4);
        int[] y = generateMultiClassLabels(300, 3);
        
        // Split data
        int trainSize = (int)(X.length * 0.8);
        double[][] XTrain = new double[trainSize][];
        double[][] XTest = new double[X.length - trainSize][];
        int[] yTrain = new int[trainSize];
        int[] yTest = new int[X.length - trainSize];
        
        System.arraycopy(X, 0, XTrain, 0, trainSize);
        System.arraycopy(X, trainSize, XTest, 0, X.length - trainSize);
        System.arraycopy(y, 0, yTrain, 0, trainSize);
        System.arraycopy(y, trainSize, yTest, 0, X.length - trainSize);
        
        System.out.println("📊 Training multi-class classifier...");
        System.out.println("Dataset: 300 samples, 4 features, 3 classes");
        System.out.println("Training samples: " + trainSize + ", Test samples: " + (X.length - trainSize));
        
        // Train model
        LogisticRegression model = new LogisticRegression();
        model.fit(XTrain, convertToDouble(yTrain));
        
        // Make predictions
        double[] predictions = model.predict(XTest);
        int[] predInt = convertToInt(predictions);
        
        // Create enhanced confusion matrix visualization
        String[] classNames = {"Setosa", "Versicolor", "Virginica"};
        ConfusionMatrix confMatrix = VisualizationFactory.createConfusionMatrix(yTest, predInt, classNames);
        confMatrix.setTitle("Multi-class Classification Results");
        confMatrix.display();
        
        System.out.println("\n" + "=".repeat(80) + "\n");
    }
    
    private static void demonstrateRegressionVisualization() {
        System.out.println("📈 DEMO 2: Regression Visualization");
        System.out.println("=" + "=".repeat(35));
        
        // Generate regression dataset
        double[][] X = generateSyntheticData(100, 3);
        double[] y = generateRegressionTargets(X);
        
        // Split data
        int trainSize = (int)(X.length * 0.8);
        double[][] XTrain = new double[trainSize][];
        double[][] XTest = new double[X.length - trainSize][];
        double[] yTrain = new double[trainSize];
        double[] yTest = new double[X.length - trainSize];
        
        System.arraycopy(X, 0, XTrain, 0, trainSize);
        System.arraycopy(X, trainSize, XTest, 0, X.length - trainSize);
        System.arraycopy(y, 0, yTrain, 0, trainSize);
        System.arraycopy(y, trainSize, yTest, 0, X.length - trainSize);
        
        System.out.println("📊 Training regression model...");
        System.out.println("Dataset: 100 samples, 3 features, continuous target");
        System.out.println("Training samples: " + trainSize + ", Test samples: " + (X.length - trainSize));
        
        // Train model
        LinearRegression model = new LinearRegression();
        model.fit(XTrain, yTrain);
        
        // Make predictions
        double[] predictions = model.predict(XTest);
        
        // Create scatter plot visualization
        ScatterPlot scatterPlot = VisualizationFactory.createScatterPlot(yTest, predictions);
        scatterPlot.setTitle("Regression: Actual vs Predicted Values");
        scatterPlot.setDimensions(80, 25); // Larger plot for better visualization
        scatterPlot.display();
        
        System.out.println("\n" + "=".repeat(80) + "\n");
    }
    
    private static void demonstrateClusteringVisualization() {
        System.out.println("🎯 DEMO 3: Clustering Visualization");
        System.out.println("=" + "=".repeat(35));
        
        // Generate clustered dataset
        double[][] data = generateClusteredData(150, 2, 3);
        int[] clusterAssignments = performSimpleClustering(data, 3);
        double[][] centroids = calculateCentroids(data, clusterAssignments, 3);
        
        System.out.println("📊 Performing clustering analysis...");
        System.out.println("Dataset: 150 samples, 2 features, 3 clusters");
        
        // Create cluster plot visualization
        Visualization clusterPlot = VisualizationFactory.createClusterPlot(data, clusterAssignments, centroids);
        clusterPlot.setTitle("K-Means Clustering Results");
        clusterPlot.display();
        
        System.out.println("\n" + "=".repeat(80) + "\n");
    }
    
    // Helper methods
    
    private static double[][] generateSyntheticData(int samples, int features) {
        double[][] data = new double[samples][features];
        java.util.Random random = new java.util.Random(42);
        
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < features; j++) {
                data[i][j] = random.nextGaussian();
            }
        }
        return data;
    }
    
    private static int[] generateMultiClassLabels(int samples, int numClasses) {
        int[] labels = new int[samples];
        java.util.Random random = new java.util.Random(42);
        
        for (int i = 0; i < samples; i++) {
            // Generate labels based on some pattern
            double sum = 0;
            for (int j = 0; j < 4; j++) {
                sum += random.nextGaussian();
            }
            if (sum < -0.5) {
                labels[i] = 0;
            } else if (sum > 0.5) {
                labels[i] = 2;
            } else {
                labels[i] = 1;
            }
        }
        return labels;
    }
    
    private static double[] generateRegressionTargets(double[][] X) {
        double[] y = new double[X.length];
        java.util.Random random = new java.util.Random(42);
        
        for (int i = 0; i < X.length; i++) {
            // Linear combination with some noise
            y[i] = 2.5 * X[i][0] - 1.5 * X[i][1] + 0.8 * X[i][2] + random.nextGaussian() * 0.1;
        }
        return y;
    }
    
    private static double[][] generateClusteredData(int samples, int features, int numClusters) {
        double[][] data = new double[samples][features];
        java.util.Random random = new java.util.Random(42);
        
        // Generate cluster centers
        double[][] centers = new double[numClusters][features];
        for (int i = 0; i < numClusters; i++) {
            for (int j = 0; j < features; j++) {
                centers[i][j] = random.nextGaussian() * 3;
            }
        }
        
        // Generate data points around centers
        int samplesPerCluster = samples / numClusters;
        for (int i = 0; i < samples; i++) {
            int cluster = Math.min(i / samplesPerCluster, numClusters - 1);
            for (int j = 0; j < features; j++) {
                data[i][j] = centers[cluster][j] + random.nextGaussian() * 0.8;
            }
        }
        
        return data;
    }
    
    private static int[] performSimpleClustering(double[][] data, int k) {
        // Simple k-means-like clustering for demonstration
        int[] assignments = new int[data.length];
        java.util.Random random = new java.util.Random(42);
        
        // Random initial assignments
        for (int i = 0; i < data.length; i++) {
            assignments[i] = random.nextInt(k);
        }
        
        // Few iterations of k-means
        for (int iter = 0; iter < 10; iter++) {
            double[][] centroids = calculateCentroids(data, assignments, k);
            
            // Reassign points to nearest centroid
            for (int i = 0; i < data.length; i++) {
                double minDist = Double.MAX_VALUE;
                int bestCluster = 0;
                
                for (int j = 0; j < k; j++) {
                    double dist = 0;
                    for (int d = 0; d < data[i].length; d++) {
                        double diff = data[i][d] - centroids[j][d];
                        dist += diff * diff;
                    }
                    
                    if (dist < minDist) {
                        minDist = dist;
                        bestCluster = j;
                    }
                }
                assignments[i] = bestCluster;
            }
        }
        
        return assignments;
    }
    
    private static double[][] calculateCentroids(double[][] data, int[] assignments, int k) {
        double[][] centroids = new double[k][data[0].length];
        int[] counts = new int[k];
        
        // Sum points in each cluster
        for (int i = 0; i < data.length; i++) {
            int cluster = assignments[i];
            if (cluster >= 0 && cluster < k) {
                counts[cluster]++;
                for (int j = 0; j < data[i].length; j++) {
                    centroids[cluster][j] += data[i][j];
                }
            }
        }
        
        // Calculate averages
        for (int i = 0; i < k; i++) {
            if (counts[i] > 0) {
                for (int j = 0; j < centroids[i].length; j++) {
                    centroids[i][j] /= counts[i];
                }
            }
        }
        
        return centroids;
    }
    
    private static double[] convertToDouble(int[] intArray) {
        double[] doubleArray = new double[intArray.length];
        for (int i = 0; i < intArray.length; i++) {
            doubleArray[i] = (double) intArray[i];
        }
        return doubleArray;
    }
    
    private static int[] convertToInt(double[] doubleArray) {
        int[] intArray = new int[doubleArray.length];
        for (int i = 0; i < doubleArray.length; i++) {
            intArray[i] = (int) Math.round(doubleArray[i]);
        }
        return intArray;
    }
}
