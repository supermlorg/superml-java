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

import org.superml.cluster.KMeans;

/**
 * Clustering algorithms example
 * Demonstrates K-Means clustering
 */
public class ClusteringExample {
    
    public static void main(String[] args) {
        System.out.println("=== SuperML 2.0.0 - Clustering Example ===\n");
        
        try {
            // Generate synthetic clustering data
            double[][] X = generateClusteringData(150, 2);
            
            System.out.println("Generated " + X.length + " samples with " + X[0].length + " features");
            
            // Apply K-Means clustering
            System.out.println("\n🎯 Applying K-Means Clustering (k=3)...");
            KMeans kmeans = new KMeans();
            kmeans.fit(X);
            
            // Get cluster assignments
            int[] clusters = kmeans.predict(X);
            
            // Analyze clusters
            System.out.println("\n=== Clustering Results ===");
            int[] clusterCounts = new int[3];
            for (int cluster : clusters) {
                if (cluster >= 0 && cluster < 3) {
                    clusterCounts[cluster]++;
                }
            }
            
            for (int i = 0; i < 3; i++) {
                System.out.println("Cluster " + i + ": " + clusterCounts[i] + " points");
            }
            
            // Show some cluster assignments
            System.out.println("\n=== Sample Cluster Assignments ===");
            for (int i = 0; i < Math.min(10, clusters.length); i++) {
                System.out.println("Point " + (i+1) + 
                    " (" + String.format("%.2f", X[i][0]) + 
                    ", " + String.format("%.2f", X[i][1]) + 
                    ") -> Cluster " + clusters[i]);
            }
            
            // Calculate within-cluster sum of squares (basic measure)
            double wcss = calculateWCSS(X, clusters, 3);
            System.out.println("\nWithin-Cluster Sum of Squares: " + String.format("%.3f", wcss));
            
            System.out.println("\n✅ Clustering example completed successfully!");
            
        } catch (Exception e) {
            System.err.println("❌ Error running clustering example: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Generate synthetic data with natural clusters
     */
    private static double[][] generateClusteringData(int samples, int features) {
        double[][] data = new double[samples][features];
        java.util.Random random = new java.util.Random(42);
        
        // Create 3 natural clusters
        int samplesPerCluster = samples / 3;
        
        for (int i = 0; i < samples; i++) {
            int clusterId = i / samplesPerCluster;
            if (clusterId >= 3) clusterId = 2; // Handle remainder
            
            // Cluster centers
            double[] centers = {
                clusterId == 0 ? -2.0 : (clusterId == 1 ? 2.0 : 0.0),
                clusterId == 0 ? -2.0 : (clusterId == 1 ? 2.0 : 3.0)
            };
            
            for (int j = 0; j < features; j++) {
                data[i][j] = centers[j] + random.nextGaussian() * 0.8;
            }
        }
        return data;
    }
    
    /**
     * Calculate Within-Cluster Sum of Squares
     */
    private static double calculateWCSS(double[][] data, int[] clusters, int k) {
        // Calculate cluster centers
        double[][] centers = new double[k][data[0].length];
        int[] counts = new int[k];
        
        for (int i = 0; i < data.length; i++) {
            int cluster = clusters[i];
            if (cluster >= 0 && cluster < k) {
                for (int j = 0; j < data[i].length; j++) {
                    centers[cluster][j] += data[i][j];
                }
                counts[cluster]++;
            }
        }
        
        for (int i = 0; i < k; i++) {
            if (counts[i] > 0) {
                for (int j = 0; j < centers[i].length; j++) {
                    centers[i][j] /= counts[i];
                }
            }
        }
        
        // Calculate WCSS
        double wcss = 0.0;
        for (int i = 0; i < data.length; i++) {
            int cluster = clusters[i];
            if (cluster >= 0 && cluster < k) {
                for (int j = 0; j < data[i].length; j++) {
                    double diff = data[i][j] - centers[cluster][j];
                    wcss += diff * diff;
                }
            }
        }
        
        return wcss;
    }
}
