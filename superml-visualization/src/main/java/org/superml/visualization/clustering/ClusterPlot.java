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

package org.superml.visualization.clustering;

import org.superml.visualization.Visualization;

/**
 * Cluster visualization for clustering algorithms
 * Shows 2D cluster assignments and centroids
 */
public class ClusterPlot implements Visualization {
    
    private final double[][] data;
    private final int[] clusterAssignments;
    private final double[][] centroids;
    private String title = "Cluster Visualization";
    private int width = 60;
    private int height = 20;
    private final char[] clusterSymbols = {'*', 'o', '+', 'x', '#', '@', '%', '&'};
    
    public ClusterPlot(double[][] data, int[] clusterAssignments, double[][] centroids) {
        this.data = data.clone();
        this.clusterAssignments = clusterAssignments.clone();
        this.centroids = centroids != null ? centroids.clone() : null;
    }
    
    @Override
    public void display() {
        System.out.println("🎯 " + title);
        System.out.println("=" + "=".repeat(title.length() + 3));
        
        if (data[0].length >= 2) {
            displayClusterPlot();
        } else {
            System.out.println("Note: Visualization requires at least 2 dimensions. Showing 1D projection.\n");
            display1DProjection();
        }
        
        displayClusterStatistics();
    }
    
    private void displayClusterPlot() {
        // Use first two dimensions for plotting
        double minX = Double.MAX_VALUE, maxX = Double.MIN_VALUE;
        double minY = Double.MAX_VALUE, maxY = Double.MIN_VALUE;
        
        // Find data range
        for (double[] point : data) {
            minX = Math.min(minX, point[0]);
            maxX = Math.max(maxX, point[0]);
            minY = Math.min(minY, point[1]);
            maxY = Math.max(maxY, point[1]);
        }
        
        // Add padding
        double xRange = maxX - minX;
        double yRange = maxY - minY;
        minX -= xRange * 0.05;
        maxX += xRange * 0.05;
        minY -= yRange * 0.05;
        maxY += yRange * 0.05;
        
        char[][] plot = new char[height][width];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                plot[i][j] = ' ';
            }
        }
        
        // Plot data points
        for (int i = 0; i < data.length; i++) {
            int x = (int) ((data[i][0] - minX) / (maxX - minX) * (width - 1));
            int y = (int) ((data[i][1] - minY) / (maxY - minY) * (height - 1));
            y = height - 1 - y; // Flip Y axis
            
            if (x >= 0 && x < width && y >= 0 && y < height) {
                int cluster = clusterAssignments[i];
                char symbol = cluster >= 0 && cluster < clusterSymbols.length ? 
                    clusterSymbols[cluster] : '?';
                plot[y][x] = symbol;
            }
        }
        
        // Plot centroids if available
        if (centroids != null) {
            for (int i = 0; i < centroids.length; i++) {
                if (centroids[i].length >= 2) {
                    int x = (int) ((centroids[i][0] - minX) / (maxX - minX) * (width - 1));
                    int y = (int) ((centroids[i][1] - minY) / (maxY - minY) * (height - 1));
                    y = height - 1 - y; // Flip Y axis
                    
                    if (x >= 0 && x < width && y >= 0 && y < height) {
                        plot[y][x] = 'C'; // Centroid marker
                    }
                }
            }
        }
        
        System.out.println("\n2D Cluster Plot:");
        System.out.printf("X-axis: Feature 1 [%.3f to %.3f]\n", minX, maxX);
        System.out.printf("Y-axis: Feature 2 [%.3f to %.3f]\n", minY, maxY);
        System.out.print("Legend: ");
        
        int numClusters = getNumClusters();
        for (int i = 0; i < Math.min(numClusters, clusterSymbols.length); i++) {
            System.out.printf("%c=Cluster%d ", clusterSymbols[i], i);
        }
        if (centroids != null) {
            System.out.print("C=Centroid");
        }
        System.out.println("\n");
        
        // Print plot
        for (int i = 0; i < height; i++) {
            System.out.print("|");
            for (int j = 0; j < width; j++) {
                System.out.print(plot[i][j]);
            }
            System.out.println("|");
        }
        
        // X-axis
        System.out.print("+");
        for (int i = 0; i < width; i++) {
            System.out.print("-");
        }
        System.out.println("+");
        System.out.println();
    }
    
    private void display1DProjection() {
        // Simple 1D visualization using first dimension
        double min = Double.MAX_VALUE, max = Double.MIN_VALUE;
        for (double[] point : data) {
            min = Math.min(min, point[0]);
            max = Math.max(max, point[0]);
        }
        
        System.out.println("1D Cluster Projection (Feature 1):");
        System.out.printf("Range: [%.3f to %.3f]\n", min, max);
        
        // Create bins
        int numBins = 50;
        char[] line = new char[numBins];
        for (int i = 0; i < numBins; i++) {
            line[i] = ' ';
        }
        
        for (int i = 0; i < data.length; i++) {
            int bin = (int) ((data[i][0] - min) / (max - min) * (numBins - 1));
            if (bin >= 0 && bin < numBins) {
                int cluster = clusterAssignments[i];
                char symbol = cluster >= 0 && cluster < clusterSymbols.length ? 
                    clusterSymbols[cluster] : '?';
                line[bin] = symbol;
            }
        }
        
        System.out.print("|");
        for (char c : line) {
            System.out.print(c);
        }
        System.out.println("|");
        System.out.println();
    }
    
    private void displayClusterStatistics() {
        int numClusters = getNumClusters();
        int[] clusterCounts = new int[numClusters];
        
        // Count points per cluster
        for (int assignment : clusterAssignments) {
            if (assignment >= 0 && assignment < numClusters) {
                clusterCounts[assignment]++;
            }
        }
        
        System.out.println("📊 Cluster Statistics:");
        System.out.println("======================");
        
        for (int i = 0; i < numClusters; i++) {
            double percentage = (double) clusterCounts[i] / data.length * 100;
            System.out.printf("Cluster %d: %d points (%.1f%%)\n", 
                i, clusterCounts[i], percentage);
        }
        
        System.out.printf("Total samples: %d\n", data.length);
        System.out.printf("Number of clusters: %d\n", numClusters);
        System.out.printf("Dimensions: %d\n", data[0].length);
        
        // Calculate WCSS if centroids available
        if (centroids != null) {
            double wcss = calculateWCSS();
            System.out.printf("Within-Cluster Sum of Squares: %.3f\n", wcss);
        }
        System.out.println();
    }
    
    private int getNumClusters() {
        int max = -1;
        for (int assignment : clusterAssignments) {
            max = Math.max(max, assignment);
        }
        return max + 1;
    }
    
    private double calculateWCSS() {
        double wcss = 0;
        
        for (int i = 0; i < data.length; i++) {
            int cluster = clusterAssignments[i];
            if (cluster >= 0 && cluster < centroids.length) {
                double distance = 0;
                for (int j = 0; j < Math.min(data[i].length, centroids[cluster].length); j++) {
                    double diff = data[i][j] - centroids[cluster][j];
                    distance += diff * diff;
                }
                wcss += distance;
            }
        }
        
        return wcss;
    }
    
    // Getters and setters
    
    public void setDimensions(int width, int height) {
        this.width = Math.max(30, width);
        this.height = Math.max(10, height);
    }
    
    @Override
    public void setTitle(String title) {
        this.title = title;
    }
    
    @Override
    public String getTitle() {
        return title;
    }
    
    @Override
    public String toString() {
        return String.format("ClusterPlot: %d samples, %d clusters, %d dimensions", 
            data.length, getNumClusters(), data[0].length);
    }
}
