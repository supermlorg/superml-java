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

import org.superml.visualization.VisualizationFactory;
import org.superml.visualization.Visualization;

/**
 * Example demonstrating XChart cluster visualization
 * Shows how to create and display cluster plots
 */
public class ClusterVisualizationExample {
    
    public static void main(String[] args) {
        System.out.println("🎯 SuperML Cluster Visualization Example");
        System.out.println("=" + "=".repeat(45));
        
        // Generate sample clustered data
        double[][] data = generateSampleData();
        int[] clusterAssignments = generateClusterAssignments();
        double[][] centroids = generateCentroids();
        
        // Demo 1: ASCII mode
        System.out.println("\n1️⃣ ASCII Cluster Visualization:");
        Visualization asciiPlot = VisualizationFactory.createDualModeClusterPlot(
            data, clusterAssignments, centroids, 
            "ASCII Cluster Plot", 
            VisualizationFactory.VisualizationMode.ASCII
        );
        asciiPlot.display();
        
        // Demo 2: XChart GUI mode (if available)
        System.out.println("\n2️⃣ XChart GUI Cluster Visualization:");
        if (VisualizationFactory.isXChartAvailable()) {
            Visualization xchartPlot = VisualizationFactory.createDualModeClusterPlot(
                data, clusterAssignments, centroids, 
                "XChart GUI Cluster Plot", 
                VisualizationFactory.VisualizationMode.XCHART
            );
            xchartPlot.display();
        } else {
            System.out.println("⚠️ XChart not available - add XChart dependency to see GUI visualization");
        }
        
        // Demo 3: Auto mode (preferred)
        System.out.println("\n3️⃣ Auto Mode (XChart with ASCII fallback):");
        Visualization autoPlot = VisualizationFactory.createClusterPlot(
            data, clusterAssignments, centroids, 
            "Auto Mode Cluster Plot"
        );
        autoPlot.display();
        
        // Show mode information
        VisualizationFactory.displayModeInfo();
    }
    
    /**
     * Generate sample 2D data points
     */
    private static double[][] generateSampleData() {
        return new double[][] {
            // Cluster 0 (bottom-left)
            {1.0, 1.0}, {1.2, 1.1}, {0.8, 0.9}, {1.1, 0.8}, {0.9, 1.2},
            {1.3, 1.0}, {0.7, 1.1}, {1.0, 0.7}, {1.4, 1.2}, {0.6, 0.8},
            
            // Cluster 1 (top-right)
            {4.0, 4.0}, {4.2, 4.1}, {3.8, 3.9}, {4.1, 3.8}, {3.9, 4.2},
            {4.3, 4.0}, {3.7, 4.1}, {4.0, 3.7}, {4.4, 4.2}, {3.6, 3.8},
            
            // Cluster 2 (top-left)
            {1.0, 4.0}, {1.2, 4.1}, {0.8, 3.9}, {1.1, 3.8}, {0.9, 4.2},
            {1.3, 4.0}, {0.7, 4.1}, {1.0, 3.7}, {1.4, 4.2}, {0.6, 3.8},
            
            // Cluster 3 (center)
            {2.5, 2.5}, {2.7, 2.6}, {2.3, 2.4}, {2.6, 2.3}, {2.4, 2.7},
            {2.8, 2.5}, {2.2, 2.6}, {2.5, 2.2}, {2.9, 2.7}, {2.1, 2.4}
        };
    }
    
    /**
     * Generate cluster assignments for sample data
     */
    private static int[] generateClusterAssignments() {
        return new int[] {
            // Cluster 0 assignments
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            // Cluster 1 assignments  
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            // Cluster 2 assignments
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            // Cluster 3 assignments
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3
        };
    }
    
    /**
     * Generate centroids for sample clusters
     */
    private static double[][] generateCentroids() {
        return new double[][] {
            {1.0, 1.0},  // Centroid for cluster 0
            {4.0, 4.0},  // Centroid for cluster 1
            {1.0, 4.0},  // Centroid for cluster 2
            {2.5, 2.5}   // Centroid for cluster 3
        };
    }
}
