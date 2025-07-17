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

package org.superml.visualization;

import org.superml.visualization.classification.ConfusionMatrix;
import org.superml.visualization.classification.XChartConfusionMatrix;
import org.superml.visualization.clustering.ClusterPlot;
import org.superml.visualization.clustering.XChartClusterPlot;
import org.superml.visualization.regression.ScatterPlot;
import org.superml.visualization.regression.XChartScatterPlot;

/**
 * Enhanced Factory for creating visualization instances with dual mode support.
 * 
 * Features:
 * - ASCII-based visualizations for terminal environments
 * - XChart-based GUI visualizations for professional output
 * - Automatic fallback from GUI to ASCII mode
 * - Simplified API for both visualization types
 * 
 * @author SuperML Framework
 * @version 2.1.0
 */
public class VisualizationFactory {
    
    /**
     * Visualization mode enumeration
     */
    public enum VisualizationMode {
        ASCII,    // Terminal-based ASCII visualizations
        XCHART,   // GUI-based XChart visualizations
        AUTO      // Automatic selection (XChart with ASCII fallback)
    }
    
    private static VisualizationMode defaultMode = VisualizationMode.AUTO;
    
    /**
     * Set the default visualization mode for the factory
     */
    public static void setDefaultMode(VisualizationMode mode) {
        defaultMode = mode;
    }
    
    /**
     * Get the current default visualization mode
     */
    public static VisualizationMode getDefaultMode() {
        return defaultMode;
    }
    
    // ═══════════════════════════════════════════════════════════════════
    // LEGACY ASCII-ONLY METHODS (PRESERVED FOR COMPATIBILITY)
    // ═══════════════════════════════════════════════════════════════════
    
    /**
     * Create a confusion matrix visualization for classification results (legacy method)
     */
    public static ConfusionMatrix createConfusionMatrix(int[] actual, int[] predicted, int numClasses) {
        return new ConfusionMatrix(actual, predicted, numClasses);
    }
    
    /**
     * Create a confusion matrix with custom class names (legacy method)
     */
    public static ConfusionMatrix createConfusionMatrix(int[] actual, int[] predicted, String[] classNames) {
        return new ConfusionMatrix(actual, predicted, classNames);
    }
    
    /**
     * Create a scatter plot for regression analysis (legacy method)
     */
    public static ScatterPlot createScatterPlot(double[] actual, double[] predicted) {
        return new ScatterPlot(actual, predicted);
    }
    
    /**
     * Create a scatter plot with title (legacy method)
     */
    public static ScatterPlot createScatterPlot(double[] actual, double[] predicted, String title) {
        ScatterPlot plot = new ScatterPlot(actual, predicted);
        plot.setTitle(title);
        return plot;
    }
    
    // ═══════════════════════════════════════════════════════════════════
    // NEW DUAL MODE METHODS WITH XCHART SUPPORT
    // ═══════════════════════════════════════════════════════════════════
    
    /**
     * Create a dual-mode confusion matrix with automatic mode selection
     */
    public static Visualization createDualModeConfusionMatrix(int[] actual, int[] predicted, int numClasses) {
        return createDualModeConfusionMatrix(actual, predicted, numClasses, "Confusion Matrix", defaultMode);
    }
    
    /**
     * Create a dual-mode confusion matrix with custom class names
     */
    public static Visualization createDualModeConfusionMatrix(int[] actual, int[] predicted, String[] classNames) {
        return createDualModeConfusionMatrix(actual, predicted, classNames, "Confusion Matrix", defaultMode);
    }
    
    /**
     * Create a dual-mode confusion matrix with specific mode and title
     */
    public static Visualization createDualModeConfusionMatrix(int[] actual, int[] predicted, int numClasses, String title, VisualizationMode mode) {
        try {
            switch (mode) {
                case ASCII:
                    ConfusionMatrix asciiMatrix = new ConfusionMatrix(actual, predicted, numClasses);
                    asciiMatrix.setTitle(title);
                    return asciiMatrix;
                    
                case XCHART:
                    XChartConfusionMatrix xchartMatrix = new XChartConfusionMatrix(actual, predicted, numClasses);
                    xchartMatrix.setTitle(title);
                    return xchartMatrix;
                    
                case AUTO:
                default:
                    // Try XChart first, fallback to ASCII on error
                    try {
                        XChartConfusionMatrix autoMatrix = new XChartConfusionMatrix(actual, predicted, numClasses);
                        autoMatrix.setTitle(title);
                        return autoMatrix;
                    } catch (Exception e) {
                        System.out.println("⚠️ XChart unavailable, using ASCII mode: " + e.getMessage());
                        ConfusionMatrix fallbackMatrix = new ConfusionMatrix(actual, predicted, numClasses);
                        fallbackMatrix.setTitle(title);
                        return fallbackMatrix;
                    }
            }
        } catch (Exception e) {
            System.err.println("❌ Error creating confusion matrix: " + e.getMessage());
            // Emergency fallback
            return new ConfusionMatrix(actual, predicted, numClasses);
        }
    }
    
    /**
     * Create a dual-mode confusion matrix with custom class names and mode
     */
    public static Visualization createDualModeConfusionMatrix(int[] actual, int[] predicted, String[] classNames, String title, VisualizationMode mode) {
        try {
            switch (mode) {
                case ASCII:
                    ConfusionMatrix asciiMatrix = new ConfusionMatrix(actual, predicted, classNames);
                    asciiMatrix.setTitle(title);
                    return asciiMatrix;
                    
                case XCHART:
                    XChartConfusionMatrix xchartMatrix = new XChartConfusionMatrix(actual, predicted, classNames);
                    xchartMatrix.setTitle(title);
                    return xchartMatrix;
                    
                case AUTO:
                default:
                    // Try XChart first, fallback to ASCII on error
                    try {
                        XChartConfusionMatrix autoMatrix = new XChartConfusionMatrix(actual, predicted, classNames);
                        autoMatrix.setTitle(title);
                        return autoMatrix;
                    } catch (Exception e) {
                        System.out.println("⚠️ XChart unavailable, using ASCII mode: " + e.getMessage());
                        ConfusionMatrix fallbackMatrix = new ConfusionMatrix(actual, predicted, classNames);
                        fallbackMatrix.setTitle(title);
                        return fallbackMatrix;
                    }
            }
        } catch (Exception e) {
            System.err.println("❌ Error creating confusion matrix: " + e.getMessage());
            // Emergency fallback
            return new ConfusionMatrix(actual, predicted, classNames);
        }
    }
    
    // ═══════════════════════════════════════════════════════════════════
    // CLUSTERING VISUALIZATION METHODS
    // ═══════════════════════════════════════════════════════════════════
    
    /**
     * Create a cluster plot using default mode
     */
    public static Visualization createClusterPlot(double[][] data, int[] clusterAssignments, double[][] centroids) {
        return createDualModeClusterPlot(data, clusterAssignments, centroids, "Cluster Analysis", defaultMode);
    }
    
    /**
     * Create a cluster plot with title using default mode
     */
    public static Visualization createClusterPlot(double[][] data, int[] clusterAssignments, double[][] centroids, String title) {
        return createDualModeClusterPlot(data, clusterAssignments, centroids, title, defaultMode);
    }
    
    /**
     * Create a cluster plot without centroids using default mode
     */
    public static Visualization createClusterPlot(double[][] data, int[] clusterAssignments) {
        return createDualModeClusterPlot(data, clusterAssignments, null, "Cluster Analysis", defaultMode);
    }
    
    /**
     * Create a dual-mode cluster plot with full control
     */
    public static Visualization createDualModeClusterPlot(double[][] data, int[] clusterAssignments, double[][] centroids, String title, VisualizationMode mode) {
        try {
            switch (mode) {
                case ASCII:
                    ClusterPlot asciiCluster = new ClusterPlot(data, clusterAssignments, centroids);
                    asciiCluster.setTitle(title);
                    return asciiCluster;
                    
                case XCHART:
                    XChartClusterPlot xchartCluster = new XChartClusterPlot(data, clusterAssignments, centroids);
                    xchartCluster.setTitle(title);
                    return xchartCluster;
                    
                case AUTO:
                default:
                    // Try XChart first, fallback to ASCII on error
                    try {
                        XChartClusterPlot autoCluster = new XChartClusterPlot(data, clusterAssignments, centroids);
                        autoCluster.setTitle(title);
                        return autoCluster;
                    } catch (Exception e) {
                        System.out.println("⚠️ XChart unavailable, using ASCII mode: " + e.getMessage());
                        ClusterPlot fallbackCluster = new ClusterPlot(data, clusterAssignments, centroids);
                        fallbackCluster.setTitle(title);
                        return fallbackCluster;
                    }
            }
        } catch (Exception e) {
            System.err.println("❌ Error creating cluster plot: " + e.getMessage());
            // Emergency fallback
            return new ClusterPlot(data, clusterAssignments, centroids);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // REGRESSION VISUALIZATION METHODS
    // ═══════════════════════════════════════════════════════════════════
    
    /**
     * Create a regression scatter plot using default mode
     */
    public static Visualization createRegressionPlot(double[] actual, double[] predicted) {
        return createDualModeScatterPlot(actual, predicted, "Regression Analysis", defaultMode);
    }
    
    /**
     * Create a regression scatter plot with title using default mode
     */
    public static Visualization createRegressionPlot(double[] actual, double[] predicted, String title) {
        return createDualModeScatterPlot(actual, predicted, title, defaultMode);
    }
    
    /**
     * Create a dual-mode regression scatter plot with full control
     */
    public static Visualization createDualModeScatterPlot(double[] actual, double[] predicted, String title, VisualizationMode mode) {
        try {
            switch (mode) {
                case ASCII:
                    ScatterPlot asciiScatter = new ScatterPlot(actual, predicted);
                    asciiScatter.setTitle(title);
                    return asciiScatter;
                    
                case XCHART:
                    XChartScatterPlot xchartScatter = new XChartScatterPlot(actual, predicted);
                    xchartScatter.setTitle(title);
                    return xchartScatter;
                    
                case AUTO:
                default:
                    // Try XChart first, fallback to ASCII on error
                    try {
                        XChartScatterPlot autoScatter = new XChartScatterPlot(actual, predicted);
                        autoScatter.setTitle(title);
                        return autoScatter;
                    } catch (Exception e) {
                        System.out.println("⚠️ XChart unavailable, using ASCII mode: " + e.getMessage());
                        ScatterPlot fallbackScatter = new ScatterPlot(actual, predicted);
                        fallbackScatter.setTitle(title);
                        return fallbackScatter;
                    }
            }
        } catch (Exception e) {
            System.err.println("❌ Error creating regression plot: " + e.getMessage());
            // Emergency fallback
            return new ScatterPlot(actual, predicted);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // UTILITY METHODS
    // ═══════════════════════════════════════════════════════════════════
    
    /**
     * Display information about available visualization modes
     */
    public static void displayModeInfo() {
        System.out.println("\n📊 SuperML Visualization Modes:");
        System.out.println("═".repeat(50));
        System.out.println("🖥️  ASCII:  Terminal-based visualizations");
        System.out.println("🎨 XCHART: Professional GUI visualizations");
        System.out.println("🚀 AUTO:   XChart with ASCII fallback");
        System.out.println("═".repeat(50));
        System.out.println("Current default mode: " + defaultMode);
        System.out.println("Use VisualizationFactory.setDefaultMode() to change");
    }
    
    /**
     * Check if XChart GUI mode is available
     */
    public static boolean isXChartAvailable() {
        try {
            Class.forName("org.knowm.xchart.CategoryChart");
            return true;
        } catch (ClassNotFoundException e) {
            return false;
        }
    }
    
    /**
     * Get recommended mode based on system capabilities
     */
    public static VisualizationMode getRecommendedMode() {
        if (isXChartAvailable()) {
            return VisualizationMode.XCHART;
        } else {
            return VisualizationMode.ASCII;
        }
    }
}
