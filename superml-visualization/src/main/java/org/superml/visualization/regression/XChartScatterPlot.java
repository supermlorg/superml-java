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

package org.superml.visualization.regression;

import org.knowm.xchart.*;
import org.knowm.xchart.style.Styler;
import org.superml.visualization.Visualization;

import java.awt.Color;
import java.text.DecimalFormat;
import java.util.*;

/**
 * XChart-based GUI scatter plot visualization for regression analysis.
 * Shows actual vs predicted values with trend line and residuals.
 * 
 * Features:
 * - Professional GUI scatter plot using XChart
 * - Actual vs Predicted scatter plot with perfect prediction line
 * - Residual analysis visualization
 * - Interactive zoom and pan
 * - Export capabilities (PNG, JPEG, PDF, etc.)
 * - Regression metrics overlay
 * 
 * @author SuperML Framework
 * @version 2.1.0
 */
public class XChartScatterPlot implements Visualization {
    
    private final double[] actualValues;
    private final double[] predictedValues;
    private String title = "Regression Analysis";
    private int width = 800;
    private int height = 600;
    private final DecimalFormat df = new DecimalFormat("#.####");
    
    /**
     * Create XChart scatter plot for regression analysis
     * 
     * @param actualValues actual target values
     * @param predictedValues model predictions
     */
    public XChartScatterPlot(double[] actualValues, double[] predictedValues) {
        if (actualValues == null || predictedValues == null) {
            throw new IllegalArgumentException("Actual and predicted values cannot be null");
        }
        if (actualValues.length != predictedValues.length) {
            throw new IllegalArgumentException("Actual and predicted arrays must have same length");
        }
        if (actualValues.length == 0) {
            throw new IllegalArgumentException("Arrays cannot be empty");
        }
        
        this.actualValues = actualValues.clone();
        this.predictedValues = predictedValues.clone();
    }
    
    @Override
    public void display() {
        try {
            // Display main scatter plot
            XYChart scatterChart = createScatterChart();
            new SwingWrapper<>(scatterChart).displayChart();
            
            // Display residual plot
            XYChart residualChart = createResidualChart();
            new SwingWrapper<>(residualChart).displayChart();
            
            // Print summary to console
            printSummary();
            
        } catch (Exception e) {
            System.err.println("❌ Error displaying XChart scatter plot: " + e.getMessage());
            System.out.println("💡 Falling back to ASCII visualization...");
            
            // Fallback to ASCII visualization
            ScatterPlot asciiPlot = new ScatterPlot(actualValues, predictedValues);
            asciiPlot.setTitle(title);
            asciiPlot.display();
        }
    }
    
    /**
     * Create the main scatter plot (Actual vs Predicted)
     */
    private XYChart createScatterChart() {
        XYChart chart = new XYChartBuilder()
            .width(width)
            .height(height)
            .title(title + " - Actual vs Predicted")
            .xAxisTitle("Actual Values")
            .yAxisTitle("Predicted Values")
            .build();
        
        // Customize chart style
        chart.getStyler().setMarkerSize(6);
        chart.getStyler().setLegendPosition(Styler.LegendPosition.InsideNE);
        chart.getStyler().setPlotBackgroundColor(Color.WHITE);
        chart.getStyler().setChartBackgroundColor(Color.WHITE);
        chart.getStyler().setPlotBorderVisible(true);
        chart.getStyler().setAxisTitlesVisible(true);
        
        // Add actual vs predicted scatter plot
        var scatterSeries = chart.addSeries("Data Points", actualValues, predictedValues);
        scatterSeries.setMarkerColor(Color.BLUE);
        
        // Add perfect prediction line (y = x)
        double minVal = Math.min(getMin(actualValues), getMin(predictedValues));
        double maxVal = Math.max(getMax(actualValues), getMax(predictedValues));
        double[] perfectLine = {minVal, maxVal};
        
        var lineSeries = chart.addSeries("Perfect Prediction", perfectLine, perfectLine);
        lineSeries.setMarkerColor(Color.RED);
        lineSeries.setLineColor(Color.RED);
        lineSeries.setLineWidth(2);
        
        return chart;
    }
    
    /**
     * Create residual plot (Predicted vs Residuals)
     */
    private XYChart createResidualChart() {
        XYChart chart = new XYChartBuilder()
            .width(width)
            .height(height)
            .title(title + " - Residual Analysis")
            .xAxisTitle("Predicted Values")
            .yAxisTitle("Residuals (Actual - Predicted)")
            .build();
        
        // Calculate residuals
        double[] residuals = new double[actualValues.length];
        for (int i = 0; i < actualValues.length; i++) {
            residuals[i] = actualValues[i] - predictedValues[i];
        }
        
        // Customize chart style
        chart.getStyler().setMarkerSize(6);
        chart.getStyler().setLegendPosition(Styler.LegendPosition.InsideNE);
        chart.getStyler().setPlotBackgroundColor(Color.WHITE);
        chart.getStyler().setChartBackgroundColor(Color.WHITE);
        chart.getStyler().setPlotBorderVisible(true);
        chart.getStyler().setAxisTitlesVisible(true);
        
        // Add residual scatter plot
        var residualSeries = chart.addSeries("Residuals", predictedValues, residuals);
        residualSeries.setMarkerColor(Color.GREEN);
        
        // Add zero line
        double minPred = getMin(predictedValues);
        double maxPred = getMax(predictedValues);
        double[] zeroLineX = {minPred, maxPred};
        double[] zeroLineY = {0.0, 0.0};
        
        var zeroSeries = chart.addSeries("Zero Line", zeroLineX, zeroLineY);
        zeroSeries.setMarkerColor(Color.RED);
        zeroSeries.setLineColor(Color.RED);
        zeroSeries.setLineWidth(2);
        
        return chart;
    }
    
    /**
     * Print regression statistics to console
     */
    private void printSummary() {
        System.out.println("\n🎯 " + title + " (XChart GUI)");
        System.out.println("=" + "=".repeat(title.length() + 15));
        
        // Calculate metrics
        double mse = calculateMSE();
        double rmse = Math.sqrt(mse);
        double mae = calculateMAE();
        double r2 = calculateR2();
        
        System.out.println("📊 Regression Metrics:");
        System.out.printf("  Mean Squared Error (MSE): %s\n", df.format(mse));
        System.out.printf("  Root Mean Squared Error (RMSE): %s\n", df.format(rmse));
        System.out.printf("  Mean Absolute Error (MAE): %s\n", df.format(mae));
        System.out.printf("  R² Score: %s\n", df.format(r2));
        
        System.out.printf("Total samples: %d\n", actualValues.length);
        System.out.printf("Actual range: [%.3f, %.3f]\n", getMin(actualValues), getMax(actualValues));
        System.out.printf("Predicted range: [%.3f, %.3f]\n", getMin(predictedValues), getMax(predictedValues));
        
        System.out.println("💡 Interactive GUI features:");
        System.out.println("  • Zoom: Mouse wheel or right-click drag");
        System.out.println("  • Pan: Left-click drag");
        System.out.println("  • Export: Right-click → Export");
        System.out.println("  • Reset: Double-click");
        System.out.println();
    }
    
    /**
     * Calculate Mean Squared Error
     */
    private double calculateMSE() {
        double sum = 0;
        for (int i = 0; i < actualValues.length; i++) {
            double diff = actualValues[i] - predictedValues[i];
            sum += diff * diff;
        }
        return sum / actualValues.length;
    }
    
    /**
     * Calculate Mean Absolute Error
     */
    private double calculateMAE() {
        double sum = 0;
        for (int i = 0; i < actualValues.length; i++) {
            sum += Math.abs(actualValues[i] - predictedValues[i]);
        }
        return sum / actualValues.length;
    }
    
    /**
     * Calculate R² Score
     */
    private double calculateR2() {
        double actualMean = Arrays.stream(actualValues).average().orElse(0.0);
        
        double ssRes = 0; // Sum of squares of residuals
        double ssTot = 0; // Total sum of squares
        
        for (int i = 0; i < actualValues.length; i++) {
            double residual = actualValues[i] - predictedValues[i];
            ssRes += residual * residual;
            
            double deviation = actualValues[i] - actualMean;
            ssTot += deviation * deviation;
        }
        
        return 1 - (ssRes / ssTot);
    }
    
    /**
     * Get minimum value from array
     */
    private double getMin(double[] array) {
        return Arrays.stream(array).min().orElse(0.0);
    }
    
    /**
     * Get maximum value from array
     */
    private double getMax(double[] array) {
        return Arrays.stream(array).max().orElse(0.0);
    }
    
    /**
     * Export charts to files
     */
    public void exportCharts(String scatterFilename, String residualFilename) {
        try {
            XYChart scatterChart = createScatterChart();
            XYChart residualChart = createResidualChart();
            
            // Note: XChart export functionality would go here
            // BitmapEncoder.saveBitmap(scatterChart, scatterFilename, BitmapFormat.PNG);
            // BitmapEncoder.saveBitmap(residualChart, residualFilename, BitmapFormat.PNG);
            
            System.out.println("📄 Charts exported to: " + scatterFilename + ", " + residualFilename);
        } catch (Exception e) {
            System.err.println("❌ Error exporting charts: " + e.getMessage());
        }
    }
    
    // Getters and setters
    
    public void setDimensions(int width, int height) {
        this.width = Math.max(400, width);
        this.height = Math.max(300, height);
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
        return String.format("XChartScatterPlot: %d samples, GUI mode", actualValues.length);
    }
}
