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

import org.superml.visualization.Visualization;

/**
 * Scatter plot visualization for regression analysis
 * Shows actual vs predicted values with ASCII-based plotting
 */
public class ScatterPlot implements Visualization {
    
    private final double[] actual;
    private final double[] predicted;
    private String title = "Actual vs Predicted";
    private int width = 60;
    private int height = 20;
    
    public ScatterPlot(double[] actual, double[] predicted) {
        this.actual = actual.clone();
        this.predicted = predicted.clone();
    }
    
    @Override
    public void display() {
        System.out.println("📈 " + title);
        System.out.println("=" + "=".repeat(title.length() + 3));
        
        displayScatterPlot();
        displayMetrics();
    }
    
    private void displayScatterPlot() {
        // Find data range
        double minActual = Double.MAX_VALUE;
        double maxActual = Double.MIN_VALUE;
        double minPred = Double.MAX_VALUE;
        double maxPred = Double.MIN_VALUE;
        
        for (int i = 0; i < actual.length; i++) {
            minActual = Math.min(minActual, actual[i]);
            maxActual = Math.max(maxActual, actual[i]);
            minPred = Math.min(minPred, predicted[i]);
            maxPred = Math.max(maxPred, predicted[i]);
        }
        
        double overallMin = Math.min(minActual, minPred);
        double overallMax = Math.max(maxActual, maxPred);
        
        // Create ASCII plot
        char[][] plot = new char[height][width];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                plot[i][j] = ' ';
            }
        }
        
        // Draw diagonal line (perfect prediction)
        for (int i = 0; i < Math.min(height, width); i++) {
            int x = (int) ((double) i / height * width);
            if (x < width) {
                plot[height - 1 - i][x] = '\\';
            }
        }
        
        // Plot points
        for (int i = 0; i < actual.length; i++) {
            int x = (int) ((actual[i] - overallMin) / (overallMax - overallMin) * (width - 1));
            int y = (int) ((predicted[i] - overallMin) / (overallMax - overallMin) * (height - 1));
            
            y = height - 1 - y; // Flip Y axis
            
            if (x >= 0 && x < width && y >= 0 && y < height) {
                if (plot[y][x] == ' ' || plot[y][x] == '\\') {
                    plot[y][x] = '*';
                } else {
                    plot[y][x] = '+'; // Multiple points
                }
            }
        }
        
        System.out.println("\nScatter Plot (Actual vs Predicted):");
        System.out.printf("Y-axis: Predicted [%.3f to %.3f]\n", overallMin, overallMax);
        System.out.printf("X-axis: Actual [%.3f to %.3f]\n", overallMin, overallMax);
        System.out.println("Legend: * = data point, + = multiple points, \\ = perfect prediction\n");
        
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
    
    private void displayMetrics() {
        double mse = calculateMSE();
        double rmse = Math.sqrt(mse);
        double mae = calculateMAE();
        double r2 = calculateR2();
        
        System.out.println("📊 Regression Metrics:");
        System.out.println("======================");
        System.out.printf("Mean Squared Error (MSE):  %.6f\n", mse);
        System.out.printf("Root Mean Squared Error:   %.6f\n", rmse);
        System.out.printf("Mean Absolute Error (MAE): %.6f\n", mae);
        System.out.printf("R² Score:                  %.6f\n", r2);
        System.out.printf("Number of samples:         %d\n", actual.length);
        System.out.println();
    }
    
    private double calculateMSE() {
        double sum = 0;
        for (int i = 0; i < actual.length; i++) {
            double diff = actual[i] - predicted[i];
            sum += diff * diff;
        }
        return sum / actual.length;
    }
    
    private double calculateMAE() {
        double sum = 0;
        for (int i = 0; i < actual.length; i++) {
            sum += Math.abs(actual[i] - predicted[i]);
        }
        return sum / actual.length;
    }
    
    private double calculateR2() {
        // Calculate mean of actual values
        double meanActual = 0;
        for (double value : actual) {
            meanActual += value;
        }
        meanActual /= actual.length;
        
        // Calculate total sum of squares and residual sum of squares
        double totalSumSquares = 0;
        double residualSumSquares = 0;
        
        for (int i = 0; i < actual.length; i++) {
            totalSumSquares += Math.pow(actual[i] - meanActual, 2);
            residualSumSquares += Math.pow(actual[i] - predicted[i], 2);
        }
        
        return 1 - (residualSumSquares / totalSumSquares);
    }
    
    // Getters and setters
    
    public void setDimensions(int width, int height) {
        this.width = Math.max(20, width);
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
        return String.format("ScatterPlot: %d samples, MSE=%.6f, R²=%.6f", 
            actual.length, calculateMSE(), calculateR2());
    }
}
