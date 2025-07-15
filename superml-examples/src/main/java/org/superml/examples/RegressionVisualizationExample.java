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

import java.util.Random;

/**
 * Example demonstrating XChart regression visualization
 * Shows how to create and display scatter plots for regression analysis
 */
public class RegressionVisualizationExample {
    
    public static void main(String[] args) {
        System.out.println("📈 SuperML Regression Visualization Example");
        System.out.println("=" + "=".repeat(47));
        
        // Generate sample regression data
        double[] actualValues = generateActualValues();
        double[] predictedValues = generatePredictedValues(actualValues);
        
        // Demo 1: ASCII mode
        System.out.println("\n1️⃣ ASCII Regression Visualization:");
        Visualization asciiPlot = VisualizationFactory.createDualModeScatterPlot(
            actualValues, predictedValues, 
            "ASCII Regression Plot", 
            VisualizationFactory.VisualizationMode.ASCII
        );
        asciiPlot.display();
        
        // Demo 2: XChart GUI mode (if available)
        System.out.println("\n2️⃣ XChart GUI Regression Visualization:");
        if (VisualizationFactory.isXChartAvailable()) {
            Visualization xchartPlot = VisualizationFactory.createDualModeScatterPlot(
                actualValues, predictedValues, 
                "XChart GUI Regression Analysis", 
                VisualizationFactory.VisualizationMode.XCHART
            );
            xchartPlot.display();
        } else {
            System.out.println("⚠️ XChart not available - add XChart dependency to see GUI visualization");
        }
        
        // Demo 3: Auto mode (preferred)
        System.out.println("\n3️⃣ Auto Mode (XChart with ASCII fallback):");
        Visualization autoPlot = VisualizationFactory.createRegressionPlot(
            actualValues, predictedValues, 
            "Auto Mode Regression Analysis"
        );
        autoPlot.display();
        
        // Demo 4: Comparison with different quality predictions
        System.out.println("\n4️⃣ Comparison: Perfect vs Noisy Predictions:");
        
        // Perfect predictions
        double[] perfectPredictions = actualValues.clone();
        Visualization perfectPlot = VisualizationFactory.createRegressionPlot(
            actualValues, perfectPredictions, 
            "Perfect Predictions (R² = 1.0)"
        );
        
        // Very noisy predictions
        double[] noisyPredictions = generateNoisyPredictions(actualValues);
        Visualization noisyPlot = VisualizationFactory.createRegressionPlot(
            actualValues, noisyPredictions, 
            "Noisy Predictions (Low R²)"
        );
        
        System.out.println("🟢 Perfect predictions visualization:");
        perfectPlot.display();
        
        System.out.println("🔴 Noisy predictions visualization:");
        noisyPlot.display();
        
        // Show mode information
        VisualizationFactory.displayModeInfo();
    }
    
    /**
     * Generate sample actual values (target variable)
     */
    private static double[] generateActualValues() {
        Random random = new Random(42); // Fixed seed for reproducibility
        double[] values = new double[50];
        
        for (int i = 0; i < values.length; i++) {
            // Generate values following y = 2x + 1 + noise
            double x = i * 0.5;
            values[i] = 2 * x + 1 + random.nextGaussian() * 0.5;
        }
        
        return values;
    }
    
    /**
     * Generate realistic predicted values (with some prediction error)
     */
    private static double[] generatePredictedValues(double[] actual) {
        Random random = new Random(123); // Different seed for prediction noise
        double[] predicted = new double[actual.length];
        
        for (int i = 0; i < actual.length; i++) {
            // Add some prediction error (model isn't perfect)
            predicted[i] = actual[i] + random.nextGaussian() * 1.2;
        }
        
        return predicted;
    }
    
    /**
     * Generate very noisy predictions to demonstrate poor model performance
     */
    private static double[] generateNoisyPredictions(double[] actual) {
        Random random = new Random(789);
        double[] noisy = new double[actual.length];
        
        for (int i = 0; i < actual.length; i++) {
            // Add significant noise to simulate poor model
            noisy[i] = actual[i] + random.nextGaussian() * 5.0 + random.nextDouble() * 10 - 5;
        }
        
        return noisy;
    }
}
