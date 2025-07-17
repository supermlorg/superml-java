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

// NOTE: AutoTrainer will be available in next release
//import org.superml.autotrainer.LinearModelAutoTrainer;
//import org.superml.autotrainer.LinearModelAutoTrainer.AutoTrainingResult;
//import org.superml.autotrainer.LinearModelAutoTrainer.ModelType;
//import org.superml.core.Regressor;
import org.superml.linear_model.Ridge;
import org.superml.linear_model.Lasso;

/**
 * Example demonstrating Ridge and Lasso regression with AutoTrainer
 * 
 * This example shows:
 * 1. Manual Ridge regression training
 * 2. Manual Lasso regression training  
 * 3. AutoTrainer optimization for Ridge
 * 4. AutoTrainer optimization for Lasso
 * 5. Performance comparison between methods
 * 
 * @author SuperML Team
 */
public class RidgeLassoExample {
    
    public static void main(String[] args) {
        System.out.println("=== Ridge and Lasso Regression with AutoTrainer ===\n");
        
        // Generate sample regression data
        double[][] X = {
            {1.0, 2.0, 3.0},
            {2.0, 3.0, 4.0},
            {3.0, 4.0, 5.0},
            {4.0, 5.0, 6.0},
            {5.0, 6.0, 7.0},
            {6.0, 7.0, 8.0},
            {7.0, 8.0, 9.0},
            {8.0, 9.0, 10.0},
            {9.0, 10.0, 11.0},
            {10.0, 11.0, 12.0}
        };
        
        // Target: y = 2*x1 + 3*x2 + 1*x3 + noise
        double[] y = {
            14.2, 19.8, 25.1, 30.5, 35.9, 41.2, 46.8, 52.1, 57.5, 62.9
        };
        
        // Test Ridge Regression
        testRidgeRegression(X, y);
        
        // Test Lasso Regression
        testLassoRegression(X, y);
        
        // Test AutoTrainer with Ridge (coming in next release)
        System.out.println("AutoTrainer Ridge optimization will be available in the next release.");
        
        // Test AutoTrainer with Lasso (coming in next release)
        System.out.println("AutoTrainer Lasso optimization will be available in the next release.");
    }
    
    private static void testRidgeRegression(double[][] X, double[] y) {
        System.out.println("--- Ridge Regression (Manual) ---");
        
        Ridge ridge = new Ridge(1.0);
        ridge.fit(X, y);
        
        double[] predictions = ridge.predict(X);
        double mse = calculateMSE(y, predictions);
        
        System.out.println("Ridge MSE: " + String.format("%.6f", mse));
        System.out.println("Ridge Coefficients: ");
        double[] coefficients = ridge.getCoefficients();
        for (int i = 0; i < coefficients.length; i++) {
            System.out.printf("  coef[%d]: %.6f%n", i, coefficients[i]);
        }
        System.out.println("Ridge Intercept: " + String.format("%.6f", ridge.getIntercept()));
        System.out.println();
    }
    
    private static void testLassoRegression(double[][] X, double[] y) {
        System.out.println("--- Lasso Regression (Manual) ---");
        
        Lasso lasso = new Lasso(0.1);
        lasso.fit(X, y);
        
        double[] predictions = lasso.predict(X);
        double mse = calculateMSE(y, predictions);
        
        System.out.println("Lasso MSE: " + String.format("%.6f", mse));
        System.out.println("Lasso Coefficients: ");
        double[] coefficients = lasso.getCoefficients();
        for (int i = 0; i < coefficients.length; i++) {
            System.out.printf("  coef[%d]: %.6f%n", i, coefficients[i]);
        }
        System.out.println("Lasso Intercept: " + String.format("%.6f", lasso.getIntercept()));
        System.out.println();
    }
    
    /*
    // AutoTrainer methods - Will be enabled when AutoTrainer module is available
    private static void testAutoTrainerRidge(double[][] X, double[] y) {
        System.out.println("--- AutoTrainer Ridge Optimization ---");
        
        LinearModelAutoTrainer autoTrainer = new LinearModelAutoTrainer();
        AutoTrainingResult result = autoTrainer.autoTrain(X, y, ModelType.RIDGE);
        
        System.out.println("Best Ridge Model Type: " + result.bestModelType);
        System.out.println("Best Ridge Score: " + String.format("%.6f", result.bestScore));
        System.out.println("Best Ridge Parameters: " + result.bestParameters);
        
        // Test predictions
        Regressor model = (Regressor) result.bestModel;
        double[] predictions = model.predict(X);
        double mse = calculateMSE(y, predictions);
        System.out.println("Best Ridge MSE: " + String.format("%.6f", mse));
        System.out.println("Optimization History: " + result.optimizationHistory.size() + " trials");
        System.out.println();
    }
    
    private static void testAutoTrainerLasso(double[][] X, double[] y) {
        System.out.println("--- AutoTrainer Lasso Optimization ---");
        
        LinearModelAutoTrainer autoTrainer = new LinearModelAutoTrainer();
        AutoTrainingResult result = autoTrainer.autoTrain(X, y, ModelType.LASSO);
        
        System.out.println("Best Lasso Model Type: " + result.bestModelType);
        System.out.println("Best Lasso Score: " + String.format("%.6f", result.bestScore));
        System.out.println("Best Lasso Parameters: " + result.bestParameters);
        
        // Test predictions
        Regressor model = (Regressor) result.bestModel;
        double[] predictions = model.predict(X);
        double mse = calculateMSE(y, predictions);
        System.out.println("Best Lasso MSE: " + String.format("%.6f", mse));
        System.out.println("Optimization History: " + result.optimizationHistory.size() + " trials");
        System.out.println();
    }
    */
    
    private static double calculateMSE(double[] actual, double[] predicted) {
        double mse = 0.0;
        for (int i = 0; i < actual.length; i++) {
            double error = actual[i] - predicted[i];
            mse += error * error;
        }
        return mse / actual.length;
    }
}
