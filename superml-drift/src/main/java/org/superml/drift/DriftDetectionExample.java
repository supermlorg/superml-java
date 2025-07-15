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

package org.superml.drift;

import java.util.Date;
import java.util.Random;

/**
 * Comprehensive example demonstrating the SuperML drift detection system.
 * Shows both data drift and concept drift detection with real-time monitoring.
 */
public class DriftDetectionExample {
    
    public static void main(String[] args) {
        System.out.println("🚀 SuperML Drift Detection System Demo");
        System.out.println("=======================================\n");
        
        // Create configuration for sensitive drift detection
        DriftConfig config = DriftConfig.sensitiveDetection();
        System.out.println("📋 Using sensitive detection configuration:");
        System.out.println(config + "\n");
        
        // Initialize dashboard
        DriftDashboard dashboard = new DriftDashboard(config);
        dashboard.startMonitoring();
        
        try {
            // 1. Data Drift Detection Example
            demonstrateDataDrift(dashboard);
            
            Thread.sleep(2000); // Pause between demonstrations
            
            // 2. Concept Drift Detection Example
            demonstrateConceptDrift(dashboard);
            
            Thread.sleep(2000); // Pause before final report
            
            // 3. Generate final monitoring report
            generateFinalReport(dashboard);
            
        } catch (Exception e) {
            System.err.println("❌ Error during demonstration: " + e.getMessage());
            e.printStackTrace();
        } finally {
            dashboard.stopMonitoring();
        }
    }
    
    /**
     * Demonstrate data drift detection across multiple features.
     */
    private static void demonstrateDataDrift(DriftDashboard dashboard) {
        System.out.println("📊 DEMONSTRATING DATA DRIFT DETECTION");
        System.out.println("=====================================\n");
        
        Random random = new Random(42); // Fixed seed for reproducibility
        
        // Generate reference data (training distribution)
        double[] referenceAge = generateNormalData(1000, 35.0, 10.0, random);
        double[] referenceIncome = generateNormalData(1000, 50000, 15000, random);
        double[] referenceCreditScore = generateNormalData(1000, 650, 100, random);
        
        System.out.println("-> Generated reference data (training distribution)");
        System.out.printf("   Age: μ=%.1f, σ=%.1f\n", mean(referenceAge), stdDev(referenceAge));
        System.out.printf("   Income: μ=%.0f, σ=%.0f\n", mean(referenceIncome), stdDev(referenceIncome));
        System.out.printf("   Credit Score: μ=%.0f, σ=%.0f\n\n", mean(referenceCreditScore), stdDev(referenceCreditScore));
        
        // Test 1: No drift (same distribution)
        System.out.println("🔍 Test 1: No drift scenario (same distribution)");
        double[] currentAge1 = generateNormalData(500, 35.0, 10.0, random);
        double[] currentIncome1 = generateNormalData(500, 50000, 15000, random);
        double[] currentCredit1 = generateNormalData(500, 650, 100, random);
        
        dashboard.checkDataDrift(referenceAge, currentAge1, "age");
        dashboard.checkDataDrift(referenceIncome, currentIncome1, "income");
        dashboard.checkDataDrift(referenceCreditScore, currentCredit1, "credit_score");
        
        // Test 2: Moderate drift (shifted mean)
        System.out.println("\n🔍 Test 2: Moderate drift scenario (shifted distributions)");
        double[] currentAge2 = generateNormalData(500, 40.0, 10.0, random); // Age shifted up
        double[] currentIncome2 = generateNormalData(500, 45000, 15000, random); // Income shifted down
        double[] currentCredit2 = generateNormalData(500, 650, 100, random); // No change
        
        dashboard.checkDataDrift(referenceAge, currentAge2, "age");
        dashboard.checkDataDrift(referenceIncome, currentIncome2, "income");
        dashboard.checkDataDrift(referenceCreditScore, currentCredit2, "credit_score");
        
        // Test 3: Severe drift (different distribution)
        System.out.println("\n🔍 Test 3: Severe drift scenario (different distributions)");
        double[] currentAge3 = generateNormalData(500, 25.0, 5.0, random); // Younger, less variance
        double[] currentIncome3 = generateBimodalData(500, random); // Bimodal distribution
        double[] currentCredit3 = generateUniformData(500, 500, 800, random); // Uniform distribution
        
        dashboard.checkDataDrift(referenceAge, currentAge3, "age");
        dashboard.checkDataDrift(referenceIncome, currentIncome3, "income");
        dashboard.checkDataDrift(referenceCreditScore, currentCredit3, "credit_score");
        
        dashboard.printDashboard();
    }
    
    /**
     * Demonstrate concept drift detection with simulated model performance degradation.
     */
    private static void demonstrateConceptDrift(DriftDashboard dashboard) {
        System.out.println("🎯 DEMONSTRATING CONCEPT DRIFT DETECTION");
        System.out.println("========================================\n");
        
        Random random = new Random(42);
        
        // Set baseline accuracy from training
        double baselineAccuracy = 0.85;
        dashboard.setBaselineAccuracy(baselineAccuracy);
        System.out.printf("📊 Baseline accuracy set to: %.3f\n\n", baselineAccuracy);
        
        // Phase 1: Stable performance (no drift)
        System.out.println("🔍 Phase 1: Stable model performance (no drift expected)");
        simulateModelPredictions(dashboard, 200, 0.85, 0.02, random, "Stable Phase");
        
        // Phase 2: Gradual performance degradation
        System.out.println("\n🔍 Phase 2: Gradual performance degradation");
        for (int batch = 0; batch < 10; batch++) {
            double currentAccuracy = 0.85 - (batch * 0.03); // Gradual decline
            simulateModelPredictions(dashboard, 50, currentAccuracy, 0.02, random, 
                                   String.format("Degradation Batch %d", batch + 1));
        }
        
        // Phase 3: Sudden performance drop (concept drift)
        System.out.println("\n🔍 Phase 3: Sudden performance drop (concept drift)");
        simulateModelPredictions(dashboard, 200, 0.55, 0.05, random, "Sudden Drop Phase");
        
        // Phase 4: Recovery after model retraining
        System.out.println("\n🔍 Phase 4: Model recovery (after retraining)");
        dashboard.setBaselineAccuracy(0.82); // New baseline after retraining
        simulateModelPredictions(dashboard, 150, 0.82, 0.02, random, "Recovery Phase");
        
        dashboard.printDashboard();
    }
    
    /**
     * Generate final monitoring report and export data.
     */
    private static void generateFinalReport(DriftDashboard dashboard) {
        System.out.println("📋 GENERATING FINAL MONITORING REPORT");
        System.out.println("=====================================\n");
        
        try {
            // Generate comprehensive report
            DriftDashboard.MonitoringReport report = dashboard.generateReport(new Date());
            
            // Export to JSON
            String jsonPath = "drift_monitoring_report.json";
            dashboard.exportReportToJson(report, jsonPath);
            
            // Export events to CSV
            String csvPath = "drift_events.csv";
            dashboard.exportEventsToCSV(csvPath);
            
            // Print summary statistics
            System.out.println("📊 SESSION SUMMARY:");
            System.out.printf("   Duration: %.1f minutes\n", report.sessionDurationMs / 60000.0);
            System.out.printf("   Data Checks: %d (%.1f%% drift rate)\n", 
                             report.totalDataChecks, report.dataDriftRate * 100);
            System.out.printf("   Concept Checks: %d (%.1f%% drift rate)\n", 
                             report.totalConceptChecks, report.conceptDriftRate * 100);
            System.out.printf("   Total Alerts: %d\n", report.totalAlerts);
            System.out.printf("   Final Accuracy: %.3f\n", report.conceptStatistics.overallAccuracy);
            
        } catch (Exception e) {
            System.err.println("❌ Error generating report: " + e.getMessage());
        }
    }
    
    /**
     * Simulate model predictions with varying accuracy and confidence.
     */
    private static void simulateModelPredictions(DriftDashboard dashboard, int numPredictions, 
                                                double targetAccuracy, double accuracyVariance, 
                                                Random random, String phaseName) {
        System.out.printf("   Simulating %d predictions for %s (target accuracy: %.3f)\n", 
                         numPredictions, phaseName, targetAccuracy);
        
        for (int i = 0; i < numPredictions; i++) {
            // Generate prediction and ground truth
            double prediction = random.nextGaussian() * 0.3 + 0.5; // Predictions around 0.5
            prediction = Math.max(0.0, Math.min(1.0, prediction)); // Clamp to [0,1]
            
            // Determine if prediction is correct based on target accuracy
            double currentAccuracy = targetAccuracy + (random.nextGaussian() * accuracyVariance);
            boolean isCorrect = random.nextDouble() < currentAccuracy;
            
            Double actualLabel = isCorrect ? prediction + (random.nextGaussian() * 0.1) : 
                                           prediction + (random.nextGaussian() * 0.5);
            actualLabel = Math.max(0.0, Math.min(1.0, actualLabel));
            
            // Generate confidence score (higher when more accurate)
            double confidence = currentAccuracy + (random.nextGaussian() * 0.1);
            confidence = Math.max(0.1, Math.min(0.99, confidence));
            
            // Record prediction with dashboard
            dashboard.checkConceptDrift(prediction, actualLabel, confidence);
        }
    }
    
    // Helper methods for data generation
    
    private static double[] generateNormalData(int size, double mean, double stdDev, Random random) {
        double[] data = new double[size];
        for (int i = 0; i < size; i++) {
            data[i] = mean + (random.nextGaussian() * stdDev);
        }
        return data;
    }
    
    private static double[] generateBimodalData(int size, Random random) {
        double[] data = new double[size];
        for (int i = 0; i < size; i++) {
            if (random.nextBoolean()) {
                data[i] = 30000 + (random.nextGaussian() * 10000); // Lower mode
            } else {
                data[i] = 80000 + (random.nextGaussian() * 20000); // Higher mode
            }
        }
        return data;
    }
    
    private static double[] generateUniformData(int size, double min, double max, Random random) {
        double[] data = new double[size];
        for (int i = 0; i < size; i++) {
            data[i] = min + (random.nextDouble() * (max - min));
        }
        return data;
    }
    
    private static double mean(double[] data) {
        double sum = 0.0;
        for (double value : data) {
            sum += value;
        }
        return sum / data.length;
    }
    
    private static double stdDev(double[] data) {
        double mean = mean(data);
        double variance = 0.0;
        for (double value : data) {
            variance += Math.pow(value - mean, 2);
        }
        return Math.sqrt(variance / data.length);
    }
}
