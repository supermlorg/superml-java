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

import org.superml.drift.DriftConfig;
import org.superml.drift.DataDriftDetector;
import org.superml.drift.DDMDriftDetector;
import org.superml.drift.EDDMDriftDetector;
import org.superml.linear_model.LogisticRegression;
import org.superml.tree.RandomForest;
import org.superml.metrics.Metrics;

/**
 * Drift Detection Example
 * Demonstrates data drift and concept drift detection capabilities
 */
public class DriftDetectionExample {
    
    public static void main(String[] args) {
        System.out.println("=== SuperML 2.0.0 - Drift Detection Example ===\n");
        
        try {
            // Example 1: Data drift detection
            demonstrateDataDrift();
            
            // Example 2: Concept drift detection
            demonstrateConceptDrift();
            
            // Example 3: Model performance monitoring
            demonstrateModelMonitoring();
            
            // Example 4: Adaptive learning with drift
            demonstrateAdaptiveLearning();
            
        } catch (Exception e) {
            System.err.println("Error in DriftDetectionExample: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void demonstrateDataDrift() {
        System.out.println("1. Data Drift Detection...");
        
        try {
            // Generate baseline data (training distribution)
            double[][] baselineData = generateStableData(500, 4);
            System.out.println("   Generated baseline data: " + baselineData.length + " samples");
            
            // Simulate data drift scenarios
            System.out.println("\n   Testing different drift scenarios:");
            
            // Scenario 1: No drift
            double[][] noDriftData = generateStableData(200, 4);
            boolean drift1 = detectDataDrift(baselineData, noDriftData);
            System.out.printf("   No Drift Scenario: %s\n", drift1 ? "DRIFT DETECTED ⚠️" : "No drift detected ✅");
            
            // Scenario 2: Gradual drift
            double[][] gradualDriftData = generateDriftingData(200, 4, 0.3);
            boolean drift2 = detectDataDrift(baselineData, gradualDriftData);
            System.out.printf("   Gradual Drift: %s\n", drift2 ? "DRIFT DETECTED ⚠️" : "No drift detected ✅");
            
            // Scenario 3: Sudden drift
            double[][] suddenDriftData = generateDriftingData(200, 4, 1.0);
            boolean drift3 = detectDataDrift(baselineData, suddenDriftData);
            System.out.printf("   Sudden Drift: %s\n", drift3 ? "DRIFT DETECTED ⚠️" : "No drift detected ✅");
            
        } catch (Exception e) {
            System.err.println("   Error in data drift detection: " + e.getMessage());
        }
    }
    
    private static void demonstrateConceptDrift() {
        System.out.println("\n2. Concept Drift Detection...");
        
        try {
            DriftConfig config = DriftConfig.balancedDetection();
            DDMDriftDetector ddmDetector = new DDMDriftDetector(config);
            EDDMDriftDetector eddmDetector = new EDDMDriftDetector(config);
            
            // Generate training data
            double[][] XTrain = generateClassificationData(400, 3);
            double[] yTrain = generateClassificationLabels(400, XTrain, false);
            
            // Train initial model
            LogisticRegression model = new LogisticRegression();
            model.fit(XTrain, yTrain);
            System.out.println("   Trained initial model on " + XTrain.length + " samples");
            
            // Simulate streaming data with concept drift
            System.out.println("\n   Monitoring model performance over time:");
            System.out.println("   Batch | Accuracy | DDM Status | EDDM Status");
            System.out.println("   ------|----------|-----------|------------");
            
            for (int batch = 1; batch <= 15; batch++) {
                // Generate new batch data
                boolean isDrifted = batch > 8; // Introduce concept drift after batch 8
                double[][] XNew = generateClassificationData(20, 3);
                double[] yNew = generateClassificationLabels(20, XNew, isDrifted);
                
                // Evaluate model on each prediction
                double[] predictions = model.predict(XNew);
                double accuracy = Metrics.accuracy(yNew, predictions);
                
                // Process each prediction for drift detection
                DDMDriftDetector.DDMResult ddmResult = null;
                EDDMDriftDetector.EDDMResult eddmResult = null;
                
                for (int i = 0; i < predictions.length; i++) {
                    boolean isError = predictions[i] != yNew[i];
                    ddmResult = ddmDetector.detectDrift(isError);
                    eddmResult = eddmDetector.detectDrift(isError);
                }
                
                String ddmStatus = ddmResult.isDrift ? "DRIFT ⚠️" : (ddmResult.isWarning ? "WARN ⚡" : "OK ✅");
                String eddmStatus = eddmResult.isDrift ? "DRIFT ⚠️" : (eddmResult.isWarning ? "WARN ⚡" : "OK ✅");
                
                System.out.printf("   %5d | %8.3f | %s | %s\n", batch, accuracy, ddmStatus, eddmStatus);
                
                // Simulate model retraining when drift is detected
                if (ddmResult.isDrift && batch > 8) {
                    System.out.println("         🔄 DDM triggered model retraining...");
                    ddmDetector.reset();
                    eddmDetector.reset();
                    model.fit(XNew, yNew); // Retrain on new data
                }
            }
            
        } catch (Exception e) {
            System.err.println("   Error in concept drift detection: " + e.getMessage());
        }
    }
    
    private static void demonstrateModelMonitoring() {
        System.out.println("\n3. Model Performance Monitoring...");
        
        try {
            // Train models on stable data
            double[][] XTrain = generateClassificationData(300, 4);
            double[] yTrain = generateClassificationLabels(300, XTrain, false);
            
            LogisticRegression lr = new LogisticRegression();
            RandomForest rf = new RandomForest();
            lr.fit(XTrain, yTrain);
            rf.fit(XTrain, yTrain);
            
            System.out.println("   Monitoring multiple models over time:");
            System.out.println("   Time | LogReg Acc | RF Acc | Alert");
            System.out.println("   -----|-----------|--------|-------");
            
            for (int time = 1; time <= 8; time++) {
                // Generate test data with increasing drift
                double driftLevel = time > 5 ? (time - 5) * 0.3 : 0.0;
                double[][] XTest = generateDriftingData(50, 4, driftLevel);
                double[] yTest = generateClassificationLabels(50, XTest, time > 5);
                
                // Evaluate both models
                double[] lrPred = lr.predict(XTest);
                double[] rfPred = rf.predict(XTest);
                double lrAcc = Metrics.accuracy(yTest, lrPred);
                double rfAcc = Metrics.accuracy(yTest, rfPred);
                
                // Alert if both models degrade significantly
                boolean alert = (lrAcc < 0.6 && rfAcc < 0.6);
                String alertStatus = alert ? "🚨 ALERT" : "✅ OK";
                
                System.out.printf("   %4d | %9.3f | %6.3f | %s\n", time, lrAcc, rfAcc, alertStatus);
            }
            
        } catch (Exception e) {
            System.err.println("   Error in model monitoring: " + e.getMessage());
        }
    }
    
    private static void demonstrateAdaptiveLearning() {
        System.out.println("\n4. Adaptive Learning with Drift...");
        
        try {
            // Initial training
            double[][] XTrain = generateClassificationData(200, 3);
            double[] yTrain = generateClassificationLabels(200, XTrain, false);
            
            RandomForest adaptiveModel = new RandomForest();
            adaptiveModel.fit(XTrain, yTrain);
            
            System.out.println("   Adaptive learning simulation:");
            System.out.println("   Stage | Data Type | Accuracy | Action");
            System.out.println("   ------|-----------|----------|--------");
            
            String[] stages = {"Initial", "Stable", "Drift Start", "Full Drift", "Adaptation", "Recovery"};
            double[] accuracies = new double[stages.length];
            
            for (int stage = 0; stage < stages.length; stage++) {
                // Generate appropriate data for each stage
                boolean isDrifted = stage >= 2;
                double driftLevel = stage >= 3 ? 0.8 : (stage == 2 ? 0.3 : 0.0);
                
                double[][] XTest = generateDriftingData(100, 3, driftLevel);
                double[] yTest = generateClassificationLabels(100, XTest, isDrifted);
                
                // Evaluate current model
                double[] predictions = adaptiveModel.predict(XTest);
                double accuracy = Metrics.accuracy(yTest, predictions);
                accuracies[stage] = accuracy;
                
                String action = "Monitor";
                if (stage == 2) action = "Detect drift";
                else if (stage == 4) action = "Retrain model";
                else if (stage == 5) action = "Validate";
                
                System.out.printf("   %-5s | %-9s | %8.3f | %s\n", 
                                stages[stage], 
                                isDrifted ? "Drifted" : "Stable", 
                                accuracy, action);
                
                // Simulate retraining at adaptation stage
                if (stage == 4) {
                    adaptiveModel.fit(XTest, yTest);
                    System.out.println("         📚 Model retrained on new distribution");
                }
            }
            
            // Show improvement after adaptation
            double improvement = accuracies[5] - accuracies[3];
            System.out.printf("\n   📈 Performance improvement after adaptation: +%.3f\n", improvement);
            
        } catch (Exception e) {
            System.err.println("   Error in adaptive learning: " + e.getMessage());
        }
    }
    
    // Helper methods for drift detection
    private static boolean detectDataDrift(double[][] baseline, double[][] current) {
        // Simplified data drift detection using statistical comparison
        try {
            DriftConfig config = new DriftConfig();
            DataDriftDetector detector = new DataDriftDetector(config);
            
            // Set reference data
            String[] featureNames = new String[baseline[0].length];
            for (int i = 0; i < featureNames.length; i++) {
                featureNames[i] = "feature_" + i;
            }
            detector.setReferenceData(baseline, featureNames, new java.util.HashSet<>());
            
            // Detect drift
            DataDriftDetector.DataDriftResult result = detector.detectDrift(current, featureNames);
            return result.isOverallDriftDetected();
        } catch (Exception e) {
            // Fallback to simple statistical test
            return simpleStatisticalDriftTest(baseline, current);
        }
    }
    
    private static boolean simpleStatisticalDriftTest(double[][] baseline, double[][] current) {
        // Simple drift test comparing means
        double[] baselineMeans = calculateMeans(baseline);
        double[] currentMeans = calculateMeans(current);
        
        for (int i = 0; i < baselineMeans.length; i++) {
            if (Math.abs(baselineMeans[i] - currentMeans[i]) > 0.5) {
                return true; // Drift detected
            }
        }
        return false;
    }
    
    private static double[] calculateMeans(double[][] data) {
        if (data.length == 0) return new double[0];
        
        double[] means = new double[data[0].length];
        for (int j = 0; j < data[0].length; j++) {
            double sum = 0.0;
            for (int i = 0; i < data.length; i++) {
                sum += data[i][j];
            }
            means[j] = sum / data.length;
        }
        return means;
    }
    
    // Data generation methods
    private static double[][] generateStableData(int samples, int features) {
        double[][] data = new double[samples][features];
        java.util.Random random = new java.util.Random(42);
        
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < features; j++) {
                data[i][j] = random.nextGaussian(); // Standard normal distribution
            }
        }
        return data;
    }
    
    private static double[][] generateDriftingData(int samples, int features, double driftLevel) {
        double[][] data = new double[samples][features];
        java.util.Random random = new java.util.Random(42);
        
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < features; j++) {
                // Add drift by shifting mean and changing variance
                double shift = driftLevel * 2.0; // Mean shift
                double scaleChange = 1.0 + driftLevel * 0.5; // Variance change
                data[i][j] = random.nextGaussian() * scaleChange + shift;
            }
        }
        return data;
    }
    
    private static double[][] generateClassificationData(int samples, int features) {
        return generateStableData(samples, features);
    }
    
    private static double[] generateClassificationLabels(int samples, double[][] X, boolean isDrifted) {
        double[] labels = new double[samples];
        java.util.Random random = new java.util.Random(42);
        
        for (int i = 0; i < samples; i++) {
            // Create labels based on features with possible concept drift
            double sum = 0.0;
            for (int j = 0; j < X[i].length; j++) {
                double weight = isDrifted ? -(j + 1) : (j + 1); // Flip relationship for drift
                sum += X[i][j] * weight;
            }
            
            // Add noise and convert to binary classification
            sum += random.nextGaussian() * 0.5;
            labels[i] = sum > 0 ? 1.0 : 0.0;
        }
        return labels;
    }
}
