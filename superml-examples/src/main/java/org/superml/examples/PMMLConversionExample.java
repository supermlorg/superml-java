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

import org.superml.pmml.PMMLConverter;

/**
 * Comprehensive examples demonstrating PMML conversion functionality in SuperML.
 * 
 * This example showcases how to:
 * 1. Convert various SuperML models to PMML format
 * 2. Validate PMML XML for correctness
 * 3. Use custom feature names and target variables
 * 4. Handle different model types (Linear, Logistic, Tree-based)
 * 5. Deploy models using PMML for cross-platform compatibility
 * 
 * @author SuperML Team
 * @version 2.1.0
 */
public class PMMLConversionExample {
    
    public static void main(String[] args) {
        System.out.println("=".repeat(70));
        System.out.println("        SuperML PMML Conversion Examples");
        System.out.println("=".repeat(70));
        
        try {
            // 1. Basic PMML Conversion
            demonstrateBasicConversion();
            
            // 2. Advanced PMML with Custom Features
            demonstrateAdvancedConversion();
            
            // 3. Model Validation and Error Handling
            demonstrateValidationAndErrorHandling();
            
            // 4. Production Deployment Scenarios
            demonstrateDeploymentScenarios();
            
            System.out.println("\n" + "=".repeat(70));
            System.out.println("✅ All PMML conversion examples completed successfully!");
            System.out.println("=".repeat(70));
            
        } catch (Exception e) {
            System.err.println("❌ Error in PMML conversion examples: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Demonstrates basic PMML conversion for different model types.
     */
    private static void demonstrateBasicConversion() {
        System.out.println("\n🔄 1. Basic PMML Conversion");
        System.out.println("=" .repeat(40));
        
        PMMLConverter converter = new PMMLConverter();
        
        try {
            // Example 1: Linear Regression (simulated with mock model)
            System.out.println("\n📈 Linear Regression to PMML:");
            LinearRegression linearModel = new LinearRegression();
            
            String linearPMML = converter.convertToXML(linearModel);
            boolean isValidLinear = converter.validatePMML(linearPMML);
            
            System.out.printf("   ✓ PMML generated: %d characters\n", linearPMML.length());
            System.out.printf("   ✓ Validation result: %s\n", isValidLinear ? "VALID" : "INVALID");
            System.out.printf("   ✓ Model type: Linear Regression\n");
            
            // Example 2: Logistic Regression
            System.out.println("\n🎯 Logistic Regression to PMML:");
            LogisticRegression logisticModel = new LogisticRegression();
            
            String logisticPMML = converter.convertToXML(logisticModel);
            boolean isValidLogistic = converter.validatePMML(logisticPMML);
            
            System.out.printf("   ✓ PMML generated: %d characters\n", logisticPMML.length());
            System.out.printf("   ✓ Validation result: %s\n", isValidLogistic ? "VALID" : "INVALID");
            System.out.printf("   ✓ Model type: Logistic Regression\n");
            
            // Example 3: Decision Tree
            System.out.println("\n🌳 Decision Tree to PMML:");
            DecisionTree treeModel = new DecisionTree();
            
            String treePMML = converter.convertToXML(treeModel);
            boolean isValidTree = converter.validatePMML(treePMML);
            
            System.out.printf("   ✓ PMML generated: %d characters\n", treePMML.length());
            System.out.printf("   ✓ Validation result: %s\n", isValidTree ? "VALID" : "INVALID");
            System.out.printf("   ✓ Model type: Decision Tree\n");
            
        } catch (Exception e) {
            System.err.println("   ❌ Error in basic conversion: " + e.getMessage());
        }
    }
    
    /**
     * Demonstrates advanced PMML conversion with custom features.
     */
    private static void demonstrateAdvancedConversion() {
        System.out.println("\n⚡ 2. Advanced PMML with Custom Features");
        System.out.println("=" .repeat(40));
        
        PMMLConverter converter = new PMMLConverter();
        
        try {
            // Custom feature names for business context
            String[] businessFeatures = {
                "customer_age", "annual_income", "credit_score", 
                "debt_to_income_ratio", "employment_years"
            };
            String targetName = "loan_approval_probability";
            
            System.out.println("\n💼 Business Model with Custom Features:");
            LogisticRegression businessModel = new LogisticRegression();
            
            String businessPMML = converter.convertToXML(businessModel, businessFeatures, targetName);
            boolean isValid = converter.validatePMML(businessPMML);
            
            System.out.printf("   ✓ Features: %s\n", String.join(", ", businessFeatures));
            System.out.printf("   ✓ Target: %s\n", targetName);
            System.out.printf("   ✓ PMML size: %d characters\n", businessPMML.length());
            System.out.printf("   ✓ Validation: %s\n", isValid ? "PASSED" : "FAILED");
            
            // Technical features for engineering model
            String[] techFeatures = {
                "sensor_temp", "pressure_reading", "vibration_level", "rpm"
            };
            String techTarget = "equipment_failure_risk";
            
            System.out.println("\n🔧 Engineering Model with Technical Features:");
            LinearRegression techModel = new LinearRegression();
            
            String techPMML = converter.convertToXML(techModel, techFeatures, techTarget);
            boolean isTechValid = converter.validatePMML(techPMML);
            
            System.out.printf("   ✓ Features: %s\n", String.join(", ", techFeatures));
            System.out.printf("   ✓ Target: %s\n", techTarget);
            System.out.printf("   ✓ PMML size: %d characters\n", techPMML.length());
            System.out.printf("   ✓ Validation: %s\n", isTechValid ? "PASSED" : "FAILED");
            
        } catch (Exception e) {
            System.err.println("   ❌ Error in advanced conversion: " + e.getMessage());
        }
    }
    
    /**
     * Demonstrates PMML validation and error handling.
     */
    private static void demonstrateValidationAndErrorHandling() {
        System.out.println("\n🛡️ 3. Validation and Error Handling");
        System.out.println("=" .repeat(40));
        
        PMMLConverter converter = new PMMLConverter();
        
        try {
            // Test various validation scenarios
            System.out.println("\n✅ PMML Validation Tests:");
            
            // Valid PMML from a model
            LinearRegression validModel = new LinearRegression();
            String validPMML = converter.convertToXML(validModel);
            System.out.printf("   ✓ Valid model PMML: %s\n", 
                converter.validatePMML(validPMML) ? "PASSED" : "FAILED");
            
            // Invalid inputs
            System.out.printf("   ✓ Null PMML validation: %s\n", 
                !converter.validatePMML(null) ? "CORRECTLY REJECTED" : "FAILED");
            
            System.out.printf("   ✓ Empty PMML validation: %s\n", 
                !converter.validatePMML("") ? "CORRECTLY REJECTED" : "FAILED");
            
            System.out.printf("   ✓ Malformed XML validation: %s\n", 
                !converter.validatePMML("<invalid>xml") ? "CORRECTLY REJECTED" : "FAILED");
            
            // Error handling tests
            System.out.println("\n❌ Error Handling Tests:");
            
            try {
                converter.convertToXML(null);
                System.out.println("   ❌ Null model: Should have thrown exception");
            } catch (IllegalArgumentException e) {
                System.out.println("   ✓ Null model: Correctly rejected - " + e.getMessage());
            }
            
            try {
                converter.convertToXML("Not a model");
                System.out.println("   ❌ Invalid model: Should have thrown exception");
            } catch (Exception e) {
                System.out.println("   ✓ Invalid model: Correctly rejected - " + e.getMessage());
            }
            
            try {
                converter.convertFromXML("<pmml></pmml>");
                System.out.println("   ❌ Import not implemented: Should have thrown exception");
            } catch (UnsupportedOperationException e) {
                System.out.println("   ✓ Import not implemented: Correctly indicated - " + e.getMessage());
            }
            
        } catch (Exception e) {
            System.err.println("   ❌ Error in validation testing: " + e.getMessage());
        }
    }
    
    /**
     * Demonstrates production deployment scenarios using PMML.
     */
    private static void demonstrateDeploymentScenarios() {
        System.out.println("\n🚀 4. Production Deployment Scenarios");
        System.out.println("=" .repeat(40));
        
        PMMLConverter converter = new PMMLConverter();
        
        try {
            // Scenario 1: Model Export for Apache Spark
            System.out.println("\n⚡ Spark MLlib Deployment:");
            RandomForest sparkModel = new RandomForest();
            String[] sparkFeatures = {"feature_1", "feature_2", "feature_3", "feature_4"};
            
            String sparkPMML = converter.convertToXML(sparkModel, sparkFeatures, "prediction");
            System.out.printf("   ✓ Model: Random Forest\n");
            System.out.printf("   ✓ Features: %d\n", sparkFeatures.length);
            System.out.printf("   ✓ PMML size: %d characters\n", sparkPMML.length());
            System.out.printf("   ✓ Ready for Spark deployment: %s\n", 
                converter.validatePMML(sparkPMML) ? "YES" : "NO");
            System.out.println("   📋 Usage: Load in Spark MLlib using JPMML-SparkML");
            
            // Scenario 2: Model Export for Python/scikit-learn
            System.out.println("\n🐍 Python scikit-learn Integration:");
            LogisticRegression pythonModel = new LogisticRegression();
            String[] pythonFeatures = {"age", "income", "education", "experience"};
            
            String pythonPMML = converter.convertToXML(pythonModel, pythonFeatures, "outcome");
            System.out.printf("   ✓ Model: Logistic Regression\n");
            System.out.printf("   ✓ Features: %d\n", pythonFeatures.length);
            System.out.printf("   ✓ PMML size: %d characters\n", pythonPMML.length());
            System.out.printf("   ✓ Python-ready: %s\n", 
                converter.validatePMML(pythonPMML) ? "YES" : "NO");
            System.out.println("   📋 Usage: Use with jpmml-evaluator in Python");
            
            // Scenario 3: Enterprise Deployment
            System.out.println("\n🏢 Enterprise Platform Deployment:");
            DecisionTree enterpriseModel = new DecisionTree();
            String[] enterpriseFeatures = {
                "transaction_amount", "account_age", "transaction_frequency", 
                "geo_location_risk", "time_of_day_factor"
            };
            
            String enterprisePMML = converter.convertToXML(enterpriseModel, 
                enterpriseFeatures, "fraud_probability");
            System.out.printf("   ✓ Model: Decision Tree\n");
            System.out.printf("   ✓ Business Features: %d\n", enterpriseFeatures.length);
            System.out.printf("   ✓ PMML size: %d characters\n", enterprisePMML.length());
            System.out.printf("   ✓ Enterprise-ready: %s\n", 
                converter.validatePMML(enterprisePMML) ? "YES" : "NO");
            System.out.println("   📋 Platforms: SAS, SPSS, Azure ML, Amazon SageMaker");
            
            // Scenario 4: A/B Testing Deployment
            System.out.println("\n🧪 A/B Testing Deployment:");
            LinearRegression modelA = new LinearRegression();
            LinearRegression modelB = new LinearRegression();
            
            String pmmlA = converter.convertToXML(modelA, new String[]{"f1", "f2"}, "target");
            String pmmlB = converter.convertToXML(modelB, new String[]{"f1", "f2"}, "target");
            
            System.out.printf("   ✓ Model A PMML: %d chars (%s)\n", pmmlA.length(),
                converter.validatePMML(pmmlA) ? "Valid" : "Invalid");
            System.out.printf("   ✓ Model B PMML: %d chars (%s)\n", pmmlB.length(),
                converter.validatePMML(pmmlB) ? "Valid" : "Invalid");
            System.out.println("   📋 Deploy both models for parallel evaluation");
            
        } catch (Exception e) {
            System.err.println("   ❌ Error in deployment scenarios: " + e.getMessage());
        }
    }
    
    // Mock model classes for demonstration (simulate SuperML models)
    
    /**
     * Mock BaseEstimator for testing purposes.
     */
    public static abstract class BaseEstimator {
        // Mock base functionality - this simulates the SuperML BaseEstimator interface
    }
    
    /**
     * Mock LinearRegression model that extends BaseEstimator.
     */
    public static class LinearRegression extends BaseEstimator {
        public double[] getCoefficients() {
            return new double[]{1.5, -0.3, 0.8, 2.1, -1.2};
        }
        
        public double getIntercept() {
            return 0.5;
        }
    }
    
    /**
     * Mock LogisticRegression model.
     */
    public static class LogisticRegression extends BaseEstimator {
        public double[] getCoefficients() {
            return new double[]{0.7, 1.2, -0.5, 0.9};
        }
        
        public double getIntercept() {
            return -0.1;
        }
        
        public double[] getClasses() {
            return new double[]{0.0, 1.0};
        }
    }
    
    /**
     * Mock DecisionTree model.
     */
    public static class DecisionTree extends BaseEstimator {
        public double[] getClasses() {
            return new double[]{0.0, 1.0, 2.0};
        }
    }
    
    /**
     * Mock RandomForest model.
     */
    public static class RandomForest extends BaseEstimator {
        public double[] getClasses() {
            return new double[]{0.0, 1.0};
        }
        
        public int getNumTrees() {
            return 100;
        }
    }
}
