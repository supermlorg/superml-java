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

package org.superml.pmml;

import org.superml.pmml.PMMLConverter;

/**
 * Simple demonstration of PMML converter functionality.
 * 
 * This class provides a basic test of the PMMLConverter without requiring
 * external test frameworks or full SuperML model dependencies.
 * 
 * @author SuperML Team
 * @version 2.1.0
 */
public class PMMLConverterDemo {
    
    public static void main(String[] args) {
        System.out.println("=== SuperML PMML Converter Demo ===\n");
        
        PMMLConverter converter = new PMMLConverter();
        
        // Test 1: Basic validation functionality
        System.out.println("1. Testing PMML Validation");
        System.out.println("===========================");
        testValidation(converter);
        
        // Test 2: Error handling
        System.out.println("\n2. Testing Error Handling");
        System.out.println("==========================");
        testErrorHandling(converter);
        
        // Test 3: Method availability
        System.out.println("\n3. Testing Method Availability");
        System.out.println("===============================");
        testMethodAvailability(converter);
        
        System.out.println("\n=== Demo Complete ===");
        System.out.println("\nTo use the PMML converter with SuperML models:");
        System.out.println("1. Train a SuperML model (LinearRegression, LogisticRegression, etc.)");
        System.out.println("2. Call converter.convertToXML(model) to generate PMML");
        System.out.println("3. Use converter.validatePMML(pmmlXml) to validate the output");
        System.out.println("4. Deploy the PMML to compatible systems");
    }
    
    /**
     * Tests the PMML validation functionality.
     */
    private static void testValidation(PMMLConverter converter) {
        // Test cases for validation
        String[] testCases = {
            null,
            "",
            "   ",
            "<invalid>xml</",
            "<?xml version=\"1.0\"?><data>test</data>",
            createSamplePMML()
        };
        
        String[] descriptions = {
            "null input",
            "empty string", 
            "whitespace only",
            "malformed XML",
            "valid XML but not PMML",
            "sample PMML structure"
        };
        
        for (int i = 0; i < testCases.length; i++) {
            try {
                boolean isValid = converter.validatePMML(testCases[i]);
                System.out.println("✓ " + descriptions[i] + ": " + (isValid ? "VALID" : "INVALID"));
            } catch (Exception e) {
                System.out.println("✗ " + descriptions[i] + ": ERROR - " + e.getMessage());
            }
        }
    }
    
    /**
     * Tests error handling for various scenarios.
     */
    private static void testErrorHandling(PMMLConverter converter) {
        // Test null model
        try {
            converter.convertToXML(null);
            System.out.println("✗ Null model: Should have thrown exception");
        } catch (IllegalArgumentException e) {
            System.out.println("✓ Null model: Correctly threw IllegalArgumentException");
        } catch (Exception e) {
            System.out.println("✗ Null model: Threw unexpected exception: " + e.getClass().getSimpleName());
        }
        
        // Test unsupported model type
        try {
            converter.convertToXML("not a model");
            System.out.println("✗ Unsupported model: Should have thrown exception");
        } catch (IllegalArgumentException e) {
            System.out.println("✓ Unsupported model: Correctly threw IllegalArgumentException");
        } catch (Exception e) {
            System.out.println("✗ Unsupported model: Threw unexpected exception: " + e.getClass().getSimpleName());
        }
        
        // Test convertFromXML (not implemented)
        try {
            converter.convertFromXML("<PMML></PMML>");
            System.out.println("✗ convertFromXML: Should have thrown exception");
        } catch (UnsupportedOperationException e) {
            System.out.println("✓ convertFromXML: Correctly threw UnsupportedOperationException");
        } catch (Exception e) {
            System.out.println("✗ convertFromXML: Threw unexpected exception: " + e.getClass().getSimpleName());
        }
    }
    
    /**
     * Tests that all required methods are available and callable.
     */
    private static void testMethodAvailability(PMMLConverter converter) {
        try {
            // Check that methods exist and are callable
            converter.getClass().getMethod("convertToXML", Object.class);
            System.out.println("✓ convertToXML(Object) method available");
            
            converter.getClass().getMethod("convertToXML", Object.class, String[].class, String.class);
            System.out.println("✓ convertToXML(Object, String[], String) method available");
            
            converter.getClass().getMethod("convertFromXML", String.class);
            System.out.println("✓ convertFromXML(String) method available");
            
            converter.getClass().getMethod("validatePMML", String.class);
            System.out.println("✓ validatePMML(String) method available");
            
            System.out.println("✓ All required methods are available");
            
        } catch (NoSuchMethodException e) {
            System.out.println("✗ Method missing: " + e.getMessage());
        }
    }
    
    /**
     * Creates a sample PMML structure for testing validation.
     */
    private static String createSamplePMML() {
        return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" +
            "<PMML version=\"4.4\">\n" +
            "  <Header>\n" +
            "    <Application name=\"SuperML Java Framework\" version=\"2.1.0\"/>\n" +
            "    <Timestamp>Sample timestamp</Timestamp>\n" +
            "  </Header>\n" +
            "  <DataDictionary numberOfFields=\"2\">\n" +
            "    <DataField name=\"feature_0\" optype=\"continuous\" dataType=\"double\"/>\n" +
            "    <DataField name=\"target\" optype=\"continuous\" dataType=\"double\"/>\n" +
            "  </DataDictionary>\n" +
            "  <RegressionModel functionName=\"regression\">\n" +
            "    <MiningSchema>\n" +
            "      <MiningField name=\"feature_0\" usageType=\"active\"/>\n" +
            "      <MiningField name=\"target\" usageType=\"target\"/>\n" +
            "    </MiningSchema>\n" +
            "    <RegressionTable intercept=\"1.0\">\n" +
            "      <NumericPredictor name=\"feature_0\" coefficient=\"2.0\"/>\n" +
            "    </RegressionTable>\n" +
            "  </RegressionModel>\n" +
            "</PMML>";
    }
}
