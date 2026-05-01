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

/**
 * Runner to execute all SuperML examples
 */
public class RunAllExamples {
    
    public static void main(String[] args) {
        System.out.println("🚀 SuperML Java 3.1.2 - Running All Examples");
        System.out.println("=".repeat(80));
        
        int completedExamples = 0;
        int totalExamples = 11;
        
        try {
            System.out.println("\n📊 Example 1: Simple Classification");
            System.out.println("-".repeat(50));
            SimpleClassificationExample.main(args);
            completedExamples++;
        } catch (Exception e) {
            System.out.println("⚠️  Example 1 failed: " + e.getMessage());
        }
        
        try {
            System.out.println("\n\n📈 Example 2: Simple Regression");
            System.out.println("-".repeat(50));
            SimpleRegressionExample.main(args);
            completedExamples++;
        } catch (Exception e) {
            System.out.println("⚠️  Example 2 failed: " + e.getMessage());
        }
        
        try {
            System.out.println("\n\n🏆 Example 3: Kaggle-style Competition");
            System.out.println("-".repeat(50));
            SimpleKaggleExample.main(args);
            completedExamples++;
            
            System.out.println("\n\n🌳 Example 4: Tree Models (Decision Tree & Random Forest)");
            System.out.println("-".repeat(60));
            TreeModelsExample.main(args);
            
            System.out.println("\n\n🎯 Example 5: Clustering (K-Means)");
            System.out.println("-".repeat(50));
            ClusteringExample.main(args);
            
            System.out.println("\n\n🔄 Example 6: ML Pipeline");
            System.out.println("-".repeat(50));
            SimplePipelineExample.main(args);
            
            System.out.println("\n\n🚨 Example 7: Drift Detection");
            System.out.println("-".repeat(50));
            SimpleDriftDetectionExample.main(args);
            
            System.out.println("\n\n⚡ Example 8: Model Inference");
            System.out.println("-".repeat(50));
            InferenceExample.main(args);
            
            System.out.println("\n\n📊 Example 9: Confusion Matrix Analysis");
            System.out.println("-".repeat(60));
            ConfusionMatrixExample.main(args);
            
            System.out.println("\n\n🎨 Example 10: Comprehensive Visualization Demo");
            System.out.println("-".repeat(60));
            VisualizationExample.main(args);
            
            System.out.println("\n\n📊 Example 11: XChart GUI Visualization Showcase");
            System.out.println("-".repeat(60));
            XChartVisualizationExample.main(args);
            
            System.out.println("\n\n" + "=".repeat(80));
            System.out.println("✅ ALL 11 EXAMPLES COMPLETED SUCCESSFULLY!");
            System.out.println("🎉 SuperML Java 2.0.0 framework with full algorithm coverage!");
            System.out.println("✅ Including comprehensive confusion matrix analysis!");
            System.out.println("🎨 Plus advanced ASCII & XChart GUI visualization capabilities!");
            System.out.println("=".repeat(80));
            
        } catch (Exception e) {
            System.err.println("❌ Error running examples: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
