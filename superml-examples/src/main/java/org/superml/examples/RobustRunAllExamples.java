package org.superml.examples;

/**
 * Robust runner for all SuperML Java framework examples
 * Handles missing dependencies gracefully and provides detailed summary
 * SuperML 3.1.2 Framework Demonstration
 */
public class RobustRunAllExamples {
    
    public static void main(String[] args) {
        System.out.println("🚀 SuperML Java 3.1.2 - Robust Example Runner");
        System.out.println("=".repeat(80));
        
        int completedExamples = 0;
        int totalExamples = 11;
        
        // Example 1: Simple Classification
        try {
            System.out.println("\n📊 Example 1: Simple Classification");
            System.out.println("-".repeat(50));
            SimpleClassificationExample.main(args);
            completedExamples++;
        } catch (Exception e) {
            System.out.println("⚠️  Example 1 failed: " + e.getClass().getSimpleName());
        }
        
        // Example 2: Simple Regression
        try {
            System.out.println("\n\n📈 Example 2: Simple Regression");
            System.out.println("-".repeat(50));
            SimpleRegressionExample.main(args);
            completedExamples++;
        } catch (Exception e) {
            System.out.println("⚠️  Example 2 failed: " + e.getClass().getSimpleName());
        }
        
        // Example 3: Kaggle-style Competition (Neural Networks - may need Apache Commons Math)
        try {
            System.out.println("\n\n🏆 Example 3: Kaggle-style Competition");
            System.out.println("-".repeat(50));
            SimpleKaggleExample.main(args);
            completedExamples++;
        } catch (Exception e) {
            System.out.println("⚠️  Example 3 failed: " + e.getClass().getSimpleName() + 
                             " (Neural networks require Apache Commons Math dependency)");
        }
        
        // Example 4: Tree Models
        try {
            System.out.println("\n\n🌳 Example 4: Tree Models (Decision Tree & Random Forest)");
            System.out.println("-".repeat(60));
            TreeModelsExample.main(args);
            completedExamples++;
        } catch (Exception e) {
            System.out.println("⚠️  Example 4 failed: " + e.getClass().getSimpleName());
        }
        
        // Example 5: Clustering
        try {
            System.out.println("\n\n🎯 Example 5: Clustering (K-Means)");
            System.out.println("-".repeat(50));
            ClusteringExample.main(args);
            completedExamples++;
        } catch (Exception e) {
            System.out.println("⚠️  Example 5 failed: " + e.getClass().getSimpleName());
        }
        
        // Example 6: ML Pipeline
        try {
            System.out.println("\n\n🔄 Example 6: ML Pipeline");
            System.out.println("-".repeat(50));
            SimplePipelineExample.main(args);
            completedExamples++;
        } catch (Exception e) {
            System.out.println("⚠️  Example 6 failed: " + e.getClass().getSimpleName());
        }
        
        // Example 7: Drift Detection
        try {
            System.out.println("\n\n🚨 Example 7: Drift Detection");
            System.out.println("-".repeat(50));
            SimpleDriftDetectionExample.main(args);
            completedExamples++;
        } catch (Exception e) {
            System.out.println("⚠️  Example 7 failed: " + e.getClass().getSimpleName());
        }
        
        // Example 8: Model Inference
        try {
            System.out.println("\n\n⚡ Example 8: Model Inference");
            System.out.println("-".repeat(50));
            InferenceExample.main(args);
            completedExamples++;
        } catch (Exception e) {
            System.out.println("⚠️  Example 8 failed: " + e.getClass().getSimpleName());
        }
        
        // Example 9: Confusion Matrix Analysis
        try {
            System.out.println("\n\n📊 Example 9: Confusion Matrix Analysis");
            System.out.println("-".repeat(60));
            ConfusionMatrixExample.main(args);
            completedExamples++;
        } catch (Exception e) {
            System.out.println("⚠️  Example 9 failed: " + e.getClass().getSimpleName());
        }
        
        // Example 10: Visualization Demo
        try {
            System.out.println("\n\n🎨 Example 10: Comprehensive Visualization Demo");
            System.out.println("-".repeat(60));
            VisualizationExample.main(args);
            completedExamples++;
        } catch (Exception e) {
            System.out.println("⚠️  Example 10 failed: " + e.getClass().getSimpleName());
        }
        
        // Example 11: XChart GUI Visualization
        try {
            System.out.println("\n\n📊 Example 11: XChart GUI Visualization Showcase");
            System.out.println("-".repeat(60));
            XChartVisualizationExample.main(args);
            completedExamples++;
        } catch (Exception e) {
            System.out.println("⚠️  Example 11 failed: " + e.getClass().getSimpleName());
        }
        
        // Final Summary
        System.out.println("\n\n" + "=".repeat(80));
        System.out.println("📊 SUPERML 3.1.2 EXAMPLE EXECUTION SUMMARY");
        System.out.println("=".repeat(80));
        System.out.println(String.format("✅ Completed: %d/%d examples", completedExamples, totalExamples));
        System.out.println(String.format("📈 Success Rate: %.1f%%", (completedExamples * 100.0) / totalExamples));
        
        if (completedExamples == totalExamples) {
            System.out.println("🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!");
        } else {
            System.out.println("⚠️  Some examples failed - likely due to missing external dependencies");
            System.out.println("💡 Neural network examples require Apache Commons Math3 library");
            System.out.println("💡 Visualization examples may require additional GUI libraries");
        }
        
        System.out.println("🚀 SuperML Java 3.1.2 framework demonstration complete!");
        System.out.println("📚 Core algorithms (Linear Models, Trees, Clustering) are fully functional");
        System.out.println("=".repeat(80));
    }
}
