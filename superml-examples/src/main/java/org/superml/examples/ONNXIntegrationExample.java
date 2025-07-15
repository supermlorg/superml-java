package org.superml.examples;

import org.superml.linear_model.LogisticRegression;
import org.superml.linear_model.LinearRegression;
import org.superml.tree.DecisionTree;
import org.superml.tree.RandomForest;
import org.superml.metrics.Metrics;

/**
 * ONNX Integration Example
 * Demonstrates integration capabilities and model export concepts
 * 
 * Note: This example focuses on demonstrating the integration workflow
 * and model serialization concepts that would be used for ONNX export.
 */
public class ONNXIntegrationExample {
    
    public static void main(String[] args) {
        System.out.println("=== SuperML 2.0.0 - ONNX Integration Example ===\n");
        
        try {
            // Demonstrate model training and evaluation workflow
            demonstrateModelWorkflow();
            
            // Show model serialization concepts
            demonstrateModelSerialization();
            
            // Demonstrate batch prediction workflows
            demonstrateBatchPrediction();
            
            // Show model metadata extraction
            demonstrateModelMetadata();
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void demonstrateModelWorkflow() {
        System.out.println("🔄 Model Training & Evaluation Workflow");
        System.out.println("=======================================");
        
        try {
            // Generate dataset for classification
            double[][] X = generateClassificationData(300, 4);
            double[] y = generateClassificationLabels(X);
            
            // Split into train/test
            int trainSize = (int)(X.length * 0.8);
            double[][] XTrain = new double[trainSize][];
            double[][] XTest = new double[X.length - trainSize][];
            double[] yTrain = new double[trainSize];
            double[] yTest = new double[X.length - trainSize];
            
            System.arraycopy(X, 0, XTrain, 0, trainSize);
            System.arraycopy(X, trainSize, XTest, 0, X.length - trainSize);
            System.arraycopy(y, 0, yTrain, 0, trainSize);
            System.arraycopy(y, trainSize, yTest, 0, X.length - trainSize);
            
            System.out.printf("📊 Dataset: %d training, %d test samples\n", trainSize, X.length - trainSize);
            
            // Train multiple models for export workflow
            System.out.println("\n🎯 Training models for export workflow:");
            
            // 1. Logistic Regression
            LogisticRegression lr = new LogisticRegression();
            lr.fit(XTrain, yTrain);
            double[] lrPred = lr.predict(XTest);
            double lrAccuracy = Metrics.accuracy(yTest, lrPred);
            System.out.printf("   Logistic Regression: %.3f accuracy\n", lrAccuracy);
            
            // 2. Decision Tree
            DecisionTree dt = new DecisionTree();
            dt.fit(XTrain, yTrain);
            double[] dtPred = dt.predict(XTest);
            double dtAccuracy = Metrics.accuracy(yTest, dtPred);
            System.out.printf("   Decision Tree: %.3f accuracy\n", dtAccuracy);
            
            // 3. Random Forest
            RandomForest rf = new RandomForest();
            rf.setNEstimators(50);
            rf.fit(XTrain, yTrain);
            double[] rfPred = rf.predict(XTest);
            double rfAccuracy = Metrics.accuracy(yTest, rfPred);
            System.out.printf("   Random Forest: %.3f accuracy\n", rfAccuracy);
            
            System.out.println("   ✅ All models trained successfully!");
            
        } catch (Exception e) {
            System.err.println("   ❌ Error in model workflow: " + e.getMessage());
        }
    }
    
    private static void demonstrateModelSerialization() {
        System.out.println("\n💾 Model Serialization Concepts");
        System.out.println("=================================");
        
        try {
            // Train a simple model
            double[][] X = generateRegressionData(100, 3);
            double[] y = generateRegressionTarget(X);
            
            LinearRegression model = new LinearRegression();
            model.fit(X, y);
            
            // Extract model parameters (coefficients and intercept)
            double[] coefficients = model.getCoefficients();
            double intercept = model.getIntercept();
            
            System.out.println("📋 Model Parameters (for ONNX export):");
            System.out.printf("   Intercept: %.6f\n", intercept);
            System.out.println("   Coefficients:");
            for (int i = 0; i < coefficients.length; i++) {
                System.out.printf("     Feature %d: %.6f\n", i, coefficients[i]);
            }
            
            // Demonstrate model metadata
            System.out.println("\n📊 Model Metadata:");
            System.out.printf("   Algorithm: %s\n", model.getClass().getSimpleName());
            System.out.printf("   Input Features: %d\n", coefficients.length);
            System.out.printf("   Model Type: Regression\n");
            System.out.printf("   Framework: SuperML Java 2.0.0\n");
            
            // Show prediction format
            double[] sampleInput = X[0];
            double prediction = model.predict(new double[][]{sampleInput})[0];
            
            System.out.println("\n🔍 Prediction Example:");
            System.out.print("   Input: [");
            for (int i = 0; i < sampleInput.length; i++) {
                System.out.printf("%.3f", sampleInput[i]);
                if (i < sampleInput.length - 1) System.out.print(", ");
            }
            System.out.printf("]\n");
            System.out.printf("   Output: %.6f\n", prediction);
            
            System.out.println("   ✅ Model parameters extracted for export!");
            
        } catch (Exception e) {
            System.err.println("   ❌ Error in serialization: " + e.getMessage());
        }
    }
    
    private static void demonstrateBatchPrediction() {
        System.out.println("\n⚡ Batch Prediction Workflow");
        System.out.println("=============================");
        
        try {
            // Train model
            double[][] XTrain = generateClassificationData(200, 3);
            double[] yTrain = generateClassificationLabels(XTrain);
            
            LogisticRegression model = new LogisticRegression();
            model.fit(XTrain, yTrain);
            
            // Generate batch data for prediction
            double[][] batchData = generateClassificationData(50, 3);
            
            System.out.printf("📦 Processing batch of %d samples...\n", batchData.length);
            
            // Batch prediction
            long startTime = System.currentTimeMillis();
            double[] predictions = model.predict(batchData);
            double[][] probabilities = model.predictProba(batchData);
            long endTime = System.currentTimeMillis();
            
            System.out.printf("⏱️  Batch prediction time: %d ms\n", endTime - startTime);
            System.out.printf("📊 Average time per sample: %.2f ms\n", 
                             (double)(endTime - startTime) / batchData.length);
            
            // Show sample predictions
            System.out.println("\n📈 Sample Predictions:");
            System.out.println("   ID | Prediction | Prob Class 0 | Prob Class 1");
            System.out.println("   ---|------------|-------------|-------------");
            
            for (int i = 0; i < Math.min(5, predictions.length); i++) {
                System.out.printf("   %2d |      %.0f     |     %.3f     |     %.3f\n", 
                                 i + 1, predictions[i], probabilities[i][0], probabilities[i][1]);
            }
            
            // Performance statistics
            int class0Count = 0, class1Count = 0;
            for (double pred : predictions) {
                if (pred == 0.0) class0Count++;
                else class1Count++;
            }
            
            System.out.printf("\n📊 Prediction Distribution:\n");
            System.out.printf("   Class 0: %d samples (%.1f%%)\n", 
                             class0Count, 100.0 * class0Count / predictions.length);
            System.out.printf("   Class 1: %d samples (%.1f%%)\n", 
                             class1Count, 100.0 * class1Count / predictions.length);
            
            System.out.println("   ✅ Batch processing completed!");
            
        } catch (Exception e) {
            System.err.println("   ❌ Error in batch prediction: " + e.getMessage());
        }
    }
    
    private static void demonstrateModelMetadata() {
        System.out.println("\n📝 Model Metadata Extraction");
        System.out.println("==============================");
        
        try {
            // Train different types of models
            double[][] X = generateClassificationData(150, 4);
            double[] y = generateClassificationLabels(X);
            
            LogisticRegression lr = new LogisticRegression();
            lr.fit(X, y);
            
            DecisionTree dt = new DecisionTree();
            dt.fit(X, y);
            
            // Extract metadata for each model
            System.out.println("🔍 Model Metadata for ONNX Export:");
            System.out.println();
            
            // Logistic Regression metadata
            System.out.println("📊 Logistic Regression:");
            System.out.printf("   - Algorithm: %s\n", lr.getClass().getSimpleName());
            System.out.printf("   - Input Shape: [batch_size, %d]\n", X[0].length);
            System.out.printf("   - Output Shape: [batch_size, %d]\n", lr.getClasses().length);
            System.out.printf("   - Classes: %s\n", java.util.Arrays.toString(lr.getClasses()));
            System.out.printf("   - Learning Rate: %.3f\n", lr.getLearningRate());
            System.out.printf("   - Max Iterations: %d\n", lr.getMaxIter());
            System.out.println("   - Supports: predict(), predict_proba()");
            
            System.out.println();
            
            // Decision Tree metadata
            System.out.println("🌳 Decision Tree:");
            System.out.printf("   - Algorithm: %s\n", dt.getClass().getSimpleName());
            System.out.printf("   - Input Shape: [batch_size, %d]\n", X[0].length);
            System.out.printf("   - Output Shape: [batch_size, %d]\n", dt.getClasses().length);
            System.out.printf("   - Classes: %s\n", java.util.Arrays.toString(dt.getClasses()));
            System.out.printf("   - Criterion: %s\n", dt.getCriterion());
            System.out.printf("   - Max Depth: %d\n", dt.getMaxDepth());
            System.out.println("   - Supports: predict(), predict_proba()");
            
            // Generate ONNX-style model info
            System.out.println("\n📋 ONNX Export Information:");
            System.out.println("   - Framework: SuperML Java 2.0.0");
            System.out.println("   - Input Format: float32[batch_size, n_features]");
            System.out.println("   - Output Format: int64[batch_size] (predictions)");
            System.out.println("   - Probability Format: float32[batch_size, n_classes]");
            System.out.println("   - Supported Operations: MatMul, Add, Sigmoid, ArgMax");
            
            System.out.println("   ✅ Metadata extraction completed!");
            
        } catch (Exception e) {
            System.err.println("   ❌ Error in metadata extraction: " + e.getMessage());
        }
    }
    
    // Data generation methods
    private static double[][] generateClassificationData(int samples, int features) {
        double[][] data = new double[samples][features];
        java.util.Random random = new java.util.Random(42);
        
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < features; j++) {
                data[i][j] = random.nextGaussian() * 2.0;
            }
        }
        return data;
    }
    
    private static double[] generateClassificationLabels(double[][] X) {
        double[] labels = new double[X.length];
        
        for (int i = 0; i < X.length; i++) {
            // Simple decision boundary: sum of features
            double sum = 0;
            for (double feature : X[i]) {
                sum += feature;
            }
            labels[i] = sum > 0 ? 1.0 : 0.0;
        }
        return labels;
    }
    
    private static double[][] generateRegressionData(int samples, int features) {
        double[][] data = new double[samples][features];
        java.util.Random random = new java.util.Random(42);
        
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < features; j++) {
                data[i][j] = random.nextGaussian() * 1.5;
            }
        }
        return data;
    }
    
    private static double[] generateRegressionTarget(double[][] X) {
        double[] y = new double[X.length];
        java.util.Random random = new java.util.Random(42);
        
        for (int i = 0; i < X.length; i++) {
            y[i] = 2.0 * X[i][0] - 1.5 * X[i][1] + 0.8 * X[i][2] + random.nextGaussian() * 0.1;
        }
        return y;
    }
}
