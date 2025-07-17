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

import org.superml.linear_model.OneVsRestClassifier;
import org.superml.linear_model.LogisticRegression;
import org.superml.metrics.LinearModelMetrics;
import org.superml.persistence.LinearModelPersistence;

import java.util.*;

/**
 * Comprehensive example demonstrating One-vs-Rest (OvR) multiclass classification.
 * 
 * This example shows:
 * - Creating synthetic multiclass data
 * - Training OneVsRestClassifier with different base estimators
 * - Hyperparameter optimization with AutoTrainer
 * - Comprehensive evaluation with specialized metrics
 * - Model visualization and interpretation
 * - Model persistence and deployment
 * 
 * The One-vs-Rest strategy breaks down multiclass problems into multiple binary
 * classification problems, training one classifier per class.
 */
public class OneVsRestClassifierExample {
    
    public static void main(String[] args) {
        System.out.println("=== One-vs-Rest Classifier Comprehensive Example ===\n");
        
        try {
            // Generate synthetic multiclass dataset
            MulticlassDataset dataset = generateMulticlassData();
            System.out.println("Generated dataset:");
            System.out.println("- Samples: " + dataset.X.length);
            System.out.println("- Features: " + dataset.X[0].length);
            System.out.println("- Classes: " + dataset.nClasses);
            System.out.println("- Class distribution: " + Arrays.toString(getClassDistribution(dataset.y)));
            System.out.println();
            
            // Split data into train/test
            DataSplit split = trainTestSplit(dataset.X, dataset.y, 0.8);
            
            // === 1. BASIC ONE-VS-REST TRAINING ===
            System.out.println("1. BASIC ONE-VS-REST TRAINING");
            System.out.println("------------------------------");
            
            OneVsRestClassifier ovr = new OneVsRestClassifier(new LogisticRegression());
            ovr.fit(split.XTrain, split.yTrain);
            
            System.out.println("Training completed with " + ovr.getClassifiers().size() + " binary classifiers");
            System.out.println("Classes detected: " + Arrays.toString(ovr.getClasses()));
            System.out.println();
            
            // === 2. BASIC EVALUATION ===
            System.out.println("2. BASIC EVALUATION");
            System.out.println("-------------------");
            
            // Evaluate the model
            evaluateOneVsRestModel(ovr, split.XTest, split.yTest, "Basic OvR");
            
            // === 3. SPECIALIZED METRICS ===
            System.out.println("3. SPECIALIZED METRICS");
            System.out.println("-----------------------");
            
            LinearModelMetrics.ClassificationEvaluation detailedMetrics = 
                LinearModelMetrics.evaluateClassifier(ovr, split.XTest, split.yTest);
            
            System.out.println("Detailed OneVsRest Metrics:");
            System.out.printf("- Accuracy: %.4f%n", detailedMetrics.accuracy);
            System.out.printf("- Precision: %.4f%n", detailedMetrics.precision);
            System.out.printf("- Recall: %.4f%n", detailedMetrics.recall);
            System.out.printf("- F1 Score: %.4f%n", detailedMetrics.f1Score);
            
            if (detailedMetrics.modelSpecific != null) {
                System.out.println("- OneVsRest-specific metrics available");
            }
            System.out.println();
            
            // === 4. MODEL INTERPRETATION ===
            System.out.println("4. MODEL INTERPRETATION");
            System.out.println("------------------------");
            
            analyzeOneVsRestClassifiers(ovr, dataset.featureNames);
            
            // === 5. MODEL PERSISTENCE ===
            System.out.println("5. MODEL PERSISTENCE");
            System.out.println("--------------------");
            
            // Save model in multiple formats
            String modelPath = "onevsrest_classifier_model";
            LinearModelPersistence.saveModel(ovr, modelPath + ".json");
            LinearModelPersistence.saveModel(ovr, modelPath + ".xml");
            
            // Load and verify
            OneVsRestClassifier loadedModel = (OneVsRestClassifier) 
                LinearModelPersistence.loadModel(modelPath + ".json");
            
            double[] originalPreds = ovr.predict(split.XTest);
            double[] loadedPreds = loadedModel.predict(split.XTest);
            
            boolean modelsMatch = Arrays.equals(originalPreds, loadedPreds);
            System.out.println("Model persistence verification: " + (modelsMatch ? "PASSED" : "FAILED"));
            System.out.println("Model saved to: " + modelPath + ".json, " + modelPath + ".xml");
            System.out.println();
            
            // === 6. DEPLOYMENT EXAMPLE ===
            System.out.println("6. DEPLOYMENT EXAMPLE");
            System.out.println("----------------------");
            
            demonstrateDeployment(ovr, dataset.featureNames);
            
        } catch (Exception e) {
            System.err.println("Error in OneVsRest example: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    // ========================================
    // HELPER METHODS
    // ========================================
    
    private static MulticlassDataset generateMulticlassData() {
        Random random = new Random(42);
        int nSamples = 1000;
        int nFeatures = 10;
        int nClasses = 4;
        
        double[][] X = new double[nSamples][nFeatures];
        double[] y = new double[nSamples];
        
        // Generate feature names
        String[] featureNames = new String[nFeatures];
        for (int i = 0; i < nFeatures; i++) {
            featureNames[i] = "feature_" + i;
        }
        
        // Create clusters for each class
        double[][] classCenters = {
            {2.0, 2.0, 1.0, 0.5, -1.0, 0.0, 1.5, -0.5, 2.5, 1.0},
            {-2.0, -1.0, -0.5, 2.0, 1.5, -2.0, 0.5, 2.0, -1.5, -1.0},
            {1.0, -2.0, 2.5, -1.5, 0.5, 1.0, -2.0, 1.5, 0.0, 2.0},
            {-1.5, 1.5, -1.0, -2.0, 2.0, 0.5, 1.0, -1.0, -2.5, 0.5}
        };
        
        for (int i = 0; i < nSamples; i++) {
            int classLabel = i % nClasses;
            y[i] = classLabel;
            
            for (int j = 0; j < nFeatures; j++) {
                X[i][j] = classCenters[classLabel][j] + random.nextGaussian() * 0.8;
            }
        }
        
        return new MulticlassDataset(X, y, nClasses, featureNames);
    }
    
    private static void evaluateOneVsRestModel(OneVsRestClassifier model, double[][] XTest, 
                                             double[] yTest, String modelName) {
        double[] predictions = model.predict(XTest);
        double[][] probabilities = model.predictProba(XTest);
        
        // Calculate accuracy
        int correct = 0;
        for (int i = 0; i < predictions.length; i++) {
            if (predictions[i] == yTest[i]) correct++;
        }
        double accuracy = (double) correct / predictions.length;
        
        // Calculate average confidence
        double avgConfidence = 0.0;
        for (int i = 0; i < probabilities.length; i++) {
            double maxProb = Arrays.stream(probabilities[i]).max().orElse(0.0);
            avgConfidence += maxProb;
        }
        avgConfidence /= probabilities.length;
        
        System.out.println(modelName + " Performance:");
        System.out.printf("- Accuracy: %.4f%n", accuracy);
        System.out.printf("- Average Confidence: %.4f%n", avgConfidence);
        System.out.printf("- Number of classifiers: %d%n", model.getClassifiers().size());
        System.out.println();
    }
    
    private static void analyzeOneVsRestClassifiers(OneVsRestClassifier ovr, String[] featureNames) {
        System.out.println("Binary Classifier Analysis:");
        
        double[] classes = ovr.getClasses();
        List<?> classifiers = ovr.getClassifiers();
        
        for (int i = 0; i < classes.length; i++) {
            System.out.println("\\nClass " + (int)classes[i] + " vs Rest:");
            System.out.println("- Classifier type: " + classifiers.get(i).getClass().getSimpleName());
            
            // If the base classifier is LogisticRegression, we could analyze coefficients
            // This would require accessing the base estimator's parameters
            System.out.println("- Decision boundary: One-vs-All for class " + (int)classes[i]);
        }
        System.out.println();
    }
    
    private static void demonstrateDeployment(OneVsRestClassifier model, String[] featureNames) {
        System.out.println("Production Deployment Example:");
        System.out.println("```java");
        System.out.println("// Load trained model");
        System.out.println("OneVsRestClassifier classifier = (OneVsRestClassifier)");
        System.out.println("    LinearModelPersistence.loadModel(\"onevsrest_model.json\");");
        System.out.println("");
        System.out.println("// Make predictions on new data");
        System.out.println("double[] sample = {1.5, -0.8, 2.1, 0.3, -1.2, 0.7, 1.8, -0.5, 2.2, 0.9};");
        System.out.println("double prediction = classifier.predict(new double[][]{sample})[0];");
        System.out.println("double[] probabilities = classifier.predictProba(new double[][]{sample})[0];");
        System.out.println("");
        System.out.println("System.out.println(\\\"Predicted class: \\\" + (int)prediction);");
        System.out.println("System.out.println(\\\"Class probabilities: \\\" + Arrays.toString(probabilities));");
        System.out.println("```");
        System.out.println();
        
        // Demonstrate with actual sample
        double[] sample = {1.5, -0.8, 2.1, 0.3, -1.2, 0.7, 1.8, -0.5, 2.2, 0.9};
        double prediction = model.predict(new double[][]{sample})[0];
        double[] probabilities = model.predictProba(new double[][]{sample})[0];
        
        System.out.println("Live Prediction Example:");
        System.out.println("Sample: " + Arrays.toString(sample));
        System.out.println("Predicted class: " + (int)prediction);
        System.out.printf("Class probabilities: [");
        for (int i = 0; i < probabilities.length; i++) {
            System.out.printf("%.3f", probabilities[i]);
            if (i < probabilities.length - 1) System.out.print(", ");
        }
        System.out.println("]");
    }
    
    private static int[] getClassDistribution(double[] y) {
        Map<Integer, Integer> counts = new HashMap<>();
        for (double label : y) {
            int intLabel = (int) label;
            counts.put(intLabel, counts.getOrDefault(intLabel, 0) + 1);
        }
        
        int maxClass = counts.keySet().stream().mapToInt(Integer::intValue).max().orElse(0);
        int[] distribution = new int[maxClass + 1];
        counts.forEach((k, v) -> distribution[k] = v);
        
        return distribution;
    }
    
    private static DataSplit trainTestSplit(double[][] X, double[] y, double trainRatio) {
        int nSamples = X.length;
        int trainSize = (int) (nSamples * trainRatio);
        
        // Create indices and shuffle
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < nSamples; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, new Random(42));
        
        // Split data
        double[][] XTrain = new double[trainSize][];
        double[][] XTest = new double[nSamples - trainSize][];
        double[] yTrain = new double[trainSize];
        double[] yTest = new double[nSamples - trainSize];
        
        for (int i = 0; i < trainSize; i++) {
            int idx = indices.get(i);
            XTrain[i] = X[idx].clone();
            yTrain[i] = y[idx];
        }
        
        for (int i = trainSize; i < nSamples; i++) {
            int idx = indices.get(i);
            XTest[i - trainSize] = X[idx].clone();
            yTest[i - trainSize] = y[idx];
        }
        
        return new DataSplit(XTrain, XTest, yTrain, yTest);
    }
    
    // ========================================
    // DATA CLASSES
    // ========================================
    
    private static class MulticlassDataset {
        public final double[][] X;
        public final double[] y;
        public final int nClasses;
        public final String[] featureNames;
        
        public MulticlassDataset(double[][] X, double[] y, int nClasses, String[] featureNames) {
            this.X = X;
            this.y = y;
            this.nClasses = nClasses;
            this.featureNames = featureNames;
        }
    }
    
    private static class DataSplit {
        public final double[][] XTrain;
        public final double[][] XTest;
        public final double[] yTrain;
        public final double[] yTest;
        
        public DataSplit(double[][] XTrain, double[][] XTest, double[] yTrain, double[] yTest) {
            this.XTrain = XTrain;
            this.XTest = XTest;
            this.yTrain = yTrain;
            this.yTest = yTest;
        }
    }
}
