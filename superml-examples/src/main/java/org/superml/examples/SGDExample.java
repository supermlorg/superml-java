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

import org.superml.linear_model.SGDClassifier;
import org.superml.linear_model.SGDRegressor;
// NOTE: AutoTrainer will be available in next release
//import org.superml.autotrainer.LinearModelAutoTrainer;
//import org.superml.autotrainer.LinearModelAutoTrainer.ModelType;
//import org.superml.autotrainer.LinearModelAutoTrainer.AutoTrainingResult;

import java.util.Random;

/**
 * Comprehensive example demonstrating SGD algorithms in SuperML
 * 
 * This example shows:
 * 1. SGDClassifier with different loss functions and regularization
 * 2. SGDRegressor with different loss functions  
 * 3. AutoTrainer integration for hyperparameter optimization
 * 4. Performance comparison between different configurations
 * 
 * @author SuperML Team
 */
public class SGDExample {
    
    public static void main(String[] args) {
        System.out.println("=== SuperML SGD Example ===");
        System.out.println();
        
        // Generate sample data
        DataGenerator generator = new DataGenerator();
        
        // Classification example
        System.out.println("1. SGD Classification Example");
        System.out.println("------------------------------");
        
        DataGenerator.ClassificationData classData = generator.generateClassificationData(1000, 20, 2);
        runClassificationExample(classData);
        
        System.out.println();
        
        // Regression example
        System.out.println("2. SGD Regression Example");
        System.out.println("--------------------------");
        
        DataGenerator.RegressionData regData = generator.generateRegressionData(1000, 15);
        runRegressionExample(regData);
        
        System.out.println();
        
        // AutoTrainer examples (coming in next release)
        System.out.println("3. AutoTrainer Integration Example");
        System.out.println("-----------------------------------");
        System.out.println("AutoTrainer integration will be available in the next release.");
        System.out.println("It will provide automated hyperparameter optimization for SGD models.");
        
        // TODO: Uncomment when AutoTrainer is available
        // runAutoTrainerExample(classData, regData);
        
        System.out.println();
        System.out.println("=== SGD Example Complete ===");
    }
    
    private static void runClassificationExample(DataGenerator.ClassificationData data) {
        System.out.println("Dataset: " + data.X.length + " samples, " + data.X[0].length + " features");
        
        // Split data
        int trainSize = (int)(data.X.length * 0.8);
        double[][] XTrain = new double[trainSize][];
        double[] yTrain = new double[trainSize];
        double[][] XTest = new double[data.X.length - trainSize][];
        double[] yTest = new double[data.X.length - trainSize];
        
        System.arraycopy(data.X, 0, XTrain, 0, trainSize);
        System.arraycopy(data.y, 0, yTrain, 0, trainSize);
        System.arraycopy(data.X, trainSize, XTest, 0, data.X.length - trainSize);
        System.arraycopy(data.y, trainSize, yTest, 0, data.X.length - trainSize);
        
        // Test different loss functions
        String[] losses = {"hinge", "log", "squared_hinge", "modified_huber"};
        String[] penalties = {"l1", "l2", "elasticnet"};
        
        for (String loss : losses) {
            for (String penalty : penalties) {
                System.out.println("\nTesting SGDClassifier: loss=" + loss + ", penalty=" + penalty);
                
                SGDClassifier sgd = new SGDClassifier()
                    .setLoss(loss)
                    .setPenalty(penalty)
                    .setAlpha(0.001)
                    .setMaxIter(1000);
                
                if ("elasticnet".equals(penalty)) {
                    sgd.setL1Ratio(0.5);
                }
                
                // Train
                long startTime = System.currentTimeMillis();
                sgd.fit(XTrain, yTrain);
                long trainTime = System.currentTimeMillis() - startTime;
                
                // Predict
                double[] predictions = sgd.predict(XTest);
                double accuracy = calculateAccuracy(predictions, yTest);
                
                System.out.printf("  Training time: %d ms\n", trainTime);
                System.out.printf("  Test accuracy: %.4f\n", accuracy);
                
                // Show class probabilities for log loss
                if ("log".equals(loss)) {
                    double[][] probabilities = sgd.predictProba(XTest);
                    System.out.printf("  Sample probabilities: [%.3f, %.3f]\n", 
                                     probabilities[0][0], probabilities[0][1]);
                }
            }
        }
    }
    
    private static void runRegressionExample(DataGenerator.RegressionData data) {
        System.out.println("Dataset: " + data.X.length + " samples, " + data.X[0].length + " features");
        
        // Split data
        int trainSize = (int)(data.X.length * 0.8);
        double[][] XTrain = new double[trainSize][];
        double[] yTrain = new double[trainSize];
        double[][] XTest = new double[data.X.length - trainSize][];
        double[] yTest = new double[data.X.length - trainSize];
        
        System.arraycopy(data.X, 0, XTrain, 0, trainSize);
        System.arraycopy(data.y, 0, yTrain, 0, trainSize);
        System.arraycopy(data.X, trainSize, XTest, 0, data.X.length - trainSize);
        System.arraycopy(data.y, trainSize, yTest, 0, data.X.length - trainSize);
        
        // Test different loss functions
        String[] losses = {"squared_loss", "huber", "epsilon_insensitive"};
        String[] penalties = {"l1", "l2", "elasticnet"};
        
        for (String loss : losses) {
            for (String penalty : penalties) {
                System.out.println("\nTesting SGDRegressor: loss=" + loss + ", penalty=" + penalty);
                
                SGDRegressor sgd = new SGDRegressor()
                    .setLoss(loss)
                    .setPenalty(penalty)
                    .setAlpha(0.001)
                    .setMaxIter(1000);
                
                if ("elasticnet".equals(penalty)) {
                    sgd.setL1Ratio(0.5);
                }
                
                if ("epsilon_insensitive".equals(loss)) {
                    sgd.setEpsilon(0.1);
                }
                
                // Train
                long startTime = System.currentTimeMillis();
                sgd.fit(XTrain, yTrain);
                long trainTime = System.currentTimeMillis() - startTime;
                
                // Predict
                double[] predictions = sgd.predict(XTest);
                double r2Score = sgd.score(XTest, yTest);
                double mse = calculateMSE(predictions, yTest);
                
                System.out.printf("  Training time: %d ms\n", trainTime);
                System.out.printf("  R² score: %.4f\n", r2Score);
                System.out.printf("  MSE: %.4f\n", mse);
            }
        }
    }
    
    /*
    // AutoTrainer method - Will be enabled when AutoTrainer module is available
    private static void runAutoTrainerExample(DataGenerator.ClassificationData classData, 
                                            DataGenerator.RegressionData regData) {
        LinearModelAutoTrainer autoTrainer = new LinearModelAutoTrainer();
        
        System.out.println("AutoTraining SGDClassifier...");
        AutoTrainingResult classResult = autoTrainer.autoTrain(
            classData.X, classData.y, ModelType.SGD_CLASSIFIER);
        
        System.out.printf("Best SGDClassifier score: %.4f\n", classResult.bestScore);
        System.out.println("Best parameters: " + classResult.bestParameters);
        System.out.printf("Training time: %d ms\n", classResult.trainingTime);
        
        System.out.println();
        
        System.out.println("AutoTraining SGDRegressor...");
        AutoTrainingResult regResult = autoTrainer.autoTrain(
            regData.X, regData.y, ModelType.SGD_REGRESSOR);
        
        System.out.printf("Best SGDRegressor score: %.4f\n", regResult.bestScore);
        System.out.println("Best parameters: " + regResult.bestParameters);
        System.out.printf("Training time: %d ms\n", regResult.trainingTime);
    }
    */
    
    private static double calculateAccuracy(double[] predictions, double[] actual) {
        int correct = 0;
        for (int i = 0; i < predictions.length; i++) {
            if (Math.round(predictions[i]) == Math.round(actual[i])) {
                correct++;
            }
        }
        return (double) correct / predictions.length;
    }
    
    private static double calculateMSE(double[] predictions, double[] actual) {
        double sum = 0.0;
        for (int i = 0; i < predictions.length; i++) {
            double error = predictions[i] - actual[i];
            sum += error * error;
        }
        return sum / predictions.length;
    }
    
    /**
     * Utility class for generating synthetic datasets
     */
    public static class DataGenerator {
        private Random random = new Random(42);
        
        public ClassificationData generateClassificationData(int nSamples, int nFeatures, int nClasses) {
            double[][] X = new double[nSamples][nFeatures];
            double[] y = new double[nSamples];
            
            // Generate features
            for (int i = 0; i < nSamples; i++) {
                for (int j = 0; j < nFeatures; j++) {
                    X[i][j] = random.nextGaussian();
                }
                
                // Generate labels based on linear combination of features
                double score = 0.0;
                for (int j = 0; j < Math.min(5, nFeatures); j++) {
                    score += X[i][j] * (j + 1) * 0.5;
                }
                y[i] = score > 0 ? 1.0 : 0.0;
            }
            
            return new ClassificationData(X, y);
        }
        
        public RegressionData generateRegressionData(int nSamples, int nFeatures) {
            double[][] X = new double[nSamples][nFeatures];
            double[] y = new double[nSamples];
            
            // Generate features
            for (int i = 0; i < nSamples; i++) {
                for (int j = 0; j < nFeatures; j++) {
                    X[i][j] = random.nextGaussian();
                }
                
                // Generate target as linear combination with noise
                double target = 0.0;
                for (int j = 0; j < nFeatures; j++) {
                    target += X[i][j] * (j + 1) * 0.3;
                }
                target += random.nextGaussian() * 0.1; // Add noise
                y[i] = target;
            }
            
            return new RegressionData(X, y);
        }
        
        public static class ClassificationData {
            public final double[][] X;
            public final double[] y;
            
            public ClassificationData(double[][] X, double[] y) {
                this.X = X;
                this.y = y;
            }
        }
        
        public static class RegressionData {
            public final double[][] X;
            public final double[] y;
            
            public RegressionData(double[][] X, double[] y) {
                this.X = X;
                this.y = y;
            }
        }
    }
}
