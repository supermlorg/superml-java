package org.superml.examples;

import org.superml.datasets.Datasets;
import org.superml.tree.DecisionTree;
import org.superml.tree.RandomForest;
import org.superml.tree.GradientBoosting;
import org.superml.metrics.Metrics;

/**
 * Example demonstrating tree-based algorithms: Decision Trees, Random Forest, and Gradient Boosting.
 */
public class TreeAlgorithmsExample {
    
    public static void main(String[] args) {
        System.out.println("=== Tree-Based Algorithms Example ===\n");
        
        // Test Classification
        demonstrateClassification();
        
        System.out.println("\n" + "=".repeat(50) + "\n");
        
        // Test Regression
        demonstrateRegression();
    }
    
    private static void demonstrateClassification() {
        System.out.println("=== CLASSIFICATION EXAMPLE ===");
        
        // Generate classification dataset
        System.out.println("Generating classification dataset...");
        Datasets.ClassificationData dataset = Datasets.makeClassification(1000, 20, 2);
        double[][] X = dataset.X;
        double[] y = new double[dataset.y.length];
        for (int i = 0; i < dataset.y.length; i++) {
            y[i] = dataset.y[i];
        }
        
        // Split data
        int trainSize = (int) (X.length * 0.8);
        double[][] XTrain = new double[trainSize][];
        double[][] XTest = new double[X.length - trainSize][];
        double[] yTrain = new double[trainSize];
        double[] yTest = new double[X.length - trainSize];
        
        System.arraycopy(X, 0, XTrain, 0, trainSize);
        System.arraycopy(X, trainSize, XTest, 0, X.length - trainSize);
        System.arraycopy(y, 0, yTrain, 0, trainSize);
        System.arraycopy(y, trainSize, yTest, 0, X.length - trainSize);
        
        System.out.println("Training set size: " + trainSize);
        System.out.println("Test set size: " + (X.length - trainSize));
        System.out.println();
        
        // 1. Decision Tree
        System.out.println("--- Decision Tree ---");
        DecisionTree dt = new DecisionTree("gini", 10);
        long startTime = System.currentTimeMillis();
        dt.fit(XTrain, yTrain);
        long trainTime = System.currentTimeMillis() - startTime;
        
        startTime = System.currentTimeMillis();
        double[] dtPreds = dt.predict(XTest);
        long predTime = System.currentTimeMillis() - startTime;
        
        double dtAccuracy = Metrics.accuracy(yTest, dtPreds);
        System.out.println("Accuracy: " + String.format("%.4f", dtAccuracy));
        System.out.println("Training time: " + trainTime + "ms");
        System.out.println("Prediction time: " + predTime + "ms");
        System.out.println();
        
        // 2. Random Forest
        System.out.println("--- Random Forest ---");
        RandomForest rf = new RandomForest(100, 10);
        startTime = System.currentTimeMillis();
        rf.fit(XTrain, yTrain);
        trainTime = System.currentTimeMillis() - startTime;
        
        startTime = System.currentTimeMillis();
        double[] rfPreds = rf.predict(XTest);
        predTime = System.currentTimeMillis() - startTime;
        
        double rfAccuracy = Metrics.accuracy(yTest, rfPreds);
        System.out.println("Accuracy: " + String.format("%.4f", rfAccuracy));
        System.out.println("Training time: " + trainTime + "ms");
        System.out.println("Prediction time: " + predTime + "ms");
        System.out.println("Number of trees: " + rf.getTrees().size());
        System.out.println();
        
        // 3. Gradient Boosting
        System.out.println("--- Gradient Boosting ---");
        GradientBoosting gb = new GradientBoosting(100, 0.1, 6);
        startTime = System.currentTimeMillis();
        gb.fit(XTrain, yTrain);
        trainTime = System.currentTimeMillis() - startTime;
        
        startTime = System.currentTimeMillis();
        double[] gbPreds = gb.predict(XTest);
        predTime = System.currentTimeMillis() - startTime;
        
        double gbAccuracy = Metrics.accuracy(yTest, gbPreds);
        System.out.println("Accuracy: " + String.format("%.4f", gbAccuracy));
        System.out.println("Training time: " + trainTime + "ms");
        System.out.println("Prediction time: " + predTime + "ms");
        System.out.println("Number of trees: " + gb.getTrees().size());
        
        // Show probability predictions for a few samples
        System.out.println("\n--- Probability Predictions (first 5 test samples) ---");
        double[][] rfProba = rf.predictProba(java.util.Arrays.copyOfRange(XTest, 0, 5));
        double[][] gbProba = gb.predictProba(java.util.Arrays.copyOfRange(XTest, 0, 5));
        
        for (int i = 0; i < 5; i++) {
            System.out.println("Sample " + (i+1) + ":");
            System.out.println("  True class: " + yTest[i]);
            System.out.println("  RF probabilities: [" + 
                String.format("%.3f", rfProba[i][0]) + ", " + 
                String.format("%.3f", rfProba[i][1]) + "]");
            System.out.println("  GB probabilities: [" + 
                String.format("%.3f", gbProba[i][0]) + ", " + 
                String.format("%.3f", gbProba[i][1]) + "]");
        }
    }
    
    private static void demonstrateRegression() {
        System.out.println("=== REGRESSION EXAMPLE ===");
        
        // Generate regression dataset
        System.out.println("Generating regression dataset...");
        Datasets.RegressionData dataset = Datasets.makeRegression(1000, 10, 1, 0.1);
        double[][] X = dataset.X;
        double[] y = dataset.y;
        
        // Split data
        int trainSize = (int) (X.length * 0.8);
        double[][] XTrain = new double[trainSize][];
        double[][] XTest = new double[X.length - trainSize][];
        double[] yTrain = new double[trainSize];
        double[] yTest = new double[X.length - trainSize];
        
        System.arraycopy(X, 0, XTrain, 0, trainSize);
        System.arraycopy(X, trainSize, XTest, 0, X.length - trainSize);
        System.arraycopy(y, 0, yTrain, 0, trainSize);
        System.arraycopy(y, trainSize, yTest, 0, X.length - trainSize);
        
        System.out.println("Training set size: " + trainSize);
        System.out.println("Test set size: " + (X.length - trainSize));
        System.out.println();
        
        // 1. Decision Tree
        System.out.println("--- Decision Tree Regression ---");
        DecisionTree dt = new DecisionTree("mse", 10);
        long startTime = System.currentTimeMillis();
        dt.fit(XTrain, yTrain);
        long trainTime = System.currentTimeMillis() - startTime;
        
        startTime = System.currentTimeMillis();
        double[] dtPreds = dt.predict(XTest);
        long predTime = System.currentTimeMillis() - startTime;
        
        double dtR2 = calculateR2Score(yTest, dtPreds);
        double dtMse = Metrics.meanSquaredError(yTest, dtPreds);
        System.out.println("R² Score: " + String.format("%.4f", dtR2));
        System.out.println("MSE: " + String.format("%.4f", dtMse));
        System.out.println("Training time: " + trainTime + "ms");
        System.out.println("Prediction time: " + predTime + "ms");
        System.out.println();
        
        // 2. Random Forest
        System.out.println("--- Random Forest Regression ---");
        RandomForest rf = new RandomForest(100, 10);
        startTime = System.currentTimeMillis();
        rf.fit(XTrain, yTrain);
        trainTime = System.currentTimeMillis() - startTime;
        
        startTime = System.currentTimeMillis();
        double[] rfPreds = rf.predict(XTest);
        predTime = System.currentTimeMillis() - startTime;
        
        double rfR2 = calculateR2Score(yTest, rfPreds);
        double rfMse = Metrics.meanSquaredError(yTest, rfPreds);
        System.out.println("R² Score: " + String.format("%.4f", rfR2));
        System.out.println("MSE: " + String.format("%.4f", rfMse));
        System.out.println("Training time: " + trainTime + "ms");
        System.out.println("Prediction time: " + predTime + "ms");
        System.out.println("Number of trees: " + rf.getTrees().size());
        System.out.println();
        
        // 3. Gradient Boosting
        System.out.println("--- Gradient Boosting Regression ---");
        GradientBoosting gb = new GradientBoosting(100, 0.1, 6);
        startTime = System.currentTimeMillis();
        gb.fit(XTrain, yTrain);
        trainTime = System.currentTimeMillis() - startTime;
        
        startTime = System.currentTimeMillis();
        double[] gbPreds = gb.predict(XTest);
        predTime = System.currentTimeMillis() - startTime;
        
        double gbR2 = calculateR2Score(yTest, gbPreds);
        double gbMse = Metrics.meanSquaredError(yTest, gbPreds);
        System.out.println("R² Score: " + String.format("%.4f", gbR2));
        System.out.println("MSE: " + String.format("%.4f", gbMse));
        System.out.println("Training time: " + trainTime + "ms");
        System.out.println("Prediction time: " + predTime + "ms");
        System.out.println("Number of trees: " + gb.getTrees().size());
        
        // Show predictions vs actual for a few samples
        System.out.println("\n--- Predictions vs Actual (first 5 test samples) ---");
        for (int i = 0; i < 5; i++) {
            System.out.println("Sample " + (i+1) + ":");
            System.out.println("  Actual: " + String.format("%.3f", yTest[i]));
            System.out.println("  DT pred: " + String.format("%.3f", dtPreds[i]));
            System.out.println("  RF pred: " + String.format("%.3f", rfPreds[i]));
            System.out.println("  GB pred: " + String.format("%.3f", gbPreds[i]));
        }
    }
    
    private static double calculateR2Score(double[] yTrue, double[] yPred) {
        double totalSumSquares = 0.0;
        double residualSumSquares = 0.0;
        double mean = java.util.Arrays.stream(yTrue).average().orElse(0.0);
        
        for (int i = 0; i < yTrue.length; i++) {
            totalSumSquares += Math.pow(yTrue[i] - mean, 2);
            residualSumSquares += Math.pow(yTrue[i] - yPred[i], 2);
        }
        
        return 1.0 - (residualSumSquares / totalSumSquares);
    }
}
