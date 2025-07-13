package org.superml.examples;

import org.superml.datasets.Datasets;
import org.superml.linear_model.LogisticRegression;
import org.superml.linear_model.OneVsRestClassifier;
import org.superml.linear_model.SoftmaxRegression;
import org.superml.metrics.Metrics;

/**
 * Example demonstrating multiclass classification with different strategies.
 */
public class MulticlassClassificationExample {
    
    public static void main(String[] args) {
        System.out.println("=== Multiclass Classification Example ===\n");
        
        // Generate multiclass dataset
        System.out.println("Generating synthetic multiclass dataset...");
        Datasets.ClassificationData data = Datasets.makeClassification(1000, 10, 4);
        
        double[][] X = data.X;
        int[] yInt = data.y;
        
        // Convert int labels to double for compatibility
        double[] y = new double[yInt.length];
        for (int i = 0; i < yInt.length; i++) {
            y[i] = yInt[i];
        }
        
        System.out.printf("Dataset: %d samples, %d features, %d classes\n", 
                X.length, X[0].length, 4);
        
        // Split data into train/test
        int trainSize = (int) (X.length * 0.7);
        double[][] XTrain = new double[trainSize][];
        double[][] XTest = new double[X.length - trainSize][];
        double[] yTrain = new double[trainSize];
        double[] yTest = new double[X.length - trainSize];
        
        for (int i = 0; i < trainSize; i++) {
            XTrain[i] = X[i];
            yTrain[i] = y[i];
        }
        
        for (int i = 0; i < X.length - trainSize; i++) {
            XTest[i] = X[trainSize + i];
            yTest[i] = y[trainSize + i];
        }
        
        System.out.printf("Train size: %d, Test size: %d\n\n", trainSize, XTest.length);
        
        // Example 1: LogisticRegression with One-vs-Rest (default)
        runLogisticRegressionOvR(XTrain, yTrain, XTest, yTest);
        
        // Example 2: LogisticRegression with Softmax (multinomial)
        runLogisticRegressionSoftmax(XTrain, yTrain, XTest, yTest);
        
        // Example 3: Direct One-vs-Rest classifier
        runOneVsRestClassifier(XTrain, yTrain, XTest, yTest);
        
        // Example 4: Direct Softmax regression
        runSoftmaxRegression(XTrain, yTrain, XTest, yTest);
        
        // Example 5: Compare probabilities
        compareProbabilities(XTrain, yTrain, XTest);
    }
    
    private static void runLogisticRegressionOvR(double[][] XTrain, double[] yTrain, 
                                                 double[][] XTest, double[] yTest) {
        System.out.println("=== LogisticRegression with One-vs-Rest ===");
        
        LogisticRegression lr = new LogisticRegression(0.01, 1000)
                .setMultiClass("ovr")
                .setC(1.0);
        
        // Train
        long startTime = System.currentTimeMillis();
        lr.fit(XTrain, yTrain);
        long trainTime = System.currentTimeMillis() - startTime;
        
        // Predict
        double[] predictions = lr.predict(XTest);
        double accuracy = lr.score(XTest, yTest);
        
        System.out.printf("Training time: %d ms\n", trainTime);
        System.out.printf("Accuracy: %.4f\n", accuracy);
        System.out.printf("Classes: %s\n", java.util.Arrays.toString(lr.getClasses()));
        System.out.println();
    }
    
    private static void runLogisticRegressionSoftmax(double[][] XTrain, double[] yTrain, 
                                                     double[][] XTest, double[] yTest) {
        System.out.println("=== LogisticRegression with Softmax (Multinomial) ===");
        
        LogisticRegression lr = new LogisticRegression(0.01, 1000)
                .setMultiClass("multinomial")
                .setC(1.0);
        
        // Train
        long startTime = System.currentTimeMillis();
        lr.fit(XTrain, yTrain);
        long trainTime = System.currentTimeMillis() - startTime;
        
        // Predict
        double[] predictions = lr.predict(XTest);
        double accuracy = lr.score(XTest, yTest);
        
        System.out.printf("Training time: %d ms\n", trainTime);
        System.out.printf("Accuracy: %.4f\n", accuracy);
        System.out.printf("Classes: %s\n", java.util.Arrays.toString(lr.getClasses()));
        System.out.println();
    }
    
    private static void runOneVsRestClassifier(double[][] XTrain, double[] yTrain, 
                                               double[][] XTest, double[] yTest) {
        System.out.println("=== Direct One-vs-Rest Classifier ===");
        
        LogisticRegression baseClassifier = new LogisticRegression(0.01, 1000);
        OneVsRestClassifier ovr = new OneVsRestClassifier(baseClassifier);
        
        // Train
        long startTime = System.currentTimeMillis();
        ovr.fit(XTrain, yTrain);
        long trainTime = System.currentTimeMillis() - startTime;
        
        // Predict
        double[] predictions = ovr.predict(XTest);
        double accuracy = ovr.score(XTest, yTest);
        
        System.out.printf("Training time: %d ms\n", trainTime);
        System.out.printf("Accuracy: %.4f\n", accuracy);
        System.out.printf("Classes: %s\n", java.util.Arrays.toString(ovr.getClasses()));
        System.out.printf("Number of classifiers: %d\n", ovr.getClassifiers().size());
        System.out.println();
    }
    
    private static void runSoftmaxRegression(double[][] XTrain, double[] yTrain, 
                                             double[][] XTest, double[] yTest) {
        System.out.println("=== Direct Softmax Regression ===");
        
        SoftmaxRegression softmax = new SoftmaxRegression(0.01, 1000)
                .setC(1.0);
        
        // Train
        long startTime = System.currentTimeMillis();
        softmax.fit(XTrain, yTrain);
        long trainTime = System.currentTimeMillis() - startTime;
        
        // Predict
        double[] predictions = softmax.predict(XTest);
        double accuracy = softmax.score(XTest, yTest);
        
        System.out.printf("Training time: %d ms\n", trainTime);
        System.out.printf("Accuracy: %.4f\n", accuracy);
        System.out.printf("Classes: %s\n", java.util.Arrays.toString(softmax.getClasses()));
        System.out.printf("Weight matrix shape: %d x %d\n", 
                softmax.getWeights().length, softmax.getWeights()[0].length);
        System.out.println();
    }
    
    private static void compareProbabilities(double[][] XTrain, double[] yTrain, double[][] XTest) {
        System.out.println("=== Probability Comparison ===");
        
        // Train both models
        LogisticRegression lrOvr = new LogisticRegression().setMultiClass("ovr");
        LogisticRegression lrSoftmax = new LogisticRegression().setMultiClass("multinomial");
        SoftmaxRegression softmax = new SoftmaxRegression();
        
        lrOvr.fit(XTrain, yTrain);
        lrSoftmax.fit(XTrain, yTrain);
        softmax.fit(XTrain, yTrain);
        
        // Get probabilities for first few test samples
        double[][] probOvr = lrOvr.predictProba(XTest);
        double[][] probSoftmax = lrSoftmax.predictProba(XTest);
        double[][] probDirectSoftmax = softmax.predictProba(XTest);
        
        System.out.println("Probabilities for first 3 test samples:");
        System.out.println("Sample | OvR      | Softmax  | Direct Softmax");
        System.out.println("-------|----------|----------|---------------");
        
        for (int i = 0; i < Math.min(3, XTest.length); i++) {
            System.out.printf("   %d   | ", i + 1);
            
            // OvR probabilities
            System.out.print("[");
            for (int j = 0; j < probOvr[i].length; j++) {
                System.out.printf("%.3f", probOvr[i][j]);
                if (j < probOvr[i].length - 1) System.out.print(",");
            }
            System.out.print("] | ");
            
            // Softmax probabilities
            System.out.print("[");
            for (int j = 0; j < probSoftmax[i].length; j++) {
                System.out.printf("%.3f", probSoftmax[i][j]);
                if (j < probSoftmax[i].length - 1) System.out.print(",");
            }
            System.out.print("] | ");
            
            // Direct Softmax probabilities
            System.out.print("[");
            for (int j = 0; j < probDirectSoftmax[i].length; j++) {
                System.out.printf("%.3f", probDirectSoftmax[i][j]);
                if (j < probDirectSoftmax[i].length - 1) System.out.print(",");
            }
            System.out.println("]");
        }
        
        System.out.println("\nNote: OvR probabilities are normalized, Softmax probabilities naturally sum to 1");
    }
}
