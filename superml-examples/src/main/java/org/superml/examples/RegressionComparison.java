package org.superml.examples;

import org.superml.linear_model.LinearRegression;
import org.superml.linear_model.Ridge;
import org.superml.linear_model.Lasso;
import org.superml.tree.DecisionTree;
import org.superml.tree.RandomForest;
import org.superml.tree.GradientBoosting;
import org.superml.metrics.Metrics;

/**
 * Regression Comparison Example
 * Compares different regression algorithms on the same dataset
 */
public class RegressionComparison {
    
    public static void main(String[] args) {
        System.out.println("=== SuperML 2.0.0 - Regression Algorithms Comparison ===\n");
        
        try {
            // Generate synthetic regression dataset
            double[][] X = generateRegressionData(500, 5);
            double[] y = generateRegressionTarget(X);
            System.out.println("📊 Generated synthetic regression dataset:");
            System.out.printf("   - %d samples, %d features\n", X.length, X[0].length);
            System.out.printf("   - Target range: [%.2f, %.2f]\n", getMin(y), getMax(y));
            
            // Compare different regression algorithms
            compareRegressionAlgorithms(X, y);
            
            // Cross-validation comparison
            performCrossValidationComparison(X, y);
            
            // Feature importance analysis
            analyzeFeatureImportance(X, y);
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void compareRegressionAlgorithms(double[][] X, double[] y) {
        System.out.println("\n🔍 Regression Algorithms Comparison");
        System.out.println("===================================");
        
        // Split data for training and testing
        int trainSize = (int) (X.length * 0.8);
        double[][] XTrain = new double[trainSize][];
        double[][] XTest = new double[X.length - trainSize][];
        double[] yTrain = new double[trainSize];
        double[] yTest = new double[X.length - trainSize];
        
        System.arraycopy(X, 0, XTrain, 0, trainSize);
        System.arraycopy(X, trainSize, XTest, 0, X.length - trainSize);
        System.arraycopy(y, 0, yTrain, 0, trainSize);
        System.arraycopy(y, trainSize, yTest, 0, X.length - trainSize);
        
        System.out.printf("📈 Training on %d samples, testing on %d samples\n\n", 
                         trainSize, X.length - trainSize);
        
        System.out.println("Algorithm             | R² Score | RMSE   | MAE    | Training Time");
        System.out.println("---------------------|----------|--------|--------|---------------");
        
        // Linear Regression
        evaluateRegressor("Linear Regression", new LinearRegression(), XTrain, yTrain, XTest, yTest);
        
        // Ridge Regression
        Ridge ridge = new Ridge();
        ridge.setAlpha(1.0);
        evaluateRegressor("Ridge Regression", ridge, XTrain, yTrain, XTest, yTest);
        
        // Lasso Regression
        Lasso lasso = new Lasso();
        lasso.setAlpha(0.1);
        evaluateRegressor("Lasso Regression", lasso, XTrain, yTrain, XTest, yTest);
        
        // Decision Tree
        DecisionTree dt = new DecisionTree();
        dt.setCriterion("mse");
        evaluateRegressor("Decision Tree", dt, XTrain, yTrain, XTest, yTest);
        
        // Random Forest
        RandomForest rf = new RandomForest();
        rf.setNEstimators(100);
        evaluateRegressor("Random Forest", rf, XTrain, yTrain, XTest, yTest);
        
        // Gradient Boosting
        GradientBoosting gb = new GradientBoosting();
        gb.setNEstimators(100);
        gb.setLearningRate(0.1);
        evaluateRegressor("Gradient Boosting", gb, XTrain, yTrain, XTest, yTest);
    }
    
    private static void evaluateRegressor(String name, Object regressor, 
                                        double[][] XTrain, double[] yTrain,
                                        double[][] XTest, double[] yTest) {
        try {
            long startTime = System.currentTimeMillis();
            
            // Train the model
            if (regressor instanceof LinearRegression) {
                ((LinearRegression) regressor).fit(XTrain, yTrain);
            } else if (regressor instanceof Ridge) {
                ((Ridge) regressor).fit(XTrain, yTrain);
            } else if (regressor instanceof Lasso) {
                ((Lasso) regressor).fit(XTrain, yTrain);
            } else if (regressor instanceof DecisionTree) {
                ((DecisionTree) regressor).fit(XTrain, yTrain);
            } else if (regressor instanceof RandomForest) {
                ((RandomForest) regressor).fit(XTrain, yTrain);
            } else if (regressor instanceof GradientBoosting) {
                ((GradientBoosting) regressor).fit(XTrain, yTrain);
            }
            
            long trainingTime = System.currentTimeMillis() - startTime;
            
            // Make predictions
            double[] predictions = null;
            if (regressor instanceof LinearRegression) {
                predictions = ((LinearRegression) regressor).predict(XTest);
            } else if (regressor instanceof Ridge) {
                predictions = ((Ridge) regressor).predict(XTest);
            } else if (regressor instanceof Lasso) {
                predictions = ((Lasso) regressor).predict(XTest);
            } else if (regressor instanceof DecisionTree) {
                predictions = ((DecisionTree) regressor).predict(XTest);
            } else if (regressor instanceof RandomForest) {
                predictions = ((RandomForest) regressor).predict(XTest);
            } else if (regressor instanceof GradientBoosting) {
                predictions = ((GradientBoosting) regressor).predict(XTest);
            }
            
            // Calculate metrics
            double r2 = Metrics.r2Score(yTest, predictions);
            double rmse = Metrics.meanSquaredError(yTest, predictions);
            rmse = Math.sqrt(rmse);
            double mae = Metrics.meanAbsoluteError(yTest, predictions);
            
            System.out.printf("%-20s | %8.4f | %6.3f | %6.3f | %10dms\n", 
                             name, r2, rmse, mae, trainingTime);
            
        } catch (Exception e) {
            System.out.printf("%-20s | %8s | %6s | %6s | %10s\n", 
                             name, "ERROR", "ERROR", "ERROR", "ERROR");
        }
    }
    
    private static void performCrossValidationComparison(double[][] X, double[] y) {
        System.out.println("\n📊 Cross-Validation Comparison (5-fold)");
        System.out.println("=======================================");
        
        System.out.println("Algorithm             | CV R² Score | CV RMSE | Std Dev");
        System.out.println("---------------------|-------------|---------|--------");
        
        try {
            // Linear Regression CV
            performCV("Linear Regression", new LinearRegression(), X, y);
            
            // Ridge Regression CV
            Ridge ridge = new Ridge();
            ridge.setAlpha(1.0);
            performCV("Ridge Regression", ridge, X, y);
            
            // Random Forest CV
            RandomForest rf = new RandomForest();
            rf.setNEstimators(50); // Fewer trees for faster CV
            performCV("Random Forest", rf, X, y);
            
        } catch (Exception e) {
            System.err.println("Error in cross-validation: " + e.getMessage());
        }
    }
    
    private static void performCV(String name, Object regressor, double[][] X, double[] y) {
        try {
            // Simplified cross-validation - manual implementation
            int folds = 5;
            int foldSize = X.length / folds;
            double[] scores = new double[folds];
            
            for (int fold = 0; fold < folds; fold++) {
                // Create train/validation split
                int start = fold * foldSize;
                int end = Math.min(start + foldSize, X.length);
                
                // Training data (all except current fold)
                double[][] XTrain = new double[X.length - (end - start)][];
                double[] yTrain = new double[X.length - (end - start)];
                double[][] XVal = new double[end - start][];
                double[] yVal = new double[end - start];
                
                // Fill validation data
                System.arraycopy(X, start, XVal, 0, end - start);
                System.arraycopy(y, start, yVal, 0, end - start);
                
                // Fill training data
                int trainIdx = 0;
                for (int i = 0; i < X.length; i++) {
                    if (i < start || i >= end) {
                        XTrain[trainIdx] = X[i];
                        yTrain[trainIdx] = y[i];
                        trainIdx++;
                    }
                }
                
                // Train and evaluate
                if (regressor instanceof LinearRegression) {
                    LinearRegression lr = new LinearRegression();
                    lr.fit(XTrain, yTrain);
                    double[] pred = lr.predict(XVal);
                    scores[fold] = Metrics.r2Score(yVal, pred);
                } else if (regressor instanceof Ridge) {
                    Ridge ridge = new Ridge();
                    ridge.setAlpha(((Ridge) regressor).getAlpha());
                    ridge.fit(XTrain, yTrain);
                    double[] pred = ridge.predict(XVal);
                    scores[fold] = Metrics.r2Score(yVal, pred);
                } else if (regressor instanceof DecisionTree) {
                    DecisionTree dt = new DecisionTree();
                    dt.setCriterion("mse");
                    dt.fit(XTrain, yTrain);
                    double[] pred = dt.predict(XVal);
                    scores[fold] = Metrics.r2Score(yVal, pred);
                }
            }
            
            double mean = calculateMean(scores);
            double std = calculateStd(scores, mean);
            double rmse = Math.sqrt(Math.abs(1.0 - mean)); // Approximate RMSE from R²
            
            System.out.printf("%-20s | %11.4f | %7.3f | %7.4f\n", 
                             name, mean, rmse, std);
            
        } catch (Exception e) {
            System.out.printf("%-20s | %11s | %7s | %7s\n", 
                             name, "ERROR", "ERROR", "ERROR");
        }
    }
    
    private static void analyzeFeatureImportance(double[][] X, double[] y) {
        System.out.println("\n🎯 Feature Importance Analysis");
        System.out.println("==============================");
        
        try {
            // Use coefficient analysis from linear models
            LinearRegression lr = new LinearRegression();
            lr.fit(X, y);
            double[] lrCoeffs = lr.getCoefficients();
            
            Ridge ridge = new Ridge();
            ridge.setAlpha(1.0);
            ridge.fit(X, y);
            double[] ridgeCoeffs = ridge.getCoefficients();
            
            System.out.println("Feature | Linear Reg | Ridge Reg  | Abs Avg");
            System.out.println("--------|------------|------------|--------");
            
            for (int i = 0; i < lrCoeffs.length; i++) {
                double absAvg = (Math.abs(lrCoeffs[i]) + Math.abs(ridgeCoeffs[i])) / 2.0;
                System.out.printf("  F%-4d | %10.6f | %10.6f | %6.4f\n", 
                                 i, lrCoeffs[i], ridgeCoeffs[i], absAvg);
            }
            
            // Find top 3 features by average absolute coefficient
            int[] topFeatures = getTopFeaturesByCoefficients(lrCoeffs, ridgeCoeffs, 3);
            System.out.println("\n🏆 Top 3 Most Important Features (by coefficient magnitude):");
            for (int i = 0; i < topFeatures.length; i++) {
                double absAvg = (Math.abs(lrCoeffs[topFeatures[i]]) + Math.abs(ridgeCoeffs[topFeatures[i]])) / 2.0;
                System.out.printf("   %d. Feature %d (avg coefficient magnitude: %.6f)\n", 
                                 i + 1, topFeatures[i], absAvg);
            }
            
        } catch (Exception e) {
            System.err.println("Error in feature importance analysis: " + e.getMessage());
        }
    }
    
    // Data generation methods
    private static double[][] generateRegressionData(int samples, int features) {
        double[][] data = new double[samples][features];
        java.util.Random random = new java.util.Random(42);
        
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < features; j++) {
                data[i][j] = random.nextGaussian() * 2.0; // Scale features
            }
        }
        return data;
    }
    
    private static double[] generateRegressionTarget(double[][] X) {
        double[] y = new double[X.length];
        java.util.Random random = new java.util.Random(42);
        
        // Create target as linear combination with some noise
        for (int i = 0; i < X.length; i++) {
            y[i] = 0.5 * X[i][0] + 1.2 * X[i][1] - 0.8 * X[i][2] + 
                   0.3 * X[i][3] + 0.9 * X[i][4] + random.nextGaussian() * 0.2;
        }
        return y;
    }
    
    // Utility methods
    private static double getMin(double[] array) {
        double min = array[0];
        for (double value : array) {
            if (value < min) min = value;
        }
        return min;
    }
    
    private static double getMax(double[] array) {
        double max = array[0];
        for (double value : array) {
            if (value > max) max = value;
        }
        return max;
    }
    
    private static double calculateMean(double[] values) {
        double sum = 0.0;
        for (double value : values) {
            sum += value;
        }
        return sum / values.length;
    }
    
    private static double calculateStd(double[] values, double mean) {
        double sum = 0.0;
        for (double value : values) {
            sum += Math.pow(value - mean, 2);
        }
        return Math.sqrt(sum / values.length);
    }
    
    private static int[] getTopFeaturesByCoefficients(double[] coeff1, double[] coeff2, int top) {
        int[] indices = new int[top];
        double[] avgMagnitudes = new double[coeff1.length];
        
        // Calculate average absolute coefficients
        for (int i = 0; i < coeff1.length; i++) {
            avgMagnitudes[i] = (Math.abs(coeff1[i]) + Math.abs(coeff2[i])) / 2.0;
        }
        
        // Find top features
        for (int i = 0; i < top; i++) {
            int maxIdx = 0;
            for (int j = 1; j < avgMagnitudes.length; j++) {
                if (avgMagnitudes[j] > avgMagnitudes[maxIdx]) {
                    maxIdx = j;
                }
            }
            indices[i] = maxIdx;
            avgMagnitudes[maxIdx] = Double.NEGATIVE_INFINITY; // Remove from consideration
        }
        return indices;
    }
}
