package com.superml.linear_model;

import com.superml.core.BaseEstimator;
import com.superml.core.Classifier;
import java.util.Arrays;

/**
 * Logistic Regression classifier.
 * Similar to sklearn.linear_model.LogisticRegression
 */
public class LogisticRegression extends BaseEstimator implements Classifier {
    
    private double[] weights;
    private double[] classes;
    private boolean fitted = false;
    
    // Hyperparameters
    private double learningRate = 0.01;
    private int maxIter = 1000;
    private double tolerance = 1e-6;
    private String solver = "lbfgs";
    private double C = 1.0; // Inverse of regularization strength
    
    public LogisticRegression() {
        params.put("learning_rate", learningRate);
        params.put("max_iter", maxIter);
        params.put("tol", tolerance);
        params.put("solver", solver);
        params.put("C", C);
    }
    
    public LogisticRegression(double learningRate, int maxIter) {
        this();
        this.learningRate = learningRate;
        this.maxIter = maxIter;
        params.put("learning_rate", learningRate);
        params.put("max_iter", maxIter);
    }
    
    @Override
    public LogisticRegression fit(double[][] X, double[] y) {
        int nFeatures = X[0].length;
        
        // Initialize weights
        weights = new double[nFeatures + 1]; // +1 for bias term
        
        // Get unique classes
        classes = Arrays.stream(y).distinct().sorted().toArray();
        
        // Add bias column to X
        double[][] XWithBias = addBiasColumn(X);
        
        // Gradient descent
        for (int iter = 0; iter < maxIter; iter++) {
            double[] predictions = predictProbabilities(XWithBias);
            double[] gradients = computeGradients(XWithBias, y, predictions);
            
            // Update weights
            for (int j = 0; j < weights.length; j++) {
                weights[j] -= learningRate * gradients[j];
            }
            
            // Check convergence
            double gradientNorm = Arrays.stream(gradients)
                .map(Math::abs)
                .max().orElse(0.0);
                
            if (gradientNorm < tolerance) {
                break;
            }
        }
        
        fitted = true;
        return this;
    }
    
    @Override
    public double[] predict(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before making predictions");
        }
        
        double[][] probabilities = predictProba(X);
        double[] predictions = new double[X.length];
        
        for (int i = 0; i < X.length; i++) {
            predictions[i] = probabilities[i][1] >= 0.5 ? classes[1] : classes[0];
        }
        
        return predictions;
    }
    
    @Override
    public double[][] predictProba(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before making predictions");
        }
        
        double[][] XWithBias = addBiasColumn(X);
        double[] probabilities = predictProbabilities(XWithBias);
        double[][] result = new double[X.length][2];
        
        for (int i = 0; i < X.length; i++) {
            result[i][0] = 1 - probabilities[i]; // Probability of class 0
            result[i][1] = probabilities[i];     // Probability of class 1
        }
        
        return result;
    }
    
    @Override
    public double[][] predictLogProba(double[][] X) {
        double[][] probabilities = predictProba(X);
        double[][] logProbabilities = new double[probabilities.length][probabilities[0].length];
        
        for (int i = 0; i < probabilities.length; i++) {
            for (int j = 0; j < probabilities[i].length; j++) {
                logProbabilities[i][j] = Math.log(probabilities[i][j]);
            }
        }
        
        return logProbabilities;
    }
    
    @Override
    public double[] getClasses() {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before accessing classes");
        }
        return Arrays.copyOf(classes, classes.length);
    }
    
    @Override
    public double score(double[][] X, double[] y) {
        double[] predictions = predict(X);
        int correct = 0;
        
        for (int i = 0; i < y.length; i++) {
            if (predictions[i] == y[i]) {
                correct++;
            }
        }
        
        return (double) correct / y.length;
    }
    
    private double[][] addBiasColumn(double[][] X) {
        int nSamples = X.length;
        int nFeatures = X[0].length;
        double[][] XWithBias = new double[nSamples][nFeatures + 1];
        
        for (int i = 0; i < nSamples; i++) {
            XWithBias[i][0] = 1.0; // Bias term
            System.arraycopy(X[i], 0, XWithBias[i], 1, nFeatures);
        }
        
        return XWithBias;
    }
    
    private double[] predictProbabilities(double[][] XWithBias) {
        double[] linearModel = new double[XWithBias.length];
        
        for (int i = 0; i < XWithBias.length; i++) {
            linearModel[i] = dotProduct(XWithBias[i], weights);
        }
        
        return sigmoid(linearModel);
    }
    
    private double[] computeGradients(double[][] XWithBias, double[] y, double[] predictions) {
        int nSamples = XWithBias.length;
        int nFeatures = XWithBias[0].length;
        double[] gradients = new double[nFeatures];
        
        for (int j = 0; j < nFeatures; j++) {
            double gradient = 0;
            for (int i = 0; i < nSamples; i++) {
                gradient += (predictions[i] - y[i]) * XWithBias[i][j];
            }
            gradients[j] = gradient / nSamples;
        }
        
        return gradients;
    }
    
    private double dotProduct(double[] x, double[] weights) {
        double result = 0.0;
        for (int i = 0; i < x.length; i++) {
            result += x[i] * weights[i];
        }
        return result;
    }
    
    private double[] sigmoid(double[] z) {
        return Arrays.stream(z).map(v -> 1.0 / (1.0 + Math.exp(-v))).toArray();
    }
    
    // Getters and setters for hyperparameters
    public double getLearningRate() { return learningRate; }
    public LogisticRegression setLearningRate(double learningRate) {
        this.learningRate = learningRate;
        params.put("learning_rate", learningRate);
        return this;
    }
    
    public int getMaxIter() { return maxIter; }
    public LogisticRegression setMaxIter(int maxIter) {
        this.maxIter = maxIter;
        params.put("max_iter", maxIter);
        return this;
    }
    
    public double getTolerance() { return tolerance; }
    public LogisticRegression setTolerance(double tolerance) {
        this.tolerance = tolerance;
        params.put("tol", tolerance);
        return this;
    }
    
    public double getC() { return C; }
    public LogisticRegression setC(double C) {
        this.C = C;
        params.put("C", C);
        return this;
    }
}
