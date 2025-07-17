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

package org.superml.linear_model;

import org.superml.core.BaseEstimator;
import org.superml.core.Classifier;
import java.util.Arrays;
import java.util.Random;

/**
 * Stochastic Gradient Descent (SGD) Classifier.
 * Similar to sklearn.linear_model.SGDClassifier
 */
public class SGDClassifier extends BaseEstimator implements Classifier {
    
    private double[] weights;
    private double intercept;
    private double[] classes;
    private boolean fitted = false;
    
    // Hyperparameters
    private String loss = "hinge"; // "hinge", "log", "modified_huber", "squared_hinge", "perceptron"
    private String penalty = "l2"; // "l1", "l2", "elasticnet", "none"
    private double alpha = 0.0001; // Regularization strength
    private double l1Ratio = 0.15; // ElasticNet mixing parameter
    private boolean fitIntercept = true;
    private int maxIter = 1000;
    private double tolerance = 1e-3;
    private boolean shuffle = true;
    private int randomState = 42;
    private double learningRate = 0.01;
    private double eta0 = 0.0; // Initial learning rate for 'constant' or 'invscaling' schedule
    private String learningRateSchedule = "optimal"; // "constant", "optimal", "invscaling", "adaptive"
    private int nIterNoChange = 5; // Number of iterations with no improvement to wait before early stopping
    private boolean earlyStoppingEnabled = false;
    private double validationFraction = 0.1;
    
    private Random random;
    
    public SGDClassifier() {
        params.put("loss", loss);
        params.put("penalty", penalty);
        params.put("alpha", alpha);
        params.put("l1_ratio", l1Ratio);
        params.put("fit_intercept", fitIntercept);
        params.put("max_iter", maxIter);
        params.put("tol", tolerance);
        params.put("shuffle", shuffle);
        params.put("random_state", randomState);
        params.put("learning_rate", learningRateSchedule);
        params.put("eta0", eta0);
        
        this.random = new Random(randomState);
    }
    
    // Fluent setters
    public SGDClassifier setLoss(String loss) {
        this.loss = loss;
        params.put("loss", loss);
        return this;
    }
    
    public SGDClassifier setPenalty(String penalty) {
        this.penalty = penalty;
        params.put("penalty", penalty);
        return this;
    }
    
    public SGDClassifier setAlpha(double alpha) {
        this.alpha = alpha;
        params.put("alpha", alpha);
        return this;
    }
    
    public SGDClassifier setL1Ratio(double l1Ratio) {
        this.l1Ratio = l1Ratio;
        params.put("l1_ratio", l1Ratio);
        return this;
    }
    
    public SGDClassifier setMaxIter(int maxIter) {
        this.maxIter = maxIter;
        params.put("max_iter", maxIter);
        return this;
    }
    
    public SGDClassifier setTolerance(double tolerance) {
        this.tolerance = tolerance;
        params.put("tol", tolerance);
        return this;
    }
    
    public SGDClassifier setLearningRate(String learningRateSchedule) {
        this.learningRateSchedule = learningRateSchedule;
        params.put("learning_rate", learningRateSchedule);
        return this;
    }
    
    public SGDClassifier setEta0(double eta0) {
        this.eta0 = eta0;
        params.put("eta0", eta0);
        return this;
    }
    
    public SGDClassifier setRandomState(int randomState) {
        this.randomState = randomState;
        this.random = new Random(randomState);
        params.put("random_state", randomState);
        return this;
    }
    
    public SGDClassifier setEarlyStoppingEnabled(boolean earlyStoppingEnabled) {
        this.earlyStoppingEnabled = earlyStoppingEnabled;
        return this;
    }
    
    @Override
    public SGDClassifier fit(double[][] X, double[] y) {
        int nSamples = X.length;
        int nFeatures = X[0].length;
        
        // Get unique classes
        this.classes = Arrays.stream(y).distinct().sorted().toArray();
        
        // Handle binary/multiclass
        if (classes.length == 2) {
            fitBinary(X, y, nSamples, nFeatures);
        } else {
            throw new UnsupportedOperationException("Multiclass classification not yet implemented for SGD");
        }
        
        this.fitted = true;
        return this;
    }
    
    private void fitBinary(double[][] X, double[] y, int nSamples, int nFeatures) {
        // Initialize weights
        weights = new double[nFeatures];
        intercept = 0.0;
        
        // Convert binary labels to {-1, +1} for SVM-style losses
        double[] yTransformed = new double[nSamples];
        for (int i = 0; i < nSamples; i++) {
            yTransformed[i] = (y[i] == classes[0]) ? -1.0 : 1.0;
        }
        
        // SGD training loop
        double bestScore = Double.NEGATIVE_INFINITY;
        int noImprovementCount = 0;
        
        for (int epoch = 0; epoch < maxIter; epoch++) {
            // Shuffle data if requested
            int[] indices = createShuffledIndices(nSamples);
            
            double epochLoss = 0.0;
            int correct = 0;
            
            for (int idx : indices) {
                double[] xi = X[idx];
                double yi = yTransformed[idx];
                
                // Calculate prediction and loss
                double prediction = computeDecisionFunction(xi);
                double loss = computeLoss(prediction, yi);
                epochLoss += loss;
                
                // Update weights using gradient
                double[] gradient = computeGradient(xi, yi, prediction);
                double currentLR = computeLearningRate(epoch);
                
                // Update weights
                for (int j = 0; j < nFeatures; j++) {
                    weights[j] -= currentLR * gradient[j];
                }
                
                // Update intercept
                if (fitIntercept) {
                    intercept -= currentLR * computeInterceptGradient(yi, prediction);
                }
                
                // Apply regularization
                applyRegularization(currentLR);
                
                // Track accuracy
                if ((prediction > 0 && yi > 0) || (prediction <= 0 && yi <= 0)) {
                    correct++;
                }
            }
            
            // Check for convergence
            double accuracy = (double) correct / nSamples;
            if (earlyStoppingEnabled) {
                if (accuracy <= bestScore + tolerance) {
                    noImprovementCount++;
                    if (noImprovementCount >= nIterNoChange) {
                        break;
                    }
                } else {
                    bestScore = accuracy;
                    noImprovementCount = 0;
                }
            }
        }
    }
    
    @Override
    public double[] predict(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before making predictions");
        }
        
        double[] predictions = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            double decision = computeDecisionFunction(X[i]);
            predictions[i] = (decision > 0) ? classes[1] : classes[0];
        }
        
        return predictions;
    }
    
    @Override
    public double[][] predictProba(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before making predictions");
        }
        
        double[][] probabilities = new double[X.length][classes.length];
        for (int i = 0; i < X.length; i++) {
            double decision = computeDecisionFunction(X[i]);
            
            if (loss.equals("log")) {
                // For logistic loss, use sigmoid function
                double sigmoid = 1.0 / (1.0 + Math.exp(-decision));
                probabilities[i][0] = 1.0 - sigmoid; // Probability of class 0
                probabilities[i][1] = sigmoid;       // Probability of class 1
            } else {
                // For other losses, use distance-based probability approximation
                double prob1 = 1.0 / (1.0 + Math.exp(-decision));
                probabilities[i][0] = 1.0 - prob1;
                probabilities[i][1] = prob1;
            }
        }
        
        return probabilities;
    }
    
    @Override
    public double[][] predictLogProba(double[][] X) {
        double[][] proba = predictProba(X);
        double[][] logProba = new double[proba.length][proba[0].length];
        
        for (int i = 0; i < proba.length; i++) {
            for (int j = 0; j < proba[i].length; j++) {
                logProba[i][j] = Math.log(Math.max(proba[i][j], 1e-15)); // Avoid log(0)
            }
        }
        
        return logProba;
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
    
    /**
     * Get decision function values (distances to hyperplane)
     */
    public double[] decisionFunction(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before computing decision function");
        }
        
        double[] decisions = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            decisions[i] = computeDecisionFunction(X[i]);
        }
        return decisions;
    }
    
    private double computeDecisionFunction(double[] x) {
        double result = intercept;
        for (int j = 0; j < weights.length; j++) {
            result += weights[j] * x[j];
        }
        return result;
    }
    
    private double computeLoss(double prediction, double trueLabel) {
        switch (loss) {
            case "hinge":
                return Math.max(0, 1 - trueLabel * prediction);
            case "log":
                return Math.log(1 + Math.exp(-trueLabel * prediction));
            case "squared_hinge":
                double hinge = Math.max(0, 1 - trueLabel * prediction);
                return hinge * hinge;
            case "modified_huber":
                if (trueLabel * prediction >= 1) {
                    return 0;
                } else if (trueLabel * prediction >= -1) {
                    return Math.pow(1 - trueLabel * prediction, 2);
                } else {
                    return -4 * trueLabel * prediction;
                }
            case "perceptron":
                return Math.max(0, -trueLabel * prediction);
            default:
                throw new IllegalArgumentException("Unknown loss function: " + loss);
        }
    }
    
    private double[] computeGradient(double[] x, double trueLabel, double prediction) {
        double[] gradient = new double[x.length];
        double lossGradient = computeLossGradient(trueLabel, prediction);
        
        for (int j = 0; j < x.length; j++) {
            gradient[j] = lossGradient * x[j];
        }
        
        return gradient;
    }
    
    private double computeLossGradient(double trueLabel, double prediction) {
        switch (loss) {
            case "hinge":
                return (trueLabel * prediction < 1) ? -trueLabel : 0;
            case "log":
                double exp = Math.exp(-trueLabel * prediction);
                return -trueLabel * exp / (1 + exp);
            case "squared_hinge":
                return (trueLabel * prediction < 1) ? -2 * trueLabel * (1 - trueLabel * prediction) : 0;
            case "modified_huber":
                if (trueLabel * prediction >= 1) {
                    return 0;
                } else if (trueLabel * prediction >= -1) {
                    return -2 * trueLabel * (1 - trueLabel * prediction);
                } else {
                    return -4 * trueLabel;
                }
            case "perceptron":
                return (trueLabel * prediction <= 0) ? -trueLabel : 0;
            default:
                throw new IllegalArgumentException("Unknown loss function: " + loss);
        }
    }
    
    private double computeInterceptGradient(double trueLabel, double prediction) {
        return computeLossGradient(trueLabel, prediction);
    }
    
    private void applyRegularization(double learningRate) {
        switch (penalty) {
            case "l2":
                for (int j = 0; j < weights.length; j++) {
                    weights[j] *= (1 - learningRate * alpha);
                }
                break;
            case "l1":
                for (int j = 0; j < weights.length; j++) {
                    double sign = Math.signum(weights[j]);
                    weights[j] -= learningRate * alpha * sign;
                }
                break;
            case "elasticnet":
                for (int j = 0; j < weights.length; j++) {
                    // L2 component
                    weights[j] *= (1 - learningRate * alpha * l1Ratio);
                    // L1 component
                    double sign = Math.signum(weights[j]);
                    weights[j] -= learningRate * alpha * (1 - l1Ratio) * sign;
                }
                break;
            case "none":
                // No regularization
                break;
        }
    }
    
    private double computeLearningRate(int iteration) {
        switch (learningRateSchedule) {
            case "constant":
                return eta0 > 0 ? eta0 : learningRate;
            case "optimal":
                // Optimal learning rate for SVM (Leon Bottou's formula)
                double typw = 1.0 / (alpha * 100); // Typical weight magnitude
                double alpha0 = 1.0 / (typw * alpha);
                return 1.0 / (alpha * (alpha0 + iteration));
            case "invscaling":
                double eta = eta0 > 0 ? eta0 : learningRate;
                return eta / Math.pow(iteration + 1, 0.5);
            case "adaptive":
                // Simple adaptive rate (decrease if no improvement)
                return learningRate / (1 + iteration * 0.001);
            default:
                return learningRate;
        }
    }
    
    private int[] createShuffledIndices(int nSamples) {
        int[] indices = new int[nSamples];
        for (int i = 0; i < nSamples; i++) {
            indices[i] = i;
        }
        
        if (shuffle) {
            // Fisher-Yates shuffle
            for (int i = nSamples - 1; i > 0; i--) {
                int j = random.nextInt(i + 1);
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
        
        return indices;
    }
    
    // Getters
    public double[] getCoefficients() {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before accessing coefficients");
        }
        return weights.clone();
    }
    
    public double getIntercept() {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before accessing intercept");
        }
        return intercept;
    }
    
    public double[] getClasses() {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before accessing classes");
        }
        return classes.clone();
    }
    
    public boolean isFitted() {
        return fitted;
    }
}
