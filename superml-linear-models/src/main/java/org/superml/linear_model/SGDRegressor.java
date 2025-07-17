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
import org.superml.core.Regressor;
import java.util.Random;

/**
 * Stochastic Gradient Descent (SGD) Regressor.
 * Similar to sklearn.linear_model.SGDRegressor
 */
public class SGDRegressor extends BaseEstimator implements Regressor {
    
    private double[] weights;
    private double intercept;
    private boolean fitted = false;
    
    // Hyperparameters
    private String loss = "squared_loss"; // "squared_loss", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"
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
    private double epsilon = 0.1; // Epsilon for epsilon-insensitive loss
    
    private Random random;
    
    public SGDRegressor() {
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
        params.put("epsilon", epsilon);
        
        this.random = new Random(randomState);
    }
    
    // Fluent setters
    public SGDRegressor setLoss(String loss) {
        this.loss = loss;
        params.put("loss", loss);
        return this;
    }
    
    public SGDRegressor setPenalty(String penalty) {
        this.penalty = penalty;
        params.put("penalty", penalty);
        return this;
    }
    
    public SGDRegressor setAlpha(double alpha) {
        this.alpha = alpha;
        params.put("alpha", alpha);
        return this;
    }
    
    public SGDRegressor setL1Ratio(double l1Ratio) {
        this.l1Ratio = l1Ratio;
        params.put("l1_ratio", l1Ratio);
        return this;
    }
    
    public SGDRegressor setMaxIter(int maxIter) {
        this.maxIter = maxIter;
        params.put("max_iter", maxIter);
        return this;
    }
    
    public SGDRegressor setTolerance(double tolerance) {
        this.tolerance = tolerance;
        params.put("tol", tolerance);
        return this;
    }
    
    public SGDRegressor setLearningRate(String learningRateSchedule) {
        this.learningRateSchedule = learningRateSchedule;
        params.put("learning_rate", learningRateSchedule);
        return this;
    }
    
    public SGDRegressor setEta0(double eta0) {
        this.eta0 = eta0;
        params.put("eta0", eta0);
        return this;
    }
    
    public SGDRegressor setRandomState(int randomState) {
        this.randomState = randomState;
        this.random = new Random(randomState);
        params.put("random_state", randomState);
        return this;
    }
    
    public SGDRegressor setEpsilon(double epsilon) {
        this.epsilon = epsilon;
        params.put("epsilon", epsilon);
        return this;
    }
    
    public SGDRegressor setEarlyStoppingEnabled(boolean earlyStoppingEnabled) {
        this.earlyStoppingEnabled = earlyStoppingEnabled;
        return this;
    }
    
    @Override
    public SGDRegressor fit(double[][] X, double[] y) {
        int nSamples = X.length;
        int nFeatures = X[0].length;
        
        // Initialize weights
        weights = new double[nFeatures];
        intercept = 0.0;
        
        // SGD training loop
        double bestLoss = Double.POSITIVE_INFINITY;
        int noImprovementCount = 0;
        
        for (int epoch = 0; epoch < maxIter; epoch++) {
            // Shuffle data if requested
            int[] indices = createShuffledIndices(nSamples);
            
            double epochLoss = 0.0;
            
            for (int idx : indices) {
                double[] xi = X[idx];
                double yi = y[idx];
                
                // Calculate prediction and loss
                double prediction = computePrediction(xi);
                double sampleLoss = computeLoss(prediction, yi);
                epochLoss += sampleLoss;
                
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
            }
            
            // Check for convergence
            epochLoss /= nSamples;
            if (earlyStoppingEnabled) {
                if (epochLoss >= bestLoss - tolerance) {
                    noImprovementCount++;
                    if (noImprovementCount >= nIterNoChange) {
                        break;
                    }
                } else {
                    bestLoss = epochLoss;
                    noImprovementCount = 0;
                }
            }
        }
        
        this.fitted = true;
        return this;
    }
    
    @Override
    public double[] predict(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before making predictions");
        }
        
        double[] predictions = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = computePrediction(X[i]);
        }
        
        return predictions;
    }
    
    @Override
    public double score(double[][] X, double[] y) {
        double[] predictions = predict(X);
        
        // Calculate R² score
        double totalSumSquares = 0.0;
        double residualSumSquares = 0.0;
        
        // Calculate mean of y
        double yMean = 0.0;
        for (double value : y) {
            yMean += value;
        }
        yMean /= y.length;
        
        // Calculate sums
        for (int i = 0; i < y.length; i++) {
            totalSumSquares += Math.pow(y[i] - yMean, 2);
            residualSumSquares += Math.pow(y[i] - predictions[i], 2);
        }
        
        return 1.0 - (residualSumSquares / totalSumSquares);
    }
    
    private double computePrediction(double[] x) {
        double result = intercept;
        for (int j = 0; j < weights.length; j++) {
            result += weights[j] * x[j];
        }
        return result;
    }
    
    private double computeLoss(double prediction, double trueValue) {
        double error = prediction - trueValue;
        
        switch (loss) {
            case "squared_loss":
                return 0.5 * error * error;
            case "huber":
                double delta = 1.35; // Huber threshold
                if (Math.abs(error) <= delta) {
                    return 0.5 * error * error;
                } else {
                    return delta * (Math.abs(error) - 0.5 * delta);
                }
            case "epsilon_insensitive":
                return Math.max(0, Math.abs(error) - epsilon);
            case "squared_epsilon_insensitive":
                double epsError = Math.max(0, Math.abs(error) - epsilon);
                return 0.5 * epsError * epsError;
            default:
                throw new IllegalArgumentException("Unknown loss function: " + loss);
        }
    }
    
    private double[] computeGradient(double[] x, double trueValue, double prediction) {
        double[] gradient = new double[x.length];
        double lossGradient = computeLossGradient(trueValue, prediction);
        
        for (int j = 0; j < x.length; j++) {
            gradient[j] = lossGradient * x[j];
        }
        
        return gradient;
    }
    
    private double computeLossGradient(double trueValue, double prediction) {
        double error = prediction - trueValue;
        
        switch (loss) {
            case "squared_loss":
                return error;
            case "huber":
                double delta = 1.35;
                if (Math.abs(error) <= delta) {
                    return error;
                } else {
                    return delta * Math.signum(error);
                }
            case "epsilon_insensitive":
                if (Math.abs(error) <= epsilon) {
                    return 0.0;
                } else {
                    return Math.signum(error);
                }
            case "squared_epsilon_insensitive":
                if (Math.abs(error) <= epsilon) {
                    return 0.0;
                } else {
                    return Math.signum(error) * (Math.abs(error) - epsilon);
                }
            default:
                throw new IllegalArgumentException("Unknown loss function: " + loss);
        }
    }
    
    private double computeInterceptGradient(double trueValue, double prediction) {
        return computeLossGradient(trueValue, prediction);
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
                // Optimal learning rate for regression
                return 1.0 / (alpha * (1.0 + iteration));
            case "invscaling":
                double eta = eta0 > 0 ? eta0 : learningRate;
                return eta / Math.pow(iteration + 1, 0.25);
            case "adaptive":
                // Simple adaptive rate
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
    
    public boolean isFitted() {
        return fitted;
    }
}
