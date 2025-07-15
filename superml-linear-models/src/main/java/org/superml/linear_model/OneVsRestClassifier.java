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
import org.superml.core.SupervisedLearner;

import java.util.*;

/**
 * One-vs-Rest (OvR) multiclass classification strategy.
 * 
 * This strategy consists of fitting one classifier per class.
 * For each classifier, the class is fitted against all the other classes.
 * In addition to its computational efficiency (only n_classes classifiers are needed),
 * one advantage of this approach is its interpretability.
 * 
 * Similar to sklearn.multiclass.OneVsRestClassifier
 */
public class OneVsRestClassifier extends BaseEstimator implements Classifier {
    
    private final SupervisedLearner baseEstimator;
    private List<SupervisedLearner> classifiers;
    private double[] classes;
    private boolean fitted = false;
    
    /**
     * Constructor with base estimator.
     * @param baseEstimator The base classifier to use for each class
     */
    public OneVsRestClassifier(SupervisedLearner baseEstimator) {
        this.baseEstimator = baseEstimator;
        this.classifiers = new ArrayList<>();
    }
    
    @Override
    public OneVsRestClassifier fit(double[][] X, double[] y) {
        // Get unique classes
        classes = Arrays.stream(y).distinct().sorted().toArray();
        int nClasses = classes.length;
        
        if (nClasses < 2) {
            throw new IllegalArgumentException("Number of classes must be at least 2");
        }
        
        // Clear previous classifiers
        classifiers.clear();
        
        // Train one classifier per class
        for (int i = 0; i < nClasses; i++) {
            double targetClass = classes[i];
            
            // Create binary labels: 1 for target class, 0 for all others
            double[] binaryY = Arrays.stream(y)
                    .map(label -> label == targetClass ? 1.0 : 0.0)
                    .toArray();
            
            // Clone the base estimator for this class
            SupervisedLearner classifier = cloneEstimator(baseEstimator);
            
            // Fit the binary classifier
            classifier.fit(X, binaryY);
            classifiers.add(classifier);
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
            // Find class with highest probability
            int maxIndex = 0;
            double maxProb = probabilities[i][0];
            
            for (int j = 1; j < probabilities[i].length; j++) {
                if (probabilities[i][j] > maxProb) {
                    maxProb = probabilities[i][j];
                    maxIndex = j;
                }
            }
            
            predictions[i] = classes[maxIndex];
        }
        
        return predictions;
    }
    
    @Override
    public double[][] predictProba(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before making predictions");
        }
        
        int nSamples = X.length;
        int nClasses = classes.length;
        double[][] probabilities = new double[nSamples][nClasses];
        
        // Get predictions from each binary classifier
        for (int i = 0; i < nClasses; i++) {
            SupervisedLearner classifier = classifiers.get(i);
            
            if (classifier instanceof Classifier) {
                // If it's a classifier, get the probability for the positive class
                double[][] binaryProba = ((Classifier) classifier).predictProba(X);
                for (int j = 0; j < nSamples; j++) {
                    probabilities[j][i] = binaryProba[j][1]; // Probability of positive class
                }
            } else {
                // If not a classifier, use the prediction directly (decision function)
                double[] predictions = classifier.predict(X);
                for (int j = 0; j < nSamples; j++) {
                    probabilities[j][i] = predictions[j];
                }
            }
        }
        
        // Normalize probabilities so they sum to 1 for each sample
        for (int i = 0; i < nSamples; i++) {
            double sum = Arrays.stream(probabilities[i]).sum();
            if (sum > 0) {
                for (int j = 0; j < nClasses; j++) {
                    probabilities[i][j] /= sum;
                }
            } else {
                // If all probabilities are 0, assign equal probability to all classes
                Arrays.fill(probabilities[i], 1.0 / nClasses);
            }
        }
        
        return probabilities;
    }
    
    @Override
    public double[][] predictLogProba(double[][] X) {
        double[][] probabilities = predictProba(X);
        double[][] logProbabilities = new double[probabilities.length][probabilities[0].length];
        
        for (int i = 0; i < probabilities.length; i++) {
            for (int j = 0; j < probabilities[i].length; j++) {
                logProbabilities[i][j] = Math.log(Math.max(probabilities[i][j], 1e-15)); // Avoid log(0)
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
    
    /**
     * Get the list of fitted classifiers.
     * @return List of classifiers, one per class
     */
    public List<SupervisedLearner> getClassifiers() {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before accessing classifiers");
        }
        return new ArrayList<>(classifiers);
    }
    
    /**
     * Clone the base estimator.
     * This is a simple implementation that works for basic estimators.
     * For more complex estimators, you might need a more sophisticated cloning mechanism.
     */
    private SupervisedLearner cloneEstimator(SupervisedLearner estimator) {
        // For now, we'll create a new instance of the same type
        // In a real implementation, you'd want a proper cloning mechanism
        if (estimator instanceof LogisticRegression) {
            LogisticRegression lr = (LogisticRegression) estimator;
            return new LogisticRegression(lr.getLearningRate(), lr.getMaxIter())
                    .setTolerance(lr.getTolerance())
                    .setC(lr.getC());
        }
        
        // For other estimators, you'd add similar cloning logic
        throw new UnsupportedOperationException("Cloning not implemented for estimator type: " + 
                estimator.getClass().getSimpleName());
    }
}
