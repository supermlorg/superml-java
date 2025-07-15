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

package org.superml.core;

/**
 * Interface for classification algorithms.
 * Similar to sklearn.base.ClassifierMixin
 */
public interface Classifier extends SupervisedLearner {
    
    /**
     * Predict class probabilities for samples.
     * @param X features to predict on
     * @return class probabilities
     */
    double[][] predictProba(double[][] X);
    
    /**
     * Predict log-probabilities for samples.
     * @param X features to predict on
     * @return log probabilities
     */
    double[][] predictLogProba(double[][] X);
    
    /**
     * Get the unique class labels.
     * @return array of class labels
     */
    double[] getClasses();
}
