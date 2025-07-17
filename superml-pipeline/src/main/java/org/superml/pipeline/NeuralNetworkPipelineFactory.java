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

package org.superml.pipeline;

import org.superml.core.BaseEstimator;
import org.superml.neural.MLPClassifier;
import org.superml.neural.CNNClassifier;
import org.superml.neural.RNNClassifier;
import org.superml.preprocessing.NeuralNetworkPreprocessor;
import org.superml.preprocessing.StandardScaler;

import java.util.Map;
import java.util.HashMap;

/**
 * Factory for creating neural network pipelines with appropriate preprocessing.
 * Provides pre-configured pipelines for MLP, CNN, and RNN models.
 */
public class NeuralNetworkPipelineFactory extends BaseEstimator {
    
    /**
     * Create an MLP pipeline with preprocessing for tabular data
     */
    public static Pipeline createMLPPipeline() {
        return createMLPPipeline(new int[]{64, 32}, "relu", 0.01, 100);
    }
    
    /**
     * Create an MLP pipeline with custom parameters
     */
    public static Pipeline createMLPPipeline(int[] hiddenLayers, String activation, 
                                           double learningRate, int maxIter) {
        Pipeline pipeline = new Pipeline();
        
        // Add preprocessing for MLP
        NeuralNetworkPreprocessor preprocessor = new NeuralNetworkPreprocessor(
            NeuralNetworkPreprocessor.NetworkType.MLP).configureMLP();
        
        // Add MLP classifier
        MLPClassifier mlp = new MLPClassifier()
            .setHiddenLayerSizes(hiddenLayers)
            .setActivation(activation)
            .setLearningRate(learningRate)
            .setMaxIter(maxIter);
        
        pipeline.addStep("preprocessor", preprocessor);
        pipeline.addStep("mlp", mlp);
        
        return pipeline;
    }
    
    /**
     * Create a CNN pipeline with preprocessing for image data
     */
    public static Pipeline createCNNPipeline(int height, int width, int channels) {
        return createCNNPipeline(height, width, channels, 0.01, 50);
    }
    
    /**
     * Create a CNN pipeline with custom parameters
     */
    public static Pipeline createCNNPipeline(int height, int width, int channels,
                                           double learningRate, int maxEpochs) {
        Pipeline pipeline = new Pipeline();
        
        // Add preprocessing for CNN
        NeuralNetworkPreprocessor preprocessor = new NeuralNetworkPreprocessor(
            NeuralNetworkPreprocessor.NetworkType.CNN).configureCNN(height, width, channels);
        
        // Add CNN classifier
        CNNClassifier cnn = new CNNClassifier()
            .setInputShape(height, width, channels)
            .setLearningRate(learningRate)
            .setMaxEpochs(maxEpochs);
        
        pipeline.addStep("preprocessor", preprocessor);
        pipeline.addStep("cnn", cnn);
        
        return pipeline;
    }
    
    /**
     * Create an RNN pipeline with preprocessing for sequence data
     */
    public static Pipeline createRNNPipeline(int sequenceLength, int featuresPerTimestep) {
        return createRNNPipeline(sequenceLength, featuresPerTimestep, 32, 2, "LSTM", 0.01, 75);
    }
    
    /**
     * Create an RNN pipeline with custom parameters
     */
    public static Pipeline createRNNPipeline(int sequenceLength, int featuresPerTimestep,
                                           int hiddenSize, int numLayers, String cellType,
                                           double learningRate, int maxEpochs) {
        Pipeline pipeline = new Pipeline();
        
        // Add preprocessing for RNN
        NeuralNetworkPreprocessor preprocessor = new NeuralNetworkPreprocessor(
            NeuralNetworkPreprocessor.NetworkType.RNN)
            .configureRNN(sequenceLength, featuresPerTimestep, false);
        
        // Add RNN classifier
        RNNClassifier rnn = new RNNClassifier()
            .setHiddenSize(hiddenSize)
            .setNumLayers(numLayers)
            .setCellType(cellType)
            .setLearningRate(learningRate)
            .setMaxEpochs(maxEpochs);
        
        pipeline.addStep("preprocessor", preprocessor);
        pipeline.addStep("rnn", rnn);
        
        return pipeline;
    }
    
    /**
     * Create a multi-modal pipeline combining different neural networks
     */
    public static Pipeline createMultiModalPipeline() {
        Pipeline pipeline = new Pipeline();
        
        // This would require custom implementation for multi-modal data
        // For now, return a basic MLP pipeline
        return createMLPPipeline();
    }
    
    /**
     * Get recommended pipeline for data type
     */
    public static Pipeline getRecommendedPipeline(String dataType, Map<String, Object> dataProperties) {
        switch (dataType.toLowerCase()) {
            case "tabular":
                return createMLPPipeline();
            case "image":
                int height = (Integer) dataProperties.getOrDefault("height", 32);
                int width = (Integer) dataProperties.getOrDefault("width", 32);
                int channels = (Integer) dataProperties.getOrDefault("channels", 1);
                return createCNNPipeline(height, width, channels);
            case "sequence":
            case "time_series":
                int seqLength = (Integer) dataProperties.getOrDefault("sequence_length", 30);
                int features = (Integer) dataProperties.getOrDefault("features_per_timestep", 1);
                return createRNNPipeline(seqLength, features);
            default:
                return createMLPPipeline(); // Default fallback
        }
    }
}
