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

package org.superml.onnx;

import org.superml.core.Estimator;
import ai.onnxruntime.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * ONNX Model Converter for exporting SuperML models to ONNX format
 * and importing ONNX models for inference.
 */
public class ONNXConverter {
    
    private static final ObjectMapper objectMapper = new ObjectMapper();
    
    /**
     * Export a SuperML model to ONNX format.
     * Note: This is a simplified implementation. Full ONNX export would require
     * implementing the ONNX protobuf schema for each algorithm type.
     */
    public static void exportModel(Estimator model, String outputPath, ExportOptions options) throws IOException {
        System.out.println("🔄 Exporting model to ONNX format: " + model.getClass().getSimpleName());
        
        // Create ONNX metadata
        ModelMetadata metadata = createMetadata(model, options);
        
        // For demonstration, we'll create a simplified ONNX-compatible structure
        // In a full implementation, this would generate proper ONNX protobuf
        String modelType = model.getClass().getSimpleName();
        
        switch (modelType) {
            case "LinearRegression":
                exportLinearModel(model, outputPath, metadata);
                break;
            case "LogisticRegression":
                exportLogisticModel(model, outputPath, metadata);
                break;
            case "DecisionTree":
                exportTreeModel(model, outputPath, metadata);
                break;
            case "RandomForest":
                exportEnsembleModel(model, outputPath, metadata);
                break;
            default:
                exportGenericModel(model, outputPath, metadata);
        }
        
        System.out.println("-> Model exported successfully to: " + outputPath);
    }
    
    /**
     * Import an ONNX model for inference.
     */
    public static ONNXModel importModel(String modelPath) throws OrtException, IOException {
        System.out.println("📥 Importing ONNX model from: " + modelPath);
        
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession session = env.createSession(modelPath, new OrtSession.SessionOptions());
        
        // Get model input/output info
        Map<String, NodeInfo> inputInfo = session.getInputInfo();
        Map<String, NodeInfo> outputInfo = session.getOutputInfo();
        
        ONNXModel model = new ONNXModel(session, inputInfo, outputInfo);
        
        System.out.println("-> ONNX model imported successfully");
        System.out.println("   - Inputs: " + inputInfo.keySet());
        System.out.println("   - Outputs: " + outputInfo.keySet());
        
        return model;
    }
    
    /**
     * Convert SuperML model predictions to ONNX-compatible format.
     */
    public static double[] convertToONNXPredictions(double[] supermlPredictions, String modelType) {
        // Handle different prediction formats
        switch (modelType.toLowerCase()) {
            case "classification":
                // Convert probabilities to class predictions if needed
                double[] classPredictions = new double[supermlPredictions.length];
                for (int i = 0; i < supermlPredictions.length; i++) {
                    classPredictions[i] = supermlPredictions[i] > 0.5 ? 1.0 : 0.0;
                }
                return classPredictions;
            case "regression":
            default:
                return supermlPredictions;
        }
    }
    
    private static ModelMetadata createMetadata(Estimator model, ExportOptions options) {
        ModelMetadata metadata = new ModelMetadata();
        metadata.modelName = options.modelName != null ? options.modelName : model.getClass().getSimpleName();
        metadata.framework = "SuperML-Java";
        metadata.version = "2.0.0";
        metadata.description = options.description;
        metadata.author = options.author;
        metadata.createdDate = new Date().toString();
        metadata.modelType = determineModelType(model);
        return metadata;
    }
    
    private static String determineModelType(Estimator model) {
        String className = model.getClass().getSimpleName().toLowerCase();
        if (className.contains("regression")) {
            return "regression";
        } else if (className.contains("classification") || className.contains("logistic") || 
                  className.contains("tree") || className.contains("forest") || 
                  className.contains("bayes") || className.contains("svm")) {
            return "classification";
        } else {
            return "unknown";
        }
    }
    
    private static void exportLinearModel(Estimator model, String outputPath, ModelMetadata metadata) throws IOException {
        // Create ONNX-like representation for linear models
        ObjectNode onnxModel = objectMapper.createObjectNode();
        onnxModel.put("ir_version", 7);
        onnxModel.put("producer_name", "SuperML-Java");
        onnxModel.put("producer_version", "2.0.0");
        onnxModel.put("model_version", 1);
        
        // Add metadata
        ObjectNode metadataNode = onnxModel.putObject("metadata_props");
        metadataNode.put("model_name", metadata.modelName);
        metadataNode.put("model_type", metadata.modelType);
        metadataNode.put("framework", metadata.framework);
        metadataNode.put("created_date", metadata.createdDate);
        
        // Add graph structure (simplified)
        ObjectNode graph = onnxModel.putObject("graph");
        graph.put("name", metadata.modelName + "_graph");
        
        // Write to file
        Path path = Paths.get(outputPath);
        Files.createDirectories(path.getParent());
        objectMapper.writerWithDefaultPrettyPrinter().writeValue(path.toFile(), onnxModel);
        
        // Also create a companion metadata file
        String metadataPath = outputPath.replace(".onnx", "_metadata.json");
        objectMapper.writerWithDefaultPrettyPrinter().writeValue(Paths.get(metadataPath).toFile(), metadata);
    }
    
    private static void exportLogisticModel(Estimator model, String outputPath, ModelMetadata metadata) throws IOException {
        exportLinearModel(model, outputPath, metadata); // Similar structure for now
    }
    
    private static void exportTreeModel(Estimator model, String outputPath, ModelMetadata metadata) throws IOException {
        // Tree models require more complex ONNX representation
        ObjectNode onnxModel = objectMapper.createObjectNode();
        onnxModel.put("ir_version", 7);
        onnxModel.put("producer_name", "SuperML-Java");
        onnxModel.put("model_version", 1);
        
        // Add tree-specific metadata
        ObjectNode metadataNode = onnxModel.putObject("metadata_props");
        metadataNode.put("model_name", metadata.modelName);
        metadataNode.put("model_type", "tree_ensemble");
        metadataNode.put("framework", metadata.framework);
        
        Path path = Paths.get(outputPath);
        Files.createDirectories(path.getParent());
        objectMapper.writerWithDefaultPrettyPrinter().writeValue(path.toFile(), onnxModel);
        
        String metadataPath = outputPath.replace(".onnx", "_metadata.json");
        objectMapper.writerWithDefaultPrettyPrinter().writeValue(Paths.get(metadataPath).toFile(), metadata);
    }
    
    private static void exportEnsembleModel(Estimator model, String outputPath, ModelMetadata metadata) throws IOException {
        exportTreeModel(model, outputPath, metadata); // Use tree structure for ensembles
    }
    
    private static void exportGenericModel(Estimator model, String outputPath, ModelMetadata metadata) throws IOException {
        System.out.println("⚠️  Generic model export - limited ONNX compatibility");
        exportLinearModel(model, outputPath, metadata);
    }
    
    /**
     * Configuration options for ONNX export.
     */
    public static class ExportOptions {
        public String modelName;
        public String description;
        public String author;
        public boolean includeTrainingData = false;
        public boolean optimizeModel = true;
        public String targetOpset = "11";
        
        public static ExportOptions defaults() {
            return new ExportOptions();
        }
        
        public ExportOptions withName(String name) {
            this.modelName = name;
            return this;
        }
        
        public ExportOptions withDescription(String desc) {
            this.description = desc;
            return this;
        }
        
        public ExportOptions withAuthor(String author) {
            this.author = author;
            return this;
        }
    }
    
    /**
     * Model metadata for ONNX export.
     */
    public static class ModelMetadata {
        public String modelName;
        public String framework;
        public String version;
        public String description;
        public String author;
        public String createdDate;
        public String modelType;
        public Map<String, Object> customProperties = new HashMap<>();
    }
    
    /**
     * Wrapper for imported ONNX models.
     */
    public static class ONNXModel implements AutoCloseable {
        private final OrtSession session;
        private final Map<String, NodeInfo> inputInfo;
        private final Map<String, NodeInfo> outputInfo;
        
        public ONNXModel(OrtSession session, Map<String, NodeInfo> inputInfo, Map<String, NodeInfo> outputInfo) {
            this.session = session;
            this.inputInfo = inputInfo;
            this.outputInfo = outputInfo;
        }
        
        /**
         * Run inference on the ONNX model.
         */
        public double[] predict(double[][] features) throws OrtException {
            // Convert input to ONNX tensor format
            String inputName = inputInfo.keySet().iterator().next();
            
            // Flatten features for ONNX (assuming single input tensor)
            float[][] floatFeatures = new float[features.length][features[0].length];
            for (int i = 0; i < features.length; i++) {
                for (int j = 0; j < features[i].length; j++) {
                    floatFeatures[i][j] = (float) features[i][j];
                }
            }
            
            OnnxTensor inputTensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), floatFeatures);
            Map<String, OnnxTensor> inputs = Collections.singletonMap(inputName, inputTensor);
            
            // Run inference
            try (OrtSession.Result result = session.run(inputs)) {
                String outputName = outputInfo.keySet().iterator().next();
                OnnxTensor outputTensor = (OnnxTensor) result.get(outputName).get();
                
                // Convert output to double array
                float[][] output = (float[][]) outputTensor.getValue();
                double[] predictions = new double[output.length];
                for (int i = 0; i < output.length; i++) {
                    predictions[i] = output[i][0]; // Assuming single output per row
                }
                
                return predictions;
            } finally {
                inputTensor.close();
            }
        }
        
        /**
         * Get information about model inputs.
         */
        public Map<String, NodeInfo> getInputInfo() {
            return inputInfo;
        }
        
        /**
         * Get information about model outputs.
         */
        public Map<String, NodeInfo> getOutputInfo() {
            return outputInfo;
        }
        
        @Override
        public void close() throws Exception {
            if (session != null) {
                session.close();
            }
        }
    }
}
