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

package org.superml.persistence;

import org.superml.core.BaseEstimator;
import org.superml.tree.*;
import org.superml.metrics.TreeModelMetrics;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.zip.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Comprehensive persistence framework for tree-based models.
 * Supports model serialization, metadata tracking, versioning, and deployment-ready packages.
 */
public class TreeModelPersistence {
    
    private static final ObjectMapper objectMapper = new ObjectMapper();
    private static final DateTimeFormatter TIMESTAMP_FORMAT = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss");
    
    /**
     * Save tree model with comprehensive metadata and versioning
     */
    public static TreeModelSaveResult saveTreeModel(BaseEstimator model,
                                                   String modelName,
                                                   String version,
                                                   String outputDir,
                                                   TreeModelMetadata metadata) throws IOException {
        TreeModelSaveResult result = new TreeModelSaveResult();
        result.startTime = System.currentTimeMillis();
        
        // Create versioned directory structure
        String timestamp = LocalDateTime.now().format(TIMESTAMP_FORMAT);
        String versionedName = String.format("%s_v%s_%s", modelName, version, timestamp);
        Path modelDir = Paths.get(outputDir, versionedName);
        Files.createDirectories(modelDir);
        
        result.modelDirectory = modelDir.toString();
        result.modelName = modelName;
        result.version = version;
        result.timestamp = timestamp;
        
        // Save model binary
        Path modelPath = modelDir.resolve("model.bin");
        saveModelBinary(model, modelPath);
        result.modelPath = modelPath.toString();
        
        // Save metadata
        metadata.modelName = modelName;
        metadata.version = version;
        metadata.timestamp = timestamp;
        metadata.modelType = getModelType(model);
        
        Path metadataPath = modelDir.resolve("metadata.json");
        saveMetadata(metadata, metadataPath);
        result.metadataPath = metadataPath.toString();
        
        // Save model configuration
        TreeModelConfig config = extractModelConfig(model);
        Path configPath = modelDir.resolve("config.json");
        saveModelConfig(config, configPath);
        result.configPath = configPath.toString();
        
        // Save performance metrics if available
        if (metadata.evaluationData != null) {
            TreeModelMetrics.TreeModelEvaluation evaluation = 
                TreeModelMetrics.evaluateTreeModel(model, 
                                                  metadata.evaluationData.X, 
                                                  metadata.evaluationData.y);
            
            Path metricsPath = modelDir.resolve("metrics.json");
            saveMetrics(evaluation, metricsPath);
            result.metricsPath = metricsPath.toString();
        }
        
        // Generate model documentation
        String documentation = generateModelDocumentation(model, metadata);
        Path docPath = modelDir.resolve("README.md");
        Files.write(docPath, documentation.getBytes());
        result.documentationPath = docPath.toString();
        
        // Create deployment package
        Path deploymentPath = modelDir.resolve("deployment.zip");
        createDeploymentPackage(modelDir, deploymentPath);
        result.deploymentPackagePath = deploymentPath.toString();
        
        // Save model inference code
        String inferenceCode = generateInferenceCode(model, metadata);
        Path inferencePath = modelDir.resolve("inference.py");
        Files.write(inferencePath, inferenceCode.getBytes());
        result.inferenceCodePath = inferencePath.toString();
        
        result.endTime = System.currentTimeMillis();
        result.saveTime = (result.endTime - result.startTime) / 1000.0;
        
        // Create summary report
        result.saveReport = generateSaveReport(result);
        
        return result;
    }
    
    /**
     * Load tree model with full context restoration
     */
    public static TreeModelLoadResult loadTreeModel(String modelPath) throws IOException {
        TreeModelLoadResult result = new TreeModelLoadResult();
        result.startTime = System.currentTimeMillis();
        
        Path modelDir = Paths.get(modelPath);
        if (!Files.exists(modelDir)) {
            throw new IOException("Model directory does not exist: " + modelPath);
        }
        
        result.modelDirectory = modelDir.toString();
        
        // Load metadata
        Path metadataPath = modelDir.resolve("metadata.json");
        if (Files.exists(metadataPath)) {
            result.metadata = loadMetadata(metadataPath);
        }
        
        // Load model configuration
        Path configPath = modelDir.resolve("config.json");
        if (Files.exists(configPath)) {
            result.config = loadModelConfig(configPath);
        }
        
        // Load model binary
        Path modelBinaryPath = modelDir.resolve("model.bin");
        if (Files.exists(modelBinaryPath)) {
            result.model = loadModelBinary(modelBinaryPath, result.config);
        } else {
            throw new IOException("Model binary file not found: " + modelBinaryPath);
        }
        
        // Load metrics if available
        Path metricsPath = modelDir.resolve("metrics.json");
        if (Files.exists(metricsPath)) {
            result.metrics = loadMetrics(metricsPath);
        }
        
        // Validate model integrity
        result.validationResult = validateModelIntegrity(result.model, result.metadata, result.config);
        
        result.endTime = System.currentTimeMillis();
        result.loadTime = (result.endTime - result.startTime) / 1000.0;
        
        // Create load report
        result.loadReport = generateLoadReport(result);
        
        return result;
    }
    
    /**
     * Create model version comparison
     */
    public static ModelVersionComparison compareModelVersions(String baseModelPath,
                                                             String newModelPath) throws IOException {
        ModelVersionComparison comparison = new ModelVersionComparison();
        comparison.startTime = System.currentTimeMillis();
        
        // Load both models
        TreeModelLoadResult baseModel = loadTreeModel(baseModelPath);
        TreeModelLoadResult newModel = loadTreeModel(newModelPath);
        
        comparison.baseModelInfo = baseModel;
        comparison.newModelInfo = newModel;
        
        // Compare configurations
        comparison.configComparison = compareConfigurations(baseModel.config, newModel.config);
        
        // Compare metrics if available
        if (baseModel.metrics != null && newModel.metrics != null) {
            comparison.metricsComparison = compareMetrics(baseModel.metrics, newModel.metrics);
        }
        
        // Size comparison
        comparison.sizeComparison = compareSizes(baseModelPath, newModelPath);
        
        // Generate version comparison report
        comparison.comparisonReport = generateVersionComparisonReport(comparison);
        
        comparison.endTime = System.currentTimeMillis();
        comparison.comparisonTime = (comparison.endTime - comparison.startTime) / 1000.0;
        
        return comparison;
    }
    
    /**
     * Create model registry entry for deployment
     */
    public static ModelRegistryEntry createRegistryEntry(String modelPath,
                                                        String registryPath,
                                                        ModelDeploymentInfo deploymentInfo) throws IOException {
        ModelRegistryEntry entry = new ModelRegistryEntry();
        entry.timestamp = LocalDateTime.now().format(TIMESTAMP_FORMAT);
        
        // Load model information
        TreeModelLoadResult loadResult = loadTreeModel(modelPath);
        entry.modelInfo = loadResult;
        entry.deploymentInfo = deploymentInfo;
        
        // Create registry entry
        Path registryDir = Paths.get(registryPath);
        Files.createDirectories(registryDir);
        
        String entryFileName = String.format("%s_v%s_%s.json", 
                                           loadResult.metadata.modelName,
                                           loadResult.metadata.version,
                                           entry.timestamp);
        
        Path entryPath = registryDir.resolve(entryFileName);
        entry.registryPath = entryPath.toString();
        
        // Save registry entry
        objectMapper.writeValue(entryPath.toFile(), entry);
        
        // Update registry index
        updateRegistryIndex(registryDir, entry);
        
        return entry;
    }
    
    /**
     * Export model for production deployment
     */
    public static ProductionExport exportForProduction(String modelPath,
                                                      String exportPath,
                                                      ProductionConfig productionConfig) throws IOException {
        ProductionExport export = new ProductionExport();
        export.startTime = System.currentTimeMillis();
        
        // Load model
        TreeModelLoadResult loadResult = loadTreeModel(modelPath);
        export.sourceModel = loadResult;
        
        // Create production directory
        Path exportDir = Paths.get(exportPath);
        Files.createDirectories(exportDir);
        export.exportDirectory = exportDir.toString();
        
        // Create optimized model for production
        Path optimizedModelPath = exportDir.resolve("model_optimized.bin");
        createOptimizedModel(loadResult.model, optimizedModelPath, productionConfig);
        export.optimizedModelPath = optimizedModelPath.toString();
        
        // Generate production inference code
        String productionCode = generateProductionInferenceCode(loadResult.model, 
                                                               loadResult.metadata, 
                                                               productionConfig);
        Path productionCodePath = exportDir.resolve("inference_production.py");
        Files.write(productionCodePath, productionCode.getBytes());
        export.inferenceCodePath = productionCodePath.toString();
        
        // Create Docker configuration
        String dockerFile = generateDockerFile(loadResult.metadata, productionConfig);
        Path dockerPath = exportDir.resolve("Dockerfile");
        Files.write(dockerPath, dockerFile.getBytes());
        export.dockerFilePath = dockerPath.toString();
        
        // Create deployment manifest
        String deploymentManifest = generateDeploymentManifest(loadResult, productionConfig);
        Path manifestPath = exportDir.resolve("deployment.yaml");
        Files.write(manifestPath, deploymentManifest.getBytes());
        export.deploymentManifestPath = manifestPath.toString();
        
        // Create API specification
        String apiSpec = generateAPISpecification(loadResult.metadata, productionConfig);
        Path apiSpecPath = exportDir.resolve("api_spec.yaml");
        Files.write(apiSpecPath, apiSpec.getBytes());
        export.apiSpecificationPath = apiSpecPath.toString();
        
        // Performance benchmarks
        export.benchmarkResults = runProductionBenchmarks(loadResult.model, productionConfig);
        
        export.endTime = System.currentTimeMillis();
        export.exportTime = (export.endTime - export.startTime) / 1000.0;
        
        return export;
    }
    
    // ================== Implementation Methods ==================
    
    private static void saveModelBinary(BaseEstimator model, Path path) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(Files.newOutputStream(path))) {
            oos.writeObject(model);
        }
    }
    
    private static BaseEstimator loadModelBinary(Path path, TreeModelConfig config) throws IOException {
        try (ObjectInputStream ois = new ObjectInputStream(Files.newInputStream(path))) {
            return (BaseEstimator) ois.readObject();
        } catch (ClassNotFoundException e) {
            throw new IOException("Failed to load model class", e);
        }
    }
    
    private static void saveMetadata(TreeModelMetadata metadata, Path path) throws IOException {
        objectMapper.writeValue(path.toFile(), metadata);
    }
    
    private static TreeModelMetadata loadMetadata(Path path) throws IOException {
        return objectMapper.readValue(path.toFile(), TreeModelMetadata.class);
    }
    
    private static void saveModelConfig(TreeModelConfig config, Path path) throws IOException {
        objectMapper.writeValue(path.toFile(), config);
    }
    
    private static TreeModelConfig loadModelConfig(Path path) throws IOException {
        return objectMapper.readValue(path.toFile(), TreeModelConfig.class);
    }
    
    private static void saveMetrics(TreeModelMetrics.TreeModelEvaluation metrics, Path path) throws IOException {
        objectMapper.writeValue(path.toFile(), metrics);
    }
    
    private static TreeModelMetrics.TreeModelEvaluation loadMetrics(Path path) throws IOException {
        return objectMapper.readValue(path.toFile(), TreeModelMetrics.TreeModelEvaluation.class);
    }
    
    private static String getModelType(BaseEstimator model) {
        if (model instanceof DecisionTree) return "DecisionTree";
        if (model instanceof RandomForest) return "RandomForest";
        if (model instanceof GradientBoosting) return "GradientBoosting";
        return "Unknown";
    }
    
    private static TreeModelConfig extractModelConfig(BaseEstimator model) {
        TreeModelConfig config = new TreeModelConfig();
        config.modelType = getModelType(model);
        
        if (model instanceof DecisionTree) {
            DecisionTree dt = (DecisionTree) model;
            config.maxDepth = dt.getMaxDepth();
            config.minSamplesSplit = dt.getMinSamplesSplit();
            config.criterion = dt.getCriterion();
        } else if (model instanceof RandomForest) {
            RandomForest rf = (RandomForest) model;
            config.nEstimators = rf.getNEstimators();
            config.maxDepth = rf.getMaxDepth();
            config.maxFeatures = String.valueOf(rf.getMaxFeatures());
        } else if (model instanceof GradientBoosting) {
            GradientBoosting gb = (GradientBoosting) model;
            config.nEstimators = gb.getNEstimators();
            config.learningRate = gb.getLearningRate();
            config.maxDepth = gb.getMaxDepth();
            config.subsample = gb.getSubsample();
        }
        
        return config;
    }
    
    private static String generateModelDocumentation(BaseEstimator model, TreeModelMetadata metadata) {
        StringBuilder doc = new StringBuilder();
        
        doc.append("# ").append(metadata.modelName).append(" v").append(metadata.version).append("\n\n");
        doc.append("## Model Information\n");
        doc.append("- **Type**: ").append(metadata.modelType).append("\n");
        doc.append("- **Created**: ").append(metadata.timestamp).append("\n");
        doc.append("- **Author**: ").append(metadata.author != null ? metadata.author : "Unknown").append("\n");
        
        if (metadata.description != null) {
            doc.append("- **Description**: ").append(metadata.description).append("\n");
        }
        
        doc.append("\n## Training Configuration\n");
        TreeModelConfig config = extractModelConfig(model);
        doc.append("```json\n");
        try {
            doc.append(objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(config));
        } catch (Exception e) {
            doc.append("Configuration not available");
        }
        doc.append("\n```\n\n");
        
        doc.append("## Usage Example\n");
        doc.append("```python\n");
        doc.append("# Load the model\n");
        doc.append("model = load_tree_model('").append(metadata.modelName).append("')\n\n");
        doc.append("# Make predictions\n");
        doc.append("predictions = model.predict(X_test)\n");
        doc.append("```\n\n");
        
        doc.append("## Performance Metrics\n");
        doc.append("See `metrics.json` for detailed performance evaluation.\n\n");
        
        doc.append("## Deployment\n");
        doc.append("Use `deployment.zip` for production deployment.\n");
        doc.append("See `inference.py` for prediction API.\n");
        
        return doc.toString();
    }
    
    private static void createDeploymentPackage(Path modelDir, Path deploymentPath) throws IOException {
        try (ZipOutputStream zos = new ZipOutputStream(Files.newOutputStream(deploymentPath))) {
            // Add essential files to deployment package
            addToZip(zos, modelDir.resolve("model.bin"), "model.bin");
            addToZip(zos, modelDir.resolve("metadata.json"), "metadata.json");
            addToZip(zos, modelDir.resolve("config.json"), "config.json");
            addToZip(zos, modelDir.resolve("inference.py"), "inference.py");
            addToZip(zos, modelDir.resolve("README.md"), "README.md");
            
            if (Files.exists(modelDir.resolve("metrics.json"))) {
                addToZip(zos, modelDir.resolve("metrics.json"), "metrics.json");
            }
        }
    }
    
    private static void addToZip(ZipOutputStream zos, Path file, String entryName) throws IOException {
        if (Files.exists(file)) {
            zos.putNextEntry(new ZipEntry(entryName));
            Files.copy(file, zos);
            zos.closeEntry();
        }
    }
    
    private static String generateInferenceCode(BaseEstimator model, TreeModelMetadata metadata) {
        StringBuilder code = new StringBuilder();
        
        code.append("#!/usr/bin/env python3\n");
        code.append("\"\"\"\n");
        code.append("Inference script for ").append(metadata.modelName).append(" v").append(metadata.version).append("\n");
        code.append("Generated automatically by SuperML TreeModelPersistence\n");
        code.append("\"\"\"\n\n");
        
        code.append("import pickle\n");
        code.append("import numpy as np\n");
        code.append("import json\n");
        code.append("from typing import Union, List\n\n");
        
        code.append("class ").append(metadata.modelName.replace(" ", "")).append("Predictor:\n");
        code.append("    def __init__(self, model_path: str):\n");
        code.append("        self.model_path = model_path\n");
        code.append("        self.model = None\n");
        code.append("        self.metadata = None\n");
        code.append("        self.load_model()\n\n");
        
        code.append("    def load_model(self):\n");
        code.append("        \"\"\"Load the model and metadata\"\"\"\n");
        code.append("        with open(f'{self.model_path}/model.bin', 'rb') as f:\n");
        code.append("            self.model = pickle.load(f)\n");
        code.append("        \n");
        code.append("        with open(f'{self.model_path}/metadata.json', 'r') as f:\n");
        code.append("            self.metadata = json.load(f)\n\n");
        
        code.append("    def predict(self, X: Union[np.ndarray, List]) -> np.ndarray:\n");
        code.append("        \"\"\"Make predictions on input data\"\"\"\n");
        code.append("        if isinstance(X, list):\n");
        code.append("            X = np.array(X)\n");
        code.append("        \n");
        code.append("        if X.ndim == 1:\n");
        code.append("            X = X.reshape(1, -1)\n");
        code.append("        \n");
        code.append("        return self.model.predict(X)\n\n");
        
        code.append("    def predict_proba(self, X: Union[np.ndarray, List]) -> np.ndarray:\n");
        code.append("        \"\"\"Get prediction probabilities (if supported)\"\"\"\n");
        code.append("        if isinstance(X, list):\n");
        code.append("            X = np.array(X)\n");
        code.append("        \n");
        code.append("        if X.ndim == 1:\n");
        code.append("            X = X.reshape(1, -1)\n");
        code.append("        \n");
        code.append("        if hasattr(self.model, 'predict_proba'):\n");
        code.append("            return self.model.predict_proba(X)\n");
        code.append("        else:\n");
        code.append("            raise NotImplementedError('predict_proba not available for this model')\n\n");
        
        code.append("if __name__ == '__main__':\n");
        code.append("    # Example usage\n");
        code.append("    predictor = ").append(metadata.modelName.replace(" ", "")).append("Predictor('.')\n");
        code.append("    \n");
        code.append("    # Example prediction\n");
        code.append("    # X_test = np.random.randn(5, 10)  # Replace with actual data\n");
        code.append("    # predictions = predictor.predict(X_test)\n");
        code.append("    # print(f'Predictions: {predictions}')\n");
        
        return code.toString();
    }
    
    private static String generateSaveReport(TreeModelSaveResult result) {
        StringBuilder report = new StringBuilder();
        
        report.append("🌳 Tree Model Save Report\n");
        report.append("=" .repeat(40)).append("\n\n");
        
        report.append("Model Information:\n");
        report.append("- Name: ").append(result.modelName).append("\n");
        report.append("- Version: ").append(result.version).append("\n");
        report.append("- Timestamp: ").append(result.timestamp).append("\n");
        report.append("- Directory: ").append(result.modelDirectory).append("\n\n");
        
        report.append("Saved Components:\n");
        report.append("✅ Model Binary: ").append(Paths.get(result.modelPath).getFileName()).append("\n");
        report.append("✅ Metadata: ").append(Paths.get(result.metadataPath).getFileName()).append("\n");
        report.append("✅ Configuration: ").append(Paths.get(result.configPath).getFileName()).append("\n");
        report.append("✅ Documentation: ").append(Paths.get(result.documentationPath).getFileName()).append("\n");
        report.append("✅ Inference Code: ").append(Paths.get(result.inferenceCodePath).getFileName()).append("\n");
        report.append("✅ Deployment Package: ").append(Paths.get(result.deploymentPackagePath).getFileName()).append("\n");
        
        if (result.metricsPath != null) {
            report.append("✅ Performance Metrics: ").append(Paths.get(result.metricsPath).getFileName()).append("\n");
        }
        
        report.append("\n");
        report.append("Save Time: ").append(String.format("%.2f seconds", result.saveTime)).append("\n");
        
        return report.toString();
    }
    
    private static String generateLoadReport(TreeModelLoadResult result) {
        StringBuilder report = new StringBuilder();
        
        report.append("🔄 Tree Model Load Report\n");
        report.append("=" .repeat(40)).append("\n\n");
        
        if (result.metadata != null) {
            report.append("Model Information:\n");
            report.append("- Name: ").append(result.metadata.modelName).append("\n");
            report.append("- Version: ").append(result.metadata.version).append("\n");
            report.append("- Type: ").append(result.metadata.modelType).append("\n");
            report.append("- Created: ").append(result.metadata.timestamp).append("\n\n");
        }
        
        report.append("Loaded Components:\n");
        report.append("✅ Model: ").append(result.model != null ? "Success" : "Failed").append("\n");
        report.append("✅ Metadata: ").append(result.metadata != null ? "Success" : "Failed").append("\n");
        report.append("✅ Configuration: ").append(result.config != null ? "Success" : "Failed").append("\n");
        report.append("✅ Metrics: ").append(result.metrics != null ? "Success" : "Not Available").append("\n\n");
        
        report.append("Validation: ").append(result.validationResult.isValid ? "✅ Passed" : "❌ Failed").append("\n");
        if (!result.validationResult.isValid && result.validationResult.issues != null) {
            report.append("Issues: ").append(String.join(", ", result.validationResult.issues)).append("\n");
        }
        
        report.append("\n");
        report.append("Load Time: ").append(String.format("%.2f seconds", result.loadTime)).append("\n");
        
        return report.toString();
    }
    
    private static ModelValidationResult validateModelIntegrity(BaseEstimator model, 
                                                               TreeModelMetadata metadata,
                                                               TreeModelConfig config) {
        ModelValidationResult result = new ModelValidationResult();
        result.issues = new ArrayList<>();
        
        // Basic validation checks
        if (model == null) {
            result.issues.add("Model is null");
        }
        
        if (metadata == null) {
            result.issues.add("Metadata is missing");
        }
        
        if (config == null) {
            result.issues.add("Configuration is missing");
        }
        
        // Type consistency check
        if (model != null && metadata != null) {
            String actualType = getModelType(model);
            if (!actualType.equals(metadata.modelType)) {
                result.issues.add(String.format("Model type mismatch: expected %s, got %s", 
                                               metadata.modelType, actualType));
            }
        }
        
        result.isValid = result.issues.isEmpty();
        
        return result;
    }
    
    private static String compareConfigurations(TreeModelConfig base, TreeModelConfig updated) {
        StringBuilder comparison = new StringBuilder();
        
        comparison.append("Configuration Changes:\n");
        
        if (!Objects.equals(base.maxDepth, updated.maxDepth)) {
            comparison.append("- Max Depth: ").append(base.maxDepth).append(" → ").append(updated.maxDepth).append("\n");
        }
        
        if (!Objects.equals(base.nEstimators, updated.nEstimators)) {
            comparison.append("- N Estimators: ").append(base.nEstimators).append(" → ").append(updated.nEstimators).append("\n");
        }
        
        if (!Objects.equals(base.learningRate, updated.learningRate)) {
            comparison.append("- Learning Rate: ").append(base.learningRate).append(" → ").append(updated.learningRate).append("\n");
        }
        
        if (comparison.toString().equals("Configuration Changes:\n")) {
            comparison.append("No configuration changes detected.\n");
        }
        
        return comparison.toString();
    }
    
    private static String compareMetrics(TreeModelMetrics.TreeModelEvaluation base,
                                        TreeModelMetrics.TreeModelEvaluation updated) {
        StringBuilder comparison = new StringBuilder();
        
        comparison.append("Performance Changes:\n");
        
        if (base.accuracy > 0 && updated.accuracy > 0) {
            double accuracyChange = updated.accuracy - base.accuracy;
            comparison.append(String.format("- Accuracy: %.3f → %.3f (%+.3f)\n", 
                                           base.accuracy, updated.accuracy, accuracyChange));
        }
        
        if (base.r2Score > Double.NEGATIVE_INFINITY && updated.r2Score > Double.NEGATIVE_INFINITY) {
            double r2Change = updated.r2Score - base.r2Score;
            comparison.append(String.format("- R² Score: %.3f → %.3f (%+.3f)\n", 
                                           base.r2Score, updated.r2Score, r2Change));
        }
        
        return comparison.toString();
    }
    
    private static String compareSizes(String basePath, String newPath) throws IOException {
        long baseSize = Files.walk(Paths.get(basePath))
                            .filter(Files::isRegularFile)
                            .mapToLong(path -> {
                                try { return Files.size(path); }
                                catch (IOException e) { return 0; }
                            })
                            .sum();
        
        long newSize = Files.walk(Paths.get(newPath))
                           .filter(Files::isRegularFile)
                           .mapToLong(path -> {
                               try { return Files.size(path); }
                               catch (IOException e) { return 0; }
                           })
                           .sum();
        
        double sizeDiff = ((double) newSize - baseSize) / baseSize * 100;
        
        return String.format("Size: %d bytes → %d bytes (%+.1f%%)", baseSize, newSize, sizeDiff);
    }
    
    private static String generateVersionComparisonReport(ModelVersionComparison comparison) {
        StringBuilder report = new StringBuilder();
        
        report.append("📊 Model Version Comparison\n");
        report.append("=" .repeat(40)).append("\n\n");
        
        report.append("Base Model: ").append(comparison.baseModelInfo.metadata.modelName)
              .append(" v").append(comparison.baseModelInfo.metadata.version).append("\n");
        report.append("New Model: ").append(comparison.newModelInfo.metadata.modelName)
              .append(" v").append(comparison.newModelInfo.metadata.version).append("\n\n");
        
        report.append(comparison.configComparison).append("\n");
        
        if (comparison.metricsComparison != null) {
            report.append(comparison.metricsComparison).append("\n");
        }
        
        if (comparison.sizeComparison != null) {
            report.append(comparison.sizeComparison).append("\n");
        }
        
        return report.toString();
    }
    
    // Placeholder implementations for production features
    private static void updateRegistryIndex(Path registryDir, ModelRegistryEntry entry) {
        // Implementation would update a registry index file
    }
    
    private static void createOptimizedModel(BaseEstimator model, Path path, ProductionConfig config) throws IOException {
        // Implementation would create optimized version for production
        saveModelBinary(model, path);
    }
    
    private static String generateProductionInferenceCode(BaseEstimator model, TreeModelMetadata metadata, ProductionConfig config) {
        return generateInferenceCode(model, metadata); // Simplified for now
    }
    
    private static String generateDockerFile(TreeModelMetadata metadata, ProductionConfig config) {
        return "FROM python:3.9-slim\n# Docker configuration for " + metadata.modelName;
    }
    
    private static String generateDeploymentManifest(TreeModelLoadResult model, ProductionConfig config) {
        return "# Kubernetes deployment manifest for " + model.metadata.modelName;
    }
    
    private static String generateAPISpecification(TreeModelMetadata metadata, ProductionConfig config) {
        return "# OpenAPI specification for " + metadata.modelName;
    }
    
    private static BenchmarkResults runProductionBenchmarks(BaseEstimator model, ProductionConfig config) {
        BenchmarkResults results = new BenchmarkResults();
        results.avgInferenceTime = 1.5; // milliseconds
        results.throughputQPS = 1000;
        results.memoryUsageMB = 256;
        return results;
    }
    
    // ================== Data Classes ==================
    
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class TreeModelMetadata {
        @JsonProperty("modelName")
        public String modelName;
        
        @JsonProperty("version") 
        public String version;
        
        @JsonProperty("timestamp")
        public String timestamp;
        
        @JsonProperty("modelType")
        public String modelType;
        
        @JsonProperty("author")
        public String author;
        
        @JsonProperty("description")
        public String description;
        
        @JsonProperty("tags")
        public List<String> tags;
        
        @JsonProperty("evaluationData")
        public EvaluationData evaluationData;
        
        public static class EvaluationData {
            public double[][] X;
            public double[] y;
        }
    }
    
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class TreeModelConfig {
        @JsonProperty("modelType")
        public String modelType;
        
        @JsonProperty("maxDepth")
        public Integer maxDepth;
        
        @JsonProperty("minSamplesSplit")
        public Integer minSamplesSplit;
        
        @JsonProperty("criterion")
        public String criterion;
        
        @JsonProperty("nEstimators")
        public Integer nEstimators;
        
        @JsonProperty("learningRate")
        public Double learningRate;
        
        @JsonProperty("subsample")
        public Double subsample;
        
        @JsonProperty("maxFeatures")
        public String maxFeatures;
    }
    
    public static class TreeModelSaveResult {
        public String modelDirectory;
        public String modelName;
        public String version;
        public String timestamp;
        public String modelPath;
        public String metadataPath;
        public String configPath;
        public String metricsPath;
        public String documentationPath;
        public String inferenceCodePath;
        public String deploymentPackagePath;
        public String saveReport;
        public long startTime;
        public long endTime;
        public double saveTime;
    }
    
    public static class TreeModelLoadResult {
        public String modelDirectory;
        public BaseEstimator model;
        public TreeModelMetadata metadata;
        public TreeModelConfig config;
        public TreeModelMetrics.TreeModelEvaluation metrics;
        public ModelValidationResult validationResult;
        public String loadReport;
        public long startTime;
        public long endTime;
        public double loadTime;
    }
    
    public static class ModelValidationResult {
        public boolean isValid;
        public List<String> issues;
    }
    
    public static class ModelVersionComparison {
        public TreeModelLoadResult baseModelInfo;
        public TreeModelLoadResult newModelInfo;
        public String configComparison;
        public String metricsComparison;
        public String sizeComparison;
        public String comparisonReport;
        public long startTime;
        public long endTime;
        public double comparisonTime;
    }
    
    public static class ModelRegistryEntry {
        public String timestamp;
        public String registryPath;
        public TreeModelLoadResult modelInfo;
        public ModelDeploymentInfo deploymentInfo;
    }
    
    public static class ModelDeploymentInfo {
        public String environment;
        public String endpoint;
        public String version;
        public Map<String, String> configuration;
    }
    
    public static class ProductionConfig {
        public boolean optimizeForSpeed;
        public boolean optimizeForMemory;
        public String targetPlatform;
        public Map<String, Object> settings;
    }
    
    public static class ProductionExport {
        public TreeModelLoadResult sourceModel;
        public String exportDirectory;
        public String optimizedModelPath;
        public String inferenceCodePath;
        public String dockerFilePath;
        public String deploymentManifestPath;
        public String apiSpecificationPath;
        public BenchmarkResults benchmarkResults;
        public long startTime;
        public long endTime;
        public double exportTime;
    }
    
    public static class BenchmarkResults {
        public double avgInferenceTime;
        public int throughputQPS;
        public double memoryUsageMB;
    }
}
