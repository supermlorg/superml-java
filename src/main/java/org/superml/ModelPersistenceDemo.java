package org.superml;

import org.superml.datasets.Datasets;
import org.superml.linear_model.LogisticRegression;
import org.superml.linear_model.LinearRegression;
import org.superml.metrics.Metrics;
import org.superml.model_selection.ModelSelection;
import org.superml.persistence.ModelManager;
import org.superml.persistence.ModelPersistence;
import org.superml.pipeline.Pipeline;
import org.superml.preprocessing.StandardScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Comprehensive demonstration of SuperML Java Model Persistence capabilities.
 * Shows how to save, load, and manage trained models with various configurations.
 */
public class ModelPersistenceDemo {
    
    private static final Logger logger = LoggerFactory.getLogger(ModelPersistenceDemo.class);
    
    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("SuperML Java - Model Persistence Demo");
        System.out.println("=".repeat(60));
        
        try {
            demonstrateBasicModelSaving();
            demonstratePipelineSaving();
            demonstrateModelManager();
            demonstrateMetadataAndInfo();
            demonstrateModelLoading();
        } catch (Exception e) {
            logger.error("Demo failed", e);
            e.printStackTrace();
        }
    }
    
    private static void demonstrateBasicModelSaving() {
        System.out.println("\n1. BASIC MODEL SAVING AND LOADING");
        System.out.println("=".repeat(40));
        
        // Load dataset - create a classification dataset instead of iris
        Datasets.Dataset dataset = Datasets.makeClassification(150, 4, 3, 42);
        ModelSelection.TrainTestSplit split = ModelSelection.trainTestSplit(dataset.data, dataset.target, 0.2, 42);
        
        // Train a model
        LogisticRegression model = new LogisticRegression()
            .setMaxIter(1000)
            .setLearningRate(0.01);
        
        model.fit(split.XTrain, split.yTrain);
        double originalAccuracy = Metrics.accuracy(split.yTest, model.predict(split.XTest));
        
        System.out.printf("Original model accuracy: %.3f\n", originalAccuracy);
        
        // Save the model
        String modelPath = "iris_classifier";
        ModelPersistence.save(model, modelPath, "Iris classification model", 
            Map.of("dataset", "iris", "accuracy", originalAccuracy));
        
        System.out.println("Model saved to: " + modelPath + ModelPersistence.DEFAULT_EXTENSION);
        
        // Load the model
        LogisticRegression loadedModel = ModelPersistence.load(modelPath, LogisticRegression.class);
        double loadedAccuracy = Metrics.accuracy(split.yTest, loadedModel.predict(split.XTest));
        
        System.out.printf("Loaded model accuracy: %.3f\n", loadedAccuracy);
        System.out.printf("Accuracies match: %s\n", originalAccuracy == loadedAccuracy ? "✓" : "✗");
        
        // Check file info
        long fileSize = ModelPersistence.getFileSize(modelPath);
        System.out.printf("Model file size: %d bytes\n", fileSize);
        
        // Clean up
        ModelPersistence.delete(modelPath);
        System.out.println("Cleaned up model file");
    }
    
    private static void demonstratePipelineSaving() {
        System.out.println("\n2. PIPELINE SAVING AND LOADING");
        System.out.println("=".repeat(40));
        
        // Create a pipeline
        Pipeline pipeline = new Pipeline()
            .addStep("scaler", new StandardScaler())
            .addStep("classifier", new LogisticRegression().setMaxIter(1000));
        
        // Train pipeline
        Datasets.Dataset dataset = Datasets.makeClassification(150, 4, 3, 42);
        ModelSelection.TrainTestSplit split = ModelSelection.trainTestSplit(dataset.data, dataset.target, 0.2, 42);
        
        pipeline.fit(split.XTrain, split.yTrain);
        double[] originalPredictions = pipeline.predict(split.XTest);
        double originalAccuracy = Metrics.accuracy(split.yTest, originalPredictions);
        
        System.out.printf("Original pipeline accuracy: %.3f\n", originalAccuracy);
        
        // Save pipeline
        String pipelinePath = "iris_pipeline";
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("steps", List.of("StandardScaler", "LogisticRegression"));
        metadata.put("dataset", "iris");
        metadata.put("accuracy", originalAccuracy);
        
        ModelPersistence.save(pipeline, pipelinePath, "Iris classification pipeline", metadata);
        System.out.println("Pipeline saved successfully");
        
        // Load pipeline
        Pipeline loadedPipeline = ModelPersistence.load(pipelinePath, Pipeline.class);
        double[] loadedPredictions = loadedPipeline.predict(split.XTest);
        double loadedAccuracy = Metrics.accuracy(split.yTest, loadedPredictions);
        
        System.out.printf("Loaded pipeline accuracy: %.3f\n", loadedAccuracy);
        
        // Clean up
        ModelPersistence.delete(pipelinePath);
        System.out.println("Cleaned up pipeline file");
    }
    
    private static void demonstrateModelManager() {
        System.out.println("\n3. MODEL MANAGER FUNCTIONALITY");
        System.out.println("=".repeat(40));
        
        ModelManager manager = new ModelManager("demo_models");
        
        // Create and save multiple models
        Datasets.Dataset irisData = Datasets.makeClassification(150, 4, 3, 42);
        Datasets.Dataset regressionData = Datasets.makeRegression(100, 5, 0.1, 42);
        
        ModelSelection.TrainTestSplit irisSplit = ModelSelection.trainTestSplit(
            irisData.data, irisData.target, 0.2, 42);
        ModelSelection.TrainTestSplit regSplit = ModelSelection.trainTestSplit(
            regressionData.data, regressionData.target, 0.2, 42);
        
        // Train and save classification model
        LogisticRegression classifier = new LogisticRegression().setMaxIter(1000);
        classifier.fit(irisSplit.XTrain, irisSplit.yTrain);
        String classifierPath = manager.saveModel(classifier, "iris");
        System.out.println("Saved classifier: " + classifierPath);
        
        // Train and save regression model
        LinearRegression regressor = new LinearRegression();
        regressor.fit(regSplit.XTrain, regSplit.yTrain);
        String regressorPath = manager.saveModel(regressor, "regression");
        System.out.println("Saved regressor: " + regressorPath);
        
        // List all models
        System.out.println("\nAll saved models:");
        List<String> models = manager.listModels();
        for (String model : models) {
            System.out.println("  - " + model);
        }
        
        // Get detailed model info
        System.out.println("\nDetailed model information:");
        List<ModelManager.ModelInfo> modelsInfo = manager.getModelsInfo();
        for (ModelManager.ModelInfo info : modelsInfo) {
            System.out.println("  " + info);
        }
        
        // Find models by class
        List<String> logisticModels = manager.findModelsByClass("LogisticRegression");
        System.out.println("\nLogistic Regression models: " + logisticModels);
        
        // Clean up all models
        for (String model : manager.listModels()) {
            manager.deleteModel(model);
        }
        System.out.println("Cleaned up all demo models");
    }
    
    private static void demonstrateMetadataAndInfo() {
        System.out.println("\n4. METADATA AND MODEL INFORMATION");
        System.out.println("=".repeat(40));
        
        // Train a model with rich metadata
        Datasets.Dataset dataset = Datasets.makeClassification(150, 4, 3, 42);
        ModelSelection.TrainTestSplit split = ModelSelection.trainTestSplit(dataset.data, dataset.target, 0.2, 42);
        
        LogisticRegression model = new LogisticRegression()
            .setMaxIter(1500)
            .setLearningRate(0.005);
        
        model.fit(split.XTrain, split.yTrain);
        double accuracy = Metrics.accuracy(split.yTest, model.predict(split.XTest));
        
        // Create rich metadata
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("dataset_name", "iris");
        metadata.put("dataset_samples", dataset.data.length);
        metadata.put("dataset_features", dataset.data[0].length);
        metadata.put("test_accuracy", accuracy);
        metadata.put("hyperparameters", model.getParams());
        metadata.put("training_notes", "Optimized for iris classification");
        
        String modelPath = "iris_detailed";
        ModelPersistence.save(model, modelPath, "Detailed iris classifier with metadata", metadata);
        
        // Read metadata without loading full model
        ModelPersistence.ModelMetadata meta = ModelPersistence.getMetadata(modelPath);
        System.out.println("Model metadata:");
        System.out.println("  Class: " + meta.modelClass);
        System.out.println("  Version: " + meta.supermlVersion);
        System.out.println("  Saved at: " + meta.savedAt);
        System.out.println("  Description: " + meta.description);
        System.out.println("  Custom metadata:");
        meta.customMetadata.forEach((key, value) -> 
            System.out.printf("    %s: %s\n", key, value));
        
        // Validate model file
        boolean isValid = ModelPersistence.isValidModelFile(modelPath);
        System.out.println("Model file is valid: " + isValid);
        
        // Clean up
        ModelPersistence.delete(modelPath);
        System.out.println("Cleaned up metadata demo model");
    }
    
    private static void demonstrateModelLoading() {
        System.out.println("\n5. ADVANCED MODEL LOADING");
        System.out.println("=".repeat(40));
        
        // Create different types of models
        Datasets.Dataset dataset = Datasets.makeClassification(150, 4, 3, 42);
        ModelSelection.TrainTestSplit split = ModelSelection.trainTestSplit(dataset.data, dataset.target, 0.2, 42);
        
        // Save multiple model types
        LogisticRegression lr = new LogisticRegression().setMaxIter(1000);
        lr.fit(split.XTrain, split.yTrain);
        ModelPersistence.save(lr, "model_lr");
        
        Pipeline pipeline = new Pipeline()
            .addStep("scaler", new StandardScaler())
            .addStep("classifier", new LogisticRegression().setMaxIter(1000));
        pipeline.fit(split.XTrain, split.yTrain);
        ModelPersistence.save(pipeline, "model_pipeline");
        
        // Load with type checking
        System.out.println("Loading LogisticRegression...");
        LogisticRegression loadedLR = ModelPersistence.load("model_lr", LogisticRegression.class);
        System.out.println("✓ Successfully loaded LogisticRegression");
        
        System.out.println("Loading Pipeline...");
        Pipeline loadedPipeline = ModelPersistence.load("model_pipeline", Pipeline.class);
        System.out.println("✓ Successfully loaded Pipeline");
        
        // Try loading with wrong type (should fail)
        System.out.println("Attempting to load Pipeline as LogisticRegression...");
        try {
            LogisticRegression wrongType = ModelPersistence.load("model_pipeline", LogisticRegression.class);
            System.out.println("✗ This should not succeed!");
        } catch (Exception e) {
            System.out.println("✓ Correctly rejected wrong type: " + e.getMessage());
        }
        
        // Load without type checking
        System.out.println("Loading without type checking...");
        var genericModel = ModelPersistence.load("model_lr");
        System.out.println("✓ Loaded as: " + genericModel.getClass().getSimpleName());
        
        // Test predictions work
        double[] lrPredictions = loadedLR.predict(split.XTest);
        double[] pipelinePredictions = loadedPipeline.predict(split.XTest);
        
        System.out.printf("LR accuracy: %.3f\n", Metrics.accuracy(split.yTest, lrPredictions));
        System.out.printf("Pipeline accuracy: %.3f\n", Metrics.accuracy(split.yTest, pipelinePredictions));
        
        // Clean up
        ModelPersistence.delete("model_lr");
        ModelPersistence.delete("model_pipeline");
        System.out.println("Cleaned up all models");
        
        System.out.println("\n" + "=".repeat(60));
        System.out.println("Model Persistence Demo Completed Successfully!");
        System.out.println("=".repeat(60));
    }
}
