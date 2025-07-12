package org.superml;

import org.superml.datasets.KaggleIntegration.KaggleCredentials;
import org.superml.datasets.KaggleTrainingManager;
import org.superml.datasets.KaggleTrainingManager.TrainingConfig;
import org.superml.datasets.KaggleTrainingManager.TrainingResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

/**
 * Example demonstrating Kaggle integration for automated ML training.
 * This example shows how to use SuperML Java with real Kaggle datasets.
 */
public class KaggleExample {
    
    private static final Logger logger = LoggerFactory.getLogger(KaggleExample.class);
    
    public static void main(String[] args) {
        logger.info("=======================================================");
        logger.info("    SuperML Java - Kaggle Integration Example");
        logger.info("=======================================================");
        
        try {
            // Example 1: Basic Kaggle training
            demonstrateBasicKaggleTraining();
            
            // Example 2: Advanced configuration
            demonstrateAdvancedTraining();
            
            // Example 3: Dataset search and exploration
            demonstrateDatasetExploration();
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            System.out.println("\nNote: This example requires valid Kaggle API credentials.");
            System.out.println("To set up Kaggle API:");
            System.out.println("1. Go to https://www.kaggle.com/account");
            System.out.println("2. Create API token and download kaggle.json");
            System.out.println("3. Place kaggle.json in ~/.kaggle/ directory");
            System.out.println("4. Set file permissions: chmod 600 ~/.kaggle/kaggle.json");
        }
    }
    
    private static void demonstrateBasicKaggleTraining() {
        System.out.println("1. BASIC KAGGLE TRAINING");
        System.out.println("========================\n");
        
        // Load Kaggle credentials from default location (~/.kaggle/kaggle.json)
        KaggleCredentials credentials = KaggleCredentials.fromDefaultLocation();
        
        // Create training manager
        KaggleTrainingManager trainer = new KaggleTrainingManager(credentials);
        
        try {
            // Train on a popular dataset (Titanic survival prediction)
            System.out.println("Training on Titanic dataset...");
            List<TrainingResult> results = trainer.trainOnDataset(
                "titanic", "titanic", "survived");
            
            // Display results
            System.out.println("\nTraining Results:");
            for (TrainingResult result : results) {
                System.out.println("  " + result);
                System.out.println("    Metrics: " + result.metrics);
            }
            
            // Get best model
            System.out.println("\nBest Model: " + results.get(0).algorithm);
            System.out.println("Best Score: " + String.format("%.4f", results.get(0).score));
            
        } finally {
            trainer.close();
        }
        
        System.out.println();
    }
    
    private static void demonstrateAdvancedTraining() {
        System.out.println("2. ADVANCED TRAINING CONFIGURATION");
        System.out.println("==================================\n");
        
        KaggleCredentials credentials = KaggleCredentials.fromDefaultLocation();
        KaggleTrainingManager trainer = new KaggleTrainingManager(credentials);
        
        try {
            // Create custom training configuration
            TrainingConfig config = new TrainingConfig()
                .setAlgorithms("logistic", "ridge")  // Only specific algorithms
                .setStandardScaler(true)             // Enable preprocessing
                .setGridSearch(true)                 // Enable hyperparameter tuning
                .setTestSize(0.3)                    // Larger test set
                .setRandomState(123)                 // Custom random seed
                .setVerbose(true);                   // Detailed output
            
            // Train with custom configuration
            System.out.println("Training with custom configuration...");
            List<TrainingResult> results = trainer.trainOnDataset(
                "uciml", "iris", "species", config);
            
            // Analyze results
            System.out.println("\nDetailed Analysis:");
            for (TrainingResult result : results) {
                System.out.printf("Algorithm: %s\n", result.algorithm);
                System.out.printf("  Score: %.4f\n", result.score);
                System.out.printf("  Training Time: %dms\n", result.trainingTimeMs);
                if (result.bestParams != null) {
                    System.out.printf("  Best Parameters: %s\n", result.bestParams);
                }
                System.out.printf("  All Metrics: %s\n", result.metrics);
                System.out.println();
            }
            
        } finally {
            trainer.close();
        }
    }
    
    private static void demonstrateDatasetExploration() {
        System.out.println("3. DATASET EXPLORATION");
        System.out.println("======================\n");
        
        KaggleCredentials credentials = KaggleCredentials.fromDefaultLocation();
        KaggleTrainingManager trainer = new KaggleTrainingManager(credentials);
        
        try {
            // Search for machine learning datasets
            System.out.println("Searching for 'classification' datasets...");
            trainer.searchDatasets("classification", 3);
            
            System.out.println("Searching for 'regression' datasets...");
            trainer.searchDatasets("regression", 3);
            
            // Quick training on a found dataset
            System.out.println("Quick training on popular dataset...");
            TrainingConfig quickConfig = new TrainingConfig()
                .setAlgorithms("logistic", "linear")
                .setGridSearch(false)  // Faster training
                .setVerbose(false);    // Less output
            
            List<TrainingResult> results = trainer.quickTrain("iris", "species", quickConfig);
            
            System.out.println("Quick training results:");
            for (TrainingResult result : results) {
                System.out.println("  " + result);
            }
            
        } finally {
            trainer.close();
        }
    }
    
    /**
     * Example of programmatic model usage after training.
     */
    public static void demonstrateModelUsage() {
        KaggleCredentials credentials = KaggleCredentials.fromDefaultLocation();
        KaggleTrainingManager trainer = new KaggleTrainingManager(credentials);
        
        try {
            // Train models
            List<TrainingResult> results = trainer.quickTrain("iris", "species");
            
            // Use the best model for predictions
            double[][] newData = {
                {5.1, 3.5, 1.4, 0.2},  // New iris measurements
                {6.2, 2.9, 4.3, 1.3},
                {7.3, 2.9, 6.3, 1.8}
            };
            
            double[] predictions = trainer.predict(results, newData);
            
            System.out.println("Predictions for new data:");
            for (int i = 0; i < predictions.length; i++) {
                System.out.printf("Sample %d: Class %.0f\n", i + 1, predictions[i]);
            }
            
        } finally {
            trainer.close();
        }
    }
}
