---
title: "Kaggle Integration Guide"
description: "Complete guide to using SuperML Java with Kaggle datasets and competitions"
layout: default
toc: true
search: true
---

# Kaggle Integration Guide

SuperML Java provides seamless integration with Kaggle, allowing you to train machine learning models on any public Kaggle dataset with just a few lines of code. This guide covers everything from setup to advanced usage.

## 🔧 Setup

### 1. Get Kaggle API Credentials

1. Go to [https://www.kaggle.com/account](https://www.kaggle.com/account)
2. Scroll to the "API" section
3. Click "Create New API Token"
4. Download the `kaggle.json` file

### 2. Install Credentials

```bash
# Create Kaggle directory
mkdir ~/.kaggle

# Move the downloaded file
mv ~/Downloads/kaggle.json ~/.kaggle/

# Set proper permissions (important for security)
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Verify Credentials

Your `kaggle.json` should look like:
```json
{
  "username": "your-kaggle-username",
  "key": "your-api-key-here"
}
```

## 🚀 Quick Start

### One-Line Training

```java
import org.superml.datasets.KaggleTrainingManager;
import org.superml.datasets.KaggleIntegration.KaggleCredentials;

public class KaggleQuickStart {
    public static void main(String[] args) {
        // Load credentials from default location
        var credentials = KaggleCredentials.fromDefaultLocation();
        var trainer = new KaggleTrainingManager(credentials);
        
        try {
            // Train on Titanic dataset - one line!
            var results = trainer.trainOnDataset("titanic", "titanic", "survived");
            
            // Show results
            System.out.println("Best model: " + results.get(0).algorithm);
            System.out.println("Accuracy: " + results.get(0).score);
            
        } finally {
            trainer.close();
        }
    }
}
```

### Custom Credentials

```java
// Load from custom location
var credentials = KaggleCredentials.fromFile("/path/to/kaggle.json");

// Or create programmatically
var credentials = new KaggleCredentials("username", "api-key");
```

## 🔍 Dataset Discovery

### Search Datasets

```java
var trainer = new KaggleTrainingManager(credentials);

// Search for datasets
trainer.searchDatasets("machine learning", 5);
trainer.searchDatasets("classification", 3);
trainer.searchDatasets("time series", 10);
```

Output:
```
Found 5 datasets:
1. owner/dataset-name
   Title: Machine Learning Dataset Collection
   Size: 45.23 MB, Downloads: 1250

2. another-owner/ml-data
   Title: Advanced ML Algorithms Dataset
   Size: 12.45 MB, Downloads: 890
...
```

### Browse by Category

```java
// Popular ML datasets
trainer.searchDatasets("iris", 3);           // Classic datasets
trainer.searchDatasets("titanic", 2);        // Survival prediction
trainer.searchDatasets("house prices", 5);   // Regression problems
trainer.searchDatasets("image classification", 4); // Computer vision
```

## ⚙️ Training Configuration

### Basic Configuration

```java
var config = new TrainingConfig()
    .setAlgorithms("logistic", "ridge", "lasso")  // Choose algorithms
    .setTestSize(0.2)                             // 20% test set
    .setRandomState(42)                           // Reproducible results
    .setVerbose(true);                            // Show progress

var results = trainer.trainOnDataset("owner", "dataset", "target", config);
```

### Advanced Configuration

```java
var config = new TrainingConfig()
    // Algorithm selection
    .setAlgorithms("logistic", "linear", "ridge", "lasso")
    
    // Preprocessing
    .setStandardScaler(true)                      // Feature scaling
    .setHandleMissingValues(true)                 // Auto-handle NaN
    
    // Hyperparameter tuning
    .setGridSearch(true)                          // Enable grid search
    .setGridSearchCV(5)                           // 5-fold cross-validation
    
    // Data splitting
    .setTestSize(0.25)                            // 25% test set
    .setValidationSize(0.15)                      // 15% validation set
    .setRandomState(123)                          // Seed for reproducibility
    
    // Output control
    .setVerbose(true)                             // Detailed logging
    .setSaveModels(true)                          // Save trained models
    .setOutputDir("./models");                    // Model save directory

var results = trainer.trainOnDataset("uciml", "iris", "species", config);
```

## 📊 Training Results

### Understanding Results

```java
var results = trainer.trainOnDataset("titanic", "titanic", "survived");

for (var result : results) {
    System.out.println("Algorithm: " + result.algorithm);
    System.out.println("Score: " + result.score);
    System.out.println("Training Time: " + result.trainingTimeMs + "ms");
    
    // Detailed metrics
    System.out.println("All Metrics: " + result.metrics);
    
    // Best hyperparameters (if grid search enabled)
    if (result.bestParams != null) {
        System.out.println("Best Parameters: " + result.bestParams);
    }
    
    // Cross-validation scores
    if (result.cvScores != null) {
        System.out.println("CV Scores: " + Arrays.toString(result.cvScores));
        System.out.println("CV Mean: " + result.cvMean);
        System.out.println("CV Std: " + result.cvStd);
    }
}
```

### Results Sorting

Results are automatically sorted by score (best first):

```java
var results = trainer.trainOnDataset("owner", "dataset", "target");

// Best model
var bestResult = results.get(0);
System.out.println("Best: " + bestResult.algorithm + " (" + bestResult.score + ")");

// Compare all models
System.out.println("\nModel Rankings:");
for (int i = 0; i < results.size(); i++) {
    var result = results.get(i);
    System.out.printf("%d. %s: %.4f\n", i + 1, result.algorithm, result.score);
}
```

## 🎯 Specific Use Cases

### Classification Tasks

```java
// Binary classification
var results = trainer.trainOnDataset("titanic", "titanic", "survived");

// Multiclass classification  
var results = trainer.trainOnDataset("uciml", "iris", "species");

// Custom config for classification
var config = new TrainingConfig()
    .setAlgorithms("logistic")                    // Best for classification
    .setGridSearch(true)                          // Optimize hyperparameters
    .setStandardScaler(true);                     // Often helps
```

### Regression Tasks

```java
// House price prediction
var results = trainer.trainOnDataset("owner", "house-prices", "price");

// Custom config for regression
var config = new TrainingConfig()
    .setAlgorithms("linear", "ridge", "lasso")    // Regression algorithms
    .setGridSearch(true)                          // Find best regularization
    .setStandardScaler(true);                     // Important for regularization
```

### Quick Prototyping

```java
// Fast training without grid search
var config = new TrainingConfig()
    .setAlgorithms("logistic", "linear")          // Just 2 algorithms
    .setGridSearch(false)                         // Skip hyperparameter tuning
    .setVerbose(false);                           // Minimal output

var results = trainer.quickTrain("iris", "species", config);
```

## 🔄 Workflow Examples

### End-to-End ML Pipeline

```java
public class KaggleMLPipeline {
    public static void main(String[] args) {
        var credentials = KaggleCredentials.fromDefaultLocation();
        var trainer = new KaggleTrainingManager(credentials);
        
        try {
            // 1. Explore datasets
            System.out.println("=== Dataset Discovery ===");
            trainer.searchDatasets("classification", 3);
            
            // 2. Configure training
            var config = new TrainingConfig()
                .setAlgorithms("logistic", "ridge", "lasso")
                .setStandardScaler(true)
                .setGridSearch(true)
                .setTestSize(0.2)
                .setVerbose(true);
            
            // 3. Train models
            System.out.println("\n=== Model Training ===");
            var results = trainer.trainOnDataset("uciml", "iris", "species", config);
            
            // 4. Analyze results
            System.out.println("\n=== Results Analysis ===");
            analyzeResults(results);
            
            // 5. Use best model
            System.out.println("\n=== Model Usage ===");
            useBestModel(trainer, results);
            
        } finally {
            trainer.close();
        }
    }
    
    private static void analyzeResults(List<TrainingResult> results) {
        System.out.println("Model Performance Comparison:");
        for (int i = 0; i < results.size(); i++) {
            var result = results.get(i);
            System.out.printf("%d. %s: %.4f (±%.4f)\n", 
                i + 1, result.algorithm, result.score, result.cvStd);
        }
        
        // Best model details
        var best = results.get(0);
        System.out.println("\nBest Model Details:");
        System.out.println("Algorithm: " + best.algorithm);
        System.out.println("Score: " + best.score);
        System.out.println("Training Time: " + best.trainingTimeMs + "ms");
        if (best.bestParams != null) {
            System.out.println("Parameters: " + best.bestParams);
        }
    }
    
    private static void useBestModel(KaggleTrainingManager trainer, 
                                   List<TrainingResult> results) {
        // Predict on new data
        double[][] newSamples = {
            {5.1, 3.5, 1.4, 0.2},  // Setosa
            {6.2, 2.9, 4.3, 1.3},  // Versicolor  
            {7.3, 2.9, 6.3, 1.8}   // Virginica
        };
        
        double[] predictions = trainer.predict(results, newSamples);
        
        String[] classes = {"Setosa", "Versicolor", "Virginica"};
        System.out.println("Predictions:");
        for (int i = 0; i < predictions.length; i++) {
            System.out.printf("Sample %d: %s\n", i + 1, classes[(int)predictions[i]]);
        }
    }
}
```

### Batch Processing Multiple Datasets

```java
public class BatchKaggleTraining {
    public static void main(String[] args) {
        var credentials = KaggleCredentials.fromDefaultLocation();
        var trainer = new KaggleTrainingManager(credentials);
        
        // Define datasets to process
        String[][] datasets = {
            {"titanic", "titanic", "survived"},
            {"uciml", "iris", "species"},
            {"uciml", "wine", "target"}
        };
        
        var config = new TrainingConfig()
            .setAlgorithms("logistic", "ridge")
            .setGridSearch(true)
            .setVerbose(false);  // Reduce output for batch processing
        
        try {
            for (String[] dataset : datasets) {
                String owner = dataset[0];
                String name = dataset[1]; 
                String target = dataset[2];
                
                System.out.println("Processing: " + owner + "/" + name);
                
                try {
                    var results = trainer.trainOnDataset(owner, name, target, config);
                    var best = results.get(0);
                    
                    System.out.printf("  Best: %s (%.4f)\n", 
                        best.algorithm, best.score);
                    
                } catch (Exception e) {
                    System.out.println("  Error: " + e.getMessage());
                }
            }
        } finally {
            trainer.close();
        }
    }
}
```

## 🚨 Error Handling & Troubleshooting

### Common Issues

```java
try {
    var results = trainer.trainOnDataset("owner", "dataset", "target");
} catch (Exception e) {
    if (e.getMessage().contains("credentials")) {
        System.err.println("Kaggle API credentials not found or invalid");
        System.err.println("Check ~/.kaggle/kaggle.json file");
    } else if (e.getMessage().contains("dataset not found")) {
        System.err.println("Dataset doesn't exist or is private");
        System.err.println("Verify owner/dataset names are correct");
    } else if (e.getMessage().contains("target column")) {
        System.err.println("Target column not found in dataset");
        System.err.println("Check available columns in the dataset");
    } else {
        System.err.println("Training error: " + e.getMessage());
        e.printStackTrace();
    }
}
```

### Dataset Validation

```java
// Check if dataset exists before training
try {
    var info = trainer.getDatasetInfo("owner", "dataset");
    System.out.println("Dataset found: " + info.title);
    System.out.println("Size: " + info.size + " bytes");
    System.out.println("Downloads: " + info.downloadCount);
    
    // Now safe to train
    var results = trainer.trainOnDataset("owner", "dataset", "target");
    
} catch (Exception e) {
    System.err.println("Dataset not available: " + e.getMessage());
}
```

## 🎯 Best Practices

### 1. Start with Popular Datasets

```java
// These are well-maintained and documented
trainer.trainOnDataset("titanic", "titanic", "survived");       // Classification
trainer.trainOnDataset("uciml", "iris", "species");             // Multiclass
trainer.trainOnDataset("housedata", "house-prices", "price");   // Regression
```

### 2. Use Appropriate Algorithms

```java
// For classification
var classificationConfig = new TrainingConfig()
    .setAlgorithms("logistic");  // Best general-purpose classifier

// For regression
var regressionConfig = new TrainingConfig()
    .setAlgorithms("linear", "ridge", "lasso");  // Compare regularization
```

### 3. Enable Preprocessing

```java
var config = new TrainingConfig()
    .setStandardScaler(true)        // Almost always helpful
    .setHandleMissingValues(true);  // Robust to data quality issues
```

### 4. Use Grid Search for Important Models

```java
var config = new TrainingConfig()
    .setGridSearch(true)            // Find optimal hyperparameters
    .setGridSearchCV(5);            // 5-fold cross-validation
```

### 5. Resource Management

```java
var trainer = new KaggleTrainingManager(credentials);
try {
    // Do training work
    var results = trainer.trainOnDataset("owner", "dataset", "target");
} finally {
    trainer.close();  // Always close to free resources
}
```

## 📈 Performance Tips

### 1. Parallel Processing

```java
// Process multiple algorithms in parallel
var config = new TrainingConfig()
    .setAlgorithms("logistic", "ridge", "lasso")  // Will train in parallel
    .setParallelTraining(true);
```

### 2. Caching

```java
// Downloaded datasets are cached locally
// Subsequent runs on same dataset will be faster
var config = new TrainingConfig()
    .setCacheDatasets(true)         // Enable caching
    .setCacheDir("./kaggle-cache"); // Custom cache location
```

### 3. Memory Management

```java
// For large datasets
var config = new TrainingConfig()
    .setMaxMemoryUsage(8_000_000_000L)  // 8GB limit
    .setStreamProcessing(true);          // Process in chunks
```

This comprehensive guide should get you started with Kaggle integration. The framework handles all the complexity of API calls, data preprocessing, and model training, letting you focus on building great ML applications! 🚀
