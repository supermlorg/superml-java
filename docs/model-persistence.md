# Model Persistence Guide

Complete guide to saving, loading, and managing trained models in SuperML Java.

## 🎯 Overview

SuperML Java provides comprehensive model persistence functionality that allows you to:

- **Save trained models** to disk with GZIP compression
- **Load models** with type safety and validation
- **Store metadata** alongside models for tracking and documentation
- **Manage collections** of models with automated organization
- **Version control** models with timestamps and descriptions
- **Cross-platform compatibility** with standard file formats

## 🚀 Quick Start

### Basic Save and Load

```java
import org.superml.persistence.ModelPersistence;
import org.superml.linear_model.LogisticRegression;

// Train your model
LogisticRegression model = new LogisticRegression().setMaxIter(1000);
model.fit(X_train, y_train);

// Save model (automatically adds .superml extension)
ModelPersistence.save(model, "my_classifier");

// Load model with type checking
LogisticRegression loadedModel = ModelPersistence.load("my_classifier", LogisticRegression.class);

// Use loaded model
double[] predictions = loadedModel.predict(X_test);
```

### Save with Metadata

```java
import java.util.Map;

// Create rich metadata
Map<String, Object> metadata = Map.of(
    "accuracy", 0.95,
    "dataset", "iris",
    "features", 4,
    "samples", 150,
    "algorithm", "LogisticRegression",
    "hyperparameters", model.getParams()
);

// Save with description and metadata
ModelPersistence.save(model, "iris_classifier", 
                     "Production iris classification model", metadata);
```

## 📦 Model File Format

SuperML models are saved in a custom format with the following features:

- **File Extension**: `.superml`
- **Compression**: GZIP compression for smaller file sizes
- **Magic Header**: Format validation and version checking
- **Metadata**: JSON-serializable custom metadata
- **Cross-Platform**: Works across different operating systems

### File Structure

```
SuperML Model File (.superml)
├── Magic Header ("SUPERML_MODEL_V1")
├── Format Version (integer)
├── Model Metadata (serialized object)
│   ├── Model class name
│   ├── SuperML version
│   ├── Save timestamp
│   ├── Description
│   └── Custom metadata map
└── Model Object (serialized with compression)
```

## 🔧 Advanced Usage

### Pipeline Persistence

```java
import org.superml.pipeline.Pipeline;
import org.superml.preprocessing.StandardScaler;

// Create and train pipeline
Pipeline pipeline = new Pipeline()
    .addStep("scaler", new StandardScaler())
    .addStep("classifier", new LogisticRegression());

pipeline.fit(X_train, y_train);

// Save pipeline (all steps are preserved)
ModelPersistence.save(pipeline, "preprocessing_pipeline");

// Load and use pipeline
Pipeline loadedPipeline = ModelPersistence.load("preprocessing_pipeline", Pipeline.class);
double[] predictions = loadedPipeline.predict(X_test);
```

### Type-Safe Loading

```java
// Correct type - works
LogisticRegression model = ModelPersistence.load("classifier", LogisticRegression.class);

// Wrong type - throws exception
try {
    LinearRegression wrongType = ModelPersistence.load("classifier", LinearRegression.class);
} catch (ModelPersistenceException e) {
    System.out.println("Type mismatch: " + e.getMessage());
}

// Load without type checking (returns BaseEstimator)
BaseEstimator genericModel = ModelPersistence.load("classifier");
```

### Metadata Operations

```java
// Read metadata without loading full model
ModelPersistence.ModelMetadata metadata = ModelPersistence.getMetadata("classifier");

System.out.println("Model class: " + metadata.modelClass);
System.out.println("Saved at: " + metadata.savedAt);
System.out.println("Description: " + metadata.description);

// Access custom metadata
Object accuracy = metadata.customMetadata.get("accuracy");
Object dataset = metadata.customMetadata.get("dataset");
```

### File Operations

```java
// Check if file is valid SuperML model
boolean isValid = ModelPersistence.isValidModelFile("my_model");

// Get file size
long sizeBytes = ModelPersistence.getFileSize("my_model");

// Delete model file
boolean deleted = ModelPersistence.delete("my_model");
```

## 📁 Model Management

For managing collections of models, use the `ModelManager` class:

### Basic Model Manager

```java
import org.superml.persistence.ModelManager;

// Create manager (uses "models" directory by default)
ModelManager manager = new ModelManager();

// Or specify custom directory
ModelManager manager = new ModelManager("my_models");
```

### Automated Model Saving

```java
// Save with automatic naming (includes timestamp)
String savedPath = manager.saveModel(model, "iris");
// Result: models/iris_LogisticRegression_20250711_143022.superml

// Save without prefix
String savedPath = manager.saveModel(model);
// Result: models/LogisticRegression_20250711_143022.superml
```

### Model Discovery and Information

```java
// List all model files
List<String> modelFiles = manager.listModels();

// Get detailed information about all models
List<ModelManager.ModelInfo> modelsInfo = manager.getModelsInfo();
for (ModelManager.ModelInfo info : modelsInfo) {
    System.out.printf("File: %s, Class: %s, Size: %s, Saved: %s%n",
        info.filename,
        info.metadata.modelClass,
        info.getFormattedFileSize(),
        info.metadata.savedAt);
}

// Find models by class type
List<String> logisticModels = manager.findModelsByClass("LogisticRegression");
```

### Model Cleanup

```java
// Delete specific model
boolean deleted = manager.deleteModel("old_model.superml");

// Clean up old models (keep only 3 most recent per type)
int deletedCount = manager.cleanupOldModels(3);
System.out.println("Deleted " + deletedCount + " old models");
```

## 🔗 Kaggle Integration

Enable automatic model saving during Kaggle training:

```java
import org.superml.datasets.KaggleTrainingManager;

KaggleTrainingManager.TrainingConfig config = new KaggleTrainingManager.TrainingConfig()
    .setSaveModels(true)                    // Enable model saving
    .setModelsDirectory("kaggle_models")    // Custom directory
    .setAlgorithms("logistic", "ridge")     // Train multiple algorithms
    .setGridSearch(true);                   // Use hyperparameter tuning

List<KaggleTrainingManager.TrainingResult> results = 
    trainer.trainOnDataset("titanic", "titanic", "survived", config);

// Each result contains the trained model and save path
for (KaggleTrainingManager.TrainingResult result : results) {
    System.out.println("Algorithm: " + result.algorithm);
    System.out.println("Score: " + result.score);
    System.out.println("Saved to: " + result.modelFilePath);
}
```

## 🛠️ Best Practices

### 1. Use Descriptive Names and Metadata

```java
Map<String, Object> metadata = Map.of(
    "purpose", "Production fraud detection",
    "dataset_version", "v2.1",
    "accuracy", 0.97,
    "precision", 0.95,
    "recall", 0.93,
    "training_date", LocalDate.now().toString(),
    "features", Arrays.asList("amount", "merchant", "time", "location"),
    "validation_strategy", "5-fold CV"
);

ModelPersistence.save(model, "fraud_detector_v2", 
                     "Production fraud detection model v2.1", metadata);
```

### 2. Organize Models by Project

```java
// Use project-specific directories
ModelManager productionModels = new ModelManager("production");
ModelManager experimentModels = new ModelManager("experiments");
ModelManager kaggleModels = new ModelManager("competitions/titanic");
```

### 3. Version Control Integration

```java
// Include git commit info in metadata
Map<String, Object> metadata = new HashMap<>();
metadata.put("git_commit", getCurrentGitCommit());
metadata.put("git_branch", getCurrentGitBranch());
metadata.put("code_version", "v1.2.3");

ModelPersistence.save(model, "model_v1_2_3", "Release version 1.2.3", metadata);
```

### 4. Model Validation After Loading

```java
// Load model
LogisticRegression loadedModel = ModelPersistence.load("classifier", LogisticRegression.class);

// Validate on test set
double[] testPredictions = loadedModel.predict(X_test);
double testAccuracy = Metrics.accuracy(y_test, testPredictions);

// Compare with expected performance
ModelPersistence.ModelMetadata metadata = ModelPersistence.getMetadata("classifier");
double expectedAccuracy = (Double) metadata.customMetadata.get("accuracy");

if (Math.abs(testAccuracy - expectedAccuracy) > 0.01) {
    System.out.println("WARNING: Model performance differs from expected!");
}
```

### 5. Model Lifecycle Management

```java
public class ModelLifecycleManager {
    private final ModelManager manager;
    
    public ModelLifecycleManager(String baseDir) {
        this.manager = new ModelManager(baseDir);
    }
    
    public String promoteToProduction(String candidateModel, double minAccuracy) {
        // Load and validate candidate
        BaseEstimator model = ModelPersistence.load(candidateModel);
        ModelPersistence.ModelMetadata meta = ModelPersistence.getMetadata(candidateModel);
        
        Double accuracy = (Double) meta.customMetadata.get("accuracy");
        if (accuracy < minAccuracy) {
            throw new IllegalArgumentException("Model accuracy too low: " + accuracy);
        }
        
        // Save to production directory
        ModelManager prodManager = new ModelManager("production");
        return prodManager.saveModel(model, "current");
    }
}
```

## ⚠️ Error Handling

### Common Exceptions

```java
try {
    // Model operations
    ModelPersistence.save(model, "test_model");
    LogisticRegression loaded = ModelPersistence.load("test_model", LogisticRegression.class);
    
} catch (ModelPersistenceException e) {
    // Handle persistence errors
    System.err.println("Model persistence error: " + e.getMessage());
    
} catch (IllegalArgumentException e) {
    // Handle invalid arguments (null model, empty path, etc.)
    System.err.println("Invalid argument: " + e.getMessage());
}
```

### File System Issues

```java
// Check if model exists before loading
String modelPath = "my_model";
if (!ModelPersistence.isValidModelFile(modelPath)) {
    System.out.println("Model file not found or invalid: " + modelPath);
    return;
}

// Handle permissions and disk space
try {
    ModelPersistence.save(largeModel, "large_model");
} catch (ModelPersistenceException e) {
    if (e.getMessage().contains("No space left")) {
        // Handle disk space issues
        cleanupOldModels();
        ModelPersistence.save(largeModel, "large_model");
    }
}
```

## 🔍 Performance Considerations

### File Size Optimization

- Models are automatically compressed with GZIP
- Typical compression ratios: 70-90% size reduction
- Pipeline models are larger due to multiple components
- Metadata adds minimal overhead (~100-500 bytes)

### Loading Performance

- Decompression is fast (usually < 100ms for typical models)
- Metadata can be read without loading the full model
- Type checking adds minimal overhead
- Consider model size for frequently-loaded models

### Memory Usage

- Only one copy of the model is kept in memory after loading
- Serialization creates temporary copies during save/load
- Large pipelines with multiple steps use more memory
- Use `ModelManager.getModelsInfo()` to check sizes before loading

## 📚 API Reference

### ModelPersistence Class

**Static Methods:**

- `save(BaseEstimator model, String filePath)` - Basic model saving
- `save(BaseEstimator model, String filePath, String description, Map<String, Object> metadata)` - Save with metadata
- `<T> T load(String filePath, Class<T> expectedClass)` - Type-safe loading
- `BaseEstimator load(String filePath)` - Generic loading
- `ModelMetadata getMetadata(String filePath)` - Read metadata only
- `boolean isValidModelFile(String filePath)` - Validate model file
- `long getFileSize(String filePath)` - Get file size
- `boolean delete(String filePath)` - Delete model file

### ModelManager Class

**Constructor:**
- `ModelManager()` - Use default "models" directory
- `ModelManager(String directory)` - Use custom directory

**Methods:**
- `String saveModel(BaseEstimator model, String prefix)` - Save with prefix
- `String saveModel(BaseEstimator model)` - Save with auto-naming
- `<T> T loadModel(String filename, Class<T> expectedClass)` - Load by filename
- `List<String> listModels()` - List all model files
- `List<ModelInfo> getModelsInfo()` - Get detailed model information
- `List<String> findModelsByClass(String className)` - Find by class type
- `boolean deleteModel(String filename)` - Delete specific model
- `int cleanupOldModels(int keepCount)` - Clean up old models

### ModelMetadata Class

**Properties:**
- `String modelClass` - Full class name
- `String supermlVersion` - Framework version
- `LocalDateTime savedAt` - Save timestamp
- `String description` - User description
- `Map<String, Object> customMetadata` - Custom metadata map

## 🔧 Integration Examples

### Spring Boot Integration

```java
@Service
public class ModelService {
    private final ModelManager modelManager;
    
    public ModelService(@Value("${app.models.directory}") String modelsDir) {
        this.modelManager = new ModelManager(modelsDir);
    }
    
    @PostConstruct
    public void loadProductionModel() {
        // Load current production model
        List<String> models = modelManager.findModelsByClass("LogisticRegression");
        if (!models.isEmpty()) {
            this.productionModel = modelManager.loadModel(models.get(0), LogisticRegression.class);
        }
    }
    
    public double[] predict(double[][] features) {
        return productionModel.predict(features);
    }
}
```

### Model A/B Testing

```java
public class ModelABTester {
    private final BaseEstimator modelA;
    private final BaseEstimator modelB;
    
    public ModelABTester(String modelAPath, String modelBPath) {
        this.modelA = ModelPersistence.load(modelAPath);
        this.modelB = ModelPersistence.load(modelBPath);
    }
    
    public double[] predict(double[][] features, boolean useModelB) {
        return useModelB ? modelB.predict(features) : modelA.predict(features);
    }
}
```

---

For more examples and advanced usage, see the [SuperML Examples](examples/basic-examples.md) documentation.

### Automatic Training Statistics Capture

The framework can automatically capture comprehensive training statistics when saving models:

```java
// Prepare test data for automatic evaluation
double[][] X_test = /* your test features */;
double[] y_test = /* your test targets */;

// Save with automatic statistics capture
ModelPersistence.saveWithStats(model, "models/iris_classifier", 
                               "Model with comprehensive stats", 
                               X_test, y_test);

// The framework automatically captures and saves:
// 📊 Performance Metrics:
//   - Classification: accuracy, precision, recall, F1-score, confusion matrix
//   - Regression: MSE, MAE, R-squared, residual statistics
// 
// 📈 Dataset Statistics:
//   - Sample count and feature count
//   - Feature value ranges and distributions
//   - Label distribution for classification
//
// ⚙️ Model Configuration:
//   - All hyperparameters and model parameters
//   - Algorithm name and task type
//   - Training configuration details
//
// 🖥️ System Information:
//   - Java version and operating system
//   - Timestamp and SuperML version
//   - Memory usage during training
```

### Enhanced Model Manager

```java
import org.superml.persistence.ModelManager;

// Initialize model manager with automatic organization
ModelManager manager = new ModelManager("./trained_models");

// Save with automatic statistics and organized file naming
String savedPath = manager.saveModelWithStats(model, "production", 
                                             "Final production model",
                                             X_test, y_test);

// The ModelManager automatically:
// - Generates unique filenames with timestamps
// - Organizes models in the specified directory
// - Captures training statistics
// - Provides model lifecycle management
```
