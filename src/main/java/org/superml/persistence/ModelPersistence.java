package org.superml.persistence;

import org.superml.core.BaseEstimator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.Map;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * Model persistence utilities for saving and loading trained models.
 * Provides functionality similar to scikit-learn's joblib.dump() and joblib.load().
 * 
 * Features:
 * - Automatic compression with GZIP
 * - Model metadata and versioning
 * - Type-safe model loading
 * - Cross-platform file paths
 * - Comprehensive error handling
 * 
 * Usage:
 * <pre>
 * // Save a model
 * LogisticRegression model = new LogisticRegression();
 * model.fit(X, y);
 * ModelPersistence.save(model, "my_model.superml");
 * 
 * // Load a model
 * LogisticRegression loadedModel = ModelPersistence.load("my_model.superml", LogisticRegression.class);
 * double[] predictions = loadedModel.predict(X_test);
 * </pre>
 */
public class ModelPersistence {
    
    private static final Logger logger = LoggerFactory.getLogger(ModelPersistence.class);
    
    /** Default file extension for SuperML model files */
    public static final String DEFAULT_EXTENSION = ".superml";
    
    /** Magic header to identify SuperML model files */
    private static final String MAGIC_HEADER = "SUPERML_MODEL_V1";
    
    /** Current format version for backwards compatibility */
    private static final int FORMAT_VERSION = 1;
    
    /**
     * Container for model metadata.
     */
    public static class ModelMetadata implements Serializable {
        private static final long serialVersionUID = 1L;
        
        public final String modelClass;
        public final String supermlVersion;
        public final LocalDateTime savedAt;
        public final String description;
        public final Map<String, Object> customMetadata;
        
        public ModelMetadata(String modelClass, String description, Map<String, Object> customMetadata) {
            this.modelClass = modelClass;
            this.supermlVersion = "1.0-SNAPSHOT";
            this.savedAt = LocalDateTime.now();
            this.description = description != null ? description : "";
            this.customMetadata = customMetadata != null ? new HashMap<>(customMetadata) : new HashMap<>();
        }
        
        @Override
        public String toString() {
            return String.format("ModelMetadata{class=%s, version=%s, saved=%s}", 
                modelClass, supermlVersion, savedAt.format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
        }
    }
    
    /**
     * Save a trained model to file with compression.
     * 
     * @param model the trained model to save
     * @param filePath path where to save the model
     * @throws ModelPersistenceException if saving fails
     */
    public static void save(BaseEstimator model, String filePath) {
        save(model, filePath, null, null);
    }
    
    /**
     * Save a trained model to file with optional description and metadata.
     * 
     * @param model the trained model to save
     * @param filePath path where to save the model
     * @param description optional description of the model
     * @param customMetadata optional custom metadata
     * @throws ModelPersistenceException if saving fails
     */
    public static void save(BaseEstimator model, String filePath, String description, 
                          Map<String, Object> customMetadata) {
        if (model == null) {
            throw new IllegalArgumentException("Model cannot be null");
        }
        if (filePath == null || filePath.trim().isEmpty()) {
            throw new IllegalArgumentException("File path cannot be null or empty");
        }
        
        // Ensure .superml extension
        String normalizedPath = ensureExtension(filePath);
        
        // Create directories if needed
        createDirectories(normalizedPath);
        
        logger.info("Saving model {} to {}", model.getClass().getSimpleName(), normalizedPath);
        
        try (FileOutputStream fos = new FileOutputStream(normalizedPath);
             GZIPOutputStream gzos = new GZIPOutputStream(fos);
             ObjectOutputStream oos = new ObjectOutputStream(gzos)) {
            
            // Write magic header
            oos.writeUTF(MAGIC_HEADER);
            oos.writeInt(FORMAT_VERSION);
            
            // Write metadata
            ModelMetadata metadata = new ModelMetadata(
                model.getClass().getName(), description, customMetadata);
            oos.writeObject(metadata);
            
            // Write the actual model
            oos.writeObject(model);
            oos.flush();
            
            logger.info("Successfully saved model to {}", normalizedPath);
            
        } catch (IOException e) {
            throw new ModelPersistenceException("Failed to save model to " + normalizedPath, e);
        }
    }
    
    /**
     * Load a model from file with type checking.
     * 
     * @param <T> the expected model type
     * @param filePath path to the saved model file
     * @param expectedClass expected class of the model
     * @return the loaded model
     * @throws ModelPersistenceException if loading fails or type mismatch
     */
    @SuppressWarnings("unchecked")
    public static <T extends BaseEstimator> T load(String filePath, Class<T> expectedClass) {
        if (filePath == null || filePath.trim().isEmpty()) {
            throw new IllegalArgumentException("File path cannot be null or empty");
        }
        if (expectedClass == null) {
            throw new IllegalArgumentException("Expected class cannot be null");
        }
        
        String normalizedPath = ensureExtension(filePath);
        
        if (!Files.exists(Paths.get(normalizedPath))) {
            throw new ModelPersistenceException("Model file not found: " + normalizedPath);
        }
        
        logger.info("Loading model from {}", normalizedPath);
        
        try (FileInputStream fis = new FileInputStream(normalizedPath);
             GZIPInputStream gzis = new GZIPInputStream(fis);
             ObjectInputStream ois = new ObjectInputStream(gzis)) {
            
            // Verify magic header
            String header = ois.readUTF();
            if (!MAGIC_HEADER.equals(header)) {
                throw new ModelPersistenceException("Invalid model file format: " + normalizedPath);
            }
            
            int version = ois.readInt();
            if (version > FORMAT_VERSION) {
                throw new ModelPersistenceException(
                    "Model file version " + version + " is newer than supported version " + FORMAT_VERSION);
            }
            
            // Read metadata
            ModelMetadata metadata = (ModelMetadata) ois.readObject();
            logger.debug("Loading model: {}", metadata);
            
            // Read and verify model
            Object modelObj = ois.readObject();
            
            if (!expectedClass.isInstance(modelObj)) {
                throw new ModelPersistenceException(
                    String.format("Expected %s but loaded %s", 
                        expectedClass.getSimpleName(), modelObj.getClass().getSimpleName()));
            }
            
            T model = (T) modelObj;
            logger.info("Successfully loaded {} from {}", expectedClass.getSimpleName(), normalizedPath);
            
            return model;
            
        } catch (IOException | ClassNotFoundException e) {
            throw new ModelPersistenceException("Failed to load model from " + normalizedPath, e);
        }
    }
    
    /**
     * Load a model without type checking (returns BaseEstimator).
     * 
     * @param filePath path to the saved model file
     * @return the loaded model as BaseEstimator
     * @throws ModelPersistenceException if loading fails
     */
    public static BaseEstimator load(String filePath) {
        return load(filePath, BaseEstimator.class);
    }
    
    /**
     * Get metadata from a model file without loading the full model.
     * 
     * @param filePath path to the model file
     * @return model metadata
     * @throws ModelPersistenceException if reading metadata fails
     */
    public static ModelMetadata getMetadata(String filePath) {
        if (filePath == null || filePath.trim().isEmpty()) {
            throw new IllegalArgumentException("File path cannot be null or empty");
        }
        
        String normalizedPath = ensureExtension(filePath);
        
        if (!Files.exists(Paths.get(normalizedPath))) {
            throw new ModelPersistenceException("Model file not found: " + normalizedPath);
        }
        
        try (FileInputStream fis = new FileInputStream(normalizedPath);
             GZIPInputStream gzis = new GZIPInputStream(fis);
             ObjectInputStream ois = new ObjectInputStream(gzis)) {
            
            // Verify magic header
            String header = ois.readUTF();
            if (!MAGIC_HEADER.equals(header)) {
                throw new ModelPersistenceException("Invalid model file format: " + normalizedPath);
            }
            
            int version = ois.readInt();
            if (version > FORMAT_VERSION) {
                throw new ModelPersistenceException(
                    "Model file version " + version + " is newer than supported version " + FORMAT_VERSION);
            }
            
            // Read and return metadata
            return (ModelMetadata) ois.readObject();
            
        } catch (IOException | ClassNotFoundException e) {
            throw new ModelPersistenceException("Failed to read metadata from " + normalizedPath, e);
        }
    }
    
    /**
     * Check if a file is a valid SuperML model file.
     * 
     * @param filePath path to check
     * @return true if valid SuperML model file
     */
    public static boolean isValidModelFile(String filePath) {
        try {
            getMetadata(filePath);
            return true;
        } catch (Exception e) {
            return false;
        }
    }
    
    /**
     * Get file size of a model file in bytes.
     * 
     * @param filePath path to the model file
     * @return file size in bytes
     * @throws ModelPersistenceException if file doesn't exist or can't be read
     */
    public static long getFileSize(String filePath) {
        String normalizedPath = ensureExtension(filePath);
        try {
            return Files.size(Paths.get(normalizedPath));
        } catch (IOException e) {
            throw new ModelPersistenceException("Failed to get file size: " + normalizedPath, e);
        }
    }
    
    /**
     * Delete a model file.
     * 
     * @param filePath path to the model file to delete
     * @return true if file was deleted, false if it didn't exist
     * @throws ModelPersistenceException if deletion fails
     */
    public static boolean delete(String filePath) {
        String normalizedPath = ensureExtension(filePath);
        try {
            return Files.deleteIfExists(Paths.get(normalizedPath));
        } catch (IOException e) {
            throw new ModelPersistenceException("Failed to delete model file: " + normalizedPath, e);
        }
    }
    
    /**
     * Save a trained model with automatic training statistics capture.
     * 
     * @param model the trained model to save
     * @param filePath path where to save the model
     * @param XTest test features for automatic evaluation
     * @param yTest test targets for automatic evaluation
     * @throws ModelPersistenceException if saving fails
     */
    public static void saveWithStats(BaseEstimator model, String filePath, 
                                   double[][] XTest, double[] yTest) {
        saveWithStats(model, filePath, null, XTest, yTest, null);
    }
    
    /**
     * Save a trained model with automatic training statistics capture and custom metadata.
     * 
     * @param model the trained model to save
     * @param filePath path where to save the model
     * @param description optional description of the model
     * @param XTest test features for automatic evaluation
     * @param yTest test targets for automatic evaluation
     * @param customMetadata optional custom metadata
     * @throws ModelPersistenceException if saving fails
     */
    public static void saveWithStats(BaseEstimator model, String filePath, String description,
                                   double[][] XTest, double[] yTest, Map<String, Object> customMetadata) {
        // Automatically capture training statistics
        Map<String, Object> autoMetadata = new HashMap<>();
        
        if (customMetadata != null) {
            autoMetadata.putAll(customMetadata);
        }
        
        try {
            // Capture model parameters
            autoMetadata.put("model_parameters", model.getParams());
            
            // Capture dataset statistics
            if (XTest != null && yTest != null) {
                autoMetadata.put("test_samples", XTest.length);
                autoMetadata.put("test_features", XTest.length > 0 ? XTest[0].length : 0);
                
                // Calculate performance metrics if model supports prediction
                if (model instanceof org.superml.core.SupervisedLearner) {
                    org.superml.core.SupervisedLearner supervisor = (org.superml.core.SupervisedLearner) model;
                    double[] predictions = supervisor.predict(XTest);
                    
                    // Determine if classification or regression
                    boolean isClassification = isClassificationTask(yTest);
                    
                    if (isClassification) {
                        autoMetadata.put("task_type", "classification");
                        autoMetadata.put("test_accuracy", org.superml.metrics.Metrics.accuracy(yTest, predictions));
                        
                        // Add precision, recall, f1 if binary classification
                        java.util.Set<Double> uniqueClasses = java.util.Arrays.stream(yTest)
                            .boxed().collect(java.util.stream.Collectors.toSet());
                        if (uniqueClasses.size() == 2) {
                            autoMetadata.put("test_precision", org.superml.metrics.Metrics.precision(yTest, predictions));
                            autoMetadata.put("test_recall", org.superml.metrics.Metrics.recall(yTest, predictions));
                            autoMetadata.put("test_f1_score", org.superml.metrics.Metrics.f1Score(yTest, predictions));
                        }
                        autoMetadata.put("unique_classes", uniqueClasses.size());
                    } else {
                        autoMetadata.put("task_type", "regression");
                        autoMetadata.put("test_mse", org.superml.metrics.Metrics.meanSquaredError(yTest, predictions));
                        autoMetadata.put("test_mae", org.superml.metrics.Metrics.meanAbsoluteError(yTest, predictions));
                        autoMetadata.put("test_r2", org.superml.metrics.Metrics.r2Score(yTest, predictions));
                    }
                    
                    // Add prediction statistics
                    autoMetadata.put("prediction_mean", java.util.Arrays.stream(predictions).average().orElse(0.0));
                    autoMetadata.put("prediction_std", calculateStandardDeviation(predictions));
                }
            }
            
            // Add system information
            autoMetadata.put("java_version", System.getProperty("java.version"));
            autoMetadata.put("os_name", System.getProperty("os.name"));
            autoMetadata.put("save_timestamp_epoch", System.currentTimeMillis());
            
        } catch (Exception e) {
            logger.warn("Failed to capture automatic training statistics: {}", e.getMessage());
            // Continue with saving even if stats capture fails
        }
        
        save(model, filePath, description, autoMetadata);
    }
    
    /**
     * Determine if the task is classification based on target values.
     */
    private static boolean isClassificationTask(double[] y) {
        java.util.Set<Double> uniqueValues = java.util.Arrays.stream(y)
            .boxed().collect(java.util.stream.Collectors.toSet());
        
        // If there are 10 or fewer unique values and they are all integers, treat as classification
        if (uniqueValues.size() <= 10) {
            return uniqueValues.stream().allMatch(val -> val == Math.floor(val));
        }
        return false;
    }
    
    /**
     * Calculate standard deviation of an array.
     */
    private static double calculateStandardDeviation(double[] values) {
        double mean = java.util.Arrays.stream(values).average().orElse(0.0);
        double variance = java.util.Arrays.stream(values)
            .map(val -> Math.pow(val - mean, 2))
            .average().orElse(0.0);
        return Math.sqrt(variance);
    }
    
    // Helper methods
    
    private static String ensureExtension(String filePath) {
        if (!filePath.toLowerCase().endsWith(DEFAULT_EXTENSION)) {
            return filePath + DEFAULT_EXTENSION;
        }
        return filePath;
    }
    
    private static void createDirectories(String filePath) {
        Path path = Paths.get(filePath);
        Path parentDir = path.getParent();
        if (parentDir != null) {
            try {
                Files.createDirectories(parentDir);
            } catch (IOException e) {
                throw new ModelPersistenceException("Failed to create directories for: " + filePath, e);
            }
        }
    }
}
