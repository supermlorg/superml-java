package com.superml.persistence;

import com.superml.core.BaseEstimator;
import com.superml.pipeline.Pipeline;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Utility class for managing collections of saved models.
 * Provides functionality for listing, organizing, and batch operations on model files.
 */
public class ModelManager {
    
    private static final Logger logger = LoggerFactory.getLogger(ModelManager.class);
    
    private final String modelsDirectory;
    
    /**
     * Create a ModelManager for the specified directory.
     * 
     * @param modelsDirectory directory where models are stored
     */
    public ModelManager(String modelsDirectory) {
        this.modelsDirectory = modelsDirectory;
        createModelsDirectory();
    }
    
    /**
     * Create a ModelManager using default "models" directory.
     */
    public ModelManager() {
        this("models");
    }
    
    /**
     * Save a model with automatic naming based on class and timestamp.
     * 
     * @param model the model to save
     * @param prefix optional prefix for the filename
     * @return the full path where the model was saved
     */
    public String saveModel(BaseEstimator model, String prefix) {
        String className = model.getClass().getSimpleName();
        String timestamp = java.time.LocalDateTime.now()
            .format(java.time.format.DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        
        String filename = prefix != null ? 
            String.format("%s_%s_%s", prefix, className, timestamp) :
            String.format("%s_%s", className, timestamp);
        
        String fullPath = Paths.get(modelsDirectory, filename).toString();
        ModelPersistence.save(model, fullPath);
        
        logger.info("Saved model to: {}", fullPath);
        return fullPath;
    }
    
    /**
     * Save a model with automatic naming.
     * 
     * @param model the model to save
     * @return the full path where the model was saved
     */
    public String saveModel(BaseEstimator model) {
        return saveModel(model, null);
    }
    
    /**
     * Load a model by filename (without path).
     * 
     * @param <T> the expected model type
     * @param filename the model filename
     * @param expectedClass expected class of the model
     * @return the loaded model
     */
    public <T extends BaseEstimator> T loadModel(String filename, Class<T> expectedClass) {
        String fullPath = Paths.get(modelsDirectory, filename).toString();
        return ModelPersistence.load(fullPath, expectedClass);
    }
    
    /**
     * List all model files in the models directory.
     * 
     * @return list of model filenames
     */
    public List<String> listModels() {
        try {
            Path dir = Paths.get(modelsDirectory);
            if (!Files.exists(dir)) {
                return List.of();
            }
            
            return Files.list(dir)
                .filter(path -> path.toString().endsWith(ModelPersistence.DEFAULT_EXTENSION))
                .map(path -> path.getFileName().toString())
                .sorted()
                .collect(Collectors.toList());
                
        } catch (Exception e) {
            logger.error("Failed to list models in directory: {}", modelsDirectory, e);
            return List.of();
        }
    }
    
    /**
     * Get detailed information about all models in the directory.
     * 
     * @return list of model information
     */
    public List<ModelInfo> getModelsInfo() {
        return listModels().stream()
            .map(filename -> {
                try {
                    String fullPath = Paths.get(modelsDirectory, filename).toString();
                    ModelPersistence.ModelMetadata metadata = ModelPersistence.getMetadata(fullPath);
                    long fileSize = ModelPersistence.getFileSize(fullPath);
                    
                    return new ModelInfo(filename, metadata, fileSize);
                } catch (Exception e) {
                    logger.warn("Failed to read metadata for model: {}", filename, e);
                    return null;
                }
            })
            .filter(info -> info != null)
            .collect(Collectors.toList());
    }
    
    /**
     * Delete a model file.
     * 
     * @param filename the model filename to delete
     * @return true if deleted successfully
     */
    public boolean deleteModel(String filename) {
        String fullPath = Paths.get(modelsDirectory, filename).toString();
        return ModelPersistence.delete(fullPath);
    }
    
    /**
     * Find models by class type.
     * 
     * @param className the class name to search for
     * @return list of matching model filenames
     */
    public List<String> findModelsByClass(String className) {
        return getModelsInfo().stream()
            .filter(info -> info.metadata.modelClass.contains(className))
            .map(info -> info.filename)
            .collect(Collectors.toList());
    }
    
    /**
     * Clean up old models, keeping only the most recent N models of each type.
     * 
     * @param keepCount number of recent models to keep per type
     * @return number of models deleted
     */
    public int cleanupOldModels(int keepCount) {
        List<ModelInfo> models = getModelsInfo();
        
        // Group by model class
        var modelsByClass = models.stream()
            .collect(Collectors.groupingBy(
                info -> info.metadata.modelClass,
                Collectors.toList()
            ));
        
        int deletedCount = 0;
        
        for (List<ModelInfo> classModels : modelsByClass.values()) {
            // Sort by save time (newest first)
            classModels.sort((a, b) -> b.metadata.savedAt.compareTo(a.metadata.savedAt));
            
            // Delete old models beyond keepCount
            for (int i = keepCount; i < classModels.size(); i++) {
                ModelInfo oldModel = classModels.get(i);
                if (deleteModel(oldModel.filename)) {
                    deletedCount++;
                    logger.info("Deleted old model: {}", oldModel.filename);
                }
            }
        }
        
        return deletedCount;
    }
    
    /**
     * Get the models directory path.
     * 
     * @return the models directory path
     */
    public String getModelsDirectory() {
        return modelsDirectory;
    }
    
    private void createModelsDirectory() {
        try {
            Files.createDirectories(Paths.get(modelsDirectory));
        } catch (Exception e) {
            throw new ModelPersistenceException("Failed to create models directory: " + modelsDirectory, e);
        }
    }
    
    /**
     * Save a model with automatic training statistics capture.
     * 
     * @param model the model to save
     * @param prefix optional prefix for the filename
     * @param XTest test features for automatic evaluation
     * @param yTest test targets for automatic evaluation
     * @return the full path where the model was saved
     */
    public String saveModelWithStats(BaseEstimator model, String prefix, 
                                   double[][] XTest, double[] yTest) {
        return saveModelWithStats(model, prefix, null, XTest, yTest, null);
    }
    
    /**
     * Save a model with automatic training statistics capture and custom metadata.
     * 
     * @param model the model to save
     * @param prefix optional prefix for the filename
     * @param description optional description
     * @param XTest test features for automatic evaluation
     * @param yTest test targets for automatic evaluation
     * @param customMetadata optional custom metadata
     * @return the full path where the model was saved
     */
    public String saveModelWithStats(BaseEstimator model, String prefix, String description,
                                   double[][] XTest, double[] yTest, Map<String, Object> customMetadata) {
        String className = model.getClass().getSimpleName();
        String timestamp = java.time.LocalDateTime.now()
            .format(java.time.format.DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        
        String filename = prefix != null ? 
            String.format("%s_%s_%s", prefix, className, timestamp) :
            String.format("%s_%s", className, timestamp);
        
        String fullPath = Paths.get(modelsDirectory, filename).toString();
        
        // Use enhanced save with automatic statistics
        ModelPersistence.saveWithStats(model, fullPath, description, XTest, yTest, customMetadata);
        
        logger.info("Saved model with training statistics to: {}", fullPath);
        return fullPath;
    }
    
    /**
     * Information about a saved model.
     */
    public static class ModelInfo {
        public final String filename;
        public final ModelPersistence.ModelMetadata metadata;
        public final long fileSizeBytes;
        
        public ModelInfo(String filename, ModelPersistence.ModelMetadata metadata, long fileSizeBytes) {
            this.filename = filename;
            this.metadata = metadata;
            this.fileSizeBytes = fileSizeBytes;
        }
        
        /**
         * Get file size in human-readable format.
         * 
         * @return formatted file size
         */
        public String getFormattedFileSize() {
            if (fileSizeBytes < 1024) {
                return fileSizeBytes + " B";
            } else if (fileSizeBytes < 1024 * 1024) {
                return String.format("%.1f KB", fileSizeBytes / 1024.0);
            } else {
                return String.format("%.1f MB", fileSizeBytes / (1024.0 * 1024.0));
            }
        }
        
        @Override
        public String toString() {
            return String.format("ModelInfo{file=%s, class=%s, size=%s, saved=%s}", 
                filename, 
                metadata.modelClass.substring(metadata.modelClass.lastIndexOf('.') + 1),
                getFormattedFileSize(),
                metadata.savedAt.format(java.time.format.DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm"))
            );
        }
    }
}
