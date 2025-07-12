package org.superml.persistence;

/**
 * Exception thrown when model persistence operations fail.
 * This includes saving, loading, and metadata operations.
 */
public class ModelPersistenceException extends RuntimeException {
    
    private static final long serialVersionUID = 1L;
    
    /**
     * Constructs a new ModelPersistenceException with the specified detail message.
     * 
     * @param message the detail message
     */
    public ModelPersistenceException(String message) {
        super(message);
    }
    
    /**
     * Constructs a new ModelPersistenceException with the specified detail message and cause.
     * 
     * @param message the detail message
     * @param cause the cause
     */
    public ModelPersistenceException(String message, Throwable cause) {
        super(message, cause);
    }
    
    /**
     * Constructs a new ModelPersistenceException with the specified cause.
     * 
     * @param cause the cause
     */
    public ModelPersistenceException(Throwable cause) {
        super(cause);
    }
}
