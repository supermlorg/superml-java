package com.superml.inference;

/**
 * Exception thrown when inference operations fail.
 */
public class InferenceException extends RuntimeException {
    
    /**
     * Create exception with message.
     * @param message error message
     */
    public InferenceException(String message) {
        super(message);
    }
    
    /**
     * Create exception with message and cause.
     * @param message error message
     * @param cause underlying cause
     */
    public InferenceException(String message, Throwable cause) {
        super(message, cause);
    }
    
    /**
     * Create exception with cause.
     * @param cause underlying cause
     */
    public InferenceException(Throwable cause) {
        super(cause);
    }
}
